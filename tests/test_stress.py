"""Prueba de carga: observación del turno, snapshot, escenarios, PDFs, runner e informe."""

import csv
from unittest.mock import patch

import httpx
import pytest

from app.config import get_settings
from app.main import app
from app.services import llm_client
from app.services.attachments import extract_text
from app.services.compression import ConversationSummary
from app.sessions import store
from evals.stress import report, run
from evals.stress.fixtures import build_pdfs
from evals.stress.scenarios import SCENARIOS, Scenario, ScenarioTurn
from tests.factories import make_info, make_phase, make_result

FORM = {"project_type": "web_saas", "detail_level": "summary", "output_format": "phases_table"}


async def fake_llm(messages, response_model, max_tokens):
    """Estimación (que menciona Navision si llega en el mensaje) o resumen."""
    if response_model is ConversationSummary:
        return ConversationSummary(
            summary="Resumen: Vega, presupuesto de 30.000 euros."
        ), make_info()
    user = messages[-1]["content"]
    technologies = ["Navision"] if "Navision" in user else ["Stripe"]
    result = make_result(
        project_name="Vega",
        technologies=technologies,
        phases=[make_phase("Diseño", 20), make_phase("Desarrollo", 60)],
    )
    return result, make_info(cost_usd=0.001)


@pytest.fixture(autouse=True)
def clean_store():
    store.clear()
    yield
    store.clear()


@pytest.fixture
def short_window(monkeypatch):
    monkeypatch.setenv("SESSION_MAX_TURNS", "2")
    get_settings.cache_clear()


# --- Observación del turno y snapshot ---


async def test_each_turn_records_an_observation_and_the_snapshot_shows_memory(short_window):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_id = (await client.post("/api/v1/sessions")).json()["session_id"]
        pdf = build_pdfs.build_pdf(build_pdfs.requirements_text(5))
        messages = [
            "Queremos estimar Vega, un portal B2B para que los bares hagan pedidos.",
            "El presupuesto máximo es de 30.000 euros.",
            "Avisos de entrega por SMS con Twilio.",
            "Informes de ventas en Power BI.",
        ]
        with patch.object(llm_client, "acomplete_structured", fake_llm):
            for i, text in enumerate(messages):
                files = [("attachments", ("a.pdf", pdf, "application/pdf"))] if i == 0 else None
                body = (
                    await client.post(
                        f"/api/v1/sessions/{session_id}/estimate",
                        data=FORM | {"transcript": text},
                        files=files,
                    )
                ).json()
                if i == 0:
                    first = body["session"]["last_turn"]
        snapshot = (await client.get(f"/api/v1/sessions/{session_id}/snapshot")).json()

    assert first["turn_index"] == 1
    assert first["attachments_total_chars"] == extract_text("a.pdf", pdf).chars
    assert first["enriched_transcript_chars"] == len(messages[0]) + first["attachments_total_chars"]
    assert (first["messages_in_window"], first["cache_hit_kind"], first["llm_calls"]) == (
        2,
        "none",
        1,
    )
    last = body["session"]["last_turn"]
    # Turno 4 con ventana de 2: el turno 2 (presupuesto) ya es un ancla; el 1, resumen
    assert (last["turn_index"], last["messages_in_window"], last["anchors_count"]) == (4, 4, 1)
    # El turno que sale (el 2) es un ancla: no se llama al resumidor en este turno
    assert last["summary_chars"] > 0 and last["llm_calls"] == 1
    assert last["cost_usd"] == pytest.approx(0.001)
    assert "30.000 euros" in snapshot["anchor_messages"][0]["content"]
    assert [m["role"] for m in snapshot["window_messages"]] == ["user", "assistant"] * 2
    assert snapshot["last_turn"] == last


# --- Escenarios ---


def test_scenarios_have_twenty_turns_and_facts():
    for scenario in SCENARIOS.values():
        assert len(scenario.turns) == 20
        assert all(len(t.transcript) >= 20 for t in scenario.turns)
        assert scenario.facts_before(1) == []
    assert SCENARIOS["pivot"].stale_facts_before(20) == ["React"]
    assert "React" not in SCENARIOS["pivot"].facts_before(20)
    assert "30.000" not in SCENARIOS["contradiction"].facts_before(9)
    assert "30.000" in SCENARIOS["contradiction"].facts_before(8)


# --- PDFs ---


def test_pdfs_have_the_requested_size_and_both_facts(tmp_path):
    paths = build_pdfs.build_all([5, 20], tmp_path)

    small = extract_text("a.pdf", paths[0].read_bytes())
    big = extract_text("b.pdf", paths[1].read_bytes())
    assert 4900 <= small.chars <= 5100 and not small.truncated
    assert build_pdfs.HEAD_FACT in small.text and build_pdfs.TAIL_FACT in small.text
    # 20 KB pasan del límite de 15.000 caracteres: el final del documento no llega
    assert big.truncated and build_pdfs.TAIL_FACT not in big.text


# --- Runner ---

TINY = Scenario(
    "tiny",
    (
        ScenarioTurn("Queremos estimar Vega, un portal B2B para bares.", "Vega"),
        ScenarioTurn("El presupuesto máximo es de 30.000 euros.", "30.000"),
        ScenarioTurn("Corrección: el presupuesto máximo es de 80.000 euros.", "80.000", "30.000"),
        ScenarioTurn("Avisos de entrega por SMS con Twilio.", "Twilio"),
    ),
)


async def test_run_stress_writes_one_row_per_turn(short_window, tmp_path):
    with patch.object(llm_client, "acomplete_structured", fake_llm):
        rows = await run.run_stress([TINY], [0, 5], repeats=1, max_turns=4)

    assert len(rows) == 8 and all(r["status_code"] == 200 for r in rows)
    with_pdf = [r for r in rows if r["attachment_kb"] == 5]
    assert with_pdf[0]["attachment_head_recall"] == 1.0
    assert with_pdf[0]["attachment_tail_recall"] == 0.0  # el modelo simulado no lo menciona
    assert with_pdf[1]["attachment_head_recall"] is None
    last = rows[3]
    assert (last["facts_checked"], last["project_name_long_term"]) == (2, 1.0)
    assert last["stale_facts_in_long_term"] == 1  # 30.000 sigue en el resumen simulado
    assert rows[0]["recall_long_term"] is None  # en el turno 1 no hay hechos anteriores

    output = tmp_path / "results.csv"
    run.write_csv(rows, output)
    run.write_csv(rows[:1], output, append=True)
    saved = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(saved) == 9 and list(saved[0]) == run.FIELDS


async def test_rate_limits_are_retried(monkeypatch):
    monkeypatch.setattr(run, "RATE_LIMIT_WAIT_S", 0)
    calls = []

    async def flaky(messages, response_model, max_tokens):
        calls.append(1)
        if len(calls) == 1:
            raise llm_client.LLMError("litellm.RateLimitError: demasiadas peticiones")
        return await fake_llm(messages, response_model, max_tokens)

    with patch.object(llm_client, "acomplete_structured", flaky):
        rows = await run.run_stress([TINY], [0], repeats=1, max_turns=1)

    assert rows[0]["status_code"] == 200 and len(calls) == 2


async def test_errors_stop_the_conversation_and_are_recorded():
    async def failing(*args, **kwargs):
        raise llm_client.LLMError("timeout")

    with patch.object(llm_client, "acomplete_structured", failing):
        rows = await run.run_stress([TINY], [0], repeats=1, max_turns=4)

    assert len(rows) == 1
    assert (rows[0]["status_code"], rows[0]["error"]) == (502, "Error al llamar al LLM: timeout")


async def test_connection_errors_are_recorded():
    class BrokenClient:
        async def post(self, url, data=None, files=None):
            if url.endswith("/sessions"):
                return httpx.Response(201, json={"session_id": "s1"})
            raise httpx.ConnectError("API caída")

    latency, cost = run.LatencyBudgetMetric(1), run.CostBudgetMetric(1)
    rows = await run.run_conversation(BrokenClient(), TINY, 0, 1, 4, latency, cost)

    assert rows == [
        {
            "scenario": "tiny",
            "attachment_kb": 0,
            "repeat": 1,
            "turn_index": 1,
            "status_code": 0,
            "error": "ConnectError('API caída')",
        }
    ]


async def test_burst_runs_simultaneous_first_turns():
    calls = []

    async def third_call_fails(messages, response_model, max_tokens):
        calls.append(1)
        if len(calls) == 3:
            raise llm_client.LLMError("timeout")
        return await fake_llm(messages, response_model, max_tokens)

    with patch.object(llm_client, "acomplete_structured", third_call_fails):
        rows = await run.run_burst([1, 2])

    assert [r["scenario"] for r in rows] == ["burst_1", "burst_2", "burst_2"]
    assert sorted(r["status_code"] for r in rows) == [200, 200, 502]


def test_http_mode_uses_the_given_url():
    assert str(run._client("http://api:8000").base_url) == "http://api:8000"


def test_main_runs_and_writes_the_csv(short_window, tmp_path, capsys):
    output = tmp_path / "out.csv"
    with patch.object(llm_client, "acomplete_structured", fake_llm):
        code = run.main(
            [
                "--scenarios",
                "contradiction",
                "--attachment-sizes",
                "0",
                "--repeats",
                "1",
                "--turns",
                "2",
                "--burst",
                "1",
                "--output",
                str(output),
            ]
        )

    assert code == 0
    assert "3 filas (0 con error)" in capsys.readouterr().out
    assert len(list(csv.DictReader(output.open(encoding="utf-8")))) == 3


def test_main_only_burst_and_failures_exit_with_1(tmp_path):
    async def failing(*args, **kwargs):
        raise llm_client.LLMError("timeout")

    with patch.object(llm_client, "acomplete_structured", failing):
        code = run.main(["--skip-scenarios", "--burst", "1", "--output", str(tmp_path / "b.csv")])

    assert code == 1


def test_main_rejects_unknown_scenarios():
    with pytest.raises(SystemExit) as exit_info:
        run.main(["--scenarios", "no_existe"])

    assert exit_info.value.code == 2


# --- Informe ---


async def test_report_renders_every_section(short_window, tmp_path, capsys):
    pivot = Scenario(
        "pivot",
        (
            ScenarioTurn("Queremos estimar Orbita, una web en React para técnicos.", "Orbita"),
            ScenarioTurn("Cambio: será una app móvil en Flutter.", "Flutter", "React"),
        ),
    )
    with patch.object(llm_client, "acomplete_structured", fake_llm):
        rows = await run.run_stress([pivot, SCENARIOS["growing"]], [0, 20], 1, max_turns=11)
        rows += await run.run_burst([2])
    rows.append({"scenario": "growing", "attachment_kb": 0, "repeat": 9, "status_code": 502})
    output = tmp_path / "results.csv"
    run.write_csv(rows, output)

    assert report.main([str(output)]) == 0
    text = capsys.readouterr().out
    for title in ("Resumen", "Curva 1", "Curva 2", "Curva 3", "Adjuntos", "obsoletos", "Ráfaga"):
        assert title in text
    assert f"{len(rows)} filas, 1 con error." in text
    assert "| growing | 0 KB | 11/12 |" in text
    assert "| pivot |" in text  # tiene un hecho obsoleto (React); growing no
    assert "Coste del turno 20 / turno 1: growing: ×—, pivot: ×—" in text


def test_report_helpers():
    assert report.percentile([], 50) is None
    assert report.percentile([1, 2, 3, 4, 5], 95) == 5
    assert report.fmt(None) == "—"
    assert report.pct(0.5) == "50 %"
    rows = [
        {"scenario": "growing", "turn_index": t, "status_code": "200", "cost_usd": c}
        for t, c in (("1", "0.001"), ("20", "0.003"))
    ]
    for r in rows:
        r.update(attachment_kb="0", repeat="1")
    assert report.turn_cost_ratio(rows) == {"growing": pytest.approx(3.0)}


def test_report_main_uses_the_default_path(monkeypatch, tmp_path, capsys):
    output = tmp_path / "results.csv"
    run.write_csv([], output)
    monkeypatch.setattr(report, "DEFAULT_OUTPUT", output)

    assert report.main([]) == 0
    assert "0 filas, 0 con error." in capsys.readouterr().out


async def test_unexpected_exceptions_are_recorded_and_stop_the_conversation():
    async def broken(*args, **kwargs):
        raise RuntimeError("fallo interno")

    with patch.object(llm_client, "acomplete_structured", broken):
        saved = []
        rows = await run.run_stress([TINY], [0], repeats=1, max_turns=4, on_rows=saved.append)

    assert len(rows) == 1
    assert (rows[0]["status_code"], rows[0]["error"]) == (0, "RuntimeError('fallo interno')")
    assert saved == [rows]


async def test_burst_records_unexpected_exceptions():
    async def broken(*args, **kwargs):
        raise RuntimeError("fallo interno")

    with patch.object(llm_client, "acomplete_structured", broken):
        rows = await run.run_burst([1])

    assert (rows[0]["status_code"], rows[0]["error"]) == (0, "RuntimeError('fallo interno')")
