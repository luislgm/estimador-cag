"""Tests del código de las evals (dataset, métricas y runner), con el LLM simulado."""

import json
from unittest.mock import patch

import pytest

from app.services import llm_client
from app.tiers import Tier
from evals import run
from evals.dataset import DATASET_PATH, EvalCase, Expected, load_dataset
from evals.metrics import (
    THRESHOLDS,
    MetricResult,
    TurnOutcome,
    aggregate,
    content_recall,
    direction,
    hours_in_range,
    memory,
    normalize,
    schema,
    scope,
    score_case,
    tier,
)
from tests.factories import make_info, make_phase, make_result

# --- Dataset ---


def test_golden_dataset_is_valid():
    cases = load_dataset()

    assert len(cases) >= 12
    assert len({c.id for c in cases}) == len(cases)
    assert any(c.expected.out_of_scope for c in cases)
    assert {c.expected.tier for c in cases} >= {Tier.EXECUTIVE, Tier.PM, Tier.DEVELOPER}
    assert {c.expected.change for c in cases} >= {"up", "down"}
    assert {c.project_type for c in cases} == {
        "mobile_app",
        "web_saas",
        "internal_tool",
        "data_pipeline",
    }


@pytest.mark.parametrize(
    ("expected", "error"),
    [
        ({"hours": [100, 50]}, "el mínimo no puede ser mayor"),
        ({"out_of_scope": True, "hours": [1, 2]}, "fuera de alcance no tiene horas"),
        ({"must_mention": [[]]}, "al menos un sinónimo"),
    ],
)
def test_invalid_expectations(expected, error):
    with pytest.raises(ValueError, match=error):
        Expected.model_validate(expected)


def test_change_needs_two_turns():
    with pytest.raises(ValueError, match="al menos dos turnos"):
        EvalCase(id="x", turns=["uno"], expected={"change": "up"})


def test_duplicated_ids_are_rejected(tmp_path):
    case = json.loads(DATASET_PATH.read_text("utf-8"))[0]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps([case, case]), encoding="utf-8")

    with pytest.raises(ValueError, match="ids repetidos"):
        load_dataset(path)


# --- Métricas ---


def case(**expected) -> EvalCase:
    turns = ["primer turno", "segundo turno"] if expected.get("change") else ["un turno"]
    return EvalCase(id="caso", turns=turns, expected=expected)


def ok(result, tier_value=None) -> TurnOutcome:
    body = {"result": result.model_dump(mode="json"), "session": {"tier": tier_value}}
    return TurnOutcome(200, body)


OUT_OF_SCOPE = make_result(
    summary="Fuera de alcance: no es un proyecto de software.",
    confidence_pct=10,
    phases=[make_phase("Sin estimar", 0)],
)


def test_normalize_ignores_case_and_accents():
    assert normalize("Código de Barras, NÓMINA") == "codigo de barras, nomina"


def test_schema_fails_when_a_turn_fails_or_is_missing():
    assert schema(case(), [ok(make_result())]).score == 1
    assert schema(case(), [TurnOutcome(502, {"detail": "x"})]).detail == "turno fallido (502)"
    assert schema(case(change="up"), [ok(make_result())]).detail == "turno fallido (sin respuesta)"


def test_scope():
    assert scope(case(), make_result()).score == 1
    assert scope(case(), OUT_OF_SCOPE).score == 0
    assert scope(case(out_of_scope=True), OUT_OF_SCOPE).score == 1
    assert scope(case(out_of_scope=True), make_result()).score == 0
    # Fuera de alcance pero con cifras: no vale
    with_figures = OUT_OF_SCOPE.model_copy(update={"phases": [make_phase("Desarrollo", 80)]})
    assert scope(case(out_of_scope=True), with_figures).score == 0


def test_hours_in_range():
    result = make_result()  # 80 h
    assert hours_in_range(case(), result) is None
    assert hours_in_range(case(hours=[50, 100]), result).score == 1
    assert hours_in_range(case(hours=[100, 200]), result).detail == "80 h en [100, 200]"


def test_content_recall_counts_synonyms_without_accents():
    result = make_result(risks=["Integración con la pasarela de PAGO", "Migración de datos"])

    metric = content_recall(
        case(must_mention=[["pago"], ["migracion"], ["chat", "mensajes"]]), result
    )

    assert metric.score == pytest.approx(2 / 3)
    assert metric.detail == "faltan: chat"
    assert content_recall(case(must_mention=[["pago"]]), result).detail == "todos"
    assert content_recall(case(), result) is None


def test_tier_metric():
    body = {"session": {"tier": "pm"}}
    assert tier(case(), body) is None
    assert tier(case(tier="pm"), body).score == 1
    assert tier(case(tier="executive"), body).score == 0


def test_memory_and_direction():
    first = make_result(
        project_name="Academia", phases=[make_phase("Diseño", 20), make_phase("Web", 80)]
    )
    second = make_result(
        project_name="academia", phases=[make_phase("Diseño", 40), make_phase("Web y app", 160)]
    )
    renamed = second.model_copy(update={"project_name": "Otra cosa"})

    assert memory(case(), [first]) is None
    assert memory(case(change="up"), [first, second]).score == 1
    assert memory(case(change="up"), [first, renamed]).score == 0
    assert direction(case(), [first]) is None
    assert direction(case(change="up"), [first, second]).score == 1
    assert direction(case(change="down"), [first, second]).score == 0
    assert (
        direction(case(change="down"), [second, first]).detail == "200 h -> 100 h (esperado down)"
    )


def test_score_case_only_counts_schema_when_a_turn_fails():
    metrics = score_case(case(hours=[1, 2]), [TurnOutcome(502, {})])

    assert [m.name for m in metrics] == ["schema"]


def test_score_case_applies_only_the_relevant_metrics():
    metrics = score_case(case(hours=[50, 100]), [ok(make_result())])

    assert [m.name for m in metrics] == ["schema", "scope", "hours_in_range"]


def test_aggregate_compares_each_mean_with_its_threshold():
    scored = [
        [MetricResult("schema", 1, ""), MetricResult("hours_in_range", 1, "")],
        [MetricResult("schema", 1, ""), MetricResult("hours_in_range", 0, "")],
    ]

    summary = aggregate(scored)

    assert summary["schema"] == {"mean": 1.0, "cases": 2, "threshold": 1.0, "passed": True}
    assert summary["hours_in_range"]["mean"] == 0.5
    assert summary["hours_in_range"]["passed"] is (0.5 >= THRESHOLDS["hours_in_range"])
    assert "tier" not in summary  # sin casos, no aparece


# --- Runner, con la app en proceso y el LLM simulado ---


async def fake_llm(messages, response_model, max_tokens):
    message = messages[-1]["content"]
    if "paella" in message:
        return OUT_OF_SCOPE, make_info()
    phase = make_phase("Web con blog y formulario de contacto", 40)
    phases = [make_phase("Diseño", 20), phase]
    return make_result(project_name="Web despacho", phases=phases), make_info()


SUBSET = ["web_corporativa_simple", "fuera_de_alcance_receta"]


async def test_run_suite_scores_every_case():
    cases = [c for c in load_dataset() if c.id in SUBSET]

    with patch.object(llm_client, "acomplete_structured", fake_llm):
        report = await run.run_suite(cases)

    assert report.passed
    assert (report.provider, report.model, report.prompt_version) == (
        "anthropic",
        "claude-haiku-4-5",
        "v1",
    )
    assert report.cost_usd == pytest.approx(2 * make_info().cost_usd)
    assert set(report.metrics) == {"schema", "scope", "hours_in_range", "content_recall"}
    web = next(c for c in report.cases if c.id == "web_corporativa_simple")
    assert web.turns[0]["total_hours"] == 60
    assert all(m.score == 1 for m in web.metrics)


async def test_failed_turns_are_reported():
    async def failing(*args, **kwargs):
        raise llm_client.LLMError("conexión rechazada")

    cases = [c for c in load_dataset() if c.id == "web_corporativa_simple"]
    with patch.object(llm_client, "acomplete_structured", failing):
        report = await run.run_suite(cases)

    assert not report.passed
    assert report.prompt_version is None
    assert report.cases[0].turns[0]["status_code"] == 502
    text = run.format_report(report)
    assert "FALLO web_corporativa_simple" in text
    assert "RESULTADO: NO pasa" in text
    assert "schema = 0.00  (turno fallido (502))" in text


async def test_connection_errors_count_as_failed_turns():
    class BrokenClient:
        async def post(self, url, data=None):
            if url.endswith("/sessions"):
                return FakeResponse({"session_id": "s1"})
            raise run.httpx.ConnectError("API caída")

    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    case_ = EvalCase(id="x", turns=["Una web sencilla para un despacho."], expected={})
    report = await run.run_case(BrokenClient(), case_)

    assert report.turns == [{"status_code": 0, "detail": "ConnectError: API caída"}]
    assert report.metrics[0].detail == "turno fallido (0)"


def test_http_mode_uses_the_given_url():
    assert str(run._client("http://api:8000").base_url) == "http://api:8000"


def test_main_prints_the_report_and_saves_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run, "REPORTS_DIR", tmp_path)

    with patch.object(llm_client, "acomplete_structured", fake_llm):
        code = run.main(["--case", SUBSET[0], "--case", SUBSET[1]])

    out = capsys.readouterr().out
    assert code == 0
    assert "RESULTADO: pasa" in out
    saved = list(tmp_path.glob("*-actor-claude-haiku-4-5.json"))
    assert len(saved) == 1
    data = json.loads(saved[0].read_text("utf-8"))
    assert data["passed"] is True and len(data["cases"]) == 2


def test_main_with_explicit_output_and_failure(tmp_path):
    async def failing(*args, **kwargs):
        raise llm_client.LLMError("timeout")

    output = tmp_path / "sub" / "informe.json"
    with patch.object(llm_client, "acomplete_structured", failing):
        code = run.main(["--case", SUBSET[0], "--output", str(output)])

    assert code == 1
    assert json.loads(output.read_text("utf-8"))["passed"] is False


def test_main_rejects_unknown_cases():
    with pytest.raises(SystemExit) as exit_info:
        run.main(["--case", "no_existe"])

    assert exit_info.value.code == 2


async def test_acb_mode_uses_the_review_endpoint():
    from app.schemas.critic import CriticIssue, CriticReview

    issue = CriticIssue(
        category="alcance", severity="alta", problem="Falta el blog.", suggestion="Añade el blog."
    )
    reviews = [
        CriticReview(verdict="revise", issues=[issue], confidence_pct=70),
        CriticReview(verdict="accept", confidence_pct=80),
    ]

    async def llm(messages, response_model, max_tokens):
        if response_model is CriticReview:
            return reviews.pop(0), make_info()
        return await fake_llm(messages, response_model, max_tokens)

    cases = [c for c in load_dataset() if c.id == "web_corporativa_simple"]
    with patch.object(llm_client, "acomplete_structured", llm):
        report = await run.run_suite(cases, mode="acb")

    turn = report.cases[0].turns[0]
    assert report.mode == "acb"
    assert (turn["acb_outcome"], turn["acb_hours"], turn["acb_llm_calls"]) == (
        "accepted",
        [60, 60],
        4,
    )
    assert "revisión: 60 -> 60 h" in run.format_report(report)
    assert run.format_report(report).startswith("Evals (acb)")
