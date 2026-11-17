"""Compresión del historial: anclas, resumen acumulado e integración en las sesiones."""

from unittest.mock import AsyncMock, patch

import httpx
import litellm
import pytest

from app.config import get_settings
from app.main import app
from app.prompts import PROMPT_VERSION, render_estimation_prompt, render_summary_prompt
from app.services import compression, llm_client
from app.services.compression import ConversationSummary, compress, detect_anchor
from app.sessions import ConversationHistory, ProjectMetadata, Turn, store
from tests.factories import make_info, make_request, make_result, tool_call_response

# --- Detección de anclas ---


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        (
            "Hemos firmado un NDA con el cliente, nada de esto puede salir fuera.",
            "confidencialidad",
        ),
        ("Hay un acuerdo de confidencialidad con la distribuidora.", "confidencialidad"),
        ("El cliente ha firmado el contrato esta mañana.", "contrato_firmado"),
        ("La propuesta ya está firmada por ambas partes.", "contrato_firmado"),
        ("We signed the contract yesterday.", "contrato_firmado"),
        ("El alcance queda cerrado: no se añade nada más.", "alcance_cerrado"),
        ("The scope is frozen for phase 1.", "alcance_cerrado"),
        ("El presupuesto máximo es de 30.000 euros.", "presupuesto_maximo"),
        ("No puede superar los 25.000 € en total.", "presupuesto_maximo"),
        ("Tiene que cumplir el RGPD porque guarda datos de clientes.", "normativa"),
        ("Los pagos con tarjeta exigen PCI-DSS.", "normativa"),
        ("La fecha límite es innegociable: marzo.", "fecha_limite"),
        ("Tiene que estar en producción antes del 1 de marzo.", "fecha_limite"),
        ("El cliente ha aceptado la propuesta de dos fases.", "compromiso_explicito"),
        ("Hemos acordado dejar la app para la segunda fase.", "compromiso_explicito"),
    ],
)
def test_detects_durable_commitments(text, rule):
    assert rule in detect_anchor(text)


@pytest.mark.parametrize(
    "text",
    [
        "Una distribuidora de bebidas quiere un portal B2B para que los bares hagan pedidos.",
        "Añade una app móvil para los repartidores. ¿Cómo cambia la estimación?",
        "El cliente confirma que el equipo será de 4 personas.",
        "Ajusta la estimación con el documento que te adjunto.",
        "¿Cuánto costaría añadir notificaciones push?",
    ],
)
def test_ordinary_messages_are_not_anchors(text):
    assert detect_anchor(text) == []


# --- Historial: ventana, anclas y resumen ---


def test_add_turn_returns_the_evicted_turns():
    history = ConversationHistory(max_turns=2)

    assert history.add_turn("u1", "a1") == []
    assert history.add_turn("u2", "a2") == []
    assert history.add_turn("u3", "a3") == [Turn("u1", "a1")]
    assert len(history) == 2


def test_anchors_go_before_the_window_and_are_capped():
    history = ConversationHistory(max_turns=1, max_anchors=2)
    history.add_turn("u9", "a9")

    assert history.add_anchor(Turn("u1", "a1")) == []
    assert history.add_anchor(Turn("u2", "a2")) == []
    assert history.add_anchor(Turn("u3", "a3")) == [Turn("u1", "a1")]  # la más antigua sale
    assert [m["content"] for m in history.messages()] == ["u2", "a2", "u3", "a3", "u9", "a9"]


def test_to_messages_list_passes_the_summary_to_build_system():
    history = ConversationHistory(max_turns=2)
    history.summary = "Se habló de un portal B2B."

    messages = history.to_messages_list(ProjectMetadata(), lambda m, s: f"SYSTEM | {s}")

    assert messages == [{"role": "system", "content": "SYSTEM | Se habló de un portal B2B."}]


# --- Prompts ---


def test_summary_block_only_appears_when_there_is_a_summary():
    request = make_request()

    without, _ = render_estimation_prompt(request, PROMPT_VERSION)
    with_summary, _ = render_estimation_prompt(
        request, PROMPT_VERSION, conversation_summary="Primero 200 h; después se añadió la app."
    )

    assert "<resumen_conversacion>" not in without
    assert (
        "<resumen_conversacion>\nPrimero 200 h; después se añadió la app.\n</resumen_conversacion>"
        in with_summary
    )


def test_summary_prompt_includes_previous_summary_and_turns():
    system, user = render_summary_prompt("Resumen previo.", [Turn("Pregunta 1", "Respuesta 1")])

    assert "resumen" in system.lower()
    assert "<resumen_anterior>\nResumen previo.\n</resumen_anterior>" in user
    assert "Usuario:\nPregunta 1" in user and "Asistente:\nRespuesta 1" in user

    _, first = render_summary_prompt(None, [])
    assert "(vacío: es el primer resumen)" in first


# --- Política de compresión ---


@pytest.fixture
def summarizer():
    mock = AsyncMock(return_value=(ConversationSummary(summary="Resumen nuevo."), make_info()))
    with patch.object(llm_client, "acomplete_structured", mock):
        yield mock


async def test_ordinary_turns_are_folded_into_the_summary(summarizer):
    history = ConversationHistory(max_turns=1, max_anchors=4)
    history.summary = "Resumen viejo."

    report = await compress(history, [Turn("Añade una app móvil, por favor.", "## Estimación")])

    assert history.summary == "Resumen nuevo."
    assert (history.summarized_turns, report.summarized, history.anchors) == (1, 1, [])
    assert report.cost_usd == make_info().cost_usd
    messages = summarizer.await_args.args[0]
    assert "Resumen viejo." in messages[1]["content"]
    assert "Añade una app móvil" in messages[1]["content"]
    assert summarizer.await_args.args[1] is ConversationSummary


async def test_commitments_become_anchors_without_calling_the_llm(summarizer):
    history = ConversationHistory(max_turns=1, max_anchors=4)
    anchor = Turn("Hemos firmado un NDA: todo es confidencial.", "Entendido.")

    report = await compress(history, [anchor])

    assert history.anchors == [anchor]
    assert report.anchored == [["confidencialidad"]]
    assert history.summary is None
    summarizer.assert_not_awaited()


async def test_anchors_that_no_longer_fit_go_to_the_summary(summarizer):
    history = ConversationHistory(max_turns=1, max_anchors=1)
    first = Turn("El presupuesto máximo es de 20.000 euros.", "Ok.")
    second = Turn("Tiene que cumplir el RGPD.", "Ok.")

    await compress(history, [first])
    await compress(history, [second])

    assert history.anchors == [second]
    assert "20.000 euros" in summarizer.await_args.args[0][1]["content"]
    assert history.summarized_turns == 1


async def test_without_anchor_slots_everything_is_summarized(summarizer):
    history = ConversationHistory(max_turns=1, max_anchors=0)

    report = await compress(history, [Turn("Hemos firmado un NDA.", "Ok.")])

    assert history.anchors == []
    assert report.summarized == 1


async def test_a_failed_summary_keeps_the_previous_one():
    history = ConversationHistory(max_turns=1)
    history.summary = "Resumen viejo."
    failing = AsyncMock(side_effect=llm_client.LLMError("timeout"))

    with patch.object(llm_client, "acomplete_structured", failing):
        report = await compress(history, [Turn("Añade login con Google.", "Ok.")])

    assert history.summary == "Resumen viejo."
    assert report.summary_failed and report.summarized == 0 and report.cost_usd == 0


async def test_nothing_to_do_without_evicted_turns(summarizer):
    report = await compress(ConversationHistory(max_turns=2), [])

    assert report == compression.CompressionReport()
    summarizer.assert_not_awaited()


# --- De extremo a extremo, con la API ---

TRANSCRIPT = "Una distribuidora de bebidas quiere un portal B2B para que los bares hagan pedidos."
FORM = {"project_type": "web_saas", "detail_level": "summary", "output_format": "phases_table"}


@pytest.fixture
async def client():
    store.clear()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    store.clear()


async def run_conversation(client, messages: list[str]):
    """Envía los mensajes en una sesión y devuelve (llamadas de estimación, de resumen, estado)."""
    estimations, summaries = [], []

    async def fake_acompletion(**params):
        tool = params["tools"][0]["function"]["name"]
        if tool == "ConversationSummary":
            summaries.append(params["messages"])
            return tool_call_response({"summary": f"Resumen {len(summaries)}."}, name=tool)
        estimations.append(params["messages"])
        return tool_call_response(make_result().model_dump(mode="json"))

    session_id = (await client.post("/api/v1/sessions")).json()["session_id"]
    with patch.object(litellm, "acompletion", fake_acompletion):
        for text in messages:
            response = await client.post(
                f"/api/v1/sessions/{session_id}/estimate", data=FORM | {"transcript": text}
            )
            assert response.status_code == 200
    state = (await client.get(f"/api/v1/sessions/{session_id}")).json()
    return estimations, summaries, state


async def test_anchor_and_summary_reach_the_llm(client):
    nda = f"{TRANSCRIPT} Hemos firmado un NDA con el cliente."
    messages = [nda] + [f"{TRANSCRIPT} Mensaje número {i}." for i in range(2, 10)]

    estimations, summaries, state = await run_conversation(client, messages)

    # 9 turnos con ventana de 6: salen el 1 (ancla), el 2 y el 3 (al resumen)
    assert (state["anchors"], state["summarized_turns"]) == (1, 2)
    assert state["conversation_summary"] == "Resumen 2."
    assert state["compression_cost_usd"] > 0
    assert len(summaries) == 2
    last = estimations[-1]
    # system + ancla (2) + ventana (6 turnos = 12) + mensaje actual
    assert len(last) == 1 + 2 + 12 + 1
    assert "Hemos firmado un NDA" in last[1]["content"]
    # La compresión va después de cada turno: el turno 9 aún ve el resumen del turno 8
    assert "<resumen_conversacion>\nResumen 1.\n</resumen_conversacion>" in last[0]["content"]


async def test_compression_can_be_disabled(client, monkeypatch):
    monkeypatch.setenv("SESSION_COMPRESSION_ENABLED", "false")
    get_settings.cache_clear()
    nda = f"{TRANSCRIPT} Hemos firmado un NDA con el cliente."
    messages = [nda] + [f"{TRANSCRIPT} Mensaje número {i}." for i in range(2, 9)]

    estimations, summaries, state = await run_conversation(client, messages)

    assert summaries == []
    assert (state["anchors"], state["summarized_turns"], state["conversation_summary"]) == (
        0,
        0,
        None,
    )
    assert len(estimations[-1]) == 1 + 12 + 1
