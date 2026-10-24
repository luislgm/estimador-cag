from pathlib import Path
from unittest.mock import patch

import anthropic
import httpx
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from app.services.llm_service import EstimationResult  # noqa: E402

APP_FILE = str(Path(__file__).parent.parent / "streamlit_app.py")
TRANSCRIPTION = (
    "El cliente quiere una web para reservar mesas en su restaurante, "
    "con pago online y un panel de gestión para el personal."
)
CHUNKS = ["## Estimación: ", "Reservas", "\n\n1. Diseño: 20 horas"]


class FakeStream:
    """Sustituye a EstimationStream: entrega el texto a trozos sin llamar a la API."""

    calls: list[str] = []
    error: Exception | None = None
    finish_reason: str = "end_turn"

    def __init__(self, transcription: str):
        FakeStream.calls.append(transcription)
        self.result = None

    def __iter__(self):
        if FakeStream.error:
            raise FakeStream.error
        yield from CHUNKS
        self.result = EstimationResult(
            "".join(CHUNKS), "claude-haiku-4-5", 1200, 300, FakeStream.finish_reason
        )


@pytest.fixture(autouse=True)
def fake_stream():
    FakeStream.calls = []
    FakeStream.error = None
    FakeStream.finish_reason = "end_turn"
    with patch("app.services.llm_service.EstimationStream", FakeStream):
        yield FakeStream


def run_app() -> AppTest:
    return AppTest.from_file(APP_FILE, default_timeout=30).run()


def send(at: AppTest, text: str) -> AppTest:
    return at.chat_input[0].set_value(text).run()


def test_starts_with_empty_chat():
    at = run_app()

    assert not at.exception
    assert at.title[0].value == "📋 Estimador CAG"
    assert len(at.chat_message) == 0
    assert at.session_state.messages == []


def test_streams_estimation_and_keeps_history(fake_stream):
    at = send(run_app(), TRANSCRIPTION)
    at = send(at, TRANSCRIPTION + " También quieren app móvil.")

    assert not at.exception
    assert len(fake_stream.calls) == 2
    assert [m["role"] for m in at.session_state.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    # El texto que llegó a trozos se guarda completo en el historial
    assert at.session_state.messages[1]["content"] == "".join(CHUNKS)
    assert "1.500 tokens (1.200 entrada + 300 salida)" in at.session_state.messages[1]["caption"]
    assert len(at.chat_message) == 4  # todo el historial sigue visible


def test_short_transcription_does_not_call_llm(fake_stream):
    at = send(run_app(), "Quiero una web")

    assert fake_stream.calls == []
    assert "demasiado corta" in at.warning[0].value


def test_llm_error_is_shown(fake_stream):
    fake_stream.error = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )

    at = send(run_app(), TRANSCRIPTION)

    assert "Error al llamar al LLM" in at.error[0].value


def test_missing_api_key_shows_config_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")

    at = run_app()

    assert "Configuración incompleta" in at.error[0].value
    assert len(at.chat_input) == 0


def test_sidebar_shows_system_prompt_and_examples():
    from app.context.examples import ESTIMATION_EXAMPLES
    from app.services.llm_service import build_system_prompt

    at = run_app()

    assert at.sidebar.code[0].value == build_system_prompt()
    titles = [e.label for e in at.sidebar.expander]
    assert len(titles) == 1 + len(ESTIMATION_EXAMPLES)  # prompt + un desplegable por ejemplo
    assert "Plataforma de Gestión de Inventario" in titles[1]
    assert "Aún no se ha hecho ninguna llamada" in at.sidebar.caption[-1].value


def test_sidebar_shows_metrics_of_last_call():
    at = send(run_app(), TRANSCRIPTION)

    metrics = {m.label: m.value for m in at.sidebar.metric}
    assert metrics["Tokens entrada"] == "1.200"
    assert metrics["Tokens salida"] == "300"
    assert metrics["Tiempo total"].endswith(" s")
    assert metrics["Primer token"].endswith(" s")
    assert at.session_state.last_metrics["model"] == "claude-haiku-4-5"


def test_new_conversation_clears_history_and_metrics():
    at = send(run_app(), TRANSCRIPTION)
    at = at.sidebar.button[0].click().run()

    assert at.session_state.messages == []
    assert at.session_state.last_metrics is None
    assert len(at.sidebar.metric) == 0


def test_shows_cost_in_metrics_and_caption():
    at = send(run_app(), TRANSCRIPTION)

    metrics = {m.label: m.value for m in at.sidebar.metric}
    assert metrics["Coste"] == "0,0027 $"  # 1.200 × 1 $/M + 300 × 5 $/M
    assert metrics["Respuesta"] == "Completa"
    assert at.session_state.messages[1]["caption"].endswith("0,0027 $")
    assert len(at.warning) == 0


def test_warns_when_estimation_is_truncated(fake_stream):
    fake_stream.finish_reason = "max_tokens"

    at = send(run_app(), TRANSCRIPTION)

    assert "se ha cortado" in at.warning[0].value
    assert {m.label: m.value for m in at.sidebar.metric}["Respuesta"] == "Cortada"
