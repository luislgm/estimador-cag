import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType  # noqa: E402

APP_FILE = str(Path(__file__).parent.parent / "streamlit_app.py")
DESCRIPTION = (
    "Una cadena de autoescuelas quiere una app móvil para matricularse, pagar paquetes "
    "de clases y reservar prácticas con su profesor."
)
HEALTH = {
    "status": "ok",
    "environment": "development",
    "provider": "anthropic",
    "model": "claude-haiku-4-5",
}


def done_payload(text, **changes):
    return {
        "text": text,
        "prompt_version": "v1",
        "model": "claude-haiku-4-5",
        "provider": "anthropic",
        "usage": {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500},
        "cost_usd": 0.0027,
        "finish_reason": "stop",
        "cache_hit": False,
        "created_at": "2026-11-01T10:00:00Z",
    } | changes


def sse_lines(events):
    for event, data in events:
        yield f"event: {event}"
        yield f"data: {json.dumps(data, ensure_ascii=False)}"
        yield ""


class FakeApi:
    """Simula la API: health y el endpoint de streaming (SSE)."""

    def __init__(self):
        self.requests = []
        self.health = HEALTH
        self.status_code = 200
        self.events = None  # por defecto, una estimación correcta en dos trozos
        self.connection_error = False

    def estimation(self, title="App autoescuelas", **done_changes):
        text = f"## Estimación: {title}\n\n1. Diseño: 20 horas"
        return [
            ("token", {"text": f"## Estimación: {title}"}),
            ("token", {"text": "\n\n1. Diseño: 20 horas"}),
            ("done", done_payload(text, **done_changes)),
        ]

    def get(self, url, **kwargs):
        if self.health is None:
            raise httpx.ConnectError("API caída")
        return httpx.Response(200, json=self.health)

    def stream(self, method, url, json=None, **kwargs):
        if self.connection_error:
            raise httpx.ConnectError("API caída")
        self.requests.append((method, url, json))
        api = self

        class Response:
            status_code = api.status_code
            text = '{"detail": "error de validación"}'

            def read(self):
                pass

            def iter_lines(self):
                yield from sse_lines(api.events or api.estimation())

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return Response()


@pytest.fixture(autouse=True)
def api(monkeypatch):
    monkeypatch.setenv("ESTIMATOR_API_URL", "http://api-test:8000")
    fake = FakeApi()
    st.cache_data.clear()
    with patch("httpx.get", fake.get), patch("httpx.stream", fake.stream):
        yield fake


def run_app() -> AppTest:
    return AppTest.from_file(APP_FILE, default_timeout=30).run()


def submit(
    at: AppTest,
    description=DESCRIPTION,
    project_type=ProjectType.MOBILE_APP,
    detail_level=DetailLevel.DETAILED,
    output_format=OutputFormat.PHASES_TABLE,
) -> AppTest:
    at.text_area[0].input(description)
    at.selectbox[0].set_value(project_type)
    at.selectbox[1].set_value(output_format)
    at.radio[0].set_value(detail_level)
    return next(b for b in at.button if b.label == "Estimar").click().run()


def all_markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_starts_with_empty_form(api):
    at = run_app()

    assert not at.exception
    assert at.title[0].value == "📋 Estimador CAG"
    assert at.text_area[0].label == "Descripción del proyecto"
    assert at.session_state.history == []
    assert any("Aún no se ha hecho ninguna estimación" in c.value for c in at.sidebar.caption)
    assert any("claude-haiku-4-5" in c.value for c in at.sidebar.caption)


def test_form_options_show_spanish_labels(api):
    at = run_app()

    assert at.selectbox[0].options == [
        "App móvil",
        "Web SaaS",
        "Herramienta interna",
        "Pipeline de datos",
    ]
    assert at.radio[0].options == ["Resumen", "Medio", "Detallado"]
    assert at.selectbox[1].options == ["Tabla por fases", "Desglose de tareas", "Narrativo"]


def test_submit_sends_estimation_request_to_api(api):
    at = submit(run_app())

    assert not at.exception
    method, url, payload = api.requests[0]
    assert (method, url) == ("POST", "http://api-test:8000/api/v1/estimate/stream")
    assert payload == {
        "description": DESCRIPTION,
        "project_type": "mobile_app",
        "detail_level": "detailed",
        "output_format": "phases_table",
    }


def test_shows_streamed_estimation_and_metrics(api):
    at = submit(run_app())

    assert "## Estimación: App autoescuelas" in all_markdown(at)
    assert len(at.session_state.history) == 1
    metrics = {m.label: m.value for m in at.sidebar.metric}
    assert metrics["Tokens entrada"] == "1.200"
    assert metrics["Coste"] == "0,0027 $"
    assert metrics["Respuesta"] == "Completa"
    assert any("prompt v1" in c.value for c in at.caption)


def test_previous_estimations_are_kept_folded(api):
    at = submit(run_app())
    api.events = api.estimation(title="Segunda estimación")
    at = submit(at, detail_level=DetailLevel.SUMMARY)

    assert len(at.session_state.history) == 2
    assert "## Estimación: Segunda estimación" in all_markdown(at)
    assert at.main.subheader[-1].value == "Estimaciones anteriores de esta sesión"
    folded = [e.label for e in at.main.expander if "App autoescuelas" in e.label]
    assert len(folded) == 1
    assert "App móvil · Detallado · Tabla por fases" in folded[0]


def test_last_estimation_is_shown_after_rerun(api):
    at = submit(run_app()).run()  # otra ejecución sin enviar el formulario

    assert "## Estimación: App autoescuelas" in all_markdown(at)
    assert len(at.session_state.history) == 1


def test_invalid_description_is_not_sent(api):
    at = submit(run_app(), description="corta")

    assert api.requests == []
    assert "Revisa el formulario" in at.error[0].value


def test_api_error_event_is_shown(api):
    api.events = [("error", {"message": "Error al llamar al LLM: conexión rechazada"})]

    at = submit(run_app())

    assert "conexión rechazada" in at.error[0].value
    assert at.session_state.history == []


def test_api_http_error_is_shown(api):
    api.status_code = 422

    at = submit(run_app())

    assert "La API respondió 422" in at.error[0].value


def test_api_unreachable(api):
    api.health = None
    api.connection_error = True

    at = submit(run_app())

    assert any("No se puede conectar con la API" in e.value for e in at.sidebar.error)
    assert any("No se puede conectar con la API" in e.value for e in at.main.error)


def test_shows_cache_hit(api):
    api.events = api.estimation(cache_hit=True, cost_usd=0.0)

    at = submit(run_app())

    assert any("desde la caché" in c.value for c in at.sidebar.caption)
    assert {m.label: m.value for m in at.sidebar.metric}["Coste"] == "0,0000 $"


def test_warns_when_estimation_is_truncated(api):
    api.events = api.estimation(finish_reason="length")

    at = submit(run_app())

    assert "se ha cortado" in at.warning[0].value
    assert {m.label: m.value for m in at.sidebar.metric}["Respuesta"] == "Cortada"


def test_new_session_clears_history(api):
    at = submit(run_app())
    at = next(b for b in at.sidebar.button if b.label == "Nueva sesión").click().run()

    assert at.session_state.history == []
    assert len(at.sidebar.metric) == 0


def test_sidebar_shows_examples_and_default_prompts(api):
    from app.prompts import render_examples

    at = run_app()

    codes = [c.value for c in at.sidebar.code]
    assert render_examples("v1") in codes
    assert any("(aquí irá la descripción del proyecto)" in c for c in codes)
    assert any("(v1)" in s.value for s in at.sidebar.subheader)


def test_sidebar_shows_prompts_of_last_estimation(api):
    from app.prompts import render_estimation_prompt
    from app.schemas.estimation import EstimationRequest

    at = submit(run_app(), output_format=OutputFormat.NARRATIVE)

    request = EstimationRequest(**at.session_state.history[-1]["request"])
    system, user = render_estimation_prompt(request)
    codes = [c.value for c in at.sidebar.code]
    assert system in codes and user in codes
    assert "texto en párrafos" in system  # el bloque del formato elegido
