from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType  # noqa: E402
from app.services.render import render_markdown  # noqa: E402
from app.tiers import Tier  # noqa: E402
from tests.factories import make_phase, make_request, make_result  # noqa: E402

APP_FILE = str(Path(__file__).parent.parent / "streamlit_app.py")
API = "http://api-test:8000/api/v1/sessions"
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


class FakeApi:
    """Simula la API: health, creación de sesiones y turnos de estimación."""

    def __init__(self):
        self.sessions_created = 0
        self.turn_requests = []
        self.health = HEALTH
        self.status_code = 200
        self.error_body = None
        self.result = make_result(project_name="App autoescuelas", team_size=3)
        self.connection_error = False
        self.session_extra = {}  # campos extra del estado de la sesión (tier, compresión...)
        self.acb = None  # traza del ciclo Actor-Critic-Boss, si la respuesta la trae

    def get(self, url, **kwargs):
        if self.health is None:
            raise httpx.ConnectError("API caída")
        return httpx.Response(200, json=self.health)

    def post(self, url, data=None, files=None, **kwargs):
        if self.connection_error:
            raise httpx.ConnectError("API caída")
        if url == API:
            self.sessions_created += 1
            return httpx.Response(201, json={"session_id": f"sesion-{self.sessions_created}"})
        self.turn_requests.append((url, data, files))
        self.last_url = url
        if self.status_code != 200:
            return httpx.Response(self.status_code, json=self.error_body)
        turn = len(self.turn_requests)
        return httpx.Response(200, json=self.turn_response(turn))

    def turn_response(self, turn):
        result = self.result
        return {
            "text": render_markdown(result, make_request()),
            "prompt_version": "v1",
            "result": result.model_dump(mode="json"),
            "model": "claude-haiku-4-5",
            "provider": "anthropic",
            "usage": {"input_tokens": 3500, "output_tokens": 1200, "total_tokens": 4700},
            "cost_usd": 0.0095,
            "finish_reason": "tool_calls",
            "cache_hit": False,
            "cache_kind": None,
            "created_at": "2026-11-08T10:00:00Z",
            "session": {
                "session_id": "sesion-1",
                "turns": turn,
                "turns_in_window": min(turn, 6),
                "max_turns": 6,
                "project_metadata": {
                    "project_name": result.project_name,
                    "assumed_team_size": result.team_size,
                    "mentioned_technologies": ["Flutter", "PostgreSQL"],
                    "agreed_scope": result.summary,
                },
                **self.session_extra,
            },
            "attachments": [],
            **({"acb": self.acb} if self.acb else {}),
        }


@pytest.fixture(autouse=True)
def api(monkeypatch):
    monkeypatch.setenv("ESTIMATOR_API_URL", "http://api-test:8000")
    fake = FakeApi()
    st.cache_data.clear()
    with patch("httpx.get", fake.get), patch("httpx.post", fake.post):
        yield fake


def run_app() -> AppTest:
    return AppTest.from_file(APP_FILE, default_timeout=30).run()


def send(
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
    return next(b for b in at.button if b.label == "Enviar").click().run()


def all_markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def sidebar_text(at: AppTest) -> str:
    return "\n".join([c.value for c in at.sidebar.caption] + [m.value for m in at.sidebar.markdown])


def test_creates_a_session_on_load(api):
    at = run_app()

    assert not at.exception
    assert api.sessions_created == 1
    assert at.session_state.session_id == "sesion-1"
    assert at.session_state.turns == []
    assert "Vacía: se rellena con la primera estimación." in sidebar_text(at)
    assert "claude-haiku-4-5" in sidebar_text(at)


def test_form_has_transcript_attachments_and_options(api):
    at = run_app()

    assert at.text_area[0].label == "Descripción del proyecto"
    assert at.selectbox[0].options == [
        "App móvil",
        "Web SaaS",
        "Herramienta interna",
        "Pipeline de datos",
    ]
    assert at.radio[0].options == ["Resumen", "Medio", "Detallado"]
    assert at.selectbox[1].options == ["Tabla por fases", "Desglose de tareas", "Narrativo"]
    assert at.get("file_uploader")[0].proto.label == "Adjuntos (PDF o Word)"


def test_turn_is_sent_to_the_session_endpoint(api):
    at = send(run_app())

    assert not at.exception
    url, data, files = api.turn_requests[0]
    assert url == f"{API}/sesion-1/estimate"
    assert data == {
        "transcript": DESCRIPTION,
        "project_type": "mobile_app",
        "detail_level": "detailed",
        "output_format": "phases_table",
    }
    assert files is None  # sin adjuntos


def test_shows_conversation_and_project_memory(api):
    at = send(run_app())

    assert "## Estimación: App autoescuelas" in all_markdown(at)
    assert len(at.chat_message) == 2  # mensaje del usuario y respuesta
    text = sidebar_text(at)
    assert "**Proyecto:** App autoescuelas" in text
    assert "**Equipo asumido:** 3 personas" in text
    assert "**Tecnologías:** Flutter, PostgreSQL" in text
    assert "se envían al LLM los últimos 1 de un máximo de 6" in text
    assert at.text_area[0].label == "Siguiente mensaje"


def test_several_turns_stay_in_the_conversation(api):
    at = send(run_app())
    at = send(at, description="Añade también una web de administración para la autoescuela.")
    at = send(at, description="Y que los profesores puedan ver su agenda desde el móvil.")

    assert len(at.session_state.turns) == 3
    assert len(at.chat_message) == 6
    assert all(url == f"{API}/sesion-1/estimate" for url, _, _ in api.turn_requests)
    assert "Turno 3" in "\n".join(c.value for c in at.main.caption)


def test_system_prompt_preview_includes_memory(api):
    at = send(run_app())

    system = next(c.value for c in at.sidebar.code if c.value.startswith("Eres un estimador"))
    assert "- Nombre del proyecto: App autoescuelas" in system


def test_new_conversation_creates_a_new_session(api):
    at = send(run_app())
    at = next(b for b in at.sidebar.button if b.label == "Nueva conversación").click().run()

    assert api.sessions_created == 2
    assert at.session_state.session_id == "sesion-2"
    assert at.session_state.turns == []
    assert "Vacía" in sidebar_text(at)


def test_lost_session_starts_a_new_one(api):
    at = run_app()
    api.status_code = 404
    api.error_body = {"detail": "Sesión no encontrada"}

    at = send(at)

    assert "La sesión ya no existe" in at.warning[0].value
    assert at.session_state.session_id == "sesion-2"


def test_invalid_description_is_not_sent(api):
    at = send(run_app(), description="corta")

    assert api.turn_requests == []
    assert "Revisa el formulario" in at.error[0].value


def test_guardrail_rejection_is_shown(api):
    api.status_code = 400
    api.error_body = {"detail": {"reason": "pii", "message": "Adjunto 'a.pdf': datos personales."}}

    at = send(run_app())

    assert "Descripción rechazada: Adjunto 'a.pdf': datos personales." in at.error[0].value
    assert at.session_state.turns == []


def test_api_error_is_shown(api):
    api.status_code = 502
    api.error_body = {"detail": "Error al llamar al LLM: conexión rechazada"}

    at = send(run_app())

    assert "La API respondió 502" in at.error[0].value


def test_api_unreachable(api):
    api.health = None
    api.connection_error = True

    at = send(run_app())

    assert any("No se puede conectar con la API" in e.value for e in at.sidebar.error)
    assert any("No se puede conectar con la API" in e.value for e in at.main.error)


def test_out_of_scope_warning(api):
    api.result = make_result(
        confidence_pct=10,
        summary="Fuera de alcance: falta saber qué funcionalidades quieren.",
        phases=[make_phase("Sin estimar", 0)],
    )

    at = send(run_app())

    assert "no ha podido estimar con suficiente confianza" in at.warning[0].value


def test_examples_are_shown(api):
    from app.prompts import render_examples

    at = run_app()

    assert render_examples("v1") in [c.value for c in at.sidebar.code]


def test_audience_selector_is_automatic_by_default(api):
    at = send(run_app())

    assert at.selectbox[2].label == "Audiencia"
    assert at.selectbox[2].options[0] == "Automática (se deduce del mensaje)"
    assert "tier" not in api.turn_requests[0][1]  # la API lo deduce


def test_chosen_audience_is_sent(api):
    at = run_app()
    at.selectbox[2].set_value(Tier.EXECUTIVE)
    send(at)

    assert api.turn_requests[0][1]["tier"] == "executive"


def test_sidebar_shows_the_audience_and_why(api):
    api.session_extra = {"tier": "pm", "tier_rule": "planificacion", "tier_inherited": True}

    at = send(run_app())

    text = sidebar_text(at)
    assert "**Audiencia:** Gestión del proyecto (PM)" in text
    assert "regla `planificacion`, mantenida del turno anterior" in text
    system = next(c.value for c in at.sidebar.code if c.value.startswith("Eres un estimador"))
    assert "quien gestiona el proyecto" in system


def test_sidebar_shows_an_override(api):
    api.session_extra = {"tier": "developer", "tier_rule": "override", "tier_inherited": False}

    text = sidebar_text(send(run_app()))

    assert "**Audiencia:** Equipo técnico (elegida en el formulario)" in text


def test_sidebar_shows_anchors_and_summary(api):
    api.session_extra = {
        "anchors": 1,
        "max_anchors": 4,
        "summarized_turns": 2,
        "conversation_summary": "Se pidió un portal B2B; luego se añadió la app.",
        "compression_cost_usd": 0.004,
    }

    at = send(run_app())

    assert "1 conservado(s) como ancla (máx. 4) y 2 en el resumen" in sidebar_text(at)
    assert "Se pidió un portal B2B; luego se añadió la app." in [
        m.value for m in at.sidebar.markdown
    ]
    system = next(c.value for c in at.sidebar.code if c.value.startswith("Eres un estimador"))
    assert "<resumen_conversacion>" in system


ACB_TRACE = {
    "iterations": [
        {
            "iteration": 1,
            "total_hours": 100,
            "verdict": "revise",
            "issues": [
                {
                    "category": "horas",
                    "severity": "alta",
                    "problem": "Pagos con solo 8 h.",
                    "suggestion": "Súbelo a 40 h.",
                }
            ],
            "decision": "revise",
        },
        {
            "iteration": 2,
            "total_hours": 140,
            "verdict": "accept",
            "issues": [],
            "decision": "accept",
        },
    ],
    "outcome": "accepted",
    "chosen_iteration": 2,
    "llm_calls": 4,
}


def test_review_checkbox_uses_the_acb_endpoint(api):
    at = run_app()
    assert at.checkbox[0].label == "Revisar con un crítico"
    at.checkbox[0].check()
    send(at)

    assert api.turn_requests[0][0] == f"{API}/sesion-1/estimate-acb"


def test_review_trace_is_shown(api):
    api.acb = ACB_TRACE

    at = send(run_app())

    expander = next(e for e in at.expander if e.label.startswith("🔍 Revisión del crítico"))
    assert expander.label == "🔍 Revisión del crítico: 100 h → 140 h · aceptada por el crítico"
    text = "\n".join(m.value for m in expander.markdown)
    assert "**Versión 2 ✅** · 140 h · crítico: accept · boss: accept" in text
    assert "**[alta] horas**: Pagos con solo 8 h. *Súbelo a 40 h.*" in text
