"""Tier de audiencia: reglas, prioridad, override, herencia, prompt e integración."""

from unittest.mock import patch

import httpx
import pytest

from app.main import app
from app.prompts import PROMPT_VERSION, render_estimation_prompt
from app.services import llm_client
from app.sessions import store
from app.tiers import Tier, TierResolution, resolve_tier
from tests.factories import make_info, make_request, make_result

# --- Reglas ---


@pytest.mark.parametrize(
    ("message", "tier", "rule"),
    [
        ("Lo presento al comité de dirección el jueves.", Tier.EXECUTIVE, "audiencia_direccion"),
        ("Necesito un resumen ejecutivo para el CEO.", Tier.EXECUTIVE, "audiencia_direccion"),
        ("Es para la gerencia, que tiene que aprobarlo.", Tier.EXECUTIVE, "audiencia_direccion"),
        ("Se lo paso al equipo técnico para que lo revise.", Tier.DEVELOPER, "audiencia_tecnica"),
        ("Lo va a revisar el CTO.", Tier.DEVELOPER, "audiencia_tecnica"),
        ("Es para el jefe de proyecto del cliente.", Tier.PM, "audiencia_pm"),
        ("El product owner quiere verlo.", Tier.PM, "audiencia_pm"),
        ("¿Cuál sería el retorno de la inversión?", Tier.EXECUTIVE, "negocio"),
        ("Necesitamos el business case.", Tier.EXECUTIVE, "negocio"),
        ("Organízalo en sprints de dos semanas con hitos claros.", Tier.PM, "planificacion"),
        ("Dame un cronograma de entregas.", Tier.PM, "planificacion"),
        ("Despliegue en Kubernetes con microservicios.", Tier.DEVELOPER, "detalle_tecnico"),
        ("Detalla los endpoints de la API REST.", Tier.DEVELOPER, "detalle_tecnico"),
    ],
)
def test_rules(message, tier, rule):
    assert resolve_tier(message) == TierResolution(tier, rule)


def test_explicit_audience_wins_over_topic_signals():
    # Habla de Kubernetes (técnico), pero lo va a leer la dirección
    message = "Para el comité de dirección: ¿cuánto supone migrar a Kubernetes?"

    assert resolve_tier(message) == TierResolution(Tier.EXECUTIVE, "audiencia_direccion")


def test_no_signals_is_default():
    message = "Una app para que los clientes de una cafetería acumulen puntos."

    assert resolve_tier(message) == TierResolution(Tier.DEFAULT, "sin_señales")


def test_override_always_wins():
    message = "Lo presento al comité de dirección."

    assert resolve_tier(message, override=Tier.DEVELOPER) == TierResolution(
        Tier.DEVELOPER, "override"
    )


def test_inherits_the_previous_tier_when_the_message_gives_no_clues():
    previous = TierResolution(Tier.PM, "planificacion")

    resolved = resolve_tier("Quita la descarga de facturas.", previous=previous)

    assert resolved == TierResolution(Tier.PM, "planificacion", inherited=True)


def test_a_new_signal_replaces_the_previous_tier():
    previous = TierResolution(Tier.PM, "planificacion")

    assert resolve_tier("Detalla la arquitectura.", previous=previous).tier == Tier.DEVELOPER


def test_default_is_not_inherited():
    previous = TierResolution(Tier.DEFAULT, "sin_señales")

    assert resolve_tier("Añade login con Google.", previous=previous) == TierResolution(
        Tier.DEFAULT, "sin_señales"
    )


# --- Prompt ---


def test_default_tier_leaves_the_prompt_unchanged():
    request = make_request()

    plain, _ = render_estimation_prompt(request, PROMPT_VERSION)
    default, _ = render_estimation_prompt(request, PROMPT_VERSION, tier=Tier.DEFAULT)

    assert default == plain
    assert "<audiencia>" not in plain


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        (Tier.EXECUTIVE, "la leerá la dirección"),
        (Tier.PM, "quien gestiona el proyecto"),
        (Tier.DEVELOPER, "el equipo técnico"),
    ],
)
def test_each_tier_adds_its_audience_block(tier, expected):
    system, _ = render_estimation_prompt(make_request(), PROMPT_VERSION, tier=tier)

    block = system.split("<audiencia>")[1].split("</audiencia>")[0]
    assert expected in block
    assert "no las reglas ni el formato" in block
    # Solo el bloque de su tier
    others = {"la leerá la dirección", "quien gestiona el proyecto", "el equipo técnico"}
    assert not any(text in block for text in others - {expected})


# --- Integración con las sesiones ---

FORM = {"project_type": "web_saas", "detail_level": "summary", "output_format": "phases_table"}
TRANSCRIPT = "Una distribuidora de bebidas quiere un portal B2B para que los bares hagan pedidos."


@pytest.fixture
async def client():
    store.clear()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    store.clear()


class RecordingLLM:
    def __init__(self):
        self.systems = []

    async def __call__(self, messages, response_model, max_tokens):
        self.systems.append(messages[0]["content"])
        return make_result(), make_info()


async def test_tier_is_resolved_inherited_and_overridden_across_turns(client):
    llm = RecordingLLM()
    session_id = (await client.post("/api/v1/sessions")).json()["session_id"]
    url = f"/api/v1/sessions/{session_id}/estimate"

    with patch.object(llm_client, "acomplete_structured", llm):
        first = await client.post(
            url, data=FORM | {"transcript": f"{TRANSCRIPT} Lo presento al comité de dirección."}
        )
        second = await client.post(
            url, data=FORM | {"transcript": "Quita la descarga de facturas, por favor."}
        )
        third = await client.post(
            url,
            data=FORM | {"transcript": "Ahora la versión para revisar.", "tier": "developer"},
        )

    states = [r.json()["session"] for r in (first, second, third)]
    assert [(s["tier"], s["tier_rule"], s["tier_inherited"]) for s in states] == [
        ("executive", "audiencia_direccion", False),
        ("executive", "audiencia_direccion", True),
        ("developer", "override", False),
    ]
    assert "la leerá la dirección" in llm.systems[0]
    assert "la leerá la dirección" in llm.systems[1]
    assert "el equipo técnico" in llm.systems[2]
    state = (await client.get(f"/api/v1/sessions/{session_id}")).json()
    assert state["tier"] == "developer"


async def test_failed_turn_does_not_change_the_tier(client):
    async def failing(*args, **kwargs):
        raise llm_client.LLMError("timeout")

    session_id = (await client.post("/api/v1/sessions")).json()["session_id"]
    with patch.object(llm_client, "acomplete_structured", failing):
        response = await client.post(
            f"/api/v1/sessions/{session_id}/estimate",
            data=FORM | {"transcript": f"{TRANSCRIPT} Es para el CEO."},
        )

    assert response.status_code == 502
    assert (await client.get(f"/api/v1/sessions/{session_id}")).json()["tier"] is None


async def test_unknown_tier_returns_422(client):
    session_id = (await client.post("/api/v1/sessions")).json()["session_id"]

    response = await client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data=FORM | {"transcript": TRANSCRIPT, "tier": "ceo"},
    )

    assert response.status_code == 422
