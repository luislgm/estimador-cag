"""Ciclo Actor-Critic-Boss: el boss, el crítico, los prompts y el endpoint."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import ValidationError

from app.main import app
from app.prompts import PROMPT_VERSION, render_critic_prompt, render_estimation_prompt
from app.schemas.critic import CriticIssue, CriticReview
from app.schemas.estimation import EstimationResult
from app.services import boss, critic, llm_client
from app.sessions import ProjectMetadata, store
from tests.factories import make_info, make_phase, make_request, make_result

HIGH = CriticIssue(
    category="horas",
    severity="alta",
    problem="La integración de pagos tiene solo 8 h.",
    suggestion="Sube la fase de pagos a unas 40 h.",
)
LOW = CriticIssue(
    category="riesgos",
    severity="baja",
    problem="Falta mencionar el RGPD.",
    suggestion="Añádelo a los riesgos.",
)
ACCEPT = CriticReview(verdict="accept", issues=[LOW], confidence_pct=80)
REVISE = CriticReview(verdict="revise", issues=[HIGH, LOW], confidence_pct=75)


def result_with(hours: int, name: str = "Proyecto") -> EstimationResult:
    return make_result(
        project_name=name, phases=[make_phase("Diseño", 20), make_phase("Desarrollo", hours - 20)]
    )


class ScriptedActor:
    """Devuelve las estimaciones en orden y guarda el feedback que recibe."""

    def __init__(self, *results):
        self.results = list(results)
        self.feedback = []

    async def __call__(self, feedback=None):
        self.feedback.append(feedback)
        item = self.results[len(self.feedback) - 1]
        if isinstance(item, Exception):
            raise item
        return item, make_info(cost_usd=0.01)


def scripted_critic(*reviews):
    """Devuelve las revisiones en orden (None: la revisión falla)."""
    pending = list(reviews)

    async def review_fn(result):
        current = pending.pop(0)
        return (current, make_info(cost_usd=0.002)) if current else (None, None)

    return review_fn


# --- Esquema del crítico ---


def test_revise_needs_a_serious_issue():
    with pytest.raises(ValidationError, match="al menos un problema de gravedad alta o media"):
        CriticReview(verdict="revise", issues=[LOW], confidence_pct=70)

    assert REVISE.serious_issues == [HIGH]
    assert ACCEPT.serious_issues == []


# --- Boss ---


async def test_accepts_at_once_when_the_critic_is_happy():
    actor = ScriptedActor(result_with(100))

    cycle = await boss.run(actor, scripted_critic(ACCEPT), max_iterations=2)

    assert cycle.result.total_hours == 100
    assert cycle.trace.outcome == "accepted"
    assert [i.decision for i in cycle.trace.iterations] == ["accept"]
    assert (cycle.trace.chosen_iteration, cycle.trace.llm_calls) == (1, 2)
    assert actor.feedback == [None]


async def test_revises_with_the_serious_issues_and_then_accepts():
    actor = ScriptedActor(result_with(100), result_with(140))

    cycle = await boss.run(actor, scripted_critic(REVISE, ACCEPT), max_iterations=2)

    assert cycle.result.total_hours == 140
    assert actor.feedback == [None, [HIGH]]  # solo los problemas graves vuelven al actor
    assert [(i.total_hours, i.decision) for i in cycle.trace.iterations] == [
        (100, "revise"),
        (140, "accept"),
    ]
    assert (cycle.trace.outcome, cycle.trace.chosen_iteration, cycle.trace.llm_calls) == (
        "accepted",
        2,
        4,
    )
    assert len(cycle.calls) == 4


async def test_accept_verdict_wins_even_with_serious_issues_listed():
    lenient = CriticReview.model_construct(verdict="accept", issues=[HIGH], confidence_pct=60)

    cycle = await boss.run(ScriptedActor(result_with(100)), scripted_critic(lenient), 3)

    assert cycle.trace.outcome == "accepted"


async def test_without_iterations_left_returns_the_version_with_fewest_serious_issues():
    two_issues = CriticReview(verdict="revise", issues=[HIGH, HIGH], confidence_pct=70)
    actor = ScriptedActor(result_with(100), result_with(150), result_with(90))

    cycle = await boss.run(actor, scripted_critic(two_issues, REVISE, two_issues), 3)

    assert cycle.trace.outcome == "max_iterations"
    assert [i.decision for i in cycle.trace.iterations] == ["revise", "revise", "stop"]
    assert cycle.trace.chosen_iteration == 2
    assert cycle.result.total_hours == 150


async def test_ties_go_to_the_latest_version():
    actor = ScriptedActor(result_with(100), result_with(150))

    cycle = await boss.run(actor, scripted_critic(REVISE, REVISE), 2)

    assert cycle.trace.chosen_iteration == 2


async def test_a_failed_review_accepts_the_estimate_as_it_is():
    cycle = await boss.run(ScriptedActor(result_with(100)), scripted_critic(None), 2)

    assert cycle.trace.outcome == "critic_failed"
    assert cycle.trace.iterations[0].verdict is None
    assert (cycle.result.total_hours, cycle.trace.llm_calls) == (100, 1)


async def test_a_failed_revision_keeps_the_best_previous_version():
    actor = ScriptedActor(result_with(100), llm_client.LLMError("timeout"))

    cycle = await boss.run(actor, scripted_critic(REVISE), 2)

    assert cycle.trace.outcome == "revision_failed"
    assert (cycle.result.total_hours, cycle.trace.chosen_iteration) == (100, 1)


async def test_a_failed_first_estimate_fails_the_turn():
    actor = ScriptedActor(llm_client.LLMError("timeout"))

    with pytest.raises(llm_client.LLMError):
        await boss.run(actor, scripted_critic(ACCEPT), 2)


# --- Crítico ---


async def test_critic_returns_the_review_and_its_cost():
    llm = AsyncMock(return_value=(REVISE, make_info()))
    metadata = ProjectMetadata(project_name="Portal B2B", agreed_scope="Pedidos y facturas")

    with patch.object(llm_client, "acomplete_structured", llm):
        review, info = await critic.review("Mensaje del usuario", result_with(100), metadata, "pm")

    assert review == REVISE and info == make_info()
    messages, response_model = llm.await_args.args[:2]
    assert response_model is CriticReview
    assert "<audiencia>pm</audiencia>" in messages[1]["content"]
    assert "Alcance acordado en turnos anteriores: Pedidos y facturas" in messages[1]["content"]


async def test_a_failing_critic_returns_nothing():
    llm = AsyncMock(side_effect=llm_client.LLMError("timeout"))

    with patch.object(llm_client, "acomplete_structured", llm):
        assert await critic.review("m", result_with(100), ProjectMetadata(), "default") == (
            None,
            None,
        )


def test_critic_prompt_contains_the_estimate_and_the_message():
    system, user = render_critic_prompt("Quiero un portal B2B.", result_with(100))

    assert "revisor sénior" in system
    assert "<mensaje_usuario>\nQuiero un portal B2B.\n</mensaje_usuario>" in user
    assert '"total_hours": 100' in user
    assert "(primer mensaje de la conversación)" in user


def test_revision_block_only_appears_with_feedback():
    request = make_request()

    plain, _ = render_estimation_prompt(request, PROMPT_VERSION)
    revised, _ = render_estimation_prompt(
        request, PROMPT_VERSION, critic_feedback=[HIGH], previous_estimate="## Estimación: X"
    )

    assert "<revision_critica>" not in plain
    block = revised.split("<revision_critica>")[1].split("</revision_critica>")[0]
    assert "## Estimación: X" in block
    assert "- [alta] horas: La integración de pagos tiene solo 8 h." in block
    assert "Sugerencia: Sube la fase de pagos a unas 40 h." in block


# --- Endpoint ---

FORM = {"project_type": "web_saas", "detail_level": "summary", "output_format": "phases_table"}
TRANSCRIPT = "Una distribuidora de bebidas quiere un portal B2B para que los bares hagan pedidos."


@pytest.fixture
async def client():
    store.clear()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    store.clear()


class FakeLLM:
    """Actor y crítico a la vez: responde según el response_model que le piden."""

    def __init__(self, estimates, reviews):
        self.estimates, self.reviews = list(estimates), list(reviews)
        self.actor_systems = []

    async def __call__(self, messages, response_model, max_tokens):
        if response_model is CriticReview:
            return self.reviews.pop(0), make_info(cost_usd=0.002)
        self.actor_systems.append(messages[0]["content"])
        return self.estimates.pop(0), make_info(cost_usd=0.01)


async def test_estimate_acb_returns_the_revised_estimate_and_its_trace(client):
    llm = FakeLLM([result_with(100, "Portal"), result_with(140, "Portal")], [REVISE, ACCEPT])
    session_id = (await client.post("/api/v1/sessions")).json()["session_id"]

    with patch.object(llm_client, "acomplete_structured", llm):
        response = await client.post(
            f"/api/v1/sessions/{session_id}/estimate-acb", data=FORM | {"transcript": TRANSCRIPT}
        )

    body = response.json()
    assert response.status_code == 200
    assert body["result"]["total_hours"] == 140
    assert body["acb"]["outcome"] == "accepted"
    assert [i["total_hours"] for i in body["acb"]["iterations"]] == [100, 140]
    assert body["acb"]["llm_calls"] == 4
    # Uso y coste: la suma de las 4 llamadas (2 del actor y 2 del crítico)
    assert body["cost_usd"] == pytest.approx(2 * 0.01 + 2 * 0.002)
    assert body["usage"]["input_tokens"] == 4 * make_info().input_tokens
    # El actor rehace con el feedback y su estimación anterior
    assert "<revision_critica>" not in llm.actor_systems[0]
    assert "La integración de pagos tiene solo 8 h." in llm.actor_systems[1]
    assert "| **Total** | **4** | **100** |" in llm.actor_systems[1]
    # En el historial solo queda la versión final
    assert body["session"]["turns"] == 1
    assert body["text"].startswith("## Estimación: Portal")


async def test_plain_estimate_has_no_trace_and_one_call(client):
    llm = FakeLLM([result_with(100)], [])
    session_id = (await client.post("/api/v1/sessions")).json()["session_id"]

    with patch.object(llm_client, "acomplete_structured", llm):
        body = (
            await client.post(
                f"/api/v1/sessions/{session_id}/estimate", data=FORM | {"transcript": TRANSCRIPT}
            )
        ).json()

    assert "acb" not in body
    assert body["cost_usd"] == 0.01


async def test_estimate_acb_shares_the_validation_of_estimate(client):
    session_id = (await client.post("/api/v1/sessions")).json()["session_id"]

    short = await client.post(
        f"/api/v1/sessions/{session_id}/estimate-acb", data=FORM | {"transcript": "corta"}
    )
    missing = await client.post(
        "/api/v1/sessions/no-existe/estimate-acb", data=FORM | {"transcript": TRANSCRIPT}
    )

    assert (short.status_code, missing.status_code) == (422, 404)


def test_combine_calls_adds_up_tokens_costs_and_attempts():
    calls = [make_info(cost_usd=0.01, attempts=2), make_info(cost_usd=None, model="otro")]

    combined = llm_client.combine_calls(calls)

    assert (combined.input_tokens, combined.output_tokens) == (7000, 2400)
    assert (combined.cost_usd, combined.attempts, combined.model) == (0.01, 3, "otro")
    assert llm_client.combine_calls([make_info(cost_usd=None)]).cost_usd is None
