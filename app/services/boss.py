"""El boss del ciclo Actor-Critic-Boss: decide qué hacer tras cada revisión.

    actor -> crítico -> boss ─┬─ sin problemas graves      -> acepta
                              ├─ problemas y quedan vueltas -> el actor rehace con el feedback
                              └─ sin vueltas                -> devuelve la mejor versión

El boss no llama al LLM: solo decide. Así su lógica es determinista, se puede probar sin
modelo y cada decisión queda en la traza. El actor y el crítico se le pasan como funciones,
de modo que no sabe nada de prompts ni de sesiones:

    actor(feedback)  -> (EstimationResult, LLMCallInfo)   feedback: problemas de la vuelta anterior
    critic(result)   -> (CriticReview | None, LLMCallInfo | None)   None: la revisión falló

Criterios:
- Solo se rehace si el crítico pide revisar y hay problemas de gravedad alta o media; los
  detalles menores no justifican otra llamada.
- Si se agotan las vueltas, se devuelve la versión con menos problemas graves (a igualdad, la
  más reciente, que ya incorpora feedback).
- Degradación: si el crítico falla, se acepta la estimación sin revisar; si falla la nueva
  versión del actor, se devuelve la mejor anterior. El turno del usuario no se rompe por el
  ciclo de revisión, solo si falla la primera estimación (como sin ACB).
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import structlog

from app.schemas.critic import ACBIteration, ACBTrace, CriticIssue, CriticReview
from app.schemas.estimation import EstimationResult
from app.services.llm_client import LLMCallInfo, LLMError

log = structlog.get_logger(module=__name__)

Actor = Callable[[list[CriticIssue] | None], Awaitable[tuple[EstimationResult, LLMCallInfo]]]
Critic = Callable[[EstimationResult], Awaitable[tuple[CriticReview | None, LLMCallInfo | None]]]


@dataclass
class BossResult:
    result: EstimationResult
    calls: list[LLMCallInfo]  # todas las llamadas al LLM (actor y crítico), en orden
    trace: ACBTrace


async def run(actor: Actor, critic: Critic, max_iterations: int) -> BossResult:
    calls: list[LLMCallInfo] = []
    versions: list[tuple[EstimationResult, int]] = []  # (estimación, problemas graves)
    iterations: list[ACBIteration] = []
    feedback: list[CriticIssue] | None = None
    outcome = "max_iterations"

    for number in range(1, max_iterations + 1):
        try:
            result, info = await actor(feedback)
        except LLMError as e:
            if not versions:
                raise  # la primera estimación falla: el turno falla, igual que sin ACB
            log.warning("acb_revision_failed", iteration=number, error=str(e))
            outcome = "revision_failed"
            break
        calls.append(info)

        review, review_info = await critic(result)
        if review_info is not None:
            calls.append(review_info)

        if review is None:
            versions.append((result, 0))
            iterations.append(
                ACBIteration(iteration=number, total_hours=result.total_hours, decision="stop")
            )
            outcome = "critic_failed"
            break

        serious = review.serious_issues
        versions.append((result, len(serious)))
        if review.verdict == "accept" or not serious:
            decision, outcome = "accept", "accepted"
        elif number == max_iterations:
            decision, outcome = "stop", "max_iterations"
        else:
            decision, feedback = "revise", serious
        iterations.append(
            ACBIteration(
                iteration=number,
                total_hours=result.total_hours,
                verdict=review.verdict,
                issues=review.issues,
                decision=decision,
            )
        )
        if decision != "revise":
            break

    if outcome in ("accepted", "critic_failed"):
        chosen = len(versions)
    else:
        # La versión con menos problemas graves; a igualdad, la más reciente
        fewest = min(serious for _, serious in versions)
        chosen = max(i for i, (_, serious) in enumerate(versions, start=1) if serious == fewest)

    trace = ACBTrace(
        iterations=iterations, outcome=outcome, chosen_iteration=chosen, llm_calls=len(calls)
    )
    log.info(
        "acb_completed",
        outcome=outcome,
        iterations=len(iterations),
        chosen_iteration=chosen,
        llm_calls=len(calls),
        hours=[i.total_hours for i in iterations],
    )
    return BossResult(result=versions[chosen - 1][0], calls=calls, trace=trace)
