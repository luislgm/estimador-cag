"""El crítico del ciclo Actor-Critic-Boss: revisa una estimación sin modificarla.

Recibe lo mismo que necesitaría un revisor humano (el mensaje del usuario con sus adjuntos,
la memoria del proyecto, la audiencia y la estimación) y devuelve una CriticReview: veredicto
y problemas concretos con su gravedad. No decide qué hacer con ella: eso es cosa del boss.

Si la revisión falla (proveedor caído, respuesta que no valida), devuelve None y el boss
acepta la estimación del actor tal cual: el ciclo de revisión no puede romper el turno.
"""

import structlog

from app.prompts import render_critic_prompt
from app.schemas.critic import CriticReview
from app.schemas.estimation import EstimationResult
from app.services import llm_client
from app.services.llm_client import LLMCallInfo
from app.sessions import ProjectMetadata

log = structlog.get_logger(module=__name__)

CRITIC_MAX_TOKENS = 2000


async def review(
    user_message: str,
    estimate: EstimationResult,
    project_metadata: ProjectMetadata,
    tier: str,
) -> tuple[CriticReview | None, LLMCallInfo | None]:
    system, user = render_critic_prompt(
        user_message, estimate, project_metadata=project_metadata, tier=tier
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        result, info = await llm_client.acomplete_structured(
            messages, CriticReview, max_tokens=CRITIC_MAX_TOKENS
        )
    except llm_client.LLMError as e:
        log.warning("critic_failed", error=str(e))
        return None, None
    log.info(
        "critic_completed",
        verdict=result.verdict,
        issues=[f"{i.severity}:{i.category}" for i in result.issues],
        confidence_pct=result.confidence_pct,
        cost_usd=info.cost_usd,
    )
    return result, info
