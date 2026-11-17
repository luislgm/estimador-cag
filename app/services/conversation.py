"""Un turno de una conversación de estimación (sesión con memoria).

    0. Tier de audiencia del turno (override, reglas sobre el mensaje o el del turno anterior)
    1. Guardrails de entrada sobre la transcripción y sobre el texto de cada adjunto
    2. Prompt versionado: system con la memoria del proyecto (project_metadata) y user con la
       transcripción y los adjuntos
    3. LLM con salida estructurada: system prompt regenerado (memoria + resumen de los turnos
       antiguos), anclas, turnos de la ventana y el mensaje actual. Con review=True, ciclo
       Actor-Critic-Boss: un crítico revisa la estimación y el actor la rehace si hace falta
    4. Guardrail de salida y markdown según el formato
    5. Se guarda el turno en el historial y se actualiza la memoria con la nueva estimación
    6. Compresión: los turnos que salen de la ventana pasan a ser anclas o al resumen

No usa las cachés: la respuesta depende del historial y de la memoria de la sesión, así que
dos peticiones con el mismo texto no son equivalentes.
"""

import structlog

from app.config import get_settings
from app.guardrails.input import InputGuardrailViolation, check_input
from app.guardrails.output import enforce_scope
from app.prompts import PROMPT_VERSION, render_estimation_prompt
from app.schemas.critic import CriticIssue
from app.schemas.estimation import EstimationRequest, EstimationResult
from app.services import boss, compression, critic, llm_client
from app.services.attachments import Attachment
from app.services.estimation import MAX_TOKENS, EstimationOutcome, build_outcome
from app.services.llm_client import LLMCallInfo
from app.services.render import render_markdown
from app.sessions import ProjectMetadata, Session
from app.tiers import Tier, resolve_tier

log = structlog.get_logger(module=__name__)


async def estimate_turn(
    session: Session,
    request: EstimationRequest,
    attachments: list[Attachment],
    tier_override: Tier | None = None,
    review: bool = False,
) -> EstimationOutcome:
    # 0. Para quién es la respuesta (solo se guarda en la sesión si el turno sale bien)
    tier = resolve_tier(request.description, previous=session.tier, override=tier_override)

    # 1. Guardrails: el texto de los adjuntos también llega al LLM, así que también se revisa
    await check_input(request.description)
    for attachment in attachments:
        try:
            await check_input(attachment.text)
        except InputGuardrailViolation as e:
            raise InputGuardrailViolation(
                f"Adjunto '{attachment.filename}': {e.message}", reason=e.reason
            ) from e

    # 2. Mensaje del usuario: la transcripción y los adjuntos. Con memoria (a partir del segundo
    #    turno) se presenta como un cambio sobre el mismo proyecto, no como uno nuevo: si no, el
    #    modelo a veces estima solo lo que se añade (lo detectaron las evals)
    _, user = render_estimation_prompt(
        request, PROMPT_VERSION, project_metadata=session.metadata, attachments=attachments
    )
    turns_sent = len(session.history)
    previous: list[EstimationResult] = []  # estimaciones del actor en este turno

    # 3. El actor: array de mensajes (system prompt regenerado con la memoria actual y el
    #    resumen de los turnos antiguos + anclas + ventana + mensaje actual) y llamada al LLM.
    #    En el ciclo Actor-Critic-Boss, el feedback del crítico va en el system prompt.
    async def actor(
        feedback: list[CriticIssue] | None = None,
    ) -> tuple[EstimationResult, LLMCallInfo]:
        def build_system(metadata: ProjectMetadata, summary: str | None) -> str:
            system, _ = render_estimation_prompt(
                request,
                PROMPT_VERSION,
                project_metadata=metadata,
                conversation_summary=summary,
                tier=tier.tier,
                critic_feedback=feedback or (),
                previous_estimate=render_markdown(previous[-1], request) if feedback else None,
            )
            return system

        messages = [
            *session.history.to_messages_list(session.metadata, build_system),
            {"role": "user", "content": user},
        ]
        result, info = await llm_client.acomplete_structured(
            messages, EstimationResult, max_tokens=MAX_TOKENS
        )
        # 4. Guardrail de salida (el crítico revisa ya la versión que vería el usuario)
        result = enforce_scope(result)
        previous.append(result)
        return result, info

    async def reviewer(result: EstimationResult):
        return await critic.review(user, result, session.metadata, tier.tier)

    if review:
        cycle = await boss.run(actor, reviewer, get_settings().acb_max_iterations)
        result, info, trace = cycle.result, llm_client.combine_calls(cycle.calls), cycle.trace
    else:
        (result, info), trace = await actor(), None
    outcome = build_outcome(request, result, info)
    outcome.acb = trace

    # 5. Historial y memoria. Como respuesta del asistente se guarda la estimación en markdown
    #    (fases, horas, totales, equipo): basta para refinarla en los turnos siguientes. No se
    #    guarda en JSON: con respuestas JSON en el historial, el modelo tiende a imitarlas y a
    #    anidar JSON como texto dentro de la herramienta, y la respuesta deja de validar. Del
    #    ciclo de revisión solo queda la versión final: las intermedias no son conversación.
    evicted = session.history.add_turn(user, outcome.text)
    session.metadata = session.metadata.updated_with(result)
    session.tier = tier

    # 6. Compresión de los turnos que salen de la ventana (sin ella, simplemente se descartan)
    if evicted and get_settings().session_compression_enabled:
        report = await compression.compress(session.history, evicted)
        session.compression_cost_usd += report.cost_usd
    log.info(
        "session_turn_completed",
        session_id=session.id,
        turn=session.history.total_turns,
        history_turns_sent=turns_sent,
        tier=tier.tier.value,
        tier_rule=tier.rule,
        tier_inherited=tier.inherited,
        review=review,
        acb_outcome=trace.outcome if trace else None,
        attachments=len(attachments),
        project_name=session.metadata.project_name,
        total_cost_eur=result.total_cost_eur,
    )
    return outcome
