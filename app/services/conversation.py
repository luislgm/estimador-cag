"""Un turno de una conversación de estimación (sesión con memoria).

    1. Guardrails de entrada sobre la transcripción y sobre el texto de cada adjunto
    2. Prompt versionado: system con la memoria del proyecto (project_metadata) y user con la
       transcripción y los adjuntos
    3. LLM con salida estructurada, enviando los turnos de la ventana del historial
    4. Guardrail de salida y markdown según el formato
    5. Se guarda el turno en el historial y se actualiza la memoria con la nueva estimación

No usa las cachés: la respuesta depende del historial y de la memoria de la sesión, así que
dos peticiones con el mismo texto no son equivalentes.
"""

import structlog

from app.guardrails.input import InputGuardrailViolation, check_input
from app.guardrails.output import enforce_scope
from app.prompts import PROMPT_VERSION, render_estimation_prompt
from app.schemas.estimation import EstimationRequest, EstimationResult
from app.services import llm_client
from app.services.attachments import Attachment
from app.services.estimation import MAX_TOKENS, EstimationOutcome, build_outcome
from app.sessions import ProjectMetadata, Session

log = structlog.get_logger(module=__name__)


async def estimate_turn(
    session: Session, request: EstimationRequest, attachments: list[Attachment]
) -> EstimationOutcome:
    # 1. Guardrails: el texto de los adjuntos también llega al LLM, así que también se revisa
    await check_input(request.description)
    for attachment in attachments:
        try:
            await check_input(attachment.text)
        except InputGuardrailViolation as e:
            raise InputGuardrailViolation(
                f"Adjunto '{attachment.filename}': {e.message}", reason=e.reason
            ) from e

    # 2. Mensaje del usuario: la transcripción y los adjuntos
    _, user = render_estimation_prompt(request, PROMPT_VERSION, attachments=attachments)

    # 3. Array de mensajes: system prompt regenerado con la memoria actual + turnos de la
    #    ventana deslizante (to_messages_list) + el mensaje actual
    def build_system(metadata: ProjectMetadata) -> str:
        system, _ = render_estimation_prompt(request, PROMPT_VERSION, project_metadata=metadata)
        return system

    messages = [
        *session.history.to_messages_list(session.metadata, build_system),
        {"role": "user", "content": user},
    ]
    turns_sent = len(session.history)
    result, info = await llm_client.acomplete_structured(
        messages, EstimationResult, max_tokens=MAX_TOKENS
    )

    # 4. Guardrail de salida y markdown
    result = enforce_scope(result)
    outcome = build_outcome(request, result, info)

    # 5. Historial y memoria. Como respuesta del asistente se guarda la estimación en markdown
    #    (fases, horas, totales, equipo): basta para refinarla en los turnos siguientes. No se
    #    guarda en JSON: con respuestas JSON en el historial, el modelo tiende a imitarlas y a
    #    anidar JSON como texto dentro de la herramienta, y la respuesta deja de validar.
    session.history.add_turn(user, outcome.text)
    session.metadata = session.metadata.updated_with(result)
    log.info(
        "session_turn_completed",
        session_id=session.id,
        turn=session.history.total_turns,
        history_turns_sent=turns_sent,
        attachments=len(attachments),
        project_name=session.metadata.project_name,
        total_cost_eur=result.total_cost_eur,
    )
    return outcome
