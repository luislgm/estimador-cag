"""Sesiones conversacionales: estimaciones en varios turnos, con memoria y adjuntos.

POST /api/v1/sessions                        crea una sesión vacía
GET  /api/v1/sessions/{session_id}           estado de la sesión (memoria y turnos)
POST /api/v1/sessions/{session_id}/estimate  un turno: transcripción + adjuntos (multipart)
POST /api/v1/sessions/{session_id}/estimate-acb  lo mismo, con revisión Actor-Critic-Boss
"""

from dataclasses import dataclass
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.guardrails.input import InputGuardrailViolation
from app.routers.estimations import response_fields
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from app.schemas.session import (
    AttachmentInfo,
    SessionACBResponse,
    SessionCreated,
    SessionEstimationResponse,
    SessionState,
)
from app.services import conversation
from app.services.attachments import MAX_FILES, AttachmentError, extract_text
from app.services.llm_client import LLMError
from app.sessions import Session, SessionNotFound, store
from app.tiers import Tier

log = structlog.get_logger(module=__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _get_session(session_id: str) -> Session:
    try:
        return store.get(session_id)
    except SessionNotFound:
        raise HTTPException(
            status_code=404,
            detail="Sesión no encontrada. Puede que el servicio se haya reiniciado: crea otra.",
        ) from None


def _state(session: Session) -> SessionState:
    return SessionState(
        session_id=session.id,
        turns=session.history.total_turns,
        turns_in_window=len(session.history),
        max_turns=session.history.max_turns,
        project_metadata=session.metadata,
        anchors=len(session.history.anchors),
        max_anchors=session.history.max_anchors,
        summarized_turns=session.history.summarized_turns,
        conversation_summary=session.history.summary,
        compression_cost_usd=round(session.compression_cost_usd, 6),
        tier=session.tier.tier if session.tier else None,
        tier_rule=session.tier.rule if session.tier else None,
        tier_inherited=session.tier.inherited if session.tier else False,
    )


@router.post("", response_model=SessionCreated, status_code=status.HTTP_201_CREATED)
async def create_session() -> SessionCreated:
    session = store.create()
    log.info("session_created", session_id=session.id)
    return SessionCreated(session_id=session.id)


@router.get("/{session_id}", response_model=SessionState)
async def get_session(session_id: str) -> SessionState:
    return _state(_get_session(session_id))


@dataclass
class TurnForm:
    """Los campos del formulario de un turno, compartidos por /estimate y /estimate-acb."""

    transcript: str
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat
    tier: Tier | None
    attachments: list[UploadFile]


def turn_form(
    transcript: Annotated[str, Form(description="Transcripción o mensaje del turno")],
    project_type: Annotated[ProjectType, Form()] = ProjectType.WEB_SAAS,
    detail_level: Annotated[DetailLevel, Form()] = DetailLevel.MEDIUM,
    output_format: Annotated[OutputFormat, Form()] = OutputFormat.PHASES_TABLE,
    tier: Annotated[
        Tier | None,
        Form(description="Audiencia. Si no se indica, se deduce del mensaje y de los turnos"),
    ] = None,
    attachments: Annotated[
        list[UploadFile] | None, File(description="PDFs o documentos Word (.docx)")
    ] = None,
) -> TurnForm:
    return TurnForm(transcript, project_type, detail_level, output_format, tier, attachments or [])


TURN_RESPONSES = {
    400: {"description": "Un guardrail ha rechazado la transcripción o un adjunto"},
    404: {"description": "La sesión no existe"},
    422: {"description": "Datos o adjuntos no válidos"},
    502: {"description": "Error al llamar al LLM"},
}


async def _run_turn(session_id: str, form: TurnForm, review: bool) -> dict:
    """Valida el turno, lo ejecuta y devuelve los campos comunes de la respuesta."""
    session = _get_session(session_id)
    try:
        # Mismas reglas que el endpoint sin sesión (20-2000 caracteres...)
        request = EstimationRequest(
            description=form.transcript,
            project_type=form.project_type,
            detail_level=form.detail_level,
            output_format=form.output_format,
        )
    except ValidationError as e:
        raise RequestValidationError(e.errors()) from e

    if len(form.attachments) > MAX_FILES:
        raise HTTPException(status_code=422, detail=f"Como mucho {MAX_FILES} adjuntos")
    try:
        extracted = [
            extract_text(f.filename or "adjunto", await f.read()) for f in form.attachments
        ]
    except AttachmentError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        outcome = await conversation.estimate_turn(
            session, request, extracted, tier_override=form.tier, review=review
        )
    except InputGuardrailViolation as e:
        raise HTTPException(
            status_code=400, detail={"reason": e.reason, "message": e.message}
        ) from e
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"Error al llamar al LLM: {e}") from e

    return {
        **response_fields(outcome),
        "session": _state(session),
        "attachments": [
            AttachmentInfo(filename=a.filename, chars=a.chars, truncated=a.truncated)
            for a in extracted
        ],
        "acb": outcome.acb,
    }


@router.post(
    "/{session_id}/estimate", response_model=SessionEstimationResponse, responses=TURN_RESPONSES
)
async def estimate_in_session(
    session_id: str, form: Annotated[TurnForm, Depends(turn_form)]
) -> SessionEstimationResponse:
    fields = await _run_turn(session_id, form, review=False)
    fields.pop("acb")
    return SessionEstimationResponse(**fields)


@router.post(
    "/{session_id}/estimate-acb", response_model=SessionACBResponse, responses=TURN_RESPONSES
)
async def estimate_in_session_with_review(
    session_id: str, form: Annotated[TurnForm, Depends(turn_form)]
) -> SessionACBResponse:
    """Igual que /estimate, pero con el ciclo Actor-Critic-Boss: un crítico revisa la
    estimación y, si encuentra problemas graves, el actor la rehace con su feedback (como
    mucho ACB_MAX_ITERATIONS vueltas). La respuesta incluye la traza del ciclo en `acb`, y el
    uso y el coste suman todas las llamadas del turno (actor y crítico)."""
    return SessionACBResponse(**await _run_turn(session_id, form, review=True))
