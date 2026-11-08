"""Sesiones conversacionales: estimaciones en varios turnos, con memoria y adjuntos.

POST /api/v1/sessions                        crea una sesión vacía
GET  /api/v1/sessions/{session_id}           estado de la sesión (memoria y turnos)
POST /api/v1/sessions/{session_id}/estimate  un turno: transcripción + adjuntos (multipart)
"""

from typing import Annotated

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.guardrails.input import InputGuardrailViolation
from app.routers.estimations import response_fields
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from app.schemas.session import (
    AttachmentInfo,
    SessionCreated,
    SessionEstimationResponse,
    SessionState,
)
from app.services import conversation
from app.services.attachments import MAX_FILES, AttachmentError, extract_text
from app.services.llm_client import LLMError
from app.sessions import Session, SessionNotFound, store

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
    )


@router.post("", response_model=SessionCreated, status_code=status.HTTP_201_CREATED)
async def create_session() -> SessionCreated:
    session = store.create()
    log.info("session_created", session_id=session.id)
    return SessionCreated(session_id=session.id)


@router.get("/{session_id}", response_model=SessionState)
async def get_session(session_id: str) -> SessionState:
    return _state(_get_session(session_id))


@router.post(
    "/{session_id}/estimate",
    response_model=SessionEstimationResponse,
    responses={
        400: {"description": "Un guardrail ha rechazado la transcripción o un adjunto"},
        404: {"description": "La sesión no existe"},
        422: {"description": "Datos o adjuntos no válidos"},
        502: {"description": "Error al llamar al LLM"},
    },
)
async def estimate_in_session(
    session_id: str,
    transcript: Annotated[str, Form(description="Transcripción o mensaje del turno")],
    project_type: Annotated[ProjectType, Form()] = ProjectType.WEB_SAAS,
    detail_level: Annotated[DetailLevel, Form()] = DetailLevel.MEDIUM,
    output_format: Annotated[OutputFormat, Form()] = OutputFormat.PHASES_TABLE,
    attachments: Annotated[
        list[UploadFile] | None, File(description="PDFs o documentos Word (.docx)")
    ] = None,
) -> SessionEstimationResponse:
    session = _get_session(session_id)
    try:
        # Mismas reglas que el endpoint sin sesión (20-2000 caracteres...)
        request = EstimationRequest(
            description=transcript,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
        )
    except ValidationError as e:
        raise RequestValidationError(e.errors()) from e

    files = attachments or []
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=422, detail=f"Como mucho {MAX_FILES} adjuntos")
    try:
        extracted = [extract_text(f.filename or "adjunto", await f.read()) for f in files]
    except AttachmentError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        outcome = await conversation.estimate_turn(session, request, extracted)
    except InputGuardrailViolation as e:
        raise HTTPException(
            status_code=400, detail={"reason": e.reason, "message": e.message}
        ) from e
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"Error al llamar al LLM: {e}") from e

    return SessionEstimationResponse(
        **response_fields(outcome),
        session=_state(session),
        attachments=[
            AttachmentInfo(filename=a.filename, chars=a.chars, truncated=a.truncated)
            for a in extracted
        ],
    )
