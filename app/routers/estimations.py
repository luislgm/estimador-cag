"""POST /api/v1/estimate: petición tipada, estimación estructurada y validada.

El router no tiene lógica de negocio (está en app/services/estimation.py): registra la
petición, llama al pipeline y traduce los errores a códigos HTTP.
"""

from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, HTTPException

from app.guardrails.input import InputGuardrailViolation
from app.prompts import PROMPT_VERSION
from app.schemas.estimation import EstimationRequest, EstimationResponse, TokenUsage
from app.services import estimation
from app.services.estimation import EstimationOutcome
from app.services.llm_client import LLMError

log = structlog.get_logger(module=__name__)

router = APIRouter(tags=["estimations"])


@router.post(
    "/estimate",
    response_model=EstimationResponse,
    responses={
        400: {"description": "Un guardrail de entrada ha rechazado la descripción"},
        502: {"description": "Error al llamar al LLM o respuesta inválida tras los reintentos"},
    },
)
async def estimate(request: EstimationRequest) -> EstimationResponse:
    log.info(
        "estimation_requested",
        prompt_version=PROMPT_VERSION,
        project_type=request.project_type.value,
        detail_level=request.detail_level.value,
        output_format=request.output_format.value,
        description_chars=len(request.description),
    )
    try:
        outcome = await estimation.estimate(request)
    except InputGuardrailViolation as e:
        raise HTTPException(
            status_code=400, detail={"reason": e.reason, "message": e.message}
        ) from e
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"Error al llamar al LLM: {e}") from e

    return EstimationResponse(**response_fields(outcome))


def response_fields(outcome: EstimationOutcome) -> dict:
    """Campos de EstimationResponse a partir del resultado del pipeline (lo reutiliza el
    endpoint de sesiones, cuya respuesta amplía EstimationResponse)."""
    info = outcome.info
    return {
        "text": outcome.text,
        "prompt_version": outcome.prompt_version,
        "result": outcome.result,
        "model": info.model,
        "provider": info.provider,
        "usage": TokenUsage(
            input_tokens=info.input_tokens,
            output_tokens=info.output_tokens,
            total_tokens=info.input_tokens + info.output_tokens,
        ),
        "cost_usd": info.cost_usd,
        "finish_reason": info.finish_reason,
        "cache_hit": outcome.cache_hit,
        "cache_kind": outcome.cache_kind,
        "created_at": datetime.now(UTC),
    }
