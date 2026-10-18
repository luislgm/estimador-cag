from datetime import UTC, datetime

import anthropic
from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.schemas.estimation import EstimationRequest, EstimationResponse, TokenUsage
from app.services.llm_service import generate_estimation

router = APIRouter(tags=["estimations"])


@router.post("/estimate", response_model=EstimationResponse)
async def estimate(request: EstimationRequest) -> EstimationResponse:
    settings = get_settings()

    try:
        result = await generate_estimation(request.transcription)
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Error al llamar al LLM: {e}") from e

    return EstimationResponse(
        estimation=result.estimation,
        model=result.model,
        provider=settings.llm_provider,
        usage=TokenUsage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
        ),
        created_at=datetime.now(UTC),
    )
