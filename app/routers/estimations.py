from datetime import UTC, datetime

import anthropic
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.llm_service import generate_estimation

router = APIRouter(tags=["estimations"])


class EstimationRequest(BaseModel):
    transcription: str = Field(
        ...,
        min_length=1,
        description="Texto de la transcripción de la reunión",
        examples=["En la reunión con el cliente se discutió la necesidad de..."],
    )


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class EstimationResponse(BaseModel):
    estimation: str
    model: str
    provider: str
    usage: TokenUsage
    created_at: datetime


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
