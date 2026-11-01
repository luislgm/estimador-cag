"""Endpoints de estimación.

POST /api/v1/estimate         -> la estimación completa en una sola respuesta JSON.
POST /api/v1/estimate/stream  -> la estimación en streaming (Server-Sent Events).
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.schemas.estimation import EstimationRequest, EstimationResponse, TokenUsage
from app.services.llm_service import (
    PROMPT_VERSION,
    EstimationResult,
    EstimationStream,
    LLMError,
    generate_estimation,
)

log = structlog.get_logger(module=__name__)

router = APIRouter(tags=["estimations"])


def _log_request(request: EstimationRequest, mode: str) -> None:
    log.info(
        "estimation_requested",
        mode=mode,
        prompt_version=PROMPT_VERSION,
        project_type=request.project_type.value,
        detail_level=request.detail_level.value,
        output_format=request.output_format.value,
        description_chars=len(request.description),
    )


def _to_response(result: EstimationResult) -> EstimationResponse:
    return EstimationResponse(
        text=result.estimation,
        prompt_version=result.prompt_version,
        model=result.model,
        provider=result.provider or get_settings().llm_provider,
        usage=TokenUsage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
        ),
        cost_usd=result.cost_usd,
        finish_reason=result.finish_reason,
        cache_hit=result.cache_hit,
        created_at=datetime.now(UTC),
    )


@router.post("/estimate", response_model=EstimationResponse)
async def estimate(request: EstimationRequest) -> EstimationResponse:
    _log_request(request, "blocking")
    try:
        result = await generate_estimation(request)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"Error al llamar al LLM: {e}") from e
    return _to_response(result)


def _sse(event: str, data: dict) -> str:
    """Un evento Server-Sent Events. Los datos van en JSON (así los saltos de línea no rompen
    el formato)."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/estimate/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
def estimate_stream(request: EstimationRequest) -> StreamingResponse:
    """La estimación en streaming, como eventos Server-Sent Events:

    - `token`: un trozo de texto (`{"text": "..."}`), según lo va generando el LLM.
    - `done`: al terminar, la respuesta completa con el mismo formato que POST /estimate.
    - `error`: si falla la llamada al LLM (`{"message": "..."}`); no llega `done`.
    """
    _log_request(request, "stream")
    stream = EstimationStream(request)

    def events() -> Iterator[str]:
        try:
            for chunk in stream:
                yield _sse("token", {"text": chunk})
        except LLMError as e:
            yield _sse("error", {"message": f"Error al llamar al LLM: {e}"})
            return
        yield _sse("done", _to_response(stream.result).model_dump(mode="json"))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
