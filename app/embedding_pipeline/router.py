"""POST /embeddings/ingest: presupuestos -> chunks estructurales -> embeddings.

    chunker.chunk(budgets) -> embedder.embed_many(chunks) -> IngestResponse

- 200: los chunks con su embedding y las estadísticas (tokens y coste estimado).
- 422: el JSON no cumple el esquema (lo gestiona FastAPI con los validadores de schemas.py).
- 503: falta OPENAI_API_KEY.
- 500: la API de embeddings ha fallado. Al cliente, un mensaje genérico; el detalle, al log.

Es un endpoint síncrono (`def`): el SDK de OpenAI que usa el embedder es síncrono, y FastAPI
lo ejecuta en su pool de hilos sin bloquear el resto de peticiones.
"""

from typing import Annotated

import openai
import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import (
    EmbeddingsNotConfigured,
    OpenAIEmbedder,
    estimate_cost_usd,
)
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse, IngestStats

log = structlog.get_logger(module=__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


def get_embedder() -> OpenAIEmbedder:
    try:
        return OpenAIEmbedder()
    except EmbeddingsNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.post(
    "/ingest",
    response_model=IngestResponse,
    responses={
        500: {"description": "Error de la API de embeddings"},
        503: {"description": "Falta OPENAI_API_KEY"},
    },
)
def ingest(
    request: IngestRequest, embedder: Annotated[OpenAIEmbedder, Depends(get_embedder)]
) -> IngestResponse:
    chunks = JSONStructuralChunker().chunk(request.budgets)
    try:
        embedded = embedder.embed_many(chunks)
    except openai.OpenAIError as e:
        log.error("embedding_ingest_failed", error_type=type(e).__name__, error=str(e))
        raise HTTPException(
            status_code=500, detail="No se han podido generar los embeddings"
        ) from e
    total_tokens = sum(c.token_count for c in chunks)
    stats = IngestStats(
        total_budgets=len(request.budgets),
        total_chunks=len(embedded),
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost_usd(total_tokens),
    )
    log.info("embedding_ingest_completed", **stats.model_dump())
    return IngestResponse(chunks=embedded, stats=stats)
