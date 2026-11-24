"""Embedder de OpenAI: text-embedding-3-small, por lotes y con reintentos.

- Modelo text-embedding-3-small con su dimensión por defecto (1536).
- embed_many envía los textos en lotes de BATCH_SIZE (la API acepta listas): una petición
  por lote, no una por chunk.
- RateLimitError: hasta 3 reintentos con esperas de 1, 2 y 4 s. El resto de errores se
  propagan (el endpoint los convierte en un 500). El cliente de OpenAI se crea sin sus
  reintentos propios para que la política sea solo esta y quede explícita.
- Cada lote deja un log con el número de chunks, los tokens (los que cuenta la API) y la
  latencia.

Usa el SDK de OpenAI directamente y no LiteLLM: los embeddings son de un proveedor concreto
(OpenAI) y el ejercicio pide su manejo de errores.
"""

import math
import time
from collections.abc import Callable

import openai
import structlog

from app.config import get_settings
from app.embedding_pipeline.chunker import EMBEDDING_MODEL
from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

log = structlog.get_logger(module=__name__)

EMBEDDING_DIMENSIONS = 1536
BATCH_SIZE = 100
RETRY_DELAYS_S = (1, 2, 4)
# Precio de text-embedding-3-small: 0,02 USD por millón de tokens de entrada (noviembre de
# 2026). Cambia con el tiempo: revisar en https://openai.com/api/pricing
PRICE_USD_PER_MTOK = 0.02


class EmbeddingsNotConfigured(Exception):
    """Falta OPENAI_API_KEY."""


def estimate_cost_usd(tokens: int) -> float:
    return round(tokens * PRICE_USD_PER_MTOK / 1_000_000, 8)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Producto escalar dividido por el producto de las normas (sin numpy)."""
    if len(a) != len(b):
        raise ValueError(f"vectores de distinta dimensión: {len(a)} y {len(b)}")
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norms == 0:
        raise ValueError("no se puede calcular la similitud de un vector nulo")
    return sum(x * y for x, y in zip(a, b, strict=True)) / norms


class OpenAIEmbedder:
    def __init__(
        self,
        client: openai.OpenAI | None = None,
        batch_size: int = BATCH_SIZE,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if client is None:
            api_key = get_settings().openai_api_key
            if not api_key:
                raise EmbeddingsNotConfigured("los embeddings necesitan OPENAI_API_KEY")
            client = openai.OpenAI(api_key=api_key, max_retries=0)
        self.client = client
        self.batch_size = batch_size
        self.sleep = sleep

    def embed_one(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        embedded = []
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            vectors = self._embed_batch([c.text for c in batch])
            embedded += [
                EmbeddedChunk(**chunk.model_dump(), embedding=vector)
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
        return embedded

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(len(RETRY_DELAYS_S) + 1):
            started = time.perf_counter()
            try:
                response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            except openai.RateLimitError:
                if attempt == len(RETRY_DELAYS_S):
                    raise
                delay = RETRY_DELAYS_S[attempt]
                log.warning("embedding_rate_limited", retry_in_s=delay, attempt=attempt + 1)
                self.sleep(delay)
                continue
            log.info(
                "embedding_batch_completed",
                chunks=len(texts),
                tokens=response.usage.total_tokens,
                latency_ms=int((time.perf_counter() - started) * 1000),
                model=EMBEDDING_MODEL,
            )
            # La API puede devolver los vectores en otro orden: se ordenan por su índice
            return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        raise AssertionError("inalcanzable")  # pragma: no cover
