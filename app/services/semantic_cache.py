"""Caché semántica: reutiliza la estimación de una descripción *parecida* (no idéntica).

Dos peticiones se consideran la misma si:

1. Coinciden en el "cubo": versión del prompt, modelo, tipo de proyecto, nivel de detalle y
   formato. Con opciones distintas nunca se comparte una entrada.
2. La similitud coseno de los embeddings de sus descripciones supera el umbral.

Cuidado en un estimador: la similitud no distingue bien un cambio de alcance. Medido con
text-embedding-3-small, "la misma web + una app móvil" puntúa más alto (0,888) que una
paráfrasis exacta (0,879). Un umbral bajo serviría estimaciones de otro alcance. Por eso:

- mode="off" (por defecto): no se usa.
- mode="log_only": calcula y registra la similitud, pero nunca sirve una respuesta. Sirve
  para calibrar el umbral con tráfico real antes de activarla.
- mode="on": sirve la respuesta guardada si la similitud supera el umbral (0,95 por defecto).

Necesita Redis Stack (módulo de búsqueda vectorial) y OPENAI_API_KEY para los embeddings.
Es tolerante a fallos: si algo falla, se registra y se pausa, sin bloquear la estimación.
"""

import json
import time
import uuid
from functools import lru_cache

import litellm
import numpy as np
import structlog
from redisvl.index import AsyncSearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag

from app.config import get_settings
from app.schemas.estimation import EstimationRequest

log = structlog.get_logger(module=__name__)

EMBEDDING_DIMS = 1536  # text-embedding-3-small
PAUSE_AFTER_FAILURE = 30


def _schema(dims: int) -> dict:
    return {
        "index": {"name": "estimations", "prefix": "estimation:semantic", "storage_type": "hash"},
        "fields": [
            {"name": "bucket", "type": "tag"},
            # "payload" no sirve como nombre: está reservado en redis-py (Document)
            {"name": "data", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": dims,
                    "distance_metric": "cosine",
                    "algorithm": "flat",
                    "datatype": "float32",
                },
            },
        ],
    }


class SemanticCache:
    def __init__(
        self,
        index: AsyncSearchIndex,
        *,
        mode: str,
        threshold: float,
        ttl: int,
        embedding_model: str,
        api_key: str | None,
    ):
        self.index = index
        self.mode = mode
        self.threshold = threshold
        self.ttl = ttl
        self.embedding_model = embedding_model
        self.api_key = api_key
        self._ready = False
        self._paused_until = 0.0

    @staticmethod
    def bucket(request: EstimationRequest, prompt_version: str, model: str) -> str:
        return ":".join(
            [
                prompt_version,
                model,
                request.project_type.value,
                request.detail_level.value,
                request.output_format.value,
            ]
        )

    async def lookup(self, request: EstimationRequest, bucket: str) -> dict | None:
        """La entrada guardada más parecida si supera el umbral (y mode="on"), o None."""
        if not await self._available():
            return None
        try:
            query = VectorQuery(
                vector=await self._embed(request.description),
                vector_field_name="embedding",
                return_fields=["data"],
                filter_expression=Tag("bucket") == bucket,
                num_results=1,
                return_score=True,
            )
            matches = await self.index.query(query)
        except Exception as e:  # noqa: BLE001 — Redis o embeddings caídos: se sigue sin caché
            self._failed("lookup", e)
            return None
        if not matches:
            log.info("semantic_cache_miss", reason="empty_bucket", mode=self.mode)
            return None
        similarity = round(1 - float(matches[0]["vector_distance"]), 4)
        hit = similarity >= self.threshold
        log.info(
            "semantic_cache_lookup",
            similarity=similarity,
            threshold=self.threshold,
            would_hit=hit,
            mode=self.mode,
        )
        if hit and self.mode == "on":
            return json.loads(matches[0]["data"])
        return None

    async def store(self, request: EstimationRequest, bucket: str, payload: dict) -> None:
        if not await self._available():
            return
        try:
            embedding = await self._embed(request.description)
            await self.index.load(
                [
                    {
                        "bucket": bucket,
                        "data": json.dumps(payload, ensure_ascii=False),
                        "embedding": np.array(embedding, dtype=np.float32).tobytes(),
                    }
                ],
                keys=[f"estimation:semantic:{uuid.uuid4().hex}"],
                ttl=self.ttl,
            )
        except Exception as e:  # noqa: BLE001
            self._failed("store", e)

    async def _embed(self, text: str) -> list[float]:
        response = await litellm.aembedding(
            model=self.embedding_model, input=[text], api_key=self.api_key
        )
        return response.data[0]["embedding"]

    async def _available(self) -> bool:
        if time.monotonic() < self._paused_until:
            return False
        if not self._ready:
            try:
                await self.index.create(overwrite=False)  # no hace nada si ya existe
            except Exception as e:  # noqa: BLE001 — p. ej. Redis sin el módulo de búsqueda
                self._failed("create_index", e)
                return False
            self._ready = True
        return True

    def _failed(self, operation: str, error: Exception) -> None:
        self._paused_until = time.monotonic() + PAUSE_AFTER_FAILURE
        log.warning(
            "semantic_cache_unavailable",
            operation=operation,
            error_type=type(error).__name__,
            error=str(error),
            paused_seconds=PAUSE_AFTER_FAILURE,
        )


@lru_cache
def get_semantic_cache() -> SemanticCache | None:
    """La caché semántica configurada, o None si está desactivada o le falta algo."""
    settings = get_settings()
    if settings.semantic_cache_mode == "off":
        return None
    if not (settings.redis_url and settings.openai_api_key):
        log.warning("semantic_cache_disabled", reason="faltan REDIS_URL u OPENAI_API_KEY")
        return None
    index = AsyncSearchIndex.from_dict(_schema(EMBEDDING_DIMS), redis_url=settings.redis_url)
    return SemanticCache(
        index,
        mode=settings.semantic_cache_mode,
        threshold=settings.semantic_cache_threshold,
        ttl=settings.cache_ttl,
        embedding_model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
