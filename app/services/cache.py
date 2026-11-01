"""Caché exact-match de respuestas del LLM en Redis.

La clave es un hash de todo lo que determina la respuesta: modelo, system prompt completo,
transcripción y max_tokens. Si cambia cualquiera de ellos (por ejemplo, el prompt o los
ejemplos CAG), la clave cambia y no se sirven respuestas antiguas.

Es opcional (solo con REDIS_URL) y tolerante a fallos: si Redis no responde, se registra un
aviso, la caché se pausa unos segundos y la aplicación sigue funcionando sin ella.
"""

import hashlib
import json
import time
from functools import lru_cache

import redis
import structlog
from redis.backoff import NoBackoff
from redis.retry import Retry

from app.config import get_settings

log = structlog.get_logger(module=__name__)

# Tras un fallo de Redis, segundos sin volver a intentarlo (evita esperar en cada petición)
PAUSE_AFTER_FAILURE = 30


class EstimationCache:
    def __init__(self, client: redis.Redis, ttl: int):
        self.client = client
        self.ttl = ttl
        self._paused_until = 0.0

    def _available(self) -> bool:
        return time.monotonic() >= self._paused_until

    def _failed(self, operation: str, error: Exception) -> None:
        self._paused_until = time.monotonic() + PAUSE_AFTER_FAILURE
        log.warning(
            "cache_unavailable",
            operation=operation,
            error=str(error),
            paused_seconds=PAUSE_AFTER_FAILURE,
        )

    @staticmethod
    def make_key(*, model: str, system: str, user: str, max_tokens: int) -> str:
        payload = json.dumps(
            {"model": model, "system": system, "user": user, "max_tokens": max_tokens},
            sort_keys=True,
            ensure_ascii=False,
        )
        return "estimation:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        if not self._available():
            return None
        try:
            value = self.client.get(key)
        except redis.RedisError as e:
            self._failed("get", e)
            return None
        return json.loads(value) if value else None

    def set(self, key: str, value: dict) -> None:
        if not self._available():
            return
        try:
            self.client.set(key, json.dumps(value, ensure_ascii=False), ex=self.ttl)
        except redis.RedisError as e:
            self._failed("set", e)


@lru_cache
def get_cache() -> EstimationCache | None:
    """La caché configurada, o None si no hay REDIS_URL."""
    settings = get_settings()
    if not settings.redis_url:
        return None
    client = redis.from_url(
        settings.redis_url,
        decode_responses=True,
        # Si Redis no responde, fallar rápido (1 s, sin reintentos) en vez de bloquear
        socket_connect_timeout=1,
        socket_timeout=1,
        retry=Retry(NoBackoff(), 0),
    )
    return EstimationCache(client, ttl=settings.cache_ttl)
