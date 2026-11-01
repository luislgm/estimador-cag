import time
from unittest.mock import patch

import fakeredis
import redis
from structlog.testing import capture_logs

from app.services.cache import EstimationCache, get_cache

KEY_ARGS = {
    "model": "anthropic/claude-haiku-4-5",
    "system": "prompt",
    "user": "t",
    "max_tokens": 10,
}


def make_cache(ttl=60):
    return EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=ttl)


def test_key_is_deterministic():
    assert EstimationCache.make_key(**KEY_ARGS) == EstimationCache.make_key(**KEY_ARGS)
    assert EstimationCache.make_key(**KEY_ARGS).startswith("estimation:")


def test_key_changes_with_any_input():
    base = EstimationCache.make_key(**KEY_ARGS)
    for field, value in [
        ("model", "openai/gpt-4o-mini"),
        ("system", "otro prompt"),  # cambiar el prompt o los ejemplos invalida la caché
        ("user", "otra transcripción"),
        ("max_tokens", 20),
    ]:
        assert EstimationCache.make_key(**KEY_ARGS | {field: value}) != base, field


def test_set_and_get_roundtrip_with_ttl():
    cache = make_cache(ttl=120)
    key = EstimationCache.make_key(**KEY_ARGS)

    assert cache.get(key) is None
    cache.set(key, {"text": "## Estimación", "tokens": 5})

    assert cache.get(key) == {"text": "## Estimación", "tokens": 5}
    assert 0 < cache.client.ttl(key) <= 120


def test_redis_errors_do_not_break_the_app():
    cache = make_cache()
    error = redis.ConnectionError("Redis no responde")

    with (
        patch.object(cache.client, "get", side_effect=error) as get,
        patch.object(cache.client, "set", side_effect=error) as set_,
        capture_logs() as logs,
    ):
        assert cache.get("estimation:x") is None
        cache.set("estimation:x", {"a": 1})  # no lanza excepción

    # Tras el primer fallo se pausa: el set ya no intenta conectar (no espera otra vez)
    assert get.call_count == 1
    assert set_.call_count == 0
    assert [(e["event"], e["operation"]) for e in logs] == [("cache_unavailable", "get")]
    assert logs[0]["paused_seconds"] == 30


def test_cache_resumes_after_pause(monkeypatch):
    cache = make_cache()
    with patch.object(cache.client, "get", side_effect=redis.ConnectionError("caído")):
        cache.get("estimation:x")
    assert not cache._available()

    clock = time.monotonic() + 31  # pasados los 30 s de pausa
    monkeypatch.setattr("app.services.cache.time.monotonic", lambda: clock)

    assert cache._available()
    cache.set("estimation:x", {"a": 1})
    assert cache.get("estimation:x") == {"a": 1}


def test_get_cache_disabled_without_redis_url():
    assert get_cache() is None


def test_get_cache_uses_settings(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("CACHE_TTL", "300")

    with patch("app.services.cache.redis.from_url") as from_url:
        cache = get_cache()

    assert cache.ttl == 300
    assert from_url.call_args.args == ("redis://localhost:6379/0",)
    assert from_url.call_args.kwargs["decode_responses"] is True
    assert from_url.call_args.kwargs["socket_timeout"] == 1
    assert cache is get_cache()  # se reutiliza la misma instancia


def test_failed_set_pauses_reads_too():
    cache = make_cache()
    with (
        patch.object(cache.client, "set", side_effect=redis.TimeoutError("timeout")),
        patch.object(cache.client, "get") as get,
        capture_logs() as logs,
    ):
        cache.set("estimation:x", {"a": 1})
        assert cache.get("estimation:x") is None

    get.assert_not_called()  # en pausa: ni siquiera intenta leer
    assert logs[0]["operation"] == "set"
