import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from structlog.testing import capture_logs

from app.services import semantic_cache
from app.services.semantic_cache import SemanticCache, get_semantic_cache
from tests.factories import make_request

PAYLOAD = {"result": {"total_cost_eur": 4000}, "info": {"model": "claude-haiku-4-5"}}


def make_cache(mode="on", threshold=0.95, matches=None):
    index = SimpleNamespace(
        create=AsyncMock(),
        query=AsyncMock(return_value=matches if matches is not None else []),
        load=AsyncMock(),
    )
    return SemanticCache(
        index,
        mode=mode,
        threshold=threshold,
        ttl=60,
        embedding_model="openai/text-embedding-3-small",
        api_key="sk-test",
    )


def match(similarity: float) -> list[dict]:
    import json

    return [{"vector_distance": str(1 - similarity), "data": json.dumps(PAYLOAD)}]


@pytest.fixture(autouse=True)
def fake_embeddings():
    response = SimpleNamespace(data=[{"embedding": [0.1] * 1536}])
    with patch.object(litellm, "aembedding", AsyncMock(return_value=response)) as embed:
        yield embed


def test_bucket_includes_version_model_and_options():
    bucket = SemanticCache.bucket(make_request(), "v1", "anthropic/claude-haiku-4-5")

    assert bucket == "v1:anthropic/claude-haiku-4-5:web_saas:medium:phases_table"
    other = SemanticCache.bucket(make_request(detail_level="summary"), "v1", "m")
    assert other != bucket


async def test_hit_above_threshold():
    cache = make_cache(matches=match(0.97))

    with capture_logs() as logs:
        assert await cache.lookup(make_request(), "bucket") == PAYLOAD

    assert logs[-1]["similarity"] == 0.97
    assert logs[-1]["would_hit"] is True


async def test_miss_below_threshold():
    cache = make_cache(matches=match(0.89))  # como "la misma web + app móvil"

    assert await cache.lookup(make_request(), "bucket") is None


async def test_log_only_never_serves():
    cache = make_cache(mode="log_only", matches=match(0.99))

    with capture_logs() as logs:
        assert await cache.lookup(make_request(), "bucket") is None

    assert logs[-1]["would_hit"] is True  # se registra para calibrar, pero no se sirve
    assert logs[-1]["mode"] == "log_only"


async def test_query_is_filtered_by_bucket(fake_embeddings):
    cache = make_cache()

    await cache.lookup(make_request(), "v1:modelo:web_saas:medium:phases_table")

    query = cache.index.query.await_args.args[0]
    # redisvl escapa los dos puntos en la consulta de etiquetas
    assert str(query.filter) == r"@bucket:{v1\:modelo\:web_saas\:medium\:phases_table}"
    fake_embeddings.assert_awaited_once()
    assert fake_embeddings.await_args.kwargs["input"] == [make_request().description]


async def test_empty_bucket_is_a_miss():
    cache = make_cache(matches=[])

    with capture_logs() as logs:
        assert await cache.lookup(make_request(), "bucket") is None

    assert logs[-1]["reason"] == "empty_bucket"


async def test_store_saves_embedding_payload_and_ttl():
    cache = make_cache()

    await cache.store(make_request(), "bucket", PAYLOAD)

    records = cache.index.load.await_args.args[0]
    assert records[0]["bucket"] == "bucket"
    assert '"total_cost_eur": 4000' in records[0]["data"]
    assert len(records[0]["embedding"]) == 1536 * 4  # float32
    assert cache.index.load.await_args.kwargs["ttl"] == 60


async def test_index_is_created_once():
    cache = make_cache()

    await cache.lookup(make_request(), "bucket")
    await cache.store(make_request(), "bucket", PAYLOAD)

    cache.index.create.assert_awaited_once_with(overwrite=False)


async def test_failures_pause_the_cache():
    cache = make_cache()
    cache.index.query.side_effect = ConnectionError("Redis no responde")

    with capture_logs() as logs:
        assert await cache.lookup(make_request(), "bucket") is None
        await cache.store(make_request(), "bucket", PAYLOAD)  # en pausa: ni lo intenta

    cache.index.load.assert_not_awaited()
    assert logs[0]["event"] == "semantic_cache_unavailable"
    assert logs[0]["operation"] == "lookup"


async def test_store_failure_is_logged():
    cache = make_cache()
    cache.index.load.side_effect = ConnectionError("Redis no responde")

    with capture_logs() as logs:
        await cache.store(make_request(), "bucket", PAYLOAD)

    assert logs[0]["operation"] == "store"


async def test_redis_without_search_module_disables_it():
    cache = make_cache()
    cache.index.create.side_effect = RuntimeError("unknown command 'FT.CREATE'")

    with capture_logs() as logs:
        assert await cache.lookup(make_request(), "bucket") is None

    assert logs[0]["operation"] == "create_index"
    cache.index.query.assert_not_awaited()


async def test_resumes_after_pause(monkeypatch):
    cache = make_cache(matches=match(0.99))
    cache.index.create.side_effect = [RuntimeError("caído"), None]
    await cache.lookup(make_request(), "bucket")

    clock = time.monotonic() + 31
    monkeypatch.setattr(semantic_cache.time, "monotonic", lambda: clock)

    assert await cache.lookup(make_request(), "bucket") == PAYLOAD


def test_disabled_by_default():
    get_semantic_cache.cache_clear()
    assert get_semantic_cache() is None


def test_needs_redis_and_openai_key(monkeypatch):
    monkeypatch.setenv("SEMANTIC_CACHE_MODE", "on")
    get_semantic_cache.cache_clear()

    with capture_logs() as logs:
        assert get_semantic_cache() is None

    assert logs[0]["event"] == "semantic_cache_disabled"


def test_built_from_settings(monkeypatch):
    monkeypatch.setenv("SEMANTIC_CACHE_MODE", "log_only")
    monkeypatch.setenv("SEMANTIC_CACHE_THRESHOLD", "0.9")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_semantic_cache.cache_clear()

    with patch.object(semantic_cache.AsyncSearchIndex, "from_dict") as from_dict:
        cache = get_semantic_cache()

    assert (cache.mode, cache.threshold, cache.api_key) == ("log_only", 0.9, "sk-test")
    assert from_dict.call_args.kwargs["redis_url"] == "redis://localhost:6379/0"
    get_semantic_cache.cache_clear()
