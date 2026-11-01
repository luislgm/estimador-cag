from unittest.mock import AsyncMock, patch

import fakeredis
import httpx
import litellm
import pytest
import redis
from structlog.testing import capture_logs

from app.config import Settings
from app.services import llm_client
from app.services.cache import EstimationCache

REAL_ACOMPLETION = litellm.acompletion
REAL_COMPLETION = litellm.completion


def mocked(real, text="## Estimación: Test"):
    """Envuelve la función real de LiteLLM con mock_response: sin red, pero con su lógica."""
    calls = []

    def fake(**params):
        calls.append(params)
        return real(**params, mock_response=text)

    fake.calls = calls
    return fake


def connection_error():
    return litellm.exceptions.APIConnectionError(
        message="conexión rechazada",
        llm_provider="anthropic",
        model="claude-haiku-4-5",
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("anthropic", "claude-haiku-4-5", "anthropic/claude-haiku-4-5"),
        ("openai", "gpt-4o-mini", "openai/gpt-4o-mini"),
        ("openai", "openai/gpt-4o-mini", "openai/gpt-4o-mini"),  # ya trae el prefijo
    ],
)
def test_model_id(monkeypatch, provider, model, expected):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("LLM_MODEL", model)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    assert llm_client.model_id(Settings()) == expected


async def test_acomplete_with_anthropic():
    fake = mocked(REAL_ACOMPLETION)
    with patch.object(litellm, "acompletion", fake), capture_logs() as logs:
        result = await llm_client.acomplete("system prompt", "transcripción", 4096)

    params = fake.calls[0]
    assert params["model"] == "anthropic/claude-haiku-4-5"
    assert params["api_key"] == "test-key"
    assert params["max_tokens"] == 4096
    assert params["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "transcripción"},
    ]
    assert result.text == "## Estimación: Test"
    assert result.provider == "anthropic"
    assert result.finish_reason == "stop"
    assert not result.truncated
    assert result.input_tokens > 0 and result.output_tokens > 0
    expected_cost = (result.input_tokens * 1 + result.output_tokens * 5) / 1_000_000
    assert result.cost_usd == pytest.approx(expected_cost)  # Haiku 4.5: 1 $ / 5 $ por millón
    completed = next(e for e in logs if e["event"] == "llm_call_completed")
    assert completed["provider"] == "anthropic"
    assert completed["mode"] == "blocking"


async def test_acomplete_with_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    fake = mocked(REAL_ACOMPLETION)
    with patch.object(litellm, "acompletion", fake):
        result = await llm_client.acomplete("system", "transcripción", 100)

    assert fake.calls[0]["model"] == "openai/gpt-4o-mini"
    assert fake.calls[0]["api_key"] == "sk-test"
    assert result.provider == "openai"
    assert result.cost_usd is not None


async def test_acomplete_reports_truncation():
    response = litellm.ModelResponse(
        model="claude-haiku-4-5",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "cort"},
                "finish_reason": "length",
            }
        ],
        usage={"prompt_tokens": 1_000, "completion_tokens": 4_096, "total_tokens": 5_096},
    )
    with (
        patch.object(litellm, "acompletion", AsyncMock(return_value=response)),
        capture_logs() as logs,
    ):
        result = await llm_client.acomplete("system", "transcripción", 4096)

    assert result.truncated
    assert result.cost_usd == 0.02148  # 1.000 × 1 $/M + 4.096 × 5 $/M
    events = [(e["event"], e["log_level"]) for e in logs if e["log_level"] != "debug"]
    assert events == [("llm_call_completed", "info"), ("llm_response_truncated", "warning")]


async def test_acomplete_wraps_provider_errors():
    with (
        patch.object(litellm, "acompletion", AsyncMock(side_effect=connection_error())),
        capture_logs() as logs,
        pytest.raises(llm_client.LLMError, match="conexión rechazada"),
    ):
        await llm_client.acomplete("system", "transcripción", 4096)

    failed = logs[-1]
    assert (failed["event"], failed["log_level"]) == ("llm_call_failed", "error")
    assert failed["error_type"] == "APIConnectionError"


def test_stream_yields_chunks_and_final_result():
    fake = mocked(REAL_COMPLETION, text="uno dos tres cuatro")
    with patch.object(litellm, "completion", fake), capture_logs() as logs:
        stream = llm_client.CompletionStream("system", "transcripción", 4096)
        assert stream.result is None
        chunks = list(stream)

    assert "".join(chunks) == "uno dos tres cuatro"
    assert len(chunks) > 1  # llega a trozos
    assert fake.calls[0]["stream"] is True
    assert fake.calls[0]["stream_options"] == {"include_usage": True}
    assert stream.result.text == "uno dos tres cuatro"
    assert stream.result.finish_reason == "stop"
    assert stream.result.input_tokens > 0 and stream.result.output_tokens > 0
    assert stream.result.cost_usd is not None
    assert next(e for e in logs if e["event"] == "llm_call_completed")["mode"] == "stream"


def test_stream_wraps_provider_errors():
    with (
        patch.object(litellm, "completion", side_effect=connection_error()),
        capture_logs() as logs,
        pytest.raises(llm_client.LLMError),
    ):
        list(llm_client.CompletionStream("system", "transcripción", 4096))

    assert logs[-1]["event"] == "llm_call_failed"
    assert logs[-1]["mode"] == "stream"


def test_unknown_model_has_no_cost():
    assert llm_client._cost_usd("anthropic/modelo-inventado", 100, 100) is None


# --- Caché exact-match ---


@pytest.fixture
def cache():
    """Caché con un Redis simulado en memoria, conectada al wrapper."""
    instance = EstimationCache(fakeredis.FakeRedis(decode_responses=True), ttl=60)
    with patch.object(llm_client, "get_cache", return_value=instance):
        yield instance


async def test_cache_miss_then_hit(cache):
    fake = mocked(REAL_ACOMPLETION)
    with patch.object(litellm, "acompletion", fake), capture_logs() as logs:
        first = await llm_client.acomplete("system", "transcripción", 4096)
        second = await llm_client.acomplete("system", "transcripción", 4096)

    assert len(fake.calls) == 1  # la segunda no llega al proveedor
    assert not first.cache_hit and first.cost_usd > 0
    assert second.cache_hit
    assert second.cost_usd == 0.0  # no se ha gastado nada
    assert second.text == first.text
    assert (second.input_tokens, second.output_tokens) == (first.input_tokens, first.output_tokens)
    hit = next(e for e in logs if e["event"] == "llm_cache_hit")
    assert hit["saved_usd"] == first.cost_usd


async def test_different_transcription_is_not_a_hit(cache):
    fake = mocked(REAL_ACOMPLETION)
    with patch.object(litellm, "acompletion", fake):
        await llm_client.acomplete("system", "transcripción A", 4096)
        result = await llm_client.acomplete("system", "transcripción B", 4096)

    assert len(fake.calls) == 2
    assert not result.cache_hit


def test_stream_hit_returns_full_text_at_once(cache):
    fake = mocked(REAL_COMPLETION, text="uno dos tres cuatro")
    with patch.object(litellm, "completion", fake):
        first = llm_client.CompletionStream("system", "transcripción", 4096)
        first_chunks = list(first)
        second = llm_client.CompletionStream("system", "transcripción", 4096)
        second_chunks = list(second)

    assert len(fake.calls) == 1
    assert len(first_chunks) > 1
    assert second_chunks == ["uno dos tres cuatro"]  # de una vez, sin llamar al LLM
    assert second.result.cache_hit and second.result.cost_usd == 0.0
    assert second.result.output_tokens == first.result.output_tokens  # tokens reales guardados


async def test_cache_is_shared_between_stream_and_blocking(cache):
    stream_fake = mocked(REAL_COMPLETION, text="estimación compartida")
    with patch.object(litellm, "completion", stream_fake):
        stream = llm_client.CompletionStream("system", "transcripción", 4096)
        list(stream)

    blocking_fake = mocked(REAL_ACOMPLETION)
    with patch.object(litellm, "acompletion", blocking_fake):
        result = await llm_client.acomplete("system", "transcripción", 4096)

    assert blocking_fake.calls == []
    assert result.cache_hit
    assert result.text == "estimación compartida"
    assert result.input_tokens == stream.result.input_tokens > 0  # no se guardan tokens a 0


async def test_truncated_responses_are_not_cached(cache):
    response = litellm.ModelResponse(
        model="claude-haiku-4-5",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "cort"},
                "finish_reason": "length",
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    )
    acompletion = AsyncMock(return_value=response)
    with patch.object(litellm, "acompletion", acompletion):
        await llm_client.acomplete("system", "transcripción", 4096)
        second = await llm_client.acomplete("system", "transcripción", 4096)

    assert acompletion.await_count == 2
    assert not second.cache_hit


async def test_works_when_redis_is_down(cache):
    error = redis.ConnectionError("Redis no responde")
    fake = mocked(REAL_ACOMPLETION)
    with (
        patch.object(cache.client, "get", side_effect=error),
        patch.object(cache.client, "set", side_effect=error),
        patch.object(litellm, "acompletion", fake),
    ):
        result = await llm_client.acomplete("system", "transcripción", 4096)

    assert result.text == "## Estimación: Test"
    assert not result.cache_hit
