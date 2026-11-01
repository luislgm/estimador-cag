from unittest.mock import AsyncMock, patch

import httpx
import litellm
import pytest
from structlog.testing import capture_logs

from app.config import Settings
from app.services import llm_client

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
