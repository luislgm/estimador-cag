from unittest.mock import patch

import httpx
import litellm
import pytest
from structlog.testing import capture_logs

from app.config import Settings
from app.schemas.estimation import EstimationResult
from app.services import llm_client
from tests.factories import make_result, tool_call_response

VALID = make_result().model_dump(mode="json")
WRONG_TOTAL = VALID | {"total_cost_eur": 9999}


class FakeLLM:
    """Sustituye a litellm.acompletion: devuelve una respuesta por intento (sin red)."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []

    async def acompletion(self, **params):
        self.calls.append(params)
        return tool_call_response(self.payloads[min(len(self.calls), len(self.payloads)) - 1])


async def complete(fake):
    # Instructor distingue las funciones asíncronas por su tipo: se pasa el método, no el objeto
    acompletion = fake.acompletion if isinstance(fake, FakeLLM) else fake
    with patch.object(litellm, "acompletion", acompletion):
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ]
        return await llm_client.acomplete_structured(messages, EstimationResult, max_tokens=8192)


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


async def test_returns_validated_result_and_metadata():
    fake = FakeLLM(VALID)
    with capture_logs() as logs:
        result, info = await complete(fake)

    assert result.model_dump() == make_result().model_dump()
    assert (info.attempts, info.input_tokens, info.output_tokens) == (1, 1000, 200)
    assert info.provider == "anthropic"
    assert info.cost_usd == pytest.approx((1000 * 1 + 200 * 5) / 1_000_000)  # Haiku 4.5
    assert next(e for e in logs if e["event"] == "llm_call_completed")["attempts"] == 1


async def test_sends_system_and_user_separately_with_the_schema_as_tool():
    fake = FakeLLM(VALID)
    await complete(fake)

    params = fake.calls[0]
    assert params["model"] == "anthropic/claude-haiku-4-5"
    assert params["api_key"] == "test-key"
    assert params["max_tokens"] == 8192
    assert params["messages"][:2] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    assert params["tools"][0]["function"]["name"] == "EstimationResult"


async def test_retries_with_the_validation_error_and_adds_up_tokens():
    fake = FakeLLM(WRONG_TOTAL, VALID)
    with capture_logs() as logs:
        result, info = await complete(fake)

    assert len(fake.calls) == 2
    retry_messages = str(fake.calls[1]["messages"])
    assert "total_cost_eur es 9999" in retry_messages  # el modelo recibe el error
    assert result.total_cost_eur == 4000
    assert info.attempts == 2
    assert (info.input_tokens, info.output_tokens) == (2000, 400)  # los dos intentos
    assert any(e["event"] == "llm_response_retried" for e in logs)


async def test_fails_after_max_retries():
    fake = FakeLLM(WRONG_TOTAL)  # siempre mal
    with (
        capture_logs() as logs,
        pytest.raises(llm_client.LLMError, match="no pasó las validaciones"),
    ):
        await complete(fake)

    assert len(fake.calls) == 1 + llm_client.MAX_RETRIES
    assert logs[-1]["event"] == "llm_call_failed"


async def test_provider_errors_become_llm_error():
    error = litellm.exceptions.APIConnectionError(
        message="conexión rechazada",
        llm_provider="anthropic",
        model="claude-haiku-4-5",
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )

    async def failing(**params):
        raise error

    with capture_logs() as logs, pytest.raises(llm_client.LLMError, match="conexión rechazada"):
        await complete(failing)

    # Se registra como error del proveedor, no como fallo de validación
    assert logs[-1]["error_type"] == "APIConnectionError"
    assert "no pasó las validaciones" not in logs[-1]["error"]


async def test_openai_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake = FakeLLM(VALID)

    _, info = await complete(fake)

    assert fake.calls[0]["model"] == "openai/gpt-4o-mini"
    assert fake.calls[0]["api_key"] == "sk-test"
    assert info.provider == "openai"


def test_unknown_model_has_no_cost():
    assert llm_client._cost_usd("anthropic/modelo-inventado", 100, 100) is None


def test_truncated_property():
    from tests.factories import make_info

    assert make_info(finish_reason="length").truncated
    assert not make_info().truncated
