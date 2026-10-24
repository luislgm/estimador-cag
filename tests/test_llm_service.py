from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import httpx
import pytest
from structlog.testing import capture_logs

from app.context.examples import ESTIMATION_EXAMPLES
from app.services import llm_service


def test_examples_have_required_fields():
    assert len(ESTIMATION_EXAMPLES) >= 2
    for example in ESTIMATION_EXAMPLES:
        assert example["meeting_summary"].strip()
        assert "## Estimación" in example["estimation"]


def test_system_prompt_includes_all_examples():
    prompt = llm_service.build_system_prompt()

    assert prompt.startswith(llm_service.SYSTEM_INSTRUCTIONS)
    assert prompt.count("<ejemplo numero=") == len(ESTIMATION_EXAMPLES)
    for example in ESTIMATION_EXAMPLES:
        assert example["meeting_summary"] in prompt
        assert example["estimation"].strip() in prompt


async def test_generate_estimation_calls_anthropic():
    fake_response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="## Estimación: Test"),
        ],
        model="claude-haiku-4-5",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        stop_reason="end_turn",
    )
    create = AsyncMock(return_value=fake_response)

    with patch.object(llm_service, "AsyncAnthropic") as client_cls:
        client_cls.return_value.messages.create = create
        result = await llm_service.generate_estimation("Transcripción de prueba")

    client_cls.assert_called_once_with(api_key="test-key", open_telemetry=False)
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["system"] == llm_service.build_system_prompt()
    assert kwargs["messages"] == [{"role": "user", "content": "Transcripción de prueba"}]

    assert result.estimation == "## Estimación: Test"
    assert result.model == "claude-haiku-4-5"
    assert (result.input_tokens, result.output_tokens) == (100, 20)


def test_estimation_stream_yields_chunks_and_result():
    final_message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="## Estimación: Test")],
        model="claude-haiku-4-5",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        stop_reason="end_turn",
    )
    stream = MagicMock()
    stream.text_stream = iter(["## Estimación", ": Test"])
    stream.get_final_message.return_value = final_message

    with patch.object(llm_service, "Anthropic") as client_cls:
        client_cls.return_value.messages.stream.return_value.__enter__.return_value = stream
        estimation_stream = llm_service.EstimationStream("Transcripción de prueba")

        assert estimation_stream.result is None
        chunks = list(estimation_stream)

    assert chunks == ["## Estimación", ": Test"]
    kwargs = client_cls.return_value.messages.stream.call_args.kwargs
    assert kwargs == llm_service._request_params("Transcripción de prueba")
    assert estimation_stream.result == llm_service.EstimationResult(
        "## Estimación: Test", "claude-haiku-4-5", 100, 20
    )


async def test_generate_estimation_reports_truncation_and_cost():
    fake_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="## Estimación: cort")],
        model="claude-haiku-4-5-20251001",
        usage=SimpleNamespace(input_tokens=1_000, output_tokens=4_096),
        stop_reason="max_tokens",
    )
    with patch.object(llm_service, "AsyncAnthropic") as client_cls:
        client_cls.return_value.messages.create = AsyncMock(return_value=fake_response)
        with capture_logs() as logs:
            result = await llm_service.generate_estimation("Transcripción de prueba")

    assert result.finish_reason == "max_tokens"
    assert result.truncated
    assert result.cost_usd == 0.02148  # 1.000 × 1 $/M + 4.096 × 5 $/M
    events = [(e["event"], e["log_level"]) for e in logs if e["log_level"] != "debug"]
    assert events == [("llm_call_completed", "info"), ("llm_response_truncated", "warning")]
    completed = next(e for e in logs if e["event"] == "llm_call_completed")
    assert completed["cost_usd"] == 0.02148
    assert completed["input_tokens"] == 1_000
    assert completed["finish_reason"] == "max_tokens"
    assert completed["module"] == "app.services.llm_service"


async def test_generate_estimation_logs_failures():
    error = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    with patch.object(llm_service, "AsyncAnthropic") as client_cls:
        client_cls.return_value.messages.create = AsyncMock(side_effect=error)
        with capture_logs() as logs, pytest.raises(anthropic.APIConnectionError):
            await llm_service.generate_estimation("Transcripción de prueba")

    failed = logs[-1]
    assert (failed["event"], failed["log_level"]) == ("llm_call_failed", "error")
    assert failed["error_type"] == "APIConnectionError"


def test_estimation_stream_logs_failures():
    error = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    with patch.object(llm_service, "Anthropic") as client_cls:
        client_cls.return_value.messages.stream.side_effect = error
        with capture_logs() as logs, pytest.raises(anthropic.APIConnectionError):
            list(llm_service.EstimationStream("Transcripción de prueba"))

    assert logs[-1]["event"] == "llm_call_failed"
    assert logs[-1]["mode"] == "stream"
