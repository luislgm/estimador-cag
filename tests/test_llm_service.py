from unittest.mock import AsyncMock, patch

from app.prompts import render_estimation_prompt
from app.schemas.estimation import EstimationRequest
from app.services import llm_client, llm_service
from app.services.llm_client import LLMResult

REQUEST = EstimationRequest(
    description="Una gestoría quiere un portal para que sus clientes suban facturas.",
    project_type="web_saas",
    detail_level="summary",
    output_format="phases_table",
)
RESULT = LLMResult(
    text="## Estimación: Test",
    model="claude-haiku-4-5",
    provider="anthropic",
    input_tokens=100,
    output_tokens=20,
    finish_reason="stop",
    cost_usd=0.0002,
)


async def test_generate_estimation_sends_versioned_prompts():
    with patch.object(llm_client, "acomplete", AsyncMock(return_value=RESULT)) as acomplete:
        result = await llm_service.generate_estimation(REQUEST)

    acomplete.assert_awaited_once_with(
        *render_estimation_prompt(REQUEST, version="v1"), llm_service.MAX_TOKENS
    )
    assert result == llm_service.EstimationResult(
        "## Estimación: Test", "claude-haiku-4-5", 100, 20, "stop", 0.0002, False, "anthropic"
    )
    assert result.prompt_version == "v1"


def test_estimation_stream_delegates_and_keeps_result():
    class FakeCompletionStream:
        calls = []

        def __init__(self, system, user, max_tokens):
            FakeCompletionStream.calls.append((system, user, max_tokens))
            self.result = None

        def __iter__(self):
            yield from ["## Estimación", ": Test"]
            self.result = RESULT

    with patch.object(llm_client, "CompletionStream", FakeCompletionStream):
        stream = llm_service.EstimationStream(REQUEST)
        chunks = list(stream)

    assert chunks == ["## Estimación", ": Test"]
    assert FakeCompletionStream.calls == [
        (*render_estimation_prompt(REQUEST), llm_service.MAX_TOKENS)
    ]
    assert stream.result.estimation == "## Estimación: Test"
    assert stream.result.provider == "anthropic"
    assert stream.result.prompt_version == "v1"


def test_truncated_property():
    assert llm_service.EstimationResult("x", "m", 1, 1, "length").truncated
    assert not llm_service.EstimationResult("x", "m", 1, 1, "stop").truncated
