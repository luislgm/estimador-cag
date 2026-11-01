from unittest.mock import AsyncMock, patch

from app.context.examples import ESTIMATION_EXAMPLES
from app.services import llm_client, llm_service
from app.services.llm_client import LLMResult

RESULT = LLMResult(
    text="## Estimación: Test",
    model="claude-haiku-4-5",
    provider="anthropic",
    input_tokens=100,
    output_tokens=20,
    finish_reason="stop",
    cost_usd=0.0002,
)


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


async def test_generate_estimation_sends_cag_prompt_and_transcription():
    with patch.object(llm_client, "acomplete", AsyncMock(return_value=RESULT)) as acomplete:
        result = await llm_service.generate_estimation("Transcripción de prueba")

    acomplete.assert_awaited_once_with(
        llm_service.build_system_prompt(), "Transcripción de prueba", llm_service.MAX_TOKENS
    )
    assert result == llm_service.EstimationResult(
        "## Estimación: Test", "claude-haiku-4-5", 100, 20, "stop", 0.0002
    )


def test_estimation_stream_delegates_and_keeps_result():
    class FakeCompletionStream:
        def __init__(self, system, user, max_tokens):
            self.args = (system, user, max_tokens)
            self.result = None

        def __iter__(self):
            yield from ["## Estimación", ": Test"]
            self.result = RESULT

    with patch.object(llm_client, "CompletionStream", FakeCompletionStream):
        stream = llm_service.EstimationStream("Transcripción de prueba")
        chunks = list(stream)

    assert chunks == ["## Estimación", ": Test"]
    assert stream.result.estimation == "## Estimación: Test"
    assert stream.result.cost_usd == 0.0002


def test_truncated_property():
    assert llm_service.EstimationResult("x", "m", 1, 1, "length").truncated
    assert not llm_service.EstimationResult("x", "m", 1, 1, "stop").truncated
