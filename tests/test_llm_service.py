from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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
    )
    create = AsyncMock(return_value=fake_response)

    with patch.object(llm_service, "AsyncAnthropic") as client_cls:
        client_cls.return_value.messages.create = create
        result = await llm_service.generate_estimation("Transcripción de prueba")

    client_cls.assert_called_once_with(api_key="test-key")
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["system"] == llm_service.build_system_prompt()
    assert kwargs["messages"] == [{"role": "user", "content": "Transcripción de prueba"}]

    assert result.estimation == "## Estimación: Test"
    assert result.model == "claude-haiku-4-5"
    assert (result.input_tokens, result.output_tokens) == (100, 20)
