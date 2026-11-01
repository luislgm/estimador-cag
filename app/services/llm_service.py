"""Estimador CAG: qué se le pide al LLM.

Construye los mensajes a partir de las plantillas versionadas (app/prompts) y delega la
llamada en app.services.llm_client, que funciona igual con OpenAI o con Anthropic.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from app.prompts import PROMPT_VERSION, render_estimation_prompt
from app.schemas.estimation import EstimationRequest
from app.services import llm_client
from app.services.llm_client import LLMError, LLMResult

__all__ = [
    "PROMPT_VERSION",
    "EstimationResult",
    "EstimationStream",
    "LLMError",
    "generate_estimation",
]

MAX_TOKENS = 4096


@dataclass
class EstimationResult:
    estimation: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str = "stop"
    cost_usd: float | None = None
    cache_hit: bool = False
    provider: str = ""
    prompt_version: str = PROMPT_VERSION

    @property
    def truncated(self) -> bool:
        """True si la estimación se cortó por alcanzar el límite de tokens de salida."""
        return self.finish_reason == "length"


def _messages(request: EstimationRequest) -> tuple[str, str]:
    return render_estimation_prompt(request, version=PROMPT_VERSION)


def _to_estimation(result: LLMResult) -> EstimationResult:
    return EstimationResult(
        estimation=result.text,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        finish_reason=result.finish_reason,
        cost_usd=result.cost_usd,
        cache_hit=result.cache_hit,
        provider=result.provider,
        prompt_version=PROMPT_VERSION,
    )


async def generate_estimation(request: EstimationRequest) -> EstimationResult:
    """Envía la petición al LLM y devuelve la estimación completa."""
    system, user = _messages(request)
    result = await llm_client.acomplete(system, user, MAX_TOKENS)
    return _to_estimation(result)


class EstimationStream:
    """Genera la estimación en streaming: al iterar devuelve el texto a trozos.

    Cuando termina la iteración, `result` contiene la estimación completa y los tokens.
    """

    def __init__(self, request: EstimationRequest):
        self.request = request
        self.result: EstimationResult | None = None

    def __iter__(self) -> Iterator[str]:
        system, user = _messages(self.request)
        stream = llm_client.CompletionStream(system, user, MAX_TOKENS)
        yield from stream
        self.result = _to_estimation(stream.result)
