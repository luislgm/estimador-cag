import time
from collections.abc import Iterator
from dataclasses import dataclass

import structlog
from anthropic import Anthropic, AnthropicError, AsyncAnthropic
from anthropic.types import Message

from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.services.pricing import estimate_cost_usd

log = structlog.get_logger(module=__name__)

SYSTEM_INSTRUCTIONS = """\
Eres un estimador de proyectos de software experto. Tu tarea es generar una estimación \
a partir de la transcripción de una reunión con un cliente.

Basa tus estimaciones en los ejemplos de estimaciones previas que se incluyen más abajo: \
úsalos como referencia de granularidad de tareas, horas, tarifa (50 €/h), tamaño de equipo \
y duración. Responde siempre en español y con el mismo formato markdown que los ejemplos:
- Título "## Estimación: <nombre del proyecto>"
- "### Desglose de tareas:" con tareas numeradas, horas y coste de cada una
- Total estimado (horas y coste), equipo recomendado y duración estimada

Si la transcripción no aclara algún punto, asume lo más razonable e indica brevemente \
las suposiciones al final de la estimación."""


def build_system_prompt() -> str:
    """Construye el system prompt con las instrucciones y los ejemplos de contexto (CAG)."""
    examples = "\n\n".join(
        f'<ejemplo numero="{i}">\n'
        f"<resumen_reunion>\n{example['meeting_summary']}\n</resumen_reunion>\n"
        f"<estimacion>\n{example['estimation'].strip()}\n</estimacion>\n"
        f"</ejemplo>"
        for i, example in enumerate(ESTIMATION_EXAMPLES, start=1)
    )
    return f"{SYSTEM_INSTRUCTIONS}\n\n<estimaciones_previas>\n{examples}\n</estimaciones_previas>"


@dataclass
class EstimationResult:
    estimation: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str = "end_turn"

    @property
    def cost_usd(self) -> float | None:
        return estimate_cost_usd(self.model, self.input_tokens, self.output_tokens)

    @property
    def truncated(self) -> bool:
        """True si la estimación se cortó por alcanzar el límite de tokens de salida."""
        return self.finish_reason == "max_tokens"


def _client_options() -> dict:
    """Opciones comunes de los clientes de Anthropic.

    `open_telemetry=False` desactiva las trazas propias del SDK: las de Phoenix las genera
    la instrumentación de OpenInference (ver app/tracing.py) y así no salen duplicadas.
    """
    return {"api_key": get_settings().anthropic_api_key, "open_telemetry": False}


def _request_params(transcription: str) -> dict:
    """Parámetros de la llamada al LLM, comunes a la versión normal y a la de streaming."""
    return {
        "model": get_settings().llm_model,
        "max_tokens": 4096,
        "system": build_system_prompt(),
        "messages": [{"role": "user", "content": transcription}],
    }


def _to_result(message: Message) -> EstimationResult:
    return EstimationResult(
        estimation="".join(block.text for block in message.content if block.type == "text"),
        model=message.model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        finish_reason=message.stop_reason or "unknown",
    )


def _log_completed(result: EstimationResult, mode: str, started: float) -> None:
    fields = {
        "mode": mode,
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "finish_reason": result.finish_reason,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }
    log.info("llm_call_completed", **fields)
    if result.truncated:
        log.warning("llm_response_truncated", **fields)


def _log_failed(error: Exception, mode: str, started: float) -> None:
    log.error(
        "llm_call_failed",
        mode=mode,
        error_type=type(error).__name__,
        error=str(error),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


async def generate_estimation(transcription: str) -> EstimationResult:
    """Envía la transcripción al LLM y devuelve la estimación generada."""
    params = _request_params(transcription)
    client = AsyncAnthropic(**_client_options())
    log.debug("llm_call_started", mode="blocking", model=params["model"])
    started = time.perf_counter()
    try:
        response = await client.messages.create(**params)
    except AnthropicError as e:
        _log_failed(e, "blocking", started)
        raise
    result = _to_result(response)
    _log_completed(result, "blocking", started)
    return result


class EstimationStream:
    """Genera la estimación en streaming: al iterar devuelve el texto a trozos.

    Cuando termina la iteración, `result` contiene la estimación completa y los tokens.
    """

    def __init__(self, transcription: str):
        self.transcription = transcription
        self.result: EstimationResult | None = None

    def __iter__(self) -> Iterator[str]:
        params = _request_params(self.transcription)
        client = Anthropic(**_client_options())
        log.debug("llm_call_started", mode="stream", model=params["model"])
        started = time.perf_counter()
        try:
            with client.messages.stream(**params) as stream:
                yield from stream.text_stream
                self.result = _to_result(stream.get_final_message())
        except AnthropicError as e:
            _log_failed(e, "stream", started)
            raise
        _log_completed(self.result, "stream", started)
