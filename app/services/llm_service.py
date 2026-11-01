"""Estimador CAG: qué se le pide al LLM.

Construye el system prompt (instrucciones + estimaciones de ejemplo) y delega la llamada en
app.services.llm_client, que funciona igual con OpenAI o con Anthropic.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from app.context.examples import ESTIMATION_EXAMPLES
from app.services import llm_client
from app.services.llm_client import LLMError, LLMResult

__all__ = [
    "EstimationResult",
    "EstimationStream",
    "LLMError",
    "build_system_prompt",
    "generate_estimation",
]

MAX_TOKENS = 4096

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
    finish_reason: str = "stop"
    cost_usd: float | None = None

    @property
    def truncated(self) -> bool:
        """True si la estimación se cortó por alcanzar el límite de tokens de salida."""
        return self.finish_reason == "length"


def _to_estimation(result: LLMResult) -> EstimationResult:
    return EstimationResult(
        estimation=result.text,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        finish_reason=result.finish_reason,
        cost_usd=result.cost_usd,
    )


async def generate_estimation(transcription: str) -> EstimationResult:
    """Envía la transcripción al LLM y devuelve la estimación generada."""
    result = await llm_client.acomplete(build_system_prompt(), transcription, MAX_TOKENS)
    return _to_estimation(result)


class EstimationStream:
    """Genera la estimación en streaming: al iterar devuelve el texto a trozos.

    Cuando termina la iteración, `result` contiene la estimación completa y los tokens.
    """

    def __init__(self, transcription: str):
        self.transcription = transcription
        self.result: EstimationResult | None = None

    def __iter__(self) -> Iterator[str]:
        stream = llm_client.CompletionStream(build_system_prompt(), self.transcription, MAX_TOKENS)
        yield from stream
        self.result = _to_estimation(stream.result)
