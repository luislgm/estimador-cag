from dataclasses import dataclass

from anthropic import AsyncAnthropic

from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES

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


async def generate_estimation(transcription: str) -> EstimationResult:
    """Envía la transcripción al LLM y devuelve la estimación generada."""
    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    response = await client.messages.create(
        model=settings.llm_model,
        max_tokens=4096,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": transcription}],
    )

    return EstimationResult(
        estimation="".join(block.text for block in response.content if block.type == "text"),
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
