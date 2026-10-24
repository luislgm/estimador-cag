"""Trazas de las llamadas al LLM con Arize Phoenix (OpenTelemetry + OpenInference).

Es opcional: solo se activa si PHOENIX_COLLECTOR_ENDPOINT está configurado. Al activarse,
cada llamada al SDK de Anthropic (messages.create y messages.stream) se envía a Phoenix con
el system prompt, la transcripción, la respuesta, los tokens y la latencia. No hace falta
tocar el código del servicio: la instrumentación envuelve el SDK automáticamente.
"""

import structlog
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from phoenix.otel import register

from app.config import Settings

log = structlog.get_logger(module=__name__)

_enabled = False


def setup_tracing(settings: Settings) -> bool:
    """Activa las trazas si hay endpoint configurado. Devuelve True si quedan activas.

    Es idempotente: Streamlit vuelve a ejecutar el script en cada interacción y no debe
    registrar la instrumentación más de una vez.
    """
    global _enabled
    if _enabled:
        return True
    if not settings.phoenix_collector_endpoint:
        log.debug("tracing_disabled", reason="PHOENIX_COLLECTOR_ENDPOINT no configurado")
        return False

    tracer_provider = register(
        endpoint=settings.phoenix_collector_endpoint,
        project_name=settings.phoenix_project_name,
        batch=True,  # envía las trazas en segundo plano, sin frenar las peticiones
        verbose=False,
    )
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
    _enabled = True
    log.info(
        "tracing_enabled",
        endpoint=settings.phoenix_collector_endpoint,
        project=settings.phoenix_project_name,
    )
    return True
