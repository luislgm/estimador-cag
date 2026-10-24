"""Configuración del logging estructurado (structlog) a partir de LOG_LEVEL y APP_ENV."""

import logging
import sys

import structlog


def setup_logging(level: str, app_env: str) -> None:
    """Configura structlog: JSON en producción y texto legible en el resto de entornos.

    Los campos ligados con `structlog.contextvars.bind_contextvars` (por ejemplo, el
    `request_id` de cada petición) se añaden automáticamente a todos los logs.
    """
    production = app_env == "production"
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso" if production else "%H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
    ]
    if production:
        # En JSON la traza de una excepción va como texto; en consola la pinta el renderer
        processors += [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ]
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        # Sin caché para que los tests puedan reconfigurar y capturar los logs
        cache_logger_on_first_use=False,
    )
