"""Guardrails de entrada: se ejecutan antes de llamar al LLM (y antes de la caché).

1. Moderación (opcional, API de OpenAI): contenido de odio, violencia, sexual...
2. Prompt injection: patrones típicos ("ignora las instrucciones anteriores"...).
3. Datos personales: emails, teléfonos, IBAN y DNI/NIE.

Política: si una capa rechaza la descripción se lanza InputGuardrailViolation (no se intenta
corregir). Las capas 2 y 3 son heurísticas con expresiones regulares: muestran el patrón,
no sustituyen a una herramienta de cumplimiento normativo.
"""

import re
from typing import Literal

import openai
import structlog

from app.config import get_settings

log = structlog.get_logger(module=__name__)

Reason = Literal["moderation", "prompt_injection", "pii"]


class InputGuardrailViolation(Exception):
    def __init__(self, message: str, *, reason: Reason):
        super().__init__(message)
        self.message = message
        self.reason = reason


_PROMPT_INJECTION_PATTERNS = [
    # Inglés
    r"ignore\s+(all\s+|the\s+)?(previous|prior|above)?\s*(instructions?|prompts?|rules?)",
    r"disregard\b.{0,40}\b(instructions?|prompts?|rules?|context)",
    r"forget\s+(everything|all|previous)",
    r"\byou\s+are\s+now\b",
    r"new\s+instructions?\s*[:.\-]",
    # Español
    r"ignora\w*\s+(todas?\s+)?(las\s+|tus\s+)?(instrucciones|reglas|indicaciones)",
    r"olvida\w*\s+(todo|todas?\s+las\s+instrucciones|lo\s+anterior)",
    r"a\s+partir\s+de\s+ahora\s+(eres|act[uú]as?|responde)",
    r"\bahora\s+eres\b",
    r"nuevas?\s+instrucci[oó]n(es)?\s*[:.\-]",
    r"(muestra|revela|repite)\w*\s+(el|tu)\s+(system\s+)?prompt",
    # Etiquetas que imitan la estructura del prompt
    r"</?\s*(system|instructions?|instrucciones|prompt)\s*>",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _PROMPT_INJECTION_PATTERNS]

_PII_PATTERNS = {
    "email": re.compile(r"[\w.%+-]+@[\w-]+(\.[\w-]+)*\.[A-Za-z]{2,}"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b"),
    "dni_nie": re.compile(r"\b[XYZ]?\d{7,8}-?[A-HJ-NP-TV-Z]\b"),
    # Móvil o fijo español (empieza por 6, 7, 8 o 9) con prefijo opcional, o internacional
    "phone": re.compile(
        r"(?<![\d.,])(?:\+34[\s.-]?)?[6789]\d{2}"
        r"(?:[\s.-]?\d{3}[\s.-]?\d{3}|(?:[\s.-]?\d{2}){3})(?!\d|[.,]\d)"
    ),
}


async def check_input(description: str) -> None:
    """Ejecuta las capas en orden y lanza InputGuardrailViolation en la primera que falle."""
    await _check_moderation(description)
    _check_prompt_injection(description)
    _check_pii(description)


async def _check_moderation(description: str) -> None:
    settings = get_settings()
    if not (settings.moderation_enabled and settings.openai_api_key):
        return
    try:
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.moderations.create(
            model="omni-moderation-latest", input=description
        )
    except openai.OpenAIError as e:
        # Si la moderación no está disponible no se bloquea la petición (fail open)
        log.warning("moderation_unavailable", error_type=type(e).__name__, error=str(e))
        return
    result = response.results[0]
    if result.flagged:
        categories = [name for name, flagged in result.categories.model_dump().items() if flagged]
        log.info("input_rejected", reason="moderation", categories=categories)
        raise InputGuardrailViolation(
            "La descripción contiene contenido no permitido "
            f"({', '.join(categories) or 'sin categoría'}).",
            reason="moderation",
        )


def _check_prompt_injection(description: str) -> None:
    for pattern in _INJECTION_RE:
        if pattern.search(description):
            log.info("input_rejected", reason="prompt_injection", pattern=pattern.pattern)
            raise InputGuardrailViolation(
                "La descripción contiene instrucciones dirigidas al modelo. Describe solo el "
                "proyecto que quieres estimar.",
                reason="prompt_injection",
            )


def _check_pii(description: str) -> None:
    found = [kind for kind, pattern in _PII_PATTERNS.items() if pattern.search(description)]
    if found:
        # No se registra el dato, solo de qué tipo es
        log.info("input_rejected", reason="pii", kinds=found)
        raise InputGuardrailViolation(
            "La descripción contiene datos personales (" + ", ".join(found) + "). "
            "Elimínalos: no son necesarios para estimar el proyecto.",
            reason="pii",
        )
