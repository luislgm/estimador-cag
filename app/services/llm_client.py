"""Cliente de LLM sobre LiteLLM + Instructor: una misma interfaz para OpenAI y Anthropic.

El proveedor y el modelo salen de la configuración (LLM_PROVIDER y LLM_MODEL). Este módulo
sabe *cómo* llamar al LLM: pide una salida estructurada (un modelo de Pydantic) y deja que
Instructor la valide y, si un validador falla, le devuelva el error al modelo para que lo
corrija. También se encarga de los tokens, el coste, los logs y los errores.
*Qué* se le pide lo deciden el orquestador y las plantillas de prompt.
"""

import os

# Precios del paquete de LiteLLM, sin descargar la tabla de internet al importar
# (arranque más rápido y sin dependencia de red). Se actualizan al actualizar LiteLLM.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import TypeVar  # noqa: E402

import instructor  # noqa: E402
import litellm  # noqa: E402
import openai  # noqa: E402
import structlog  # noqa: E402
from instructor.core import InstructorRetryException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402

litellm.suppress_debug_info = True

log = structlog.get_logger(module=__name__)

# Intentos extra si la respuesta no pasa los validadores (en total: 1 + MAX_RETRIES)
MAX_RETRIES = 2

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Fallo al llamar al proveedor de LLM: credenciales, red, límite de uso, o una respuesta
    que no pasa los validadores ni tras los reintentos."""


@dataclass
class LLMCallInfo:
    """Metadatos de una llamada (con todos sus intentos)."""

    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    finish_reason: str  # normalizado por LiteLLM: "stop", "tool_calls", "length"...
    cost_usd: float | None
    attempts: int = 1

    @property
    def truncated(self) -> bool:
        """True si la última respuesta se cortó por alcanzar el límite de tokens de salida."""
        return self.finish_reason == "length"


def model_id(settings: Settings) -> str:
    """Identificador de LiteLLM: "<proveedor>/<modelo>", p. ej. "anthropic/claude-haiku-4-5"."""
    if "/" in settings.llm_model:
        return settings.llm_model
    return f"{settings.llm_provider}/{settings.llm_model}"


def _api_key(settings: Settings) -> str | None:
    if settings.llm_provider == "openai":
        return settings.openai_api_key
    return settings.anthropic_api_key


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Coste según la tabla de precios de LiteLLM; None si no conoce el modelo."""
    try:
        input_cost, output_cost = litellm.cost_per_token(
            model=model, prompt_tokens=input_tokens, completion_tokens=output_tokens
        )
    except Exception:  # noqa: BLE001 — modelo sin precio en la tabla
        return None
    return round(input_cost + output_cost, 6)


def _structured_client() -> instructor.AsyncInstructor:
    # Se crea en cada llamada (es barato) para usar la función litellm.acompletion vigente:
    # así la ve envuelta la instrumentación de Phoenix y la pueden sustituir los tests.
    # Modo TOOLS: el esquema se envía como herramienta obligatoria. Funciona con Haiku 4.5 y
    # gpt-4o-mini; los modelos de Claude más recientes no admiten forzar una herramienta y
    # necesitarían otro modo (p. ej. instructor.Mode.JSON).
    return instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.TOOLS)


async def acomplete_structured(
    messages: list[dict], response_model: type[T], max_tokens: int
) -> tuple[T, LLMCallInfo]:
    """Pide al LLM una respuesta con la forma de `response_model`, ya validada.

    `messages` es el array completo que se envía: el system prompt, los turnos anteriores de
    la conversación (si los hay) y el mensaje actual del usuario.
    """
    settings = get_settings()
    model = model_id(settings)
    client = _structured_client()
    attempts = 0

    def count_attempt(*args, **kwargs) -> None:
        nonlocal attempts
        attempts += 1

    client.on("completion:kwargs", count_attempt)

    log.debug("llm_call_started", model=model, response_model=response_model.__name__)
    started = time.perf_counter()
    try:
        result, raw = await client.create_with_completion(
            model=model,
            api_key=_api_key(settings),
            max_tokens=max_tokens,
            messages=messages,
            response_model=response_model,
            max_retries=MAX_RETRIES,
        )
    except InstructorRetryException as e:
        # Instructor envuelve también los errores del proveedor (red, credenciales, límites):
        # se distinguen de los de validación por la causa original
        if isinstance(e.__cause__, openai.OpenAIError):  # LiteLLM traduce los de todos
            raise _failed(e.__cause__, attempts, started) from e
        # El detalle de cada intento va al log; al cliente, solo un resumen
        log.debug("llm_validation_attempts", detail=str(e))
        message = f"la respuesta no pasó las validaciones tras {attempts} intentos"
        raise _failed(LLMError(message), attempts, started) from e

    usage = raw.usage  # Instructor acumula los tokens de todos los intentos
    info = LLMCallInfo(
        model=raw.model or model,
        provider=settings.llm_provider,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        finish_reason=(raw.choices[0].finish_reason if raw.choices else None) or "unknown",
        cost_usd=None,
        attempts=max(attempts, 1),
    )
    info.cost_usd = _cost_usd(model, info.input_tokens, info.output_tokens)
    _log_completed(info, started)
    return result, info


def _log_completed(info: LLMCallInfo, started: float) -> None:
    fields = {
        "provider": info.provider,
        "model": info.model,
        "input_tokens": info.input_tokens,
        "output_tokens": info.output_tokens,
        "cost_usd": info.cost_usd,
        "finish_reason": info.finish_reason,
        "attempts": info.attempts,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }
    log.info("llm_call_completed", **fields)
    if info.attempts > 1:
        log.warning("llm_response_retried", **fields)


def _failed(error: Exception, attempts: int, started: float) -> LLMError:
    log.error(
        "llm_call_failed",
        error_type=type(error).__name__,
        error=str(error),
        attempts=attempts,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return error if isinstance(error, LLMError) else LLMError(str(error))
