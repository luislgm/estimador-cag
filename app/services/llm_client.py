"""Cliente de LLM sobre LiteLLM: una misma interfaz para OpenAI y Anthropic.

El proveedor y el modelo salen de la configuración (LLM_PROVIDER y LLM_MODEL). Este módulo
sabe *cómo* llamar al LLM (normal o en streaming) y se encarga de los tokens, el coste,
el motivo de fin, los logs y los errores. *Qué* se le pide lo decide llm_service.py.
"""

import os

# Precios del paquete de LiteLLM, sin descargar la tabla de internet al importar
# (arranque más rápido y sin dependencia de red). Se actualizan al actualizar LiteLLM.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import time  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from dataclasses import asdict, dataclass, replace  # noqa: E402

import litellm  # noqa: E402
import openai  # noqa: E402
import structlog  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402
from app.services.cache import EstimationCache, get_cache  # noqa: E402

litellm.suppress_debug_info = True

log = structlog.get_logger(module=__name__)


class LLMError(Exception):
    """Fallo al llamar al proveedor de LLM: credenciales, red, límite de uso, modelo..."""


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    finish_reason: str  # normalizado por LiteLLM: "stop", "length", "content_filter"...
    cost_usd: float | None
    cache_hit: bool = False

    @property
    def truncated(self) -> bool:
        """True si la respuesta se cortó por alcanzar el límite de tokens de salida."""
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


def _call_params(system: str, user: str, max_tokens: int) -> dict:
    settings = get_settings()
    return {
        "model": model_id(settings),
        "api_key": _api_key(settings),
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Coste según la tabla de precios de LiteLLM; None si no conoce el modelo."""
    try:
        input_cost, output_cost = litellm.cost_per_token(
            model=model, prompt_tokens=input_tokens, completion_tokens=output_tokens
        )
    except Exception:  # noqa: BLE001 — modelo sin precio en la tabla
        return None
    return round(input_cost + output_cost, 6)


def _to_result(text: str, model: str, usage, finish_reason: str | None) -> LLMResult:
    settings = get_settings()
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    return LLMResult(
        text=text,
        model=model,
        provider=settings.llm_provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason or "unknown",
        cost_usd=_cost_usd(model_id(settings), input_tokens, output_tokens),
    )


def _log_completed(result: LLMResult, mode: str, started: float) -> None:
    fields = {
        "mode": mode,
        "provider": result.provider,
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


def _failed(error: Exception, mode: str, started: float) -> LLMError:
    log.error(
        "llm_call_failed",
        mode=mode,
        error_type=type(error).__name__,
        error=str(error),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    return LLMError(str(error))


def _cache_key(params: dict) -> str:
    system, user = (m["content"] for m in params["messages"])
    return EstimationCache.make_key(
        model=params["model"], system=system, user=user, max_tokens=params["max_tokens"]
    )


def _from_cache(params: dict, mode: str) -> LLMResult | None:
    """Respuesta guardada para esta misma petición, o None si no hay (o no hay caché)."""
    cache = get_cache()
    if cache is None or (cached := cache.get(_cache_key(params))) is None:
        return None
    original = LLMResult(**cached)
    log.info(
        "llm_cache_hit",
        mode=mode,
        provider=original.provider,
        model=original.model,
        input_tokens=original.input_tokens,
        output_tokens=original.output_tokens,
        saved_usd=original.cost_usd,
    )
    # No se ha llamado al LLM: no hay coste. Los tokens se conservan como información.
    return replace(original, cost_usd=0.0, cache_hit=True)


def _save_to_cache(params: dict, result: LLMResult) -> None:
    cache = get_cache()
    # Una respuesta cortada no se guarda: se volvería a servir incompleta
    if cache is not None and not result.truncated:
        cache.set(_cache_key(params), asdict(result) | {"cache_hit": False})


async def acomplete(system: str, user: str, max_tokens: int) -> LLMResult:
    """Llamada completa (sin streaming): espera a la respuesta entera."""
    params = _call_params(system, user, max_tokens)
    if cached := _from_cache(params, "blocking"):
        return cached
    log.debug("llm_call_started", mode="blocking", model=params["model"])
    started = time.perf_counter()
    try:
        response = await litellm.acompletion(**params)
    except openai.OpenAIError as e:  # LiteLLM traduce los errores de todos los proveedores
        raise _failed(e, "blocking", started) from e
    choice = response.choices[0]
    result = _to_result(
        choice.message.content or "", response.model, response.usage, choice.finish_reason
    )
    _log_completed(result, "blocking", started)
    _save_to_cache(params, result)
    return result


class CompletionStream:
    """Llamada en streaming: al iterar devuelve el texto a trozos según llega.

    Cuando termina la iteración, `result` tiene el texto completo, los tokens y el coste.
    """

    def __init__(self, system: str, user: str, max_tokens: int):
        self.params = _call_params(system, user, max_tokens) | {
            "stream": True,
            "stream_options": {"include_usage": True},  # tokens reales al final del stream
        }
        self.result: LLMResult | None = None

    def __iter__(self) -> Iterator[str]:
        if cached := _from_cache(self.params, "stream"):
            # Respuesta ya conocida: se entrega entera de una vez, sin llamar al LLM
            self.result = cached
            yield cached.text
            return
        log.debug("llm_call_started", mode="stream", model=self.params["model"])
        started = time.perf_counter()
        parts: list[str] = []
        model, usage, finish_reason = self.params["model"], None, None
        try:
            for chunk in litellm.completion(**self.params):
                model = chunk.model or model
                usage = getattr(chunk, "usage", None) or usage
                if chunk.choices:
                    finish_reason = chunk.choices[0].finish_reason or finish_reason
                    if text := chunk.choices[0].delta.content:
                        parts.append(text)
                        yield text
        except openai.OpenAIError as e:
            raise _failed(e, "stream", started) from e
        self.result = _to_result("".join(parts), model, usage, finish_reason)
        _log_completed(self.result, "stream", started)
        _save_to_cache(self.params, self.result)
