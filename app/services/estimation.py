"""Orquestador del pipeline de estimación. El router no tiene lógica: solo traduce errores.

    1. Guardrails de entrada (moderación, prompt injection, datos personales)
    2. Renderizar el prompt versionado (system + user)
    3. Caché exacta              -> si acierta, se devuelve sin llamar al LLM (coste 0)
    4. Caché semántica           -> descripción parecida y mismas opciones (si está activada)
    5. LLM con salida estructurada (Instructor valida y reintenta si un validador falla)
    6. Guardrail de salida (deja sin cifras las estimaciones de baja confianza)
    7. Markdown según el formato pedido
    8. Guardar en las cachés

Los guardrails van antes que las cachés: una descripción con datos personales o con un intento
de prompt injection no debe servirse nunca desde una caché. La exacta va antes que la semántica
porque es más barata (no necesita calcular un embedding). Solo se guardan resultados que han
pasado los validadores y el guardrail de salida.
"""

from dataclasses import asdict, dataclass, replace

import structlog

from app.config import get_settings
from app.guardrails.input import check_input
from app.guardrails.output import enforce_scope
from app.prompts import PROMPT_VERSION, render_estimation_prompt
from app.schemas.estimation import EstimationRequest, EstimationResult
from app.services import llm_client
from app.services.cache import EstimationCache, get_cache
from app.services.llm_client import LLMCallInfo
from app.services.render import render_markdown
from app.services.semantic_cache import SemanticCache, get_semantic_cache

log = structlog.get_logger(module=__name__)

MAX_TOKENS = 8192  # una estimación detallada en JSON puede ser larga


@dataclass
class EstimationOutcome:
    result: EstimationResult
    text: str
    info: LLMCallInfo
    prompt_version: str = PROMPT_VERSION
    cache_kind: str | None = None  # "exact", "semantic" o None si se ha llamado al LLM

    @property
    def cache_hit(self) -> bool:
        return self.cache_kind is not None


async def estimate(request: EstimationRequest) -> EstimationOutcome:
    # 1. Guardrails de entrada: lanzan InputGuardrailViolation si rechazan la descripción
    await check_input(request.description)

    # 2. Prompt versionado
    system, user = render_estimation_prompt(request, version=PROMPT_VERSION)
    model = llm_client.model_id(get_settings())

    # 3. Caché exacta
    cache = get_cache()
    key = EstimationCache.make_key(model=model, system=system, user=user, max_tokens=MAX_TOKENS)
    if cache is not None and (cached := cache.get(key)) is not None:
        return _from_cache(request, cached, "exact")

    # 4. Caché semántica
    semantic = get_semantic_cache()
    bucket = SemanticCache.bucket(request, PROMPT_VERSION, model)
    if semantic is not None and (cached := await semantic.lookup(request, bucket)) is not None:
        return _from_cache(request, cached, "semantic")

    # 5. LLM con salida estructurada y validada
    result, info = await llm_client.acomplete_structured(
        system, user, EstimationResult, max_tokens=MAX_TOKENS
    )

    # 6. Guardrail de salida
    result = enforce_scope(result)
    log.info(
        "estimation_generated",
        prompt_version=PROMPT_VERSION,
        confidence_pct=result.confidence_pct,
        phases=len(result.phases),
        total_hours=result.total_hours,
        total_cost_eur=result.total_cost_eur,
        out_of_scope=result.out_of_scope,
    )

    # 8. Guardar en las cachés (solo resultados validados y filtrados)
    payload = {"result": result.model_dump(mode="json"), "info": asdict(info)}
    if cache is not None:
        cache.set(key, payload)
    if semantic is not None:
        await semantic.store(request, bucket, payload)

    return _outcome(request, result, info)


def _from_cache(request: EstimationRequest, cached: dict, kind: str) -> EstimationOutcome:
    info = LLMCallInfo(**cached["info"])
    log.info("estimation_cache_hit", kind=kind, saved_usd=info.cost_usd)
    # No se ha llamado al LLM: coste 0. Los tokens se conservan como información.
    result = EstimationResult.model_validate(cached["result"])
    return _outcome(request, result, replace(info, cost_usd=0.0), cache_kind=kind)


def _outcome(
    request: EstimationRequest,
    result: EstimationResult,
    info: LLMCallInfo,
    cache_kind: str | None = None,
) -> EstimationOutcome:
    # 7. Markdown según el formato pedido
    return EstimationOutcome(
        result=result,
        text=render_markdown(result, request),
        info=info,
        cache_kind=cache_kind,
    )
