"""Guardrail de salida: red de seguridad por encima de los validadores de Pydantic.

Los validadores ya obligan a declarar "Fuera de alcance:" cuando la confianza es baja.
Este filtro garantiza además que una estimación fuera de alcance no lleve cifras: si el
modelo no puede estimar con un mínimo de confianza, no debe dar números que parezcan fiables.

Política: filtro. Nunca lanza excepciones; siempre devuelve una estimación bien formada.
"""

import structlog

from app.schemas.estimation import (
    LOW_CONFIDENCE_THRESHOLD,
    EstimationResult,
    Phase,
)

log = structlog.get_logger(module=__name__)

PLACEHOLDER_PHASE = Phase(
    name="Sin estimar",
    hours=0,
    cost_eur=0,
    duration_weeks=1,
    summary="No se puede dimensionar sin más información.",
)


def enforce_scope(result: EstimationResult) -> EstimationResult:
    """Si la estimación es de baja confianza, la deja sin cifras y con el aviso al principio."""
    if result.confidence_pct >= LOW_CONFIDENCE_THRESHOLD and not result.out_of_scope:
        return result
    # Los validadores ya garantizan que el resumen empieza por "Fuera de alcance:"
    if result.total_cost_eur:
        log.info(
            "output_filtered",
            reason="out_of_scope_with_figures",
            confidence_pct=result.confidence_pct,
            total_cost_eur=result.total_cost_eur,
        )
    return EstimationResult(
        project_name=result.project_name,
        summary=result.summary,
        confidence_pct=min(result.confidence_pct, LOW_CONFIDENCE_THRESHOLD - 1),
        phases=[PLACEHOLDER_PHASE],
        team=[],
        risks=result.risks,
        total_hours=0,
        total_cost_eur=0,
        total_duration_weeks=1,
    )
