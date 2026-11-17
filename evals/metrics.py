"""Métricas de las evals: cada una puntúa un caso entre 0 y 1, o no aplica (None).

- schema: todos los turnos responden 200 con un EstimationResult válido (todos los casos).
- scope: fuera de alcance cuando toca, sin cifras, y solo entonces (todos los casos).
- hours_in_range: las horas totales caen en el rango esperado (casos con `hours`).
- content_recall: proporción de conceptos esperados que aparecen (casos con conceptos).
- tier: la audiencia deducida es la esperada (casos con `tier`).
- memory: el project_name se mantiene en todos los turnos (casos de varios turnos).
- direction: las horas suben o bajan como se pidió (casos con `change`).

La suite pasa si la media de cada métrica alcanza su umbral (THRESHOLDS). Las métricas
deterministas del contrato (esquema, alcance, tier, memoria) exigen el 100 %; las que dependen
del criterio del modelo (horas, conceptos, dirección) tienen margen.

La coherencia de los números (totales = suma de fases, coste = horas × 50) no se mide aquí:
no la calcula el modelo, sino el código (campos calculados del esquema).
"""

import json
import unicodedata
from dataclasses import dataclass
from typing import Any

from app.schemas.estimation import EstimationResult
from evals.dataset import EvalCase

THRESHOLDS = {
    "schema": 1.0,
    "scope": 1.0,
    "hours_in_range": 0.7,
    "content_recall": 0.7,
    "tier": 1.0,
    "memory": 1.0,
    "direction": 0.5,
}


@dataclass
class TurnOutcome:
    status_code: int
    body: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.status_code == 200

    @property
    def result(self) -> EstimationResult:
        return EstimationResult.model_validate(self.body["result"])


@dataclass
class MetricResult:
    name: str
    score: float
    detail: str

    @property
    def passed(self) -> bool:
        return self.score >= 1


def normalize(text: str) -> str:
    """Minúsculas y sin tildes, para comparar conceptos."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def schema(case: EvalCase, turns: list[TurnOutcome]) -> MetricResult:
    failed = [t for t in turns if not t.ok]
    if len(turns) < len(case.turns) or failed:
        status = failed[0].status_code if failed else "sin respuesta"
        return MetricResult("schema", 0.0, f"turno fallido ({status})")
    return MetricResult("schema", 1.0, "ok")


def scope(case: EvalCase, last: EstimationResult) -> MetricResult:
    expected = case.expected.out_of_scope
    correct = last.out_of_scope == expected and (not expected or last.total_hours == 0)
    detail = f"esperado {'fuera' if expected else 'dentro'}, obtenido " + (
        "fuera" if last.out_of_scope else "dentro"
    )
    return MetricResult("scope", float(correct), detail)


def hours_in_range(case: EvalCase, last: EstimationResult) -> MetricResult | None:
    if case.expected.hours is None:
        return None
    low, high = case.expected.hours
    inside = low <= last.total_hours <= high
    return MetricResult("hours_in_range", float(inside), f"{last.total_hours} h en [{low}, {high}]")


def content_recall(case: EvalCase, last: EstimationResult) -> MetricResult | None:
    concepts = case.expected.must_mention
    if not concepts:
        return None
    text = normalize(json.dumps(last.model_dump(mode="json"), ensure_ascii=False))
    missing = [c[0] for c in concepts if not any(normalize(s) in text for s in c)]
    score = (len(concepts) - len(missing)) / len(concepts)
    detail = "todos" if not missing else "faltan: " + ", ".join(missing)
    return MetricResult("content_recall", score, detail)


def tier(case: EvalCase, last_body: dict[str, Any]) -> MetricResult | None:
    if case.expected.tier is None:
        return None
    got = last_body["session"].get("tier")
    return MetricResult(
        "tier", float(got == case.expected.tier), f"esperado {case.expected.tier}, obtenido {got}"
    )


def memory(case: EvalCase, results: list[EstimationResult]) -> MetricResult | None:
    if len(case.turns) < 2:
        return None
    names = [r.project_name for r in results]
    stable = len({normalize(n) for n in names}) == 1
    return MetricResult("memory", float(stable), " -> ".join(names))


def direction(case: EvalCase, results: list[EstimationResult]) -> MetricResult | None:
    if case.expected.change is None:
        return None
    first, last = results[0].total_hours, results[-1].total_hours
    correct = last > first if case.expected.change == "up" else last < first
    return MetricResult(
        "direction", float(correct), f"{first} h -> {last} h (esperado {case.expected.change})"
    )


def score_case(case: EvalCase, turns: list[TurnOutcome]) -> list[MetricResult]:
    """Todas las métricas que aplican al caso. Si algún turno falla, solo cuenta schema: el
    resto no se puede medir (y schema ya hace fallar la suite)."""
    first = schema(case, turns)
    if first.score < 1:
        return [first]
    results = [t.result for t in turns]
    last = results[-1]
    candidates = [
        first,
        scope(case, last),
        hours_in_range(case, last),
        content_recall(case, last),
        tier(case, turns[-1].body),
        memory(case, results),
        direction(case, results),
    ]
    return [m for m in candidates if m is not None]


def aggregate(scored: list[list[MetricResult]]) -> dict[str, dict[str, Any]]:
    """Media, número de casos y si pasa el umbral, por métrica."""
    summary = {}
    for name, threshold in THRESHOLDS.items():
        scores = [m.score for case in scored for m in case if m.name == name]
        if not scores:
            continue
        mean = sum(scores) / len(scores)
        summary[name] = {
            "mean": round(mean, 3),
            "cases": len(scores),
            "threshold": threshold,
            "passed": mean >= threshold,
        }
    return summary
