"""Métricas de la prueba de carga: presupuestos de latencia y coste, y deriva de la memoria.

Están aquí y no en evals/metrics.py porque evalúan otra cosa: las de evals/metrics.py
puntúan un EstimationResult frente a un caso del golden dataset; estas puntúan la
observación de un turno (TurnObservation, el evento turn_observed) y el snapshot de la
memoria de la sesión (GET /sessions/{id}/snapshot). Reutilizan MetricResult (name, score,
detail y passed) para que el informe las trate igual.

Deterministas a propósito: nada de embeddings ni LLM como juez. MemoryDriftMetric busca el
hecho literal, sin distinguir mayúsculas, tildes ni el punto de los miles ("30.000" casa con
"30000").
"""

import json
import re
from typing import Any

from evals.metrics import MetricResult, normalize

LOCATIONS = ("summary", "anchors", "metadata", "window")
LONG_TERM = ("summary", "anchors", "metadata")  # lo que sobrevive cuando el turno sale


def _compact(text: str) -> str:
    """normalize() y sin el separador de miles: '30.000' -> '30000'."""
    return re.sub(r"(?<=\d)\.(?=\d{3})", "", normalize(text))


class LatencyBudgetMetric:
    """1.0 si latency_ms <= budget_ms; 0.0 si no. Convierte el SLA en un test."""

    def __init__(self, budget_ms: int):
        self.budget_ms = budget_ms

    def evaluate(self, observation: dict[str, Any]) -> MetricResult:
        latency = observation["latency_ms"]
        return MetricResult(
            "latency_budget",
            float(latency <= self.budget_ms),
            f"{latency} ms (presupuesto {self.budget_ms} ms)",
        )


class CostBudgetMetric:
    """1.0 si cost_usd <= budget_usd; 0.0 si no (o si no se conoce el coste)."""

    def __init__(self, budget_usd: float):
        self.budget_usd = budget_usd

    def evaluate(self, observation: dict[str, Any]) -> MetricResult:
        cost = observation["cost_usd"]
        if cost is None:
            return MetricResult("cost_budget", 0.0, "coste desconocido")
        return MetricResult(
            "cost_budget",
            float(cost <= self.budget_usd),
            f"{cost:.6f} USD (presupuesto {self.budget_usd} USD)",
        )


def snapshot_texts(snapshot: dict[str, Any]) -> dict[str, str]:
    """El texto de cada parte de la memoria de la sesión, para buscar hechos."""
    return {
        "summary": snapshot.get("conversation_summary") or "",
        "anchors": "\n".join(m["content"] for m in snapshot.get("anchor_messages", [])),
        "metadata": json.dumps(snapshot.get("project_metadata", {}), ensure_ascii=False),
        "window": "\n".join(m["content"] for m in snapshot.get("window_messages", [])),
    }


class MemoryDriftMetric:
    """1.0 si el hecho declarado en un turno anterior aparece en alguna de las partes de la
    memoria indicadas (`where`) del snapshot actual; 0.0 si no.

    Por defecto busca en la memoria a largo plazo (resumen, anclas y metadata), que es lo que
    queda cuando el turno sale de la ventana. Con "window" incluido mide todo el contexto que
    recibe el LLM.
    """

    def __init__(self, fact: str, where: tuple[str, ...] | list[str] = LONG_TERM):
        unknown = set(where) - set(LOCATIONS)
        if unknown:
            raise ValueError(f"partes de la memoria desconocidas: {sorted(unknown)}")
        self.fact = fact
        self.where = tuple(where)

    def found_in(self, snapshot: dict[str, Any]) -> list[str]:
        needle = _compact(self.fact)
        texts = snapshot_texts(snapshot)
        return [part for part in self.where if needle in _compact(texts[part])]

    def evaluate(self, snapshot: dict[str, Any]) -> MetricResult:
        found = self.found_in(snapshot)
        detail = f"'{self.fact}' en {', '.join(found)}" if found else f"'{self.fact}' olvidado"
        return MetricResult("memory_drift", float(bool(found)), detail)
