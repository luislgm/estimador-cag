"""Métricas de la prueba de carga: una que pasa, una que falla y una en el límite."""

import pytest

from evals.stress.metrics import (
    LOCATIONS,
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
    snapshot_texts,
)

SNAPSHOT = {
    "conversation_summary": "Se estima Vega, un portal B2B. Primero 30.000 €; luego se amplió.",
    "anchor_messages": [
        {"role": "user", "content": "El presupuesto máximo es de 80.000 euros."},
        {"role": "assistant", "content": "## Estimación: Vega"},
    ],
    "project_metadata": {"project_name": "Vega", "mentioned_technologies": ["Stripe", "Flutter"]},
    "window_messages": [
        {"role": "user", "content": "Avisos de entrega por SMS con Twilio."},
        {"role": "assistant", "content": "## Estimación: Vega"},
    ],
}


# --- LatencyBudgetMetric ---


def test_latency_within_budget_passes():
    result = LatencyBudgetMetric(4000).evaluate({"latency_ms": 2500})

    assert (result.name, result.score, result.passed) == ("latency_budget", 1.0, True)
    assert result.detail == "2500 ms (presupuesto 4000 ms)"


def test_latency_over_budget_fails():
    assert not LatencyBudgetMetric(4000).evaluate({"latency_ms": 4001}).passed


def test_latency_exactly_at_the_budget_passes():
    assert LatencyBudgetMetric(4000).evaluate({"latency_ms": 4000}).passed


# --- CostBudgetMetric ---


def test_cost_within_budget_passes():
    assert CostBudgetMetric(0.002).evaluate({"cost_usd": 0.0013}).passed


def test_cost_over_budget_fails():
    result = CostBudgetMetric(0.002).evaluate({"cost_usd": 0.0051})

    assert not result.passed
    assert result.detail == "0.005100 USD (presupuesto 0.002 USD)"


def test_cost_exactly_at_the_budget_passes_and_unknown_cost_fails():
    assert CostBudgetMetric(0.002).evaluate({"cost_usd": 0.002}).passed
    unknown = CostBudgetMetric(0.002).evaluate({"cost_usd": None})
    assert (unknown.passed, unknown.detail) == (False, "coste desconocido")


# --- MemoryDriftMetric ---


def test_fact_in_long_term_memory_passes():
    result = MemoryDriftMetric("Flutter").evaluate(SNAPSHOT)

    assert result.passed
    assert result.detail == "'Flutter' en metadata"


def test_forgotten_fact_fails():
    result = MemoryDriftMetric("Mailchimp").evaluate(SNAPSHOT)

    assert not result.passed
    assert result.detail == "'Mailchimp' olvidado"


def test_fact_only_in_the_window_is_not_long_term_memory():
    # Límite: el hecho está en el contexto que recibe el LLM, pero aún no en la memoria que
    # sobrevive cuando el turno sale de la ventana
    assert not MemoryDriftMetric("Twilio").evaluate(SNAPSHOT).passed
    assert MemoryDriftMetric("Twilio", LOCATIONS).evaluate(SNAPSHOT).passed


def test_match_ignores_case_accents_and_thousand_separators():
    assert MemoryDriftMetric("vega").found_in(SNAPSHOT) == ["summary", "anchors", "metadata"]
    assert MemoryDriftMetric("30000").found_in(SNAPSHOT) == ["summary"]
    assert MemoryDriftMetric("80.000", ["anchors"]).evaluate(SNAPSHOT).passed
    assert MemoryDriftMetric("MAXIMO", ["anchors"]).evaluate(SNAPSHOT).passed


def test_unknown_memory_part_is_rejected():
    with pytest.raises(ValueError, match="desconocidas"):
        MemoryDriftMetric("Vega", ["historial"])


def test_empty_snapshot_has_empty_texts():
    assert snapshot_texts({}) == {"summary": "", "anchors": "", "metadata": "{}", "window": ""}
