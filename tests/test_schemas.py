import pytest
from pydantic import ValidationError

from app.schemas.estimation import EstimationResult
from tests.factories import make_phase, make_result


def test_valid_result():
    result = make_result()

    assert result.total_hours == 80
    assert result.total_cost_eur == 4000
    assert not result.out_of_scope


def test_phase_cost_is_computed_from_the_hourly_rate():
    assert make_phase("Desarrollo", hours=40).cost_eur == 2000


def test_totals_are_computed_from_the_phases():
    result = make_result(phases=[make_phase("Diseño", 20), make_phase("Desarrollo", 60)])

    assert (result.total_hours, result.total_cost_eur) == (80, 4000)


def test_figures_sent_by_the_model_are_ignored():
    # Si el modelo manda costes o totales (mal sumados), no cuentan: se recalculan
    data = make_result().model_dump(mode="json")
    data["total_hours"] = 9999
    data["phases"][0]["cost_eur"] = 1

    result = EstimationResult.model_validate(data)

    assert (result.total_hours, result.phases[0].cost_eur) == (80, 1000)


def test_the_llm_does_not_see_computed_fields_but_the_response_does():
    to_llm = EstimationResult.model_json_schema()
    response = EstimationResult.model_json_schema(mode="serialization")

    assert {"total_hours", "total_cost_eur"}.isdisjoint(to_llm["properties"])
    assert "cost_eur" not in to_llm["$defs"]["Phase"]["properties"]
    assert {"total_hours", "total_cost_eur"} <= set(response["properties"])
    assert {"total_hours", "total_cost_eur"} <= set(make_result().model_dump())


def test_low_confidence_requires_out_of_scope_prefix():
    with pytest.raises(ValidationError, match="debe empezar por 'Fuera de alcance:'"):
        make_result(confidence_pct=20)


def test_low_confidence_with_prefix_is_valid():
    result = make_result(confidence_pct=20, summary="Fuera de alcance: falta saber qué se quiere.")

    assert result.out_of_scope


def test_an_estimate_needs_several_phases():
    with pytest.raises(ValidationError, match="al menos 2 fases y solo hay 1"):
        make_result(phases=[make_phase("Descubrimiento", 32)])


def test_out_of_scope_has_a_single_placeholder_phase():
    result = make_result(
        confidence_pct=10,
        summary="Fuera de alcance: falta saber qué se quiere.",
        phases=[make_phase("Sin estimar", 0)],
    )

    assert result.out_of_scope and len(result.phases) == 1


def test_at_least_one_phase():
    with pytest.raises(ValidationError):
        make_result(phases=[])
