import pytest
from pydantic import ValidationError

from tests.factories import make_phase, make_result


def test_valid_result():
    result = make_result()

    assert result.total_hours == 80
    assert result.total_cost_eur == 4000
    assert not result.out_of_scope


def test_phase_cost_must_match_hourly_rate():
    with pytest.raises(ValidationError, match="a 50 €/h deberían ser 2000 €"):
        make_phase("Desarrollo", hours=40, cost_eur=1800)


def test_total_cost_must_be_the_sum_of_phases():
    with pytest.raises(ValidationError, match="las fases suman 4000 € pero total_cost_eur es 5000"):
        make_result(total_cost_eur=5000)


def test_total_hours_must_be_the_sum_of_phases():
    with pytest.raises(ValidationError, match="las fases suman 80 h pero total_hours es 100"):
        make_result(total_hours=100)


def test_low_confidence_requires_out_of_scope_prefix():
    with pytest.raises(ValidationError, match="debe empezar por 'Fuera de alcance:'"):
        make_result(confidence_pct=20)


def test_low_confidence_with_prefix_is_valid():
    result = make_result(confidence_pct=20, summary="Fuera de alcance: falta saber qué se quiere.")

    assert result.out_of_scope


def test_at_least_one_phase():
    with pytest.raises(ValidationError):
        make_result(phases=[], total_hours=0, total_cost_eur=0)
