from app.services.pricing import estimate_cost_usd


def test_known_model():
    assert estimate_cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000) == 6.0


def test_dated_model_id_uses_base_price():
    assert estimate_cost_usd("claude-haiku-4-5-20251001", 1_746, 706) == 0.005276


def test_unknown_model_returns_none():
    assert estimate_cost_usd("gpt-4o-mini", 100, 100) is None
