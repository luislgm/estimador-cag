"""Coste estimado de una llamada al LLM a partir de los tokens consumidos."""

import re

# Precio en USD por millón de tokens: (entrada, salida). Actualizar si cambian las tarifas.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

# La API devuelve el identificador con fecha: "claude-haiku-4-5-20251001"
_DATE_SUFFIX = re.compile(r"-\d{8}$")


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Devuelve el coste en USD, o None si el modelo no está en la tabla de precios."""
    prices = MODEL_PRICES.get(_DATE_SUFFIX.sub("", model))
    if prices is None:
        return None
    input_price, output_price = prices
    return round((input_tokens * input_price + output_tokens * output_price) / 1_000_000, 6)
