"""El golden dataset como suite de pytest, contra el LLM real.

Fuera por defecto (cuesta dinero). Se lanza con:

    uv run pytest -m eval -s

Usa el proveedor y el modelo del .env, o de las variables de entorno si se pasan
(LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini uv run pytest -m eval -s).
"""

import os
from pathlib import Path

import pytest

from app.config import get_settings
from evals.dataset import load_dataset
from evals.run import format_report, run_suite, save_report
from tests.conftest import ENV_VARS

ROOT = Path(__file__).resolve().parent.parent
# Las variables de entorno reales, antes de que el conftest las sustituya por las de prueba
REAL_ENV = {var: os.environ[var] for var in ENV_VARS if var in os.environ}


@pytest.mark.eval
async def test_golden_dataset(monkeypatch):
    # Deshace el aislamiento del conftest: configuración real (.env + entorno)
    monkeypatch.chdir(ROOT)
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for var, value in REAL_ENV.items():
        monkeypatch.setenv(var, value)
    get_settings.cache_clear()

    report = await run_suite(load_dataset())

    print("\n" + format_report(report))
    print(f"\nInforme: {save_report(report)}")
    failed = [name for name, m in report.metrics.items() if not m["passed"]]
    assert report.passed, f"métricas por debajo del umbral: {failed}"
