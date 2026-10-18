import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "APP_ENV",
    "LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def test_env(monkeypatch, tmp_path):
    """Aísla los tests del .env local y fija una configuración conocida."""
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # directorio sin .env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-4-5")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    return TestClient(app)
