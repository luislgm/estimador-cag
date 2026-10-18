import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER")
    monkeypatch.delenv("LLM_MODEL")
    monkeypatch.setenv("OPENAI_API_KEY", "x")

    settings = Settings()

    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.app_env == "development"
    assert settings.log_level == "DEBUG"


def test_reads_environment():
    settings = Settings()

    assert settings.llm_provider == "anthropic"
    assert settings.llm_model == "claude-haiku-4-5"
    assert settings.anthropic_api_key == "test-key"


def test_anthropic_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")

    with pytest.raises(ValidationError, match="ANTHROPIC_API_KEY"):
        Settings()


def test_openai_requires_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings()


def test_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    with pytest.raises(ValidationError):
        Settings()
