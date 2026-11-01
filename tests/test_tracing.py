from unittest.mock import patch

import pytest

from app import tracing
from app.config import Settings


@pytest.fixture(autouse=True)
def reset_tracing(monkeypatch):
    monkeypatch.setattr(tracing, "_enabled", False)


def test_disabled_without_endpoint():
    with (
        patch.object(tracing, "register") as register,
        patch.object(tracing, "LiteLLMInstrumentor") as instrumentor,
    ):
        assert tracing.setup_tracing(Settings()) is False

    register.assert_not_called()
    instrumentor.assert_not_called()


def test_enabled_with_endpoint(monkeypatch):
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")

    with (
        patch.object(tracing, "register") as register,
        patch.object(tracing, "LiteLLMInstrumentor") as instrumentor,
    ):
        assert tracing.setup_tracing(Settings()) is True

    register.assert_called_once_with(
        endpoint="http://localhost:6006/v1/traces",
        project_name="estimador-cag",
        batch=True,
        verbose=False,
        set_global_tracer_provider=False,
    )
    instrumentor.return_value.instrument.assert_called_once_with(
        tracer_provider=register.return_value
    )


def test_is_idempotent(monkeypatch):
    """Streamlit reejecuta el script en cada interacción: solo se instrumenta una vez."""
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")

    with (
        patch.object(tracing, "register") as register,
        patch.object(tracing, "LiteLLMInstrumentor") as instrumentor,
    ):
        assert tracing.setup_tracing(Settings()) is True
        assert tracing.setup_tracing(Settings()) is True

    register.assert_called_once()
    instrumentor.return_value.instrument.assert_called_once()


def test_custom_project_name(monkeypatch):
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006/v1/traces")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "estimador-pruebas")

    with (
        patch.object(tracing, "register") as register,
        patch.object(tracing, "LiteLLMInstrumentor"),
    ):
        tracing.setup_tracing(Settings())

    assert register.call_args.kwargs["project_name"] == "estimador-pruebas"
