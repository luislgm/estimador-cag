from unittest.mock import AsyncMock, patch

import structlog
from structlog.testing import capture_logs

from app.services.llm_service import EstimationResult, LLMError

TRANSCRIPTION = (
    "El cliente quiere una web para reservar mesas en su restaurante, "
    "con pago online y un panel de gestión para el personal."
)


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "environment": "development",
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
    }


def test_estimate_returns_estimation(client):
    result = EstimationResult("## Estimación: Test", "claude-haiku-4-5", 1200, 300)

    with patch(
        "app.routers.estimations.generate_estimation", AsyncMock(return_value=result)
    ) as mock:
        response = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPTION})

    mock.assert_awaited_once_with(TRANSCRIPTION)
    assert response.status_code == 200
    body = response.json()
    assert body["estimation"] == "## Estimación: Test"
    assert body["model"] == "claude-haiku-4-5"
    assert body["provider"] == "anthropic"
    assert body["usage"] == {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500}
    assert "created_at" in body


def test_estimate_requires_transcription(client):
    assert client.post("/api/v1/estimate", json={}).status_code == 422


def test_estimate_rejects_empty_transcription(client):
    assert client.post("/api/v1/estimate", json={"transcription": ""}).status_code == 422


def test_estimate_rejects_short_transcription(client):
    with patch("app.routers.estimations.generate_estimation") as mock:
        response = client.post("/api/v1/estimate", json={"transcription": "Quiero una web"})

    assert response.status_code == 422
    mock.assert_not_called()  # no se gasta una llamada al LLM


def test_estimate_ignores_surrounding_whitespace_in_length(client):
    padded = " " * 40 + "Quiero una web" + " " * 40
    assert client.post("/api/v1/estimate", json={"transcription": padded}).status_code == 422


def test_estimate_strips_transcription(client):
    result = EstimationResult("## Estimación: Test", "claude-haiku-4-5", 1, 1)

    with patch(
        "app.routers.estimations.generate_estimation", AsyncMock(return_value=result)
    ) as mock:
        client.post("/api/v1/estimate", json={"transcription": f"  {TRANSCRIPTION}\n"})

    mock.assert_awaited_once_with(TRANSCRIPTION)


def test_estimate_returns_502_when_llm_fails(client):
    error = LLMError("conexión rechazada")

    with patch("app.routers.estimations.generate_estimation", AsyncMock(side_effect=error)):
        response = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPTION})

    assert response.status_code == 502
    assert "Error al llamar al LLM" in response.json()["detail"]


def test_openapi_docs(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/v1/estimate", "/health"} <= paths.keys()


def test_estimate_includes_cost_and_finish_reason(client):
    result = EstimationResult(
        "## Estimación", "claude-haiku-4-5-20251001", 1_746, 706, "length", 0.005276
    )

    with patch("app.routers.estimations.generate_estimation", AsyncMock(return_value=result)):
        body = client.post("/api/v1/estimate", json={"transcription": TRANSCRIPTION}).json()

    assert body["cost_usd"] == 0.005276
    assert body["finish_reason"] == "length"


def test_startup_logs_application_started():
    from fastapi.testclient import TestClient

    from app.main import app

    # setup_logging reconfigura structlog y anularía la captura: se simula
    with patch("app.main.setup_logging") as setup, capture_logs() as logs, TestClient(app):
        pass

    setup.assert_called_once_with("DEBUG", "development")
    events = [e["event"] for e in logs if e["log_level"] != "debug"]
    assert events == ["application_started", "application_shutdown"]
    assert next(e for e in logs if e["event"] == "application_started")["model"] == (
        "claude-haiku-4-5"
    )
    assert any(e["event"] == "tracing_disabled" for e in logs)  # sin PHOENIX_COLLECTOR_ENDPOINT


def test_request_id_is_generated_and_returned(client):
    with capture_logs() as logs:
        response = client.post("/api/v1/estimate", json={})

    assert len(response.headers["X-Request-ID"]) == 12
    completed = next(e for e in logs if e["event"] == "request_completed")
    assert completed["status_code"] == 422
    assert completed["log_level"] == "info"


def test_request_id_from_client_is_reused(client):
    response = client.get("/health", headers={"X-Request-ID": "mi-peticion-1"})

    assert response.headers["X-Request-ID"] == "mi-peticion-1"


def test_health_requests_are_logged_at_debug(client):
    with capture_logs() as logs:
        client.get("/health")

    assert next(e for e in logs if e["event"] == "request_completed")["log_level"] == "debug"


def test_request_id_is_added_to_llm_logs(client):
    result = EstimationResult("## Estimación", "claude-haiku-4-5", 1, 1)

    async def fake_generate(transcription):
        structlog.get_logger().info("llm_call_completed")
        return result

    with (
        patch("app.routers.estimations.generate_estimation", fake_generate),
        capture_logs(processors=[structlog.contextvars.merge_contextvars]) as logs,
    ):
        response = client.post(
            "/api/v1/estimate",
            json={"transcription": TRANSCRIPTION},
            headers={"X-Request-ID": "abc123"},
        )

    llm_log = next(e for e in logs if e["event"] == "llm_call_completed")
    assert llm_log["request_id"] == "abc123"
    assert llm_log["path"] == "/api/v1/estimate"
    assert response.status_code == 200
