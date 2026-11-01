import json
from unittest.mock import AsyncMock, patch

import pytest
import structlog
from structlog.testing import capture_logs

from app.schemas.estimation import EstimationRequest
from app.services.llm_service import EstimationResult, LLMError

DESCRIPTION = (
    "El cliente quiere una web para reservar mesas en su restaurante, "
    "con pago online y un panel de gestión para el personal."
)
BODY = {
    "description": DESCRIPTION,
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}
RESULT = EstimationResult(
    "## Estimación: Test", "claude-haiku-4-5", 1200, 300, "stop", 0.0027, False, "anthropic"
)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """Convierte el cuerpo de una respuesta SSE en una lista de (evento, datos)."""
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


class FakeStream:
    """Sustituye a EstimationStream en el endpoint de streaming."""

    error: Exception | None = None

    def __init__(self, request):
        self.request = request
        self.result = None

    def __iter__(self):
        yield "## Estimación"
        if FakeStream.error:
            raise FakeStream.error
        yield ": Test"
        self.result = RESULT


@pytest.fixture
def fake_stream():
    FakeStream.error = None
    with patch("app.routers.estimations.EstimationStream", FakeStream):
        yield FakeStream


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "environment": "development",
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
    }


# --- POST /api/v1/estimate ---


def test_estimate_returns_contract_response(client):
    with patch(
        "app.routers.estimations.generate_estimation", AsyncMock(return_value=RESULT)
    ) as mock:
        response = client.post("/api/v1/estimate", json=BODY)

    assert response.status_code == 200
    assert mock.await_args.args == (EstimationRequest(**BODY),)
    body = response.json()
    assert body["text"] == "## Estimación: Test"
    assert body["prompt_version"] == "v1"
    assert body["model"] == "claude-haiku-4-5"
    assert body["provider"] == "anthropic"
    assert body["usage"] == {"input_tokens": 1200, "output_tokens": 300, "total_tokens": 1500}
    assert body["cost_usd"] == 0.0027
    assert body["finish_reason"] == "stop"
    assert body["cache_hit"] is False
    assert "created_at" in body


@pytest.mark.parametrize(
    "changes",
    [
        {"description": "corta"},  # menos de 20 caracteres
        {"description": "x" * 2001},  # más de 2000
        {"description": " " * 10 + "Quiero web" + " " * 10},  # los espacios no cuentan
        {"project_type": "videojuego"},  # valor que no está en el enum
        {"detail_level": "extremo"},
        {"output_format": "pdf"},
    ],
)
def test_estimate_rejects_invalid_requests(client, changes):
    with patch("app.routers.estimations.generate_estimation") as mock:
        response = client.post("/api/v1/estimate", json=BODY | changes)

    assert response.status_code == 422
    mock.assert_not_called()  # no se gasta una llamada al LLM


@pytest.mark.parametrize("field", ["description", "project_type", "detail_level", "output_format"])
def test_estimate_requires_all_fields(client, field):
    body = {k: v for k, v in BODY.items() if k != field}
    assert client.post("/api/v1/estimate", json=body).status_code == 422


def test_estimate_strips_description(client):
    with patch(
        "app.routers.estimations.generate_estimation", AsyncMock(return_value=RESULT)
    ) as mock:
        client.post("/api/v1/estimate", json=BODY | {"description": f"  {DESCRIPTION}\n"})

    assert mock.await_args.args[0].description == DESCRIPTION


def test_estimate_returns_502_when_llm_fails(client):
    error = LLMError("conexión rechazada")
    with patch("app.routers.estimations.generate_estimation", AsyncMock(side_effect=error)):
        response = client.post("/api/v1/estimate", json=BODY)

    assert response.status_code == 502
    assert "Error al llamar al LLM" in response.json()["detail"]


def test_estimate_reports_cache_hit(client):
    result = EstimationResult("## Estimación", "claude-haiku-4-5", 1_746, 706, "stop", 0.0, True)
    with patch("app.routers.estimations.generate_estimation", AsyncMock(return_value=result)):
        body = client.post("/api/v1/estimate", json=BODY).json()

    assert body["cache_hit"] is True
    assert body["cost_usd"] == 0.0


def test_estimate_logs_request_parameters(client):
    with (
        patch("app.routers.estimations.generate_estimation", AsyncMock(return_value=RESULT)),
        capture_logs() as logs,
    ):
        client.post("/api/v1/estimate", json=BODY)

    requested = next(e for e in logs if e["event"] == "estimation_requested")
    assert requested["mode"] == "blocking"
    assert requested["prompt_version"] == "v1"
    assert requested["project_type"] == "web_saas"
    assert requested["detail_level"] == "medium"
    assert requested["output_format"] == "phases_table"
    assert requested["description_chars"] == len(DESCRIPTION)


# --- POST /api/v1/estimate/stream ---


def test_stream_sends_tokens_then_done(client, fake_stream):
    response = client.post("/api/v1/estimate/stream", json=BODY)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert events[:2] == [("token", {"text": "## Estimación"}), ("token", {"text": ": Test"})]
    done_event, done = events[-1]
    assert done_event == "done"
    assert done["text"] == "## Estimación: Test"
    assert done["prompt_version"] == "v1"
    assert done["usage"]["total_tokens"] == 1500
    assert done["cost_usd"] == 0.0027


def test_stream_sends_error_event_when_llm_fails(client, fake_stream):
    fake_stream.error = LLMError("conexión rechazada")

    events = parse_sse(client.post("/api/v1/estimate/stream", json=BODY).text)

    assert [e for e, _ in events] == ["token", "error"]  # no llega "done"
    assert "conexión rechazada" in events[-1][1]["message"]


def test_stream_validates_like_blocking_endpoint(client, fake_stream):
    response = client.post("/api/v1/estimate/stream", json=BODY | {"description": "corta"})

    assert response.status_code == 422


def test_stream_keeps_line_breaks_in_tokens(client):
    class MultilineStream(FakeStream):
        def __iter__(self):
            yield "## Estimación\n\n| Fase |"
            self.result = RESULT

    with patch("app.routers.estimations.EstimationStream", MultilineStream):
        events = parse_sse(client.post("/api/v1/estimate/stream", json=BODY).text)

    assert events[0] == ("token", {"text": "## Estimación\n\n| Fase |"})


def test_openapi_docs(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/v1/estimate", "/api/v1/estimate/stream", "/health"} <= paths.keys()


# --- Arranque, logs y request_id ---


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
    async def fake_generate(request):
        structlog.get_logger().info("llm_call_completed")
        return RESULT

    with (
        patch("app.routers.estimations.generate_estimation", fake_generate),
        capture_logs(processors=[structlog.contextvars.merge_contextvars]) as logs,
    ):
        response = client.post("/api/v1/estimate", json=BODY, headers={"X-Request-ID": "abc123"})

    llm_log = next(e for e in logs if e["event"] == "llm_call_completed")
    assert llm_log["request_id"] == "abc123"
    assert llm_log["path"] == "/api/v1/estimate"
    assert response.status_code == 200


def test_estimate_sends_system_and_user_as_separate_messages(client):
    """De extremo a extremo: POST /estimate -> loader -> wrapper -> LiteLLM, sin concatenar."""
    import litellm

    from app.prompts import render_estimation_prompt

    real_acompletion = litellm.acompletion
    calls = []

    async def fake_acompletion(**params):
        calls.append(params)
        return await real_acompletion(**params, mock_response="## Estimación: Test")

    with patch.object(litellm, "acompletion", fake_acompletion):
        response = client.post("/api/v1/estimate", json=BODY)

    system, user = render_estimation_prompt(EstimationRequest(**BODY), version="v1")
    assert calls[0]["messages"] == [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    assert response.json()["text"] == "## Estimación: Test"
    assert response.json()["prompt_version"] == "v1"
