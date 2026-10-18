from unittest.mock import AsyncMock, patch

import anthropic
import httpx

from app.services.llm_service import EstimationResult


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
        response = client.post("/api/v1/estimate", json={"transcription": "Quiero una web"})

    mock.assert_awaited_once_with("Quiero una web")
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


def test_estimate_returns_502_when_llm_fails(client):
    error = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )

    with patch("app.routers.estimations.generate_estimation", AsyncMock(side_effect=error)):
        response = client.post("/api/v1/estimate", json={"transcription": "Quiero una web"})

    assert response.status_code == 502
    assert "Error al llamar al LLM" in response.json()["detail"]


def test_openapi_docs(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/v1/estimate", "/health"} <= paths.keys()
