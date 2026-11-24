"""Pipeline de embeddings: esquemas, chunker, embedder, endpoint y compare.py (sin red)."""

import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import get_settings
from app.embedding_pipeline import embedder as embedder_module
from app.embedding_pipeline.chunker import JSONStructuralChunker, count_tokens
from app.embedding_pipeline.embedder import (
    EmbeddingsNotConfigured,
    OpenAIEmbedder,
    cosine_similarity,
    estimate_cost_usd,
)
from app.embedding_pipeline.router import get_embedder
from app.embedding_pipeline.schemas import Budget, IngestRequest
from app.main import app

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = json.loads((ROOT / "data" / "budgets_sample.json").read_text("utf-8"))


def rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    return openai.RateLimitError(
        "rate limit", response=httpx.Response(429, request=request), body=None
    )


class FakeEmbeddings:
    """Imita client.embeddings.create: un vector por texto (devuelto en orden inverso, para
    comprobar que el embedder reordena por índice) y errores programados."""

    def __init__(self, errors=()):
        self.errors = list(errors)
        self.calls = []

    def create(self, model, input):
        self.calls.append((model, list(input)))
        if self.errors:
            raise self.errors.pop(0)
        data = [
            SimpleNamespace(index=i, embedding=[float(len(text)), float(i), 1.0])
            for i, text in enumerate(input)
        ]
        return SimpleNamespace(data=list(reversed(data)), usage=SimpleNamespace(total_tokens=7))


def fake_embedder(errors=(), batch_size=100) -> tuple[OpenAIEmbedder, FakeEmbeddings]:
    fake = FakeEmbeddings(errors)
    sleeps = []
    embedder = OpenAIEmbedder(
        client=SimpleNamespace(embeddings=fake), batch_size=batch_size, sleep=sleeps.append
    )
    embedder.sleeps = sleeps
    return embedder, fake


# --- Esquemas ---


def test_sample_budgets_are_valid():
    request = IngestRequest(budgets=SAMPLE)

    assert len(request.budgets) == 15
    assert sum(len(b.components) for b in request.budgets) == 52


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda b: b["client_metadata"].update(sector="gaming"), "sector"),
        (lambda b: b.update(total_estimated_hours=1), "los componentes suman"),
        (lambda b: b["components"][0].update(dependencies=["NOPE-999"]), "inexistentes"),
        (lambda b: b["components"].append(copy.deepcopy(b["components"][0])), "repetidos"),
        (lambda b: b.update(budget_id="BUDGET-1"), "budget_id"),
    ],
)
def test_invalid_budgets_are_rejected(change, message):
    budget = copy.deepcopy(SAMPLE[0])
    change(budget)

    with pytest.raises(ValidationError, match=message):
        Budget.model_validate(budget)


def test_duplicated_budgets_in_a_request_are_rejected():
    with pytest.raises(ValidationError, match="budget_id repetidos"):
        IngestRequest(budgets=[SAMPLE[0], SAMPLE[0]])


# --- Chunker ---


def test_one_chunk_per_component_with_parent_context():
    budget = Budget.model_validate(SAMPLE[0])

    chunks = JSONStructuralChunker().chunk([budget])

    assert len(chunks) == len(budget.components)
    first = chunks[0]
    assert first.chunk_id == "BUD-2024-001::AUTH-001"
    assert first.text.startswith(
        "[Project: Mobile banking API with OAuth 2.0 authentication and PSD2 compliance]\n"
        "[Client sector: finance | Year: 2024 | Main tech: ruby_on_rails]\n\n"
        "Component: OAuth 2.0 authentication backend\n"
    )
    assert "Tech stack: ruby_on_rails, postgresql, redis" in first.text
    assert first.text.endswith("Complexity: high\nEstimated hours: 120")
    assert first.metadata == {
        "budget_id": "BUD-2024-001",
        "component_id": "AUTH-001",
        "client_sector": "finance",
        "main_technology": "ruby_on_rails",
        "year": 2024,
        "complexity": "high",
        "estimated_hours": 120,
    }
    assert first.token_count == count_tokens(first.text) > 0


def test_token_counting_uses_the_embedding_model_encoding():
    assert count_tokens("hello world") == 2


# --- Embedder ---


def test_embed_many_sends_batches_and_keeps_the_order():
    chunks = JSONStructuralChunker().chunk([Budget.model_validate(b) for b in SAMPLE[:3]])
    embedder, fake = fake_embedder(batch_size=4)

    embedded = embedder.embed_many(chunks)

    assert [len(texts) for _, texts in fake.calls] == [4, 4, len(chunks) - 8][: len(fake.calls)]
    assert all(model == "text-embedding-3-small" for model, _ in fake.calls)
    assert [e.chunk_id for e in embedded] == [c.chunk_id for c in chunks]
    # El vector de cada chunk es el suyo, aunque la API los devuelva desordenados
    assert all(e.embedding[0] == len(e.text) for e in embedded)


def test_rate_limits_are_retried_with_exponential_waits():
    embedder, fake = fake_embedder(errors=[rate_limit_error(), rate_limit_error()])

    assert embedder.embed_one("hola") == [4.0, 0.0, 1.0]
    assert embedder.sleeps == [1, 2]
    assert len(fake.calls) == 3


def test_rate_limit_gives_up_after_three_retries():
    embedder, _ = fake_embedder(errors=[rate_limit_error() for _ in range(4)])

    with pytest.raises(openai.RateLimitError):
        embedder.embed_one("hola")
    assert embedder.sleeps == [1, 2, 4]


def test_other_errors_are_not_retried():
    request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
    error = openai.AuthenticationError(
        "bad key", response=httpx.Response(401, request=request), body=None
    )
    embedder, fake = fake_embedder(errors=[error])

    with pytest.raises(openai.AuthenticationError):
        embedder.embed_one("hola")
    assert len(fake.calls) == 1


def test_embedder_needs_an_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    with pytest.raises(EmbeddingsNotConfigured):
        OpenAIEmbedder()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_settings.cache_clear()
    assert isinstance(OpenAIEmbedder().client, openai.OpenAI)


def test_cost_and_cosine_similarity():
    assert estimate_cost_usd(1_000_000) == embedder_module.PRICE_USD_PER_MTOK == 0.02
    assert cosine_similarity([1, 0], [1, 0]) == 1
    assert cosine_similarity([1, 0], [0, 1]) == 0
    assert math.isclose(cosine_similarity([1, 1], [1, 0]), 1 / math.sqrt(2))
    with pytest.raises(ValueError, match="dimensión"):
        cosine_similarity([1], [1, 2])
    with pytest.raises(ValueError, match="nulo"):
        cosine_similarity([0, 0], [1, 1])


# --- Endpoint ---


@pytest.fixture
def api():
    embedder, fake = fake_embedder()
    app.dependency_overrides[get_embedder] = lambda: embedder
    yield TestClient(app), fake
    app.dependency_overrides.clear()


def test_ingest_returns_embedded_chunks_and_stats(api):
    client, fake = api

    response = client.post("/embeddings/ingest", json={"budgets": SAMPLE[:2]})

    body = response.json()
    assert response.status_code == 200
    n = sum(len(b["components"]) for b in SAMPLE[:2])
    assert len(body["chunks"]) == n and len(fake.calls) == 1  # un solo lote
    tokens = sum(c["token_count"] for c in body["chunks"])
    assert body["stats"] == {
        "total_budgets": 2,
        "total_chunks": n,
        "total_tokens": tokens,
        "estimated_cost_usd": estimate_cost_usd(tokens),
    }
    assert body["chunks"][0]["embedding"] == [float(len(body["chunks"][0]["text"])), 0.0, 1.0]


def test_ingest_validation_errors_return_422(api):
    client, _ = api
    budget = copy.deepcopy(SAMPLE[0])
    budget["client_metadata"]["sector"] = "gaming"

    assert client.post("/embeddings/ingest", json={"budgets": [budget]}).status_code == 422
    assert client.post("/embeddings/ingest", json={"budgets": []}).status_code == 422


def test_embedding_api_errors_return_a_generic_500(api):
    client, fake = api
    fake.errors = [openai.APIConnectionError(request=httpx.Request("POST", "https://x"))]

    response = client.post("/embeddings/ingest", json={"budgets": SAMPLE[:1]})

    assert response.status_code == 500
    assert response.json() == {"detail": "No se han podido generar los embeddings"}


def test_ingest_without_api_key_returns_503(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()

    response = TestClient(app).post("/embeddings/ingest", json={"budgets": SAMPLE[:1]})

    assert response.status_code == 503 and "OPENAI_API_KEY" in response.json()["detail"]


# --- compare.py ---


def load_compare():
    spec = importlib.util.spec_from_file_location("compare", ROOT / "scripts" / "compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_prints_the_cosine_similarity(capsys):
    embedder, _ = fake_embedder()

    assert load_compare().main(["--text-a", "abcd", "--text-b", "ab"], embedder=embedder) == 0

    out = capsys.readouterr().out
    expected = cosine_similarity([4.0, 0.0, 1.0], [2.0, 0.0, 1.0])
    assert out == f"Text A: abcd\nText B: ab\nCosine similarity: {expected:.4f}\n"
