"""Ingesta por API: persistencia, endpoints, trabajos en segundo plano y migraciones."""

from pathlib import Path
from unittest.mock import patch

import pytest
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError

from alembic import command
from app import persistence
from app.config import get_settings
from app.ingestion.pii import hash_value
from app.main import app
from app.persistence import Base, JobRepository, PseudonymMapping, SqlMappingStore

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def database(monkeypatch, tmp_path):
    url = f"sqlite:///{tmp_path / 'ingest.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("DATA_DIR", str(ROOT / "data"))
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.setenv("PII_SALT", "sal-api")
    get_settings.cache_clear()
    persistence.session_factory.cache_clear()
    Base.metadata.create_all(create_engine(url))
    yield persistence.session_factory()
    persistence.session_factory.cache_clear()


@pytest.fixture
def api():
    return TestClient(app)


# --- Persistencia ---


def test_mapping_store_keeps_and_forgets(database):
    store = SqlMappingStore(database)

    assert store.get_or_create("PERSON", "h1", "Ana Ruiz") == "Ana Ruiz"
    assert store.get_or_create("PERSON", "h1", "Otro Nombre") == "Ana Ruiz"
    assert store.forget("PERSON", "h1") and not store.forget("PERSON", "h1")


def test_mapping_store_uses_the_existing_row_on_a_race(database):
    store = SqlMappingStore(database)
    store.get_or_create("PERSON", "h1", "Ana Ruiz")
    real_find = store._find
    calls = []

    def find_misses_once(session, entity_type, original_hash):
        calls.append(1)
        return None if len(calls) == 1 else real_find(session, entity_type, original_hash)

    with patch.object(store, "_find", find_misses_once):
        assert store.get_or_create("PERSON", "h1", "Otro Nombre") == "Ana Ruiz"
    assert len(calls) == 2


def test_unique_constraint_on_mappings(database):
    with database() as session:
        session.add(PseudonymMapping(entity_type="PERSON", original_hash="h", pseudonym="a"))
        session.add(PseudonymMapping(entity_type="PERSON", original_hash="h", pseudonym="b"))
        with pytest.raises(IntegrityError):
            session.commit()


def test_job_lifecycle(database):
    jobs = JobRepository(database)
    job_id = jobs.create("budgets_2024")
    assert jobs.get(job_id).status == "pending"

    jobs.mark_running(job_id)
    jobs.complete(job_id, {"documents": 3})

    job = jobs.get(job_id)
    assert (job.status, job.report) == ("completed", {"documents": 3})
    assert job.started_at is not None and job.finished_at is not None
    assert jobs.get("no-existe") is None


def test_without_database_url_there_is_no_session_factory():
    persistence.session_factory.cache_clear()
    assert persistence.session_factory() is None


# --- Endpoints ---


def test_sources_lists_the_catalog(api, database):
    sources = {s["name"]: s for s in api.get("/api/v1/ingestion/sources").json()}

    assert sources["budgets_2024"]["files"] == 7 and sources["budgets_2024"]["contains_pii"]
    assert sources["crm_contacts_export"]["decision"] == "exclude"
    assert "RGPD" in sources["crm_contacts_export"]["decision_reason"]


def test_run_ingests_in_the_background_and_reports(api, database, tmp_path):
    response = api.post("/api/v1/ingestion/runs", json={"source": "budgets_2024"})

    assert response.status_code == 202
    job = api.get(f"/api/v1/ingestion/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "completed"  # TestClient ejecuta la tarea antes de volver
    assert job["report"]["documents"] == 3 and job["report"]["repaired"] == 1
    assert (tmp_path / "processed" / "budgets_2024.jsonl").exists()
    with database() as session:  # los seudónimos se guardan en la base de datos
        assert session.query(PseudonymMapping).count() >= 3


def test_documents_of_a_job_are_pseudonymized(api, database):
    job_id = api.post("/api/v1/ingestion/runs", json={"source": "meeting_transcripts"}).json()[
        "job_id"
    ]

    documents = api.get(f"/api/v1/ingestion/jobs/{job_id}/documents?limit=1").json()

    assert len(documents) == 1 and documents[0]["kind"] == "transcript"
    assert "María García" not in documents[0]["text"]
    assert "participants" in documents[0]["metadata"]


def test_documents_need_a_completed_job(api, database):
    jobs = JobRepository(database)
    pending = jobs.create("budgets_2024")

    assert api.get("/api/v1/ingestion/jobs/no-existe/documents").status_code == 404
    assert api.get(f"/api/v1/ingestion/jobs/{pending}/documents").status_code == 409


def test_excluded_or_unknown_sources_are_rejected_at_once(api, database):
    excluded = api.post("/api/v1/ingestion/runs", json={"source": "crm_contacts_export"})
    unknown = api.post("/api/v1/ingestion/runs", json={"source": "no_existe"})

    assert excluded.status_code == 400 and "exclude" in excluded.json()["detail"]
    assert unknown.status_code == 400


def test_a_failing_run_is_recorded(api, database):
    with patch("app.routers.ingestion.ingest_source", side_effect=RuntimeError("disco lleno")):
        response = api.post("/api/v1/ingestion/runs", json={"source": "rate_card_2024"})

    job = api.get(f"/api/v1/ingestion/jobs/{response.json()['job_id']}").json()
    assert (job["status"], job["error"]) == ("failed", "RuntimeError: disco lleno")


def test_unknown_job_returns_404(api, database):
    assert api.get("/api/v1/ingestion/jobs/no-existe").status_code == 404


def test_without_database_the_api_returns_503(api):
    persistence.session_factory.cache_clear()

    assert api.post("/api/v1/ingestion/runs", json={"source": "x"}).status_code == 503
    assert api.get("/api/v1/ingestion/jobs/x").status_code == 503


def test_forgetting_a_pseudonym(api, database):
    store = SqlMappingStore(database)
    store.get_or_create("PERSON", hash_value(b"sal-api", "PERSON", "María García"), "Ana Ruiz")
    body = {"entity_type": "PERSON", "value": "maría garcía"}

    first = api.request("DELETE", "/api/v1/ingestion/pseudonyms", json=body)
    again = api.request("DELETE", "/api/v1/ingestion/pseudonyms", json=body)

    assert (first.json(), again.json()) == ({"forgotten": True}, {"forgotten": False})


def test_forgetting_needs_a_salt_and_a_known_entity_type(api, database, monkeypatch):
    wrong_type = {"entity_type": "LOCATION", "value": "Valencia"}
    assert api.request("DELETE", "/api/v1/ingestion/pseudonyms", json=wrong_type).status_code == 422

    monkeypatch.delenv("PII_SALT")
    get_settings.cache_clear()
    body = {"entity_type": "PERSON", "value": "María García"}
    assert api.request("DELETE", "/api/v1/ingestion/pseudonyms", json=body).status_code == 503


# --- Migraciones ---


def test_migrations_create_exactly_the_model_schema(tmp_path):
    url = f"sqlite:///{tmp_path / 'migrated.db'}"
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url)

    command.upgrade(config, "head")

    with create_engine(url).connect() as connection:
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []
    command.downgrade(config, "base")
