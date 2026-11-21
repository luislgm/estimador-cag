"""Página de ingesta de Streamlit, con la API simulada."""

from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).resolve().parent.parent / "pages" / "1_Ingesta.py")
API = "http://api-test:8000/api/v1/ingestion"

SOURCES = [
    {
        "name": "budgets_2024",
        "description": "Presupuestos",
        "owner": "Ventas",
        "format": "budget_json",
        "decision": "include",
        "decision_reason": None,
        "quality": 3.25,
        "contains_pii": True,
        "files": 7,
    },
    {
        "name": "crm_contacts_export",
        "description": "CRM",
        "owner": "Marketing",
        "format": "csv",
        "decision": "exclude",
        "decision_reason": "Sin base legal (RGPD)",
        "quality": 4.0,
        "contains_pii": True,
        "files": 0,
    },
]
REPORT = {
    "source": "budgets_2024",
    "files": 7,
    "documents": 3,
    "repaired": 1,
    "quarantined": {"BUDGET-2024-0003@v1": ["client_name: not_nullable"]},
    "discarded": {"BUDGET-2024-0004@v1": ["total_amount: in_range"]},
    "deduplicated": ["BUDGET-2024-0005@v1"],
    "pii_replacements": {"PERSON": 1},
}


class FakeApi:
    def __init__(self):
        self.statuses = ["running", "completed"]
        self.run_status = 202
        self.job_error = None
        self.forgotten = True
        self.down = False
        self.posts = []

    def job(self):
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return {
            "job_id": "job-12345678",
            "source": "budgets_2024",
            "status": status,
            "report": REPORT if status == "completed" else None,
            "error": self.job_error,
        }

    def get(self, url, **kwargs):
        if self.down:
            raise httpx.ConnectError("API caída")
        if url.endswith("/sources"):
            return httpx.Response(200, json=SOURCES)
        if "/documents" in url:
            docs = [
                {"id": "budgets_2024:B1", "kind": "budget", "text": "Presupuesto…", "metadata": {}}
            ]
            return httpx.Response(200, json=docs)
        return httpx.Response(200, json=self.job())

    def post(self, url, json=None, **kwargs):
        self.posts.append(json)
        if self.run_status != 202:
            return httpx.Response(self.run_status, json={"detail": "está en 'exclude'"})
        return httpx.Response(202, json={"job_id": "job-12345678", "status": "pending"})

    def request(self, method, url, json=None, **kwargs):
        if json["value"] == "rompe":
            return httpx.Response(503, text="sin base de datos")
        return httpx.Response(200, json={"forgotten": self.forgotten})


@pytest.fixture(autouse=True)
def api(monkeypatch):
    monkeypatch.setenv("ESTIMATOR_API_URL", "http://api-test:8000")
    monkeypatch.setenv("INGESTION_POLL_SECONDS", "0")  # sondeo sin espera
    fake = FakeApi()
    with (
        patch("httpx.get", fake.get),
        patch("httpx.post", fake.post),
        patch("httpx.request", fake.request),
    ):
        yield fake


def run_page() -> AppTest:
    return AppTest.from_file(PAGE, default_timeout=30).run()


def ingest(at: AppTest, source="budgets_2024") -> AppTest:
    at.selectbox[0].set_value(source)
    return next(b for b in at.button if b.label == "Ingerir").click().run()


def test_shows_the_catalog(api):
    at = run_page()

    assert not at.exception
    table = at.dataframe[0].value
    assert list(table["Fuente"]) == ["budgets_2024", "crm_contacts_export"]
    assert list(table["Decisión"]) == ["✅ include", "⛔ exclude"]
    assert table["Motivo"][1] == "Sin base legal (RGPD)"


def test_ingesting_waits_for_the_job_and_shows_the_report(api):
    at = ingest(run_page())

    assert api.posts == [{"source": "budgets_2024"}]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {
        "Archivos": "7",
        "Documentos": "3",
        "Reparados": "1",
        "Cuarentena": "1",
        "Descartados": "1",
    }
    labels = [e.label for e in at.expander]
    assert "🟡 En cuarentena (recuperables con revisión): 1" in labels
    assert "budgets_2024:B1 (budget)" in labels
    captions = " ".join(c.value for c in at.caption)
    assert "Persona: 1" in captions and "BUDGET-2024-0005@v1" in captions


def test_rejected_source_shows_the_reason(api):
    api.run_status = 400

    at = ingest(run_page(), "crm_contacts_export")

    assert "La API ha rechazado la ingesta (400): está en 'exclude'" in at.error[0].value


def test_failed_job_shows_the_error(api):
    api.statuses = ["failed"]
    api.job_error = "RuntimeError: disco lleno"

    assert "disco lleno" in ingest(run_page()).error[0].value


def test_unfinished_job_after_the_polling_limit(api):
    api.statuses = ["running"]  # nunca termina: se agota POLL_LIMIT

    at = ingest(run_page())

    assert "sigue en estado 'running'" in at.info[0].value


def test_forgetting_a_pseudonym(api):
    at = run_page()
    at.text_input[0].input("María García López")
    at = at.button[-1].click().run()
    assert "Olvidado" in at.success[0].value

    api.forgotten = False
    at.text_input[0].input("Otra Persona")
    at = at.button[-1].click().run()
    assert "No había ningún seudónimo" in at.info[0].value

    at.text_input[0].input("rompe")
    at = at.button[-1].click().run()
    assert "No se ha podido (503): sin base de datos" in at.error[0].value


def test_api_down(api):
    api.down = True

    at = run_page()

    assert "No se puede conectar con la API" in at.error[0].value
