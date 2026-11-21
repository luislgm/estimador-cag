"""Ingesta: catálogo, carga, parsers, limpieza, orquestador y decisión de arquitectura."""

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from openpyxl import Workbook
from pydantic import ValidationError

from app.config import get_settings
from app.ingestion import architecture, catalog, cleaning, orchestrator
from app.ingestion.catalog import DataCatalog, UnknownSource, load_catalog
from app.ingestion.documents import LoadedBlob
from app.ingestion.loaders import FileSystemLoader
from app.ingestion.parsers import (
    ParseContext,
    Parser,
    ParserRegistry,
    RateCardXlsxParser,
    TranscriptTxtParser,
    UnsupportedFormat,
    default_registry,
)
from app.ingestion.pii import InMemoryMappingStore, Pseudonymizer

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def repo_data(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(DATA))
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "processed"))
    get_settings.cache_clear()
    return tmp_path / "processed"


@pytest.fixture
def seed_catalog() -> DataCatalog:
    return load_catalog(DATA / "catalog.yaml")


def source_dict(**changes) -> dict:
    base = {
        "name": "fuente",
        "description": "d",
        "owner": "o",
        "path": "*.txt",
        "format": "transcript_txt",
        "refresh": "weekly",
        "decision": "include",
        "quality": {"completeness": 3, "consistency": 3, "actuality": 3, "reliability": 3},
        "sensitivity": {"contains_pii": False},
    }
    return base | changes


# --- Catálogo ---


def test_seed_catalog_has_every_decision(seed_catalog):
    decisions = {s.name: s.decision for s in seed_catalog.sources}

    assert decisions["budgets_2024"] == "include"
    assert decisions["transcripts_legacy"] == "review"
    assert decisions["crm_contacts_export"] == "exclude"
    assert seed_catalog.get("rate_card_2024").quality.average == 4.5
    with pytest.raises(UnknownSource):
        seed_catalog.get("no_existe")


def test_excluding_a_source_needs_a_reason():
    with pytest.raises(ValidationError, match="necesita un decision_reason"):
        catalog.CatalogSource.model_validate(source_dict(decision="exclude"))


def test_duplicated_sources_are_rejected():
    with pytest.raises(ValidationError, match="fuentes repetidas"):
        DataCatalog(version="1", sources=[source_dict(), source_dict()])


def test_inventory_lists_decisions_and_files(repo_data, capsys):
    assert catalog.main() == 0
    text = capsys.readouterr().out
    assert "budgets_2024           include      3.25 sí          7" in text
    assert "crm_contacts_export    exclude" in text and "RGPD" in text


# --- Carga y parsers ---


def context(source_name="budgets_2024") -> ParseContext:
    source = load_catalog(DATA / "catalog.yaml").get(source_name)
    return ParseContext(source, "v-test", datetime(2026, 11, 21, tzinfo=UTC))


def test_loader_reads_the_source_files_in_order(seed_catalog):
    blobs = FileSystemLoader(DATA).load(seed_catalog.get("budgets_2024"))

    assert [b.path.name for b in blobs][:2] == ["BUDGET-2024-0001.json", "BUDGET-2024-0002.json"]
    assert len(blobs) == 7 and all(b.format == "budget_json" for b in blobs)


def test_budget_parser_normalizes_and_renders_text():
    blob = LoadedBlob(
        path=DATA / "seed/budgets/BUDGET-2024-0002.json",
        content=(DATA / "seed/budgets/BUDGET-2024-0002.json").read_bytes(),
        format="budget_json",
    )

    [doc] = default_registry().for_format("budget_json").parse(blob, context())

    assert doc.id == "budgets_2024:BUDGET-2024-0002@v1"
    assert (doc.kind, doc.source_version) == ("budget", "v-test")
    assert doc.metadata["client_name"] == "Beta Norte Seguros"
    assert doc.metadata["total_amount"] == 12500.0 and doc.metadata["currency"] == "EUR"
    assert "Total: 250 h, 12500.0 EUR." in doc.text
    assert len(doc.content_hash) == 64


def test_transcript_parser_reads_the_header_into_metadata():
    blob = LoadedBlob(
        path=Path("reunion.txt"),
        content="Fecha: 2025-01-15\nCliente: Acme\nHola: qué tal\nTexto.".encode(),
        format="transcript_txt",
    )

    [doc] = TranscriptTxtParser().parse(blob, context("meeting_transcripts"))

    assert doc.metadata == {"file": "reunion.txt", "date": "2025-01-15", "client": "Acme"}
    assert doc.text == "Hola: qué tal\nTexto."


def test_rate_card_parser_makes_one_document_per_role():
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Rol", "tarifa_hora", "moneda"])
    sheet.append(["QA", 40, None])
    sheet.append([None, None, None])  # fila vacía: se ignora
    buffer = io.BytesIO()
    workbook.save(buffer)
    blob = LoadedBlob(path=Path("t.xlsx"), content=buffer.getvalue(), format="rate_card_xlsx")

    [doc] = RateCardXlsxParser().parse(blob, context("rate_card_2024"))

    assert doc.text == "Tarifa de QA: 40 EUR por hora."
    assert doc.metadata == {"role": "QA", "hourly_rate": 40, "currency": "EUR"}


def test_registry_rejects_unknown_formats():
    registry = default_registry()

    assert registry.formats == {"budget_json", "transcript_txt", "rate_card_xlsx"}
    assert all(isinstance(p, Parser) for p in registry._by_format.values())
    with pytest.raises(UnsupportedFormat, match="csv"):
        registry.for_format("csv")


# --- Limpieza ---


def raw_budget(**changes) -> dict:
    base = json.loads((DATA / "seed/budgets/BUDGET-2024-0001.json").read_text("utf-8"))
    return base | changes


def test_repairs_are_applied_and_recorded():
    record = cleaning.normalize_budget(
        raw_budget(currency="eur", total_amount="12.500,00", date="02/04/2024", client_name=" X ")
    )

    assert (record["currency"], record["total_amount"], record["date"]) == (
        "EUR",
        12500.0,
        "2024-04-02",
    )
    assert record["client_name"] == "X"
    assert len(record["repairs"]) == 4


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), (1500, 1500.0), ("1500.5", 1500.5), ("15 000 €", 15000.0), ("mil", None)],
)
def test_amount_parsing(value, expected):
    assert cleaning._amount(value)[0] == expected


def test_unreadable_dates_become_none():
    assert cleaning._date("15 de marzo")[0] is None
    assert cleaning._date(None) == (None, False)


def test_seed_budgets_go_to_the_right_buckets(seed_catalog):
    records = [
        cleaning.normalize_budget(json.loads(b.content))
        for b in FileSystemLoader(DATA).load(seed_catalog.get("budgets_2024"))
    ]

    result = cleaning.validate_budgets(records)

    assert sorted((r["budget_id"], r["version"]) for r in result.valid) == [
        ("BUDGET-2024-0001", 1),
        ("BUDGET-2024-0002", 1),
        ("BUDGET-2024-0005", 2),
    ]
    assert result.repaired == 1
    assert result.quarantined == {"BUDGET-2024-0003@v1": ["client_name: not_nullable"]}
    assert result.discarded == {
        "BUDGET-2024-0004@v1": ["total_amount: in_range"],
        "BUDG-24-6@v1": ["budget_id: str_matches"],
    }
    assert result.deduplicated == ["BUDGET-2024-0005@v1"]


def test_a_missing_and_a_wrong_value_discards_the_row():
    record = cleaning.normalize_budget(raw_budget(client_name=None, currency="XXX"))

    result = cleaning.validate_budgets([record])

    assert list(result.discarded.values()) == [["client_name: not_nullable", "currency: isin"]]


def test_older_version_after_newer_is_deduplicated_and_empty_input_is_fine():
    new = cleaning.normalize_budget(raw_budget(version=3))
    old = cleaning.normalize_budget(raw_budget(version=2))

    result = cleaning.validate_budgets([new, old])

    assert [r["version"] for r in result.valid] == [3]
    assert result.deduplicated == ["BUDGET-2024-0001@v2"]
    assert cleaning.validate_budgets([]).valid == []


# --- Orquestador ---


@pytest.fixture
def pseudonymizer() -> Pseudonymizer:
    return Pseudonymizer(b"sal-de-test", InMemoryMappingStore())


def run(seed_catalog, name, pseudonymizer=None, output_dir=None):
    return orchestrator.ingest_source(
        seed_catalog,
        name,
        loader=FileSystemLoader(DATA),
        registry=default_registry(),
        pseudonymizer=pseudonymizer,
        output_dir=output_dir,
    )


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("no_existe", "no está en el catálogo"),
        ("transcripts_legacy", "está en 'review'"),
        ("crm_contacts_export", "está en 'exclude'"),
        ("budgets_2024", "no hay PII_SALT"),
    ],
)
def test_sources_that_cannot_be_ingested_are_rejected(seed_catalog, name, message):
    with pytest.raises(orchestrator.IngestionRejected, match=message):
        run(seed_catalog, name)


def test_unsupported_formats_are_rejected(seed_catalog):
    with pytest.raises(orchestrator.IngestionRejected, match="no hay parser"):
        orchestrator.check_source(seed_catalog, "budgets_2024", ParserRegistry([]), True)


def test_budgets_are_cleaned_and_pseudonymized(seed_catalog, pseudonymizer, tmp_path):
    documents, report = run(seed_catalog, "budgets_2024", pseudonymizer, tmp_path)

    assert (report.files, report.parsed, report.documents) == (7, 7, 3)
    assert report.repaired == 1 and len(report.discarded) == 2
    assert report.pii_replacements == {"PERSON": 1}  # el nombre en las notas de 0001
    acme = next(d for d in documents if d.metadata["budget_id"] == "BUDGET-2024-0001")
    assert acme.metadata["client_name"] == "Acme Logística"  # la empresa no es dato personal
    assert "María García" not in acme.text and "María García" not in acme.metadata["notes"]
    assert acme.metadata["contact_name"] in acme.text  # mismo seudónimo en texto y metadata
    assert "@acme-logistica.es" not in acme.metadata["contact_email"]
    lines = Path(report.output).read_text("utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == [d.id for d in documents]


def test_same_person_has_the_same_pseudonym_across_sources(seed_catalog, pseudonymizer):
    budgets, _ = run(seed_catalog, "budgets_2024", pseudonymizer)
    transcripts, report = run(seed_catalog, "meeting_transcripts", pseudonymizer)

    maria = next(d for d in budgets if d.metadata["client_code"] == "CLI-0042")
    acme = next(d for d in transcripts if "acme" in d.id)
    assert maria.metadata["contact_name"] in acme.text
    assert maria.metadata["contact_name"] in acme.metadata["participants"]
    assert report.pii_replacements["EMAIL_ADDRESS"] == 2
    all_text = " ".join(d.text + json.dumps(d.metadata, ensure_ascii=False) for d in transcripts)
    for real in ("María García", "Raúl Ibáñez", "Ana López", "612 345 678", "@betanorte.com"):
        assert real not in all_text


def test_sources_without_pii_do_not_need_a_pseudonymizer(seed_catalog):
    documents, report = run(seed_catalog, "rate_card_2024")

    assert report.documents == 6 and report.pii_replacements == {} and report.output is None
    assert documents[0].text == "Tarifa de Desarrollador backend: 50 EUR por hora."


def test_cli_ingests_every_included_source(repo_data, monkeypatch, capsys):
    monkeypatch.setenv("PII_SALT", "sal-cli")
    get_settings.cache_clear()

    assert orchestrator.main(["--all"]) == 0

    reports = capsys.readouterr().out
    assert reports.count('"source":') == 3
    assert sorted(p.name for p in repo_data.iterdir()) == [
        "budgets_2024.jsonl",
        "meeting_transcripts.jsonl",
        "rate_card_2024.jsonl",
    ]


def test_cli_reports_rejected_sources_and_needs_a_source(repo_data, capsys):
    assert orchestrator.main(["budgets_2024"]) == 1  # sin PII_SALT
    assert "rechazada: la fuente 'budgets_2024' tiene datos personales" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        orchestrator.main([])


# --- Arquitectura ---


def test_small_corpus_fits_cag_and_large_one_does_not():
    small, large = architecture.evaluate_cag(1_000), architecture.evaluate_cag(150_000)

    assert small.viable and small.estimated_latency_ms == 8_400
    assert not large.viable
    assert not (large.context_window_ok or large.lost_in_the_middle_ok)


def test_recommendation_axes():
    viable = architecture.evaluate_cag(1_000)
    plain = dict(refresh_days=30, needs_traceability=False, needs_access_control=False)

    assert architecture.recommend_architecture(viable, **plain).architecture == "CAG"
    hybrid = architecture.recommend_architecture(
        viable, refresh_days=1, needs_traceability=True, needs_access_control=True
    )
    assert hybrid.architecture == "Hybrid" and len(hybrid.reasons) == 3
    rag = architecture.recommend_architecture(architecture.evaluate_cag(20_000), **plain)
    assert rag.architecture == "RAG" and "latencia" in rag.reasons[0]


def test_architecture_cli_uses_the_real_corpus(repo_data, capsys):
    assert architecture.corpus_tokens() > 0
    assert architecture.main(["--scale", "50"]) == 0

    out = capsys.readouterr().out
    assert "×50" in out and "Recomendación: RAG" in out
