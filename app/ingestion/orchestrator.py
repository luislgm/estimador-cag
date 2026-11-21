"""Orquestador: una ejecución de ingesta de una fuente del catálogo, con su informe.

    uv run python -m app.ingestion.orchestrator budgets_2024        # una fuente
    uv run python -m app.ingestion.orchestrator --all               # todas las incluidas

    1. La fuente tiene que existir y estar en `include`; si tiene datos personales, hace falta
       PII_SALT. Si no, IngestionRejected (en la API, un 400) antes de tocar ningún archivo.
    2. Carga los archivos y los convierte en Document con el parser de su formato.
    3. Presupuestos: valida los registros (reparar / cuarentena / descartar) y deduplica.
    4. Datos personales: seudonimiza el texto y los campos de metadata que los contienen.
    5. Escribe los documentos en JSONL (<processed_dir>/<fuente>.jsonl) y devuelve el informe.

Es síncrono: en la API se ejecuta en segundo plano (BackgroundTasks), con el estado del
trabajo en la base de datos.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.ingestion.catalog import (
    CatalogDecision,
    CatalogSource,
    DataCatalog,
    UnknownSource,
    load_catalog,
)
from app.ingestion.cleaning import validate_budgets
from app.ingestion.documents import Document
from app.ingestion.loaders import FileSystemLoader
from app.ingestion.parsers import ParseContext, ParserRegistry, default_registry
from app.ingestion.pii import InMemoryMappingStore, MappingStore, Pseudonymizer

# Campos de metadata con datos personales de tipo conocido, y campos de texto libre que
# pueden contenerlos (se analizan con Presidio). El resto (empresa, importes...) no se toca.
PII_FIELDS = {
    "contact_name": "PERSON",
    "contact_email": "EMAIL_ADDRESS",
    "contact_phone": "PHONE_NUMBER",
}
PII_TEXT_FIELDS = ("participants", "notes")


class IngestionRejected(Exception):
    """La fuente no se puede ingerir: no existe, no está en include, falta la sal..."""


@dataclass
class IngestionReport:
    source: str
    source_version: str
    files: int = 0
    parsed: int = 0
    documents: int = 0
    repaired: int = 0
    quarantined: dict[str, list[str]] = field(default_factory=dict)
    discarded: dict[str, list[str]] = field(default_factory=dict)
    deduplicated: list[str] = field(default_factory=list)
    pii_replacements: dict[str, int] = field(default_factory=dict)
    output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_source(
    catalog: DataCatalog, source_name: str, registry: ParserRegistry, has_salt: bool
) -> CatalogSource:
    """La fuente, si se puede ingerir; IngestionRejected con el motivo si no."""
    try:
        source = catalog.get(source_name)
    except UnknownSource:
        raise IngestionRejected(f"la fuente '{source_name}' no está en el catálogo") from None
    if source.decision != CatalogDecision.INCLUDE:
        raise IngestionRejected(
            f"la fuente '{source_name}' está en '{source.decision}': {source.decision_reason}"
        )
    if source.format not in registry.formats:
        raise IngestionRejected(f"no hay parser para el formato '{source.format}'")
    if source.sensitivity.contains_pii and not has_salt:
        raise IngestionRejected(
            f"la fuente '{source_name}' tiene datos personales y no hay PII_SALT configurada"
        )
    return source


def _pseudonymize(document: Document, pseudonymizer: Pseudonymizer, counts: dict[str, int]):
    known = {
        document.metadata[key]: entity
        for key, entity in PII_FIELDS.items()
        if document.metadata.get(key)
    }
    result = pseudonymizer.pseudonymize(document.text, known)
    metadata = dict(document.metadata)
    for key, entity in PII_FIELDS.items():
        if metadata.get(key):
            metadata[key] = pseudonymizer.pseudonym(entity, metadata[key])
    for key in PII_TEXT_FIELDS:
        if metadata.get(key):
            metadata[key] = pseudonymizer.pseudonymize(metadata[key], known).text
    for entity, n in result.counts.items():
        counts[entity] = counts.get(entity, 0) + n
    return document.model_copy(update={"text": result.text, "metadata": metadata})


def ingest_source(
    catalog: DataCatalog,
    source_name: str,
    *,
    loader: FileSystemLoader,
    registry: ParserRegistry,
    pseudonymizer: Pseudonymizer | None = None,
    output_dir: Path | None = None,
) -> tuple[list[Document], IngestionReport]:
    source = check_source(catalog, source_name, registry, pseudonymizer is not None)
    report = IngestionReport(source=source.name, source_version=catalog.version)
    context = ParseContext(source, catalog.version, datetime.now(UTC))
    parser = registry.for_format(source.format)

    blobs = loader.load(source)
    documents = [doc for blob in blobs for doc in parser.parse(blob, context)]
    report.files, report.parsed = len(blobs), len(documents)

    budgets = [d for d in documents if d.kind == "budget"]
    if budgets:
        validation = validate_budgets([d.metadata for d in budgets])
        keep = {(r["budget_id"], r["version"]) for r in validation.valid}
        documents = [
            d
            for d in documents
            if d.kind != "budget" or (d.metadata["budget_id"], d.metadata["version"]) in keep
        ]
        report.repaired = validation.repaired
        report.quarantined = validation.quarantined
        report.discarded = validation.discarded
        report.deduplicated = validation.deduplicated

    if source.sensitivity.contains_pii:
        documents = [_pseudonymize(d, pseudonymizer, report.pii_replacements) for d in documents]
    report.documents = len(documents)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{source.name}.jsonl"
        lines = (d.model_dump_json() for d in documents)
        path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        report.output = str(path)
    return documents, report


def main(argv: list[str] | None = None, store: MappingStore | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingesta de una fuente del catálogo")
    parser.add_argument("source", nargs="?", help="Nombre de la fuente en el catálogo")
    parser.add_argument("--all", action="store_true", help="Todas las fuentes incluidas")
    args = parser.parse_args(argv)
    if not args.source and not args.all:
        parser.error("indica una fuente o --all")

    settings = get_settings()
    catalog = load_catalog()
    names = (
        [s.name for s in catalog.sources if s.decision == CatalogDecision.INCLUDE]
        if args.all
        else [args.source]
    )
    salt = settings.pii_salt.get_secret_value().encode() if settings.pii_salt else b""
    pseudonymizer = Pseudonymizer(salt, store or InMemoryMappingStore()) if salt else None
    loader, registry = FileSystemLoader(settings.data_dir), default_registry()
    status = 0
    for name in names:
        try:
            _, report = ingest_source(
                catalog,
                name,
                loader=loader,
                registry=registry,
                pseudonymizer=pseudonymizer,
                output_dir=settings.processed_dir,
            )
        except IngestionRejected as e:
            print(f"{name}: rechazada: {e}", file=sys.stderr)
            status = 1
            continue
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
