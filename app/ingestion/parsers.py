"""Parsers: de los bytes de cada formato a Document, el contrato común.

Un Parser es cualquier clase con `formats` y `parse(blob, context)` (un Protocol, no una
clase base: los parsers no comparten implementación, y así un parser falso en los tests no
tiene que heredar de nada). El registro elige el parser por el formato de la fuente.

- budget_json: un presupuesto por archivo. Lo normaliza (reparaciones de cleaning.py) y deja
  el registro en metadata para que el orquestador lo valide.
- transcript_txt: una transcripción por archivo, con una cabecera opcional
  (Fecha / Cliente / Participantes) que pasa a metadata.
- rate_card_xlsx: la tabla de tarifas (rol, tarifa por hora, moneda), un documento por rol.
"""

import io
import json
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Protocol, runtime_checkable

from openpyxl import load_workbook

from app.ingestion.catalog import CatalogSource
from app.ingestion.cleaning import normalize_budget
from app.ingestion.documents import Document, LoadedBlob

HEADER_FIELDS = {"fecha": "date", "cliente": "client", "participantes": "participants"}


@dataclass(frozen=True)
class ParseContext:
    """Lo que el parser necesita y no está en el archivo. Se crea una vez por ejecución, así
    la versión y la fecha de ingesta son las mismas para todos los documentos."""

    source: CatalogSource
    source_version: str
    ingested_at: datetime


@runtime_checkable
class Parser(Protocol):
    formats: ClassVar[set[str]]

    def parse(self, blob: LoadedBlob, context: ParseContext) -> list[Document]: ...


class UnsupportedFormat(Exception):
    pass


def _document(context: ParseContext, key: str, kind: str, text: str, metadata: dict) -> Document:
    return Document(
        id=f"{context.source.name}:{key}",
        source=context.source.name,
        source_version=context.source_version,
        kind=kind,
        text=text,
        metadata=metadata,
        ingested_at=context.ingested_at,
    )


def budget_text(record: dict) -> str:
    """El presupuesto como texto para indexar (sin datos de contacto: van aparte)."""
    items = "\n".join(f"- {i['concept']}: {i['hours']} h" for i in record["items"])
    amount = record["total_amount"]
    return (
        f"Presupuesto {record['budget_id']} (versión {record['version']}) para "
        f"{record['client_name'] or 'cliente sin nombre'}, {record['date'] or 'sin fecha'}.\n"
        f"Total: {record['total_hours']} h, "
        f"{amount if amount is not None else '?'} {record['currency'] or ''}.\n"
        f"Partidas:\n{items}\n"
        f"Notas: {record['notes']}"
    ).strip()


class BudgetJsonParser:
    formats: ClassVar[set[str]] = {"budget_json"}

    def parse(self, blob: LoadedBlob, context: ParseContext) -> list[Document]:
        record = normalize_budget(json.loads(blob.content.decode("utf-8")))
        key = f"{record['budget_id']}@v{record['version']}"
        return [_document(context, key, "budget", budget_text(record), record)]


class TranscriptTxtParser:
    formats: ClassVar[set[str]] = {"transcript_txt"}

    def parse(self, blob: LoadedBlob, context: ParseContext) -> list[Document]:
        lines = blob.content.decode("utf-8").strip().splitlines()
        metadata: dict = {"file": blob.path.name}
        body_start = 0
        for i, line in enumerate(lines):
            name, sep, value = line.partition(":")
            field = HEADER_FIELDS.get(name.strip().lower())
            if not sep or field is None:
                break
            metadata[field] = value.strip()
            body_start = i + 1
        body = "\n".join(lines[body_start:]).strip()
        return [_document(context, blob.path.stem, "transcript", body, metadata)]


class RateCardXlsxParser:
    formats: ClassVar[set[str]] = {"rate_card_xlsx"}

    def parse(self, blob: LoadedBlob, context: ParseContext) -> list[Document]:
        sheet = load_workbook(io.BytesIO(blob.content), read_only=True).active
        rows = sheet.iter_rows(values_only=True)
        header = [str(h).strip().lower() for h in next(rows)]
        documents = []
        for row in rows:
            data = dict(zip(header, row, strict=False))
            if not data.get("rol"):
                continue
            role, rate, currency = data["rol"], data["tarifa_hora"], data.get("moneda") or "EUR"
            text = f"Tarifa de {role}: {rate} {currency} por hora."
            metadata = {"role": role, "hourly_rate": rate, "currency": currency}
            documents.append(_document(context, role.lower(), "rate_card", text, metadata))
        return documents


class ParserRegistry:
    def __init__(self, parsers: list[Parser]):
        self._by_format = {fmt: parser for parser in parsers for fmt in parser.formats}

    @property
    def formats(self) -> set[str]:
        return set(self._by_format)

    def for_format(self, fmt: str) -> Parser:
        try:
            return self._by_format[fmt]
        except KeyError:
            raise UnsupportedFormat(f"no hay parser para el formato '{fmt}'") from None


def default_registry() -> ParserRegistry:
    return ParserRegistry([BudgetJsonParser(), TranscriptTxtParser(), RateCardXlsxParser()])
