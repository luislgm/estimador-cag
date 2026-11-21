"""Catálogo de fuentes de datos: el contrato entre lo que existe y lo que se ingiere.

    uv run python -m app.ingestion.catalog          # inventario: decisión, calidad, PII y archivos

Cada fuente tiene una decisión:

- include: entra en la ingesta.
- review: tiene problemas conocidos y espera la revisión de su responsable. Se conserva,
  pero no se ingiere. Se mantiene esta categoría intermedia en lugar de forzar sí/no.
- exclude: fuera, con un motivo obligatorio. No ingerir algo es una decisión que hay que
  poder defender, no un olvido.

La calidad se puntúa de 1 a 5 en cuatro dimensiones (completitud, consistencia, actualidad
y fiabilidad), y `sensitivity` dice si la fuente tiene datos personales: el orquestador la
usa para decidir si los documentos pasan por la seudonimización.
"""

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import get_settings


class CatalogDecision(StrEnum):
    INCLUDE = "include"
    REVIEW = "review"
    EXCLUDE = "exclude"


class QualityScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completeness: int = Field(..., ge=1, le=5)
    consistency: int = Field(..., ge=1, le=5)
    actuality: int = Field(..., ge=1, le=5)
    reliability: int = Field(..., ge=1, le=5)

    @property
    def average(self) -> float:
        return (self.completeness + self.consistency + self.actuality + self.reliability) / 4


class Sensitivity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contains_pii: bool
    pii_types: list[str] = Field(default_factory=list)


class CatalogSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., pattern=r"^[a-z0-9_]+$")
    description: str
    owner: str
    path: str = Field(..., description="Patrón glob relativo al directorio de datos")
    format: str
    refresh: str
    decision: CatalogDecision
    decision_reason: str | None = None
    quality: QualityScore
    sensitivity: Sensitivity

    @model_validator(mode="after")
    def excluding_needs_a_reason(self) -> "CatalogSource":
        if self.decision != CatalogDecision.INCLUDE and not self.decision_reason:
            raise ValueError(f"{self.name}: '{self.decision}' necesita un decision_reason")
        return self


class UnknownSource(Exception):
    pass


class DataCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    sources: list[CatalogSource]

    @model_validator(mode="after")
    def unique_names(self) -> "DataCatalog":
        names = [s.name for s in self.sources]
        duplicated = sorted({n for n in names if names.count(n) > 1})
        if duplicated:
            raise ValueError(f"fuentes repetidas en el catálogo: {duplicated}")
        return self

    def get(self, name: str) -> CatalogSource:
        for source in self.sources:
            if source.name == name:
                return source
        raise UnknownSource(name)


def load_catalog(path: Path | None = None) -> DataCatalog:
    path = path or get_settings().data_dir / "catalog.yaml"
    return DataCatalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def inventory(catalog: DataCatalog, data_dir: Path) -> str:
    """Tabla del catálogo: decisión, calidad media, PII y archivos encontrados."""
    lines = [
        f"Catálogo {catalog.version} ({data_dir})",
        "",
        f"{'Fuente':<22} {'Decisión':<9} {'Calidad':>7} {'PII':<4} {'Archivos':>8}  Motivo",
    ]
    for s in catalog.sources:
        files = len(list(data_dir.glob(s.path)))
        pii = "sí" if s.sensitivity.contains_pii else "no"
        lines.append(
            f"{s.name:<22} {s.decision:<9} {s.quality.average:>7.2f} {pii:<4} {files:>8}  "
            f"{s.decision_reason or ''}"
        )
    return "\n".join(lines)


def main() -> int:
    data_dir = get_settings().data_dir
    print(inventory(load_catalog(), data_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
