"""Contratos de los endpoints de ingesta."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.ingestion.pii import PII_ENTITIES


class SourceInfo(BaseModel):
    name: str
    description: str
    owner: str
    format: str
    decision: Literal["include", "review", "exclude"]
    decision_reason: str | None
    quality: float = Field(..., description="Media de las cuatro dimensiones de calidad (1-5)")
    contains_pii: bool
    files: int = Field(..., description="Archivos que hay ahora mismo para la fuente")


class IngestionRunRequest(BaseModel):
    source: str = Field(..., description="Nombre de la fuente en el catálogo")


class IngestionJobView(BaseModel):
    job_id: str
    source: str
    status: Literal["pending", "running", "completed", "failed"]
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    report: dict[str, Any] | None = Field(None, description="Informe de la ejecución")
    error: str | None = None


class ForgetRequest(BaseModel):
    entity_type: Literal[PII_ENTITIES] = Field(..., description="Tipo de dato personal")  # type: ignore[valid-type]
    value: str = Field(..., min_length=1, description="El dato original (no se guarda)")


class ForgetResponse(BaseModel):
    forgotten: bool = Field(..., description="False si no había ningún seudónimo para ese dato")
