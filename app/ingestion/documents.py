"""El contrato común de la ingesta: lo que entra (LoadedBlob) y lo que sale (Document).

Todos los parsers, sea cual sea el formato, devuelven Document. Así el resto del pipeline
(limpieza, seudonimización, y después la indexación de RAG) no sabe nada de JSON, TXT o
Excel. `content_hash` permite deduplicar: el mismo texto da el mismo hash.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field


class LoadedBlob(BaseModel):
    """Un archivo leído del disco, todavía sin interpretar."""

    path: Path
    content: bytes
    format: str


class Document(BaseModel):
    id: str = Field(..., description="Identificador estable: fuente + clave del documento")
    source: str
    source_version: str = Field(..., description="Versión del catálogo con la que se ingirió")
    kind: Literal["budget", "transcript", "rate_card"]
    text: str = Field(..., description="Texto listo para indexar")
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime

    @computed_field
    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()
