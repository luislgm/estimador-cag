"""Carga de archivos del disco según el patrón de cada fuente del catálogo.

Separado de los parsers a propósito: el loader sabe *dónde* están los bytes (hoy, el disco;
mañana, un bucket o una API) y el parser sabe *qué significan*.
"""

from pathlib import Path

from app.ingestion.catalog import CatalogSource
from app.ingestion.documents import LoadedBlob


class FileSystemLoader:
    def __init__(self, root: Path):
        self.root = root

    def load(self, source: CatalogSource) -> list[LoadedBlob]:
        """Los archivos de la fuente, en orden alfabético (ejecuciones reproducibles)."""
        return [
            LoadedBlob(path=path, content=path.read_bytes(), format=source.format)
            for path in sorted(self.root.glob(source.path))
            if path.is_file()
        ]
