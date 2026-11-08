"""Adjuntos: extracción local de texto de PDFs (pypdf) y documentos Word (python-docx).

Es el "camino B" del ejercicio: el texto se extrae en el servicio y se envía como parte del
prompt, en lugar de subir el archivo al LLM. Así funciona igual con cualquier proveedor
(LiteLLM), los guardrails pueden revisar el contenido de los documentos y se prepara el
terreno para trocear documentos en RAG. A cambio, se pierden las imágenes y los diagramas.
"""

import io
from dataclasses import dataclass

import docx
from pypdf import PdfReader
from pypdf.errors import PyPdfError

MAX_FILES = 5
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB por archivo
MAX_CHARS_PER_FILE = 15_000  # para no disparar los tokens (y el coste) de cada llamada
SUPPORTED_EXTENSIONS = (".pdf", ".docx")


class AttachmentError(ValueError):
    """Un adjunto no se puede procesar: formato no admitido, demasiado grande o dañado."""


@dataclass
class Attachment:
    filename: str
    text: str
    truncated: bool = False

    @property
    def chars(self) -> int:
        return len(self.text)


def extract_text(filename: str, content: bytes) -> Attachment:
    """El texto de un adjunto, recortado a MAX_CHARS_PER_FILE si es muy largo."""
    name = filename.lower()
    if not name.endswith(SUPPORTED_EXTENSIONS):
        raise AttachmentError(
            f"'{filename}': formato no admitido (solo {', '.join(SUPPORTED_EXTENSIONS)})"
        )
    if len(content) > MAX_FILE_BYTES:
        raise AttachmentError(f"'{filename}': supera el tamaño máximo de 5 MB")
    try:
        text = _pdf_text(content) if name.endswith(".pdf") else _docx_text(content)
    except (PyPdfError, ValueError, KeyError, OSError) as e:
        raise AttachmentError(f"'{filename}': no se puede leer el archivo ({e})") from e
    text = text.strip() or "(el documento no contiene texto extraíble, p. ej. un escaneado)"
    truncated = len(text) > MAX_CHARS_PER_FILE
    if truncated:
        text = text[:MAX_CHARS_PER_FILE] + "\n[... documento recortado ...]"
    return Attachment(filename=filename, text=text, truncated=truncated)


def _pdf_text(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _docx_text(content: bytes) -> str:
    document = docx.Document(io.BytesIO(content))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:  # las tablas no están en .paragraphs
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)
