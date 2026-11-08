import pytest

from app.services import attachments
from app.services.attachments import AttachmentError, extract_text
from tests.factories import make_docx, make_pdf


def test_pdf_text():
    pdf = make_pdf("Requisitos tecnicos", "Integracion con SAP Business One.")

    attachment = extract_text("requisitos.pdf", pdf)

    assert attachment.text == "Requisitos tecnicos\nIntegracion con SAP Business One."
    assert attachment.filename == "requisitos.pdf"
    assert not attachment.truncated


def test_docx_text_includes_tables():
    document = make_docx("Propuesta previa", table=[["Fase", "Horas"], ["Diseño", "40"]])

    attachment = extract_text("propuesta.DOCX", document)  # la extensión no distingue mayúsculas

    assert attachment.text == "Propuesta previa\nFase | Horas\nDiseño | 40"


def test_unsupported_format():
    with pytest.raises(AttachmentError, match="formato no admitido"):
        extract_text("notas.txt", b"hola")


def test_file_too_large(monkeypatch):
    monkeypatch.setattr(attachments, "MAX_FILE_BYTES", 10)

    with pytest.raises(AttachmentError, match="tamaño máximo"):
        extract_text("grande.pdf", make_pdf("texto"))


def test_corrupt_file():
    with pytest.raises(AttachmentError, match="no se puede leer"):
        extract_text("roto.pdf", b"esto no es un PDF")


def test_pdf_without_text():
    attachment = extract_text("escaneado.pdf", make_pdf())

    assert "no contiene texto extraíble" in attachment.text


def test_long_documents_are_truncated(monkeypatch):
    monkeypatch.setattr(attachments, "MAX_CHARS_PER_FILE", 20)

    attachment = extract_text("largo.docx", make_docx("x" * 100))

    assert attachment.truncated
    assert attachment.text.startswith("x" * 20)
    assert "documento recortado" in attachment.text
