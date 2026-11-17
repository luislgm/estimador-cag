"""PDFs sintéticos de tamaño calibrado para la prueba de carga de adjuntos.

    uv run python -m evals.stress.fixtures.build_pdfs      # attach_5kb.pdf ... attach_100kb.pdf

El tamaño es el del texto que se extrae (1 KB = 1.000 caracteres), que es lo que llega al
prompt; el del archivo PDF es algo mayor. Son deterministas, así que no se versionan: el
runner los regenera.

Cada documento es un pliego de requisitos con dos hechos marcados para medir el recall:

- HEAD_FACT al principio (integración con Navision)
- TAIL_FACT al final (informes en Tableau)

El servicio recorta cada adjunto a MAX_CHARS_PER_FILE (15.000 caracteres): a partir de unos
15 KB, el final del documento (y con él TAIL_FACT) no llega al modelo.

No usa reportlab ni fpdf2: escribe el PDF a mano (texto en Helvetica, varias páginas), como
la factoría de los tests, para no añadir dependencias.
"""

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent
SIZES_KB = (5, 20, 50, 100)
HEAD_FACT = "Navision"
TAIL_FACT = "Tableau"
LINES_PER_PAGE = 50

_ACTIONS = ("consultar", "crear", "modificar", "aprobar", "exportar", "archivar", "notificar")
_OBJECTS = (
    "los pedidos",
    "las facturas",
    "los clientes",
    "las tarifas",
    "los albaranes",
    "las incidencias",
    "los usuarios",
    "las rutas",
)


def requirements_text(kb: int) -> str:
    """Texto de ~kb * 1000 caracteres: hecho inicial, requisitos de relleno y hecho final."""
    head = (
        f"PLIEGO DE REQUISITOS. Requisito clave: integracion obligatoria con el ERP {HEAD_FACT} "
        "para sincronizar clientes, pedidos y stock."
    )
    tail = f"Requisito final: los informes de direccion se generan en {TAIL_FACT}."
    target = kb * 1000 - len(head) - len(tail) - 2
    lines, size, i = [], 0, 1
    while size < target:
        action, obj = _ACTIONS[i % len(_ACTIONS)], _OBJECTS[i % len(_OBJECTS)]
        line = f"RF-{i:04d}. El sistema debe permitir {action} {obj} con registro de auditoria."
        lines.append(line)
        size += len(line) + 1
        i += 1
    return "\n".join([head, *lines, tail])


def build_pdf(text: str) -> bytes:
    """PDF de varias páginas con el texto, línea a línea, que pypdf sabe extraer."""
    lines = text.splitlines()
    pages = [lines[i : i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]
    n = len(pages)
    font_id = 3 + 2 * n
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [%s] /Count %d >>"
        % (b" ".join(b"%d 0 R" % (3 + 2 * p) for p in range(n)), n),
    ]
    for p, page_lines in enumerate(pages):
        escaped = (
            line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            for line in page_lines
        )
        stream = "BT /F1 9 Tf 40 800 Td 14 TL " + " ".join(f"({line}) '" for line in escaped)
        content = (stream + " ET").encode("latin-1")
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>" % (font_id, 4 + 2 * p)
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )
    pdf = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    pdf += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return pdf


def fixture_path(kb: int, directory: Path = FIXTURES_DIR) -> Path:
    return directory / f"attach_{kb}kb.pdf"


def build_all(sizes=SIZES_KB, directory: Path = FIXTURES_DIR) -> list[Path]:
    paths = []
    for kb in sizes:
        path = fixture_path(kb, directory)
        path.write_bytes(build_pdf(requirements_text(kb)))
        paths.append(path)
    return paths


if __name__ == "__main__":  # pragma: no cover
    for path in build_all():
        print(f"{path.name}: {path.stat().st_size / 1024:.0f} KB")
