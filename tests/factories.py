"""Datos de prueba compartidos por los tests."""

import json

import litellm

from app.schemas.estimation import EstimationRequest, EstimationResult, Phase
from app.services.llm_client import LLMCallInfo

DESCRIPTION = (
    "El cliente quiere una web para reservar mesas en su restaurante, "
    "con pago online y un panel de gestión para el personal."
)
REQUEST_BODY = {
    "description": DESCRIPTION,
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table",
}


def make_request(**changes) -> EstimationRequest:
    return EstimationRequest(**REQUEST_BODY | changes)


def make_phase(name="Desarrollo", hours=40, **changes) -> Phase:
    fields = {
        "name": name,
        "hours": hours,
        "cost_eur": hours * 50,
        "duration_weeks": 2,
        "summary": f"Trabajo de {name.lower()}.",
    }
    return Phase(**fields | changes)


def make_result(phases=None, **changes) -> EstimationResult:
    phases = phases or [make_phase("Diseño", 20), make_phase("Desarrollo", 60)]
    fields = {
        "project_name": "Reservas para restaurante",
        "summary": "Web de reservas con pago online y panel para el personal.",
        "confidence_pct": 75,
        "phases": phases,
        "team": ["1 desarrollador full-stack", "1 diseñador (part-time)"],
        "risks": ["Integración con la pasarela de pago"],
        "total_hours": sum(p.hours for p in phases),
        "total_cost_eur": sum(p.cost_eur for p in phases),
        "total_duration_weeks": 4,
    }
    return EstimationResult(**fields | changes)


def make_info(**changes) -> LLMCallInfo:
    fields = {
        "model": "claude-haiku-4-5",
        "provider": "anthropic",
        "input_tokens": 3500,
        "output_tokens": 1200,
        "finish_reason": "tool_calls",
        "cost_usd": 0.0095,
        "attempts": 1,
    }
    return LLMCallInfo(**fields | changes)


def tool_call_response(payload: dict, prompt_tokens=1000, completion_tokens=200):
    """Respuesta de LiteLLM con una llamada a la herramienta de Instructor (sin red)."""
    return litellm.ModelResponse(
        model="claude-haiku-4-5",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "EstimationResult",
                                "arguments": json.dumps(payload, ensure_ascii=False),
                            },
                        }
                    ],
                },
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


def make_pdf(*lines: str) -> bytes:
    """Un PDF mínimo con texto (fuente Helvetica), que pypdf sabe leer."""
    stream = (
        "BT /F1 11 Tf 50 780 Td 14 TL "
        + " ".join(
            "(" + line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ") '"
            for line in lines
        )
        + " ET"
    )
    content = stream.encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
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


def make_docx(*paragraphs: str, table: list[list[str]] | None = None) -> bytes:
    """Un documento Word con párrafos y, opcionalmente, una tabla."""
    import io

    import docx

    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    if table:
        word_table = document.add_table(rows=len(table), cols=len(table[0]))
        for r, row in enumerate(table):
            for c, value in enumerate(row):
                word_table.cell(r, c).text = value
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()
