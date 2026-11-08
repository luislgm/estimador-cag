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
