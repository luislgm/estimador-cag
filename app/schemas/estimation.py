"""Contrato entre el cliente (Streamlit) y el servicio IA para las estimaciones."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"


class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"


class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"


class EstimationRequest(BaseModel):
    # Quita espacios al principio y al final antes de validar la longitud
    model_config = ConfigDict(str_strip_whitespace=True)

    description: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        description="Descripción del proyecto o transcripción de la reunión (20-2000 caracteres)",
        examples=[
            "Una cadena de autoescuelas quiere una app móvil para matricularse, pagar paquetes "
            "de clases y reservar prácticas con su profesor."
        ],
    )
    project_type: ProjectType = Field(..., description="Tipo de proyecto")
    detail_level: DetailLevel = Field(..., description="Nivel de detalle de la estimación")
    output_format: OutputFormat = Field(..., description="Formato de la estimación")


class TokenUsage(BaseModel):
    input_tokens: int = Field(..., description="Tokens enviados al modelo")
    output_tokens: int = Field(..., description="Tokens generados por el modelo")
    total_tokens: int = Field(..., description="Suma de tokens de entrada y salida")


class EstimationResponse(BaseModel):
    text: str = Field(..., description="Estimación generada en formato markdown")
    prompt_version: str = Field(..., description="Versión de la plantilla de prompt usada")

    # Metadatos adicionales de la llamada (no los pide el contrato, pero el cliente los usa)
    model: str = Field(..., description="Modelo que ha generado la estimación")
    provider: str = Field(..., description="Proveedor del LLM")
    usage: TokenUsage = Field(..., description="Consumo de tokens de la llamada")
    cost_usd: float | None = Field(
        ..., description="Coste estimado en USD (null si el modelo no tiene precio configurado)"
    )
    finish_reason: str = Field(
        ...,
        description="Motivo de fin de la generación. 'length' indica estimación cortada",
        examples=["stop"],
    )
    cache_hit: bool = Field(
        ..., description="True si la respuesta viene de la caché (sin llamar al LLM, coste 0)"
    )
    created_at: datetime = Field(..., description="Fecha y hora de la estimación (UTC)")
