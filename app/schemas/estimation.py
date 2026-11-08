"""Contrato entre el cliente (Streamlit) y el servicio IA para las estimaciones."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


# --- Estimación estructurada (salida del LLM, validada con Pydantic) ---

HOURLY_RATE_EUR = 50
OUT_OF_SCOPE_PREFIX = "Fuera de alcance:"
LOW_CONFIDENCE_THRESHOLD = 30


class Phase(BaseModel):
    """Una fase del plan de trabajo."""

    name: str = Field(..., min_length=1, max_length=80, description="Nombre de la fase")
    hours: int = Field(..., ge=0, le=5000, description="Horas de trabajo de la fase")
    cost_eur: int = Field(..., ge=0, le=250_000, description=f"Coste: horas × {HOURLY_RATE_EUR}")
    duration_weeks: int = Field(..., ge=1, le=52, description="Semanas de calendario")
    summary: str = Field(..., min_length=5, max_length=600, description="Qué se hace en la fase")
    assumptions: list[str] = Field(
        default_factory=list, description="Asunciones de la fase (obligatorias en 'detailed')"
    )

    @model_validator(mode="after")
    def cost_matches_rate(self) -> "Phase":
        expected = self.hours * HOURLY_RATE_EUR
        if self.cost_eur != expected:
            raise ValueError(
                f"la fase '{self.name}' cuesta {self.cost_eur} € pero tiene {self.hours} h: "
                f"a {HOURLY_RATE_EUR} €/h deberían ser {expected} €"
            )
        return self


class EstimationResult(BaseModel):
    """Estimación estructurada. Los validadores son reglas de negocio que el LLM no puede
    saltarse: si una falla, Instructor le devuelve el error y le pide que la corrija.

    El orden de los campos es deliberado: las fases van antes que los totales para que el
    modelo fije primero las cifras de cada fase y después solo tenga que sumarlas.
    """

    project_name: str = Field(..., min_length=3, max_length=120, description="Nombre corto")
    summary: str = Field(..., min_length=10, max_length=1500, description="Resumen del proyecto")
    confidence_pct: int = Field(..., ge=0, le=100, description="Confianza en la estimación (%)")
    phases: list[Phase] = Field(..., min_length=1, max_length=15)
    team: list[str] = Field(default_factory=list, description="Equipo recomendado (roles)")
    team_size: int = Field(default=0, ge=0, le=100, description="Personas del equipo recomendado")
    technologies: list[str] = Field(
        default_factory=list, description="Tecnologías mencionadas en la conversación o asumidas"
    )
    risks: list[str] = Field(default_factory=list, description="Riesgos principales")
    total_hours: int = Field(..., ge=0, le=50_000, description="Suma de las horas de las fases")
    total_cost_eur: int = Field(..., ge=0, le=2_500_000, description="Suma de los costes")
    total_duration_weeks: int = Field(..., ge=1, le=104, description="Duración total (semanas)")

    @model_validator(mode="after")
    def totals_match_phases(self) -> "EstimationResult":
        hours = sum(p.hours for p in self.phases)
        cost = sum(p.cost_eur for p in self.phases)
        errors = []
        if hours != self.total_hours:
            errors.append(f"las fases suman {hours} h pero total_hours es {self.total_hours}")
        if cost != self.total_cost_eur:
            errors.append(f"las fases suman {cost} € pero total_cost_eur es {self.total_cost_eur}")
        if errors:
            raise ValueError("; ".join(errors) + ". Calcula los totales sumando las fases.")
        return self

    @model_validator(mode="after")
    def low_confidence_is_declared(self) -> "EstimationResult":
        if self.confidence_pct < LOW_CONFIDENCE_THRESHOLD and not self.summary.startswith(
            OUT_OF_SCOPE_PREFIX
        ):
            raise ValueError(
                f"con confidence_pct < {LOW_CONFIDENCE_THRESHOLD} el resumen debe empezar por "
                f"'{OUT_OF_SCOPE_PREFIX}' y explicar qué falta para poder estimar"
            )
        return self

    @property
    def out_of_scope(self) -> bool:
        return self.summary.startswith(OUT_OF_SCOPE_PREFIX)


class TokenUsage(BaseModel):
    input_tokens: int = Field(..., description="Tokens enviados al modelo")
    output_tokens: int = Field(..., description="Tokens generados por el modelo")
    total_tokens: int = Field(..., description="Suma de tokens de entrada y salida")


class EstimationResponse(BaseModel):
    text: str = Field(
        ..., description="Estimación en markdown, generada a partir de `result` según output_format"
    )
    prompt_version: str = Field(..., description="Versión de la plantilla de prompt usada")
    result: EstimationResult = Field(..., description="Estimación estructurada y validada")

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
    cache_kind: Literal["exact", "semantic"] | None = Field(
        default=None, description="De qué caché viene la respuesta, si viene de alguna"
    )
    created_at: datetime = Field(..., description="Fecha y hora de la estimación (UTC)")
