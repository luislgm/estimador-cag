"""Contrato entre el cliente (Streamlit) y el servicio IA para las estimaciones."""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


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
MIN_PHASES = 2  # mínimo de fases de una estimación dentro de alcance


class Phase(BaseModel):
    """Una fase del plan de trabajo.

    El coste no lo da el modelo: se calcula (horas × tarifa). Es un computed_field, así que
    sale en el JSON de la respuesta pero no está en el esquema que se envía al LLM.
    """

    name: str = Field(..., min_length=1, max_length=80, description="Nombre de la fase")
    hours: int = Field(..., ge=0, le=5000, description="Horas de trabajo de la fase")
    duration_weeks: int = Field(..., ge=1, le=52, description="Semanas de calendario")
    summary: str = Field(..., min_length=5, max_length=600, description="Qué se hace en la fase")
    assumptions: list[str] = Field(
        default_factory=list, description="Asunciones de la fase (obligatorias en 'detailed')"
    )

    @computed_field(description=f"Coste: horas × {HOURLY_RATE_EUR} €/h")
    @property
    def cost_eur(self) -> int:
        return self.hours * HOURLY_RATE_EUR


class EstimationResult(BaseModel):
    """Estimación estructurada. Los validadores son reglas de negocio que el LLM no puede
    saltarse: si una falla, Instructor le devuelve el error y le pide que la corrija.

    La aritmética no se le pide al modelo: el coste de cada fase y los totales de horas y de
    coste se calculan a partir de las fases (computed_field). Antes se le pedía sumar y un
    validador lo comprobaba, pero las evals mostraron que los modelos pequeños fallan la suma
    (gpt-4o-mini, 2 de 13 casos, incluso tras los reintentos) y el turno acababa en error.
    Sumar es lo que el código hace bien. La duración total sí la da el modelo: las fases
    pueden solaparse, así que no es una suma.
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
    total_duration_weeks: int = Field(
        ..., ge=1, le=104, description="Duración total en semanas de calendario"
    )

    @model_validator(mode="after")
    def estimate_has_several_phases(self) -> "EstimationResult":
        # Una estimación real tiene varias fases (el prompt pide de 3 a más de 10 según el nivel
        # de detalle). Las evals detectaron que gpt-4o-mini a veces corta la lista y devuelve
        # solo la primera fase; con este validador se le pide que la complete.
        if not self.out_of_scope and len(self.phases) < MIN_PHASES:
            raise ValueError(
                f"una estimación necesita al menos {MIN_PHASES} fases y solo hay "
                f"{len(self.phases)}: devuelve todas las fases del proyecto completo"
            )
        return self

    @computed_field(description="Suma de las horas de las fases")
    @property
    def total_hours(self) -> int:
        return sum(p.hours for p in self.phases)

    @computed_field(description="Suma de los costes de las fases")
    @property
    def total_cost_eur(self) -> int:
        return sum(p.cost_eur for p in self.phases)

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
