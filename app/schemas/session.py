"""Contratos de los endpoints de sesiones conversacionales."""

from pydantic import BaseModel, Field

from app.schemas.critic import ACBTrace
from app.schemas.estimation import EstimationResponse
from app.sessions import ProjectMetadata
from app.tiers import Tier


class SessionCreated(BaseModel):
    session_id: str = Field(..., description="Identificador de la sesión (UUID v4)")


class SessionState(BaseModel):
    session_id: str
    turns: int = Field(..., description="Turnos totales de la conversación")
    turns_in_window: int = Field(..., description="Turnos que se envían al LLM (ventana)")
    max_turns: int = Field(..., description="Tamaño de la ventana deslizante")
    project_metadata: ProjectMetadata = Field(..., description="Memoria del proyecto")
    anchors: int = Field(..., description="Turnos antiguos conservados literales (anclas)")
    max_anchors: int = Field(..., description="Máximo de anclas")
    summarized_turns: int = Field(..., description="Turnos antiguos integrados en el resumen")
    conversation_summary: str | None = Field(
        None, description="Resumen acumulado de los turnos que salieron de la ventana"
    )
    compression_cost_usd: float = Field(
        ..., description="Coste acumulado de las llamadas de resumen (aparte del de cada turno)"
    )
    tier: Tier | None = Field(None, description="Audiencia del último turno")
    tier_rule: str | None = Field(
        None, description="Qué la decidió: override, el nombre de una regla o sin_señales"
    )
    tier_inherited: bool = Field(False, description="True si se mantuvo del turno anterior")


class AttachmentInfo(BaseModel):
    filename: str
    chars: int = Field(..., description="Caracteres de texto extraídos")
    truncated: bool = Field(..., description="True si el texto se recortó por ser muy largo")


class SessionEstimationResponse(EstimationResponse):
    """La estimación del turno, más el estado de la sesión tras el turno."""

    session: SessionState
    attachments: list[AttachmentInfo] = Field(default_factory=list)


class SessionACBResponse(SessionEstimationResponse):
    """La estimación revisada por el ciclo Actor-Critic-Boss, con su traza. `usage` y
    `cost_usd` suman todas las llamadas del turno (actor y crítico)."""

    acb: ACBTrace
