"""Contratos de los endpoints de sesiones conversacionales."""

from pydantic import BaseModel, Field

from app.schemas.estimation import EstimationResponse
from app.sessions import ProjectMetadata


class SessionCreated(BaseModel):
    session_id: str = Field(..., description="Identificador de la sesión (UUID v4)")


class SessionState(BaseModel):
    session_id: str
    turns: int = Field(..., description="Turnos totales de la conversación")
    turns_in_window: int = Field(..., description="Turnos que se envían al LLM (ventana)")
    max_turns: int = Field(..., description="Tamaño de la ventana deslizante")
    project_metadata: ProjectMetadata = Field(..., description="Memoria del proyecto")


class AttachmentInfo(BaseModel):
    filename: str
    chars: int = Field(..., description="Caracteres de texto extraídos")
    truncated: bool = Field(..., description="True si el texto se recortó por ser muy largo")


class SessionEstimationResponse(EstimationResponse):
    """La estimación del turno, más el estado de la sesión tras el turno."""

    session: SessionState
    attachments: list[AttachmentInfo] = Field(default_factory=list)
