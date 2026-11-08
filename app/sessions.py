"""Estado de las sesiones conversacionales: historial y memoria del proyecto.

Dos conceptos separados a propósito:

- Historial (ConversationHistory): los mensajes user/assistant que viajan al LLM en cada
  llamada. Usa una ventana deslizante: solo se envían los últimos N turnos.
- Memoria (ProjectMetadata): los hechos del proyecto en curso (nombre, equipo, tecnologías,
  alcance). Se inyecta en el system prompt en cada turno, así que se conserva aunque los
  turnos donde apareció ya hayan salido de la ventana.

Las sesiones viven en un diccionario en memoria del proceso: sin base de datos ni Redis.
Se acepta esa volatilidad en esta fase porque simplifica el ejercicio, y sus costes son
conocidos: las sesiones se pierden al reiniciar el servicio y no se comparten entre varios
procesos o réplicas. Para producción habría que llevarlas a Redis o a una base de datos.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock

from pydantic import BaseModel, Field

from app.config import get_settings
from app.schemas.estimation import EstimationResult


class ProjectMetadata(BaseModel):
    """Hechos conocidos del proyecto en curso (la "memoria" de la sesión)."""

    project_name: str | None = Field(default=None, description="Nombre del proyecto")
    assumed_team_size: int | None = Field(default=None, description="Personas en el equipo")
    mentioned_technologies: list[str] = Field(
        default_factory=list, description="Tecnologías mencionadas o asumidas"
    )
    agreed_scope: str | None = Field(default=None, description="Alcance acordado hasta ahora")

    def is_empty(self) -> bool:
        return self == ProjectMetadata()

    def updated_with(self, result: EstimationResult) -> "ProjectMetadata":
        """Memoria actualizada con los hechos de la última estimación.

        Se extraen de la salida estructurada (validada), sin una llamada extra al LLM. Cada
        campo refleja el estado más reciente: el modelo recibe la memoria anterior en el system
        prompt y la refina, así que su última lista de tecnologías ya está depurada. Sumar las
        listas de todos los turnos dejaría suposiciones superadas ("PostgreSQL o MySQL" junto
        a "PostgreSQL" una vez confirmado). Una estimación "fuera de alcance" no aporta hechos
        fiables y no cambia la memoria.
        """
        if result.out_of_scope:
            return self
        return ProjectMetadata(
            project_name=result.project_name,
            assumed_team_size=result.team_size or self.assumed_team_size,
            mentioned_technologies=result.technologies or self.mentioned_technologies,
            agreed_scope=result.summary,
        )


class ConversationHistory:
    """Historial con ventana deslizante: como mucho `max_turns` turnos (par user + assistant).

    El system prompt no se guarda como un mensaje más: siempre va el primero, pero se regenera
    en cada llamada a partir de la memoria actual del proyecto (ver to_messages_list), así que
    la ventana nunca lo descarta y siempre refleja los últimos hechos conocidos.
    """

    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self._turns: list[tuple[str, str]] = []
        self.total_turns = 0  # incluidos los que ya han salido de la ventana

    def add_turn(self, user: str, assistant: str) -> None:
        self._turns.append((user, assistant))
        self.total_turns += 1
        if len(self._turns) > self.max_turns:
            # Se descartan los pares más antiguos
            self._turns = self._turns[-self.max_turns :]

    def messages(self) -> list[dict]:
        """Los turnos de la ventana como mensajes, sin system prompt."""
        return [
            message
            for user, assistant in self._turns
            for message in (
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            )
        ]

    def to_messages_list(
        self,
        project_metadata: "ProjectMetadata",
        build_system: Callable[["ProjectMetadata"], str],
    ) -> list[dict]:
        """El array messages listo para el LLM: el system prompt, regenerado a partir del
        project_metadata actual, seguido de los turnos de la ventana.

        `build_system` construye el system prompt a partir de la memoria (renderiza la
        plantilla con las opciones del turno: tipo, nivel de detalle y formato).
        """
        system = build_system(project_metadata)
        return [{"role": "system", "content": system}, *self.messages()]

    def __len__(self) -> int:
        return len(self._turns)


@dataclass
class Session:
    id: str
    history: ConversationHistory
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class SessionNotFound(Exception):
    pass


class SessionStore:
    """Sesiones en memoria del proceso, indexadas por session_id."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()

    def create(self) -> Session:
        session = Session(
            id=str(uuid.uuid4()),
            history=ConversationHistory(max_turns=get_settings().session_max_turns),
        )
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionNotFound(session_id) from None

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()


store = SessionStore()
