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
from app.tiers import TierResolution


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


@dataclass(frozen=True)
class Turn:
    """Un turno de la conversación: el mensaje del usuario y la respuesta del asistente."""

    user: str
    assistant: str

    def as_messages(self) -> list[dict]:
        return [
            {"role": "user", "content": self.user},
            {"role": "assistant", "content": self.assistant},
        ]


class ConversationHistory:
    """Historial con ventana deslizante, más anclas y un resumen de lo que ya salió de ella.

    Tres partes:

    - Ventana: los últimos `max_turns` turnos, completos.
    - Anclas: turnos antiguos que contienen un compromiso que no se puede olvidar (NDA,
      contrato firmado, alcance cerrado, normativa...). Se conservan literales fuera de la
      ventana, como mucho `max_anchors`.
    - Resumen: un texto acumulado con el resto de turnos que ya salieron de la ventana.

    Esta clase solo guarda el estado. Qué turno es un ancla y cómo se resume lo decide la
    política de compresión (app/services/compression.py), que recibe los turnos que expulsa
    add_turn().

    El system prompt no se guarda como un mensaje más: siempre va el primero, pero se regenera
    en cada llamada a partir de la memoria actual del proyecto y del resumen (ver
    to_messages_list), así que nunca se descarta y siempre refleja los últimos hechos.
    """

    def __init__(self, max_turns: int, max_anchors: int = 0):
        self.max_turns = max_turns
        self.max_anchors = max_anchors
        self._turns: list[Turn] = []
        self.anchors: list[Turn] = []
        self.summary: str | None = None
        self.summarized_turns = 0  # turnos absorbidos por el resumen
        self.total_turns = 0  # incluidos los que ya han salido de la ventana

    def add_turn(self, user: str, assistant: str) -> list[Turn]:
        """Añade un turno y devuelve los que salen de la ventana (los más antiguos)."""
        self._turns.append(Turn(user, assistant))
        self.total_turns += 1
        overflow = len(self._turns) - self.max_turns
        if overflow <= 0:
            return []
        evicted, self._turns = self._turns[:overflow], self._turns[overflow:]
        return evicted

    def add_anchor(self, turn: Turn) -> list[Turn]:
        """Conserva un turno como ancla. Devuelve las anclas que ya no caben (las más
        antiguas), para que pasen al resumen: así el prompt no crece sin límite."""
        self.anchors.append(turn)
        overflow = len(self.anchors) - self.max_anchors
        if overflow <= 0:
            return []
        dropped, self.anchors = self.anchors[:overflow], self.anchors[overflow:]
        return dropped

    def messages(self) -> list[dict]:
        """Anclas y turnos de la ventana como mensajes, en orden cronológico, sin system prompt.

        Las anclas son siempre más antiguas que la ventana, así que van delante y se mantiene
        la alternancia user/assistant.
        """
        return [message for turn in (*self.anchors, *self._turns) for message in turn.as_messages()]

    def to_messages_list(
        self,
        project_metadata: "ProjectMetadata",
        build_system: Callable[["ProjectMetadata", str | None], str],
    ) -> list[dict]:
        """El array messages listo para el LLM: el system prompt, regenerado a partir del
        project_metadata actual (y del resumen de los turnos antiguos), seguido de las anclas
        y los turnos de la ventana.

        `build_system(project_metadata, summary)` construye el system prompt (renderiza la
        plantilla con las opciones del turno: tipo, nivel de detalle y formato).
        """
        system = build_system(project_metadata, self.summary)
        return [{"role": "system", "content": system}, *self.messages()]

    def __len__(self) -> int:
        """Turnos en la ventana."""
        return len(self._turns)


class TurnObservation(BaseModel):
    """Todo lo observable de un turno en un único registro (evento turn_observed).

    Junta lo que antes estaba repartido en varios logs (llamada al LLM, compresión, tier...)
    para poder volcar una fila de CSV por turno sin reconciliar eventos ni timestamps.
    """

    turn_index: int = Field(..., description="Número de turno en la sesión (desde 1)")
    session_id: str
    enriched_transcript_chars: int = Field(..., description="Transcripción + texto de adjuntos")
    attachments_total_chars: int = Field(..., description="Texto extraído de los adjuntos")
    messages_in_window: int = Field(..., description="Mensajes de la ventana tras compresión")
    anchors_count: int
    summary_chars: int
    tokens_in: int = Field(..., description="Tokens de entrada del turno (todas sus llamadas)")
    tokens_out: int
    cost_usd: float | None = Field(..., description="Coste del turno (todas sus llamadas)")
    latency_ms: int = Field(..., description="Duración del turno completo en el servicio")
    cache_hit_kind: str = Field(..., description="none | exact | semantic")
    last_resolved_tier: str
    llm_calls: int = Field(..., description="Llamadas al LLM del turno (estimación + resumen)")


@dataclass
class Session:
    id: str
    history: ConversationHistory
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    compression_cost_usd: float = 0.0  # coste acumulado de las llamadas de resumen
    tier: TierResolution | None = None  # audiencia del último turno
    last_turn: TurnObservation | None = None  # observación del último turno
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class SessionNotFound(Exception):
    pass


class SessionStore:
    """Sesiones en memoria del proceso, indexadas por session_id."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()

    def create(self) -> Session:
        settings = get_settings()
        session = Session(
            id=str(uuid.uuid4()),
            history=ConversationHistory(
                max_turns=settings.session_max_turns,
                max_anchors=settings.session_max_anchors,
            ),
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
