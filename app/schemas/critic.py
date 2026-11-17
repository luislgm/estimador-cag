"""Contratos del ciclo Actor-Critic-Boss: la revisión del crítico y la traza del ciclo."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Severity = Literal["alta", "media", "baja"]
SERIOUS: tuple[Severity, ...] = ("alta", "media")


class CriticIssue(BaseModel):
    """Un problema concreto de la estimación, con lo que habría que cambiar."""

    category: Literal["alcance", "horas", "fases", "equipo", "riesgos", "enfoque"] = Field(
        ..., description="Qué parte de la estimación falla"
    )
    severity: Severity = Field(
        ...,
        description=(
            "alta: cambia el total más de un 25 % o falta algo que se pidió; media: se nota pero "
            "es acotado; baja: detalle menor"
        ),
    )
    problem: str = Field(..., min_length=5, max_length=400, description="Qué está mal y dónde")
    suggestion: str = Field(..., min_length=5, max_length=400, description="Qué cambiar")


class CriticReview(BaseModel):
    """Salida estructurada del crítico."""

    verdict: Literal["accept", "revise"] = Field(
        ..., description="accept si no hay problemas de gravedad alta o media; revise si los hay"
    )
    issues: list[CriticIssue] = Field(default_factory=list, max_length=8)
    confidence_pct: int = Field(..., ge=0, le=100, description="Confianza en la propia revisión")

    @model_validator(mode="after")
    def revise_needs_serious_issues(self) -> "CriticReview":
        if self.verdict == "revise" and not self.serious_issues:
            raise ValueError(
                "con verdict 'revise' tiene que haber al menos un problema de gravedad alta o "
                "media; si solo hay detalles menores, el veredicto es 'accept'"
            )
        return self

    @property
    def serious_issues(self) -> list[CriticIssue]:
        return [issue for issue in self.issues if issue.severity in SERIOUS]


class ACBIteration(BaseModel):
    """Una vuelta del ciclo: la estimación del actor y lo que dijo el crítico."""

    iteration: int = Field(..., ge=1)
    total_hours: int
    verdict: Literal["accept", "revise"] | None = Field(
        None, description="Veredicto del crítico (None si la revisión falló)"
    )
    issues: list[CriticIssue] = Field(default_factory=list)
    decision: Literal["accept", "revise", "stop"] = Field(
        ..., description="Lo que decidió el boss tras esta vuelta"
    )


class ACBTrace(BaseModel):
    """Traza completa del ciclo, para ver por qué la respuesta es la que es."""

    iterations: list[ACBIteration]
    outcome: Literal["accepted", "max_iterations", "critic_failed", "revision_failed"] = Field(
        ...,
        description=(
            "accepted: el crítico aceptó; max_iterations: se agotaron las vueltas; critic_failed: "
            "la revisión falló y se usa la estimación sin revisar; revision_failed: la nueva "
            "versión del actor falló y se usa la mejor anterior"
        ),
    )
    chosen_iteration: int = Field(..., ge=1, description="Vuelta cuya estimación se devuelve")
    llm_calls: int = Field(..., description="Llamadas al LLM del turno (actor + crítico)")
