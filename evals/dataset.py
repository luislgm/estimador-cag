"""El golden dataset: casos de referencia con lo que se espera de cada estimación.

Cada caso es una conversación de uno o más turnos (se envían a una sesión, como haría un
usuario) y las expectativas sobre la respuesta del último turno:

- hours: rango [mínimo, máximo] de horas totales razonable para ese proyecto. Es amplio a
  propósito: no se busca la cifra exacta, sino detectar estimaciones disparatadas.
- must_mention: conceptos que la estimación debe cubrir. Cada concepto es una lista de
  sinónimos; basta con que aparezca uno (sin distinguir mayúsculas ni tildes).
- out_of_scope: si la petición no es un proyecto de software y debe rechazarse.
- tier: la audiencia que debe deducirse del mensaje.
- change: en conversaciones de varios turnos, si las horas deben subir o bajar.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType
from app.tiers import Tier

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


class Expected(BaseModel):
    out_of_scope: bool = False
    hours: tuple[int, int] | None = None
    must_mention: list[list[str]] = Field(default_factory=list)
    tier: Tier | None = None
    change: Literal["up", "down"] | None = None

    @model_validator(mode="after")
    def check(self) -> "Expected":
        if self.hours and self.hours[0] > self.hours[1]:
            raise ValueError("hours: el mínimo no puede ser mayor que el máximo")
        if self.out_of_scope and (self.hours or self.must_mention):
            raise ValueError("un caso fuera de alcance no tiene horas ni conceptos esperados")
        if any(not synonyms for synonyms in self.must_mention):
            raise ValueError("must_mention: cada concepto necesita al menos un sinónimo")
        return self


class EvalCase(BaseModel):
    id: str
    turns: list[str] = Field(..., min_length=1)
    project_type: ProjectType = ProjectType.WEB_SAAS
    detail_level: DetailLevel = DetailLevel.MEDIUM
    output_format: OutputFormat = OutputFormat.PHASES_TABLE
    expected: Expected

    @model_validator(mode="after")
    def check(self) -> "EvalCase":
        if self.expected.change and len(self.turns) < 2:
            raise ValueError(f"{self.id}: 'change' necesita al menos dos turnos")
        return self


def load_dataset(path: Path = DATASET_PATH) -> list[EvalCase]:
    cases = [EvalCase.model_validate(item) for item in json.loads(path.read_text("utf-8"))]
    ids = [case.id for case in cases]
    duplicated = {i for i in ids if ids.count(i) > 1}
    if duplicated:
        raise ValueError(f"ids repetidos en el dataset: {sorted(duplicated)}")
    return cases
