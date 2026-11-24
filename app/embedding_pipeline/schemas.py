"""Contratos del pipeline de embeddings (Pydantic v2).

El esquema de entrada es el de los presupuestos históricos normalizados del programa
(data/budgets_sample.json). Los validadores rechazan con un 422 lo que no tendría sentido
embeber: un sector desconocido, unas horas totales que no cuadran con los componentes o
una dependencia a un componente que no existe en el presupuesto.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Sector = Literal["finance", "ecommerce", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]


class ClientMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    sector: Sector
    country: str = Field(..., pattern=r"^[A-Z]{2}$", description="Código ISO 3166-1 alfa-2")


class BudgetComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    component_id: str = Field(..., pattern=r"^[A-Z0-9]+-\d{3}$", examples=["AUTH-001"])
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    tech_stack: list[str] = Field(default_factory=list)
    estimated_hours: int = Field(..., ge=0)
    complexity: Complexity
    dependencies: list[str] = Field(default_factory=list)


class Budget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budget_id: str = Field(..., pattern=r"^BUD-\d{4}-\d{3}$", examples=["BUD-2024-014"])
    client_metadata: ClientMetadata
    project_summary: str = Field(..., min_length=1)
    main_technology: str = Field(..., min_length=1)
    year: int = Field(..., ge=2000, le=2100)
    total_estimated_hours: int = Field(..., ge=0)
    components: list[BudgetComponent] = Field(..., min_length=1)

    @model_validator(mode="after")
    def consistent(self) -> "Budget":
        ids = [c.component_id for c in self.components]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.budget_id}: component_id repetidos")
        missing = {d for c in self.components for d in c.dependencies} - set(ids)
        if missing:
            raise ValueError(
                f"{self.budget_id}: dependencias a componentes inexistentes: {sorted(missing)}"
            )
        hours = sum(c.estimated_hours for c in self.components)
        if hours != self.total_estimated_hours:
            raise ValueError(
                f"{self.budget_id}: total_estimated_hours es {self.total_estimated_hours} pero "
                f"los componentes suman {hours}"
            )
        return self


class Chunk(BaseModel):
    chunk_id: str = Field(..., description="{budget_id}::{component_id}, trazable al origen")
    text: str = Field(..., description="Lo que se embebe: contexto del presupuesto + componente")
    metadata: dict[str, Any] = Field(..., description="Campos filtrables; no se embeben")
    token_count: int = Field(..., ge=0, description="Tokens del texto (tiktoken)")


class EmbeddedChunk(Chunk):
    embedding: list[float]


class IngestRequest(BaseModel):
    budgets: list[Budget] = Field(..., min_length=1)

    @model_validator(mode="after")
    def unique_budgets(self) -> "IngestRequest":
        ids = [b.budget_id for b in self.budgets]
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        if duplicated:
            raise ValueError(f"budget_id repetidos: {duplicated}")
        return self


class IngestStats(BaseModel):
    total_budgets: int
    total_chunks: int
    total_tokens: int
    estimated_cost_usd: float


class IngestResponse(BaseModel):
    chunks: list[EmbeddedChunk]
    stats: IngestStats
