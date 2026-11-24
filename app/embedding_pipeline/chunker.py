"""Chunker estructural para presupuestos JSON: un componente = un chunk.

No parte por tamaño ni solapa: confía en la estructura del documento. Un componente es una
unidad con sentido propio (qué se hace, con qué tecnologías, cuántas horas), y es justo lo
que querremos recuperar al estimar un proyecto parecido.

El texto de cada chunk empieza con el contexto del presupuesto padre (contextual chunk
header): sin él, un componente "Authentication backend" perdería a qué proyecto, sector y
año pertenece, y dos componentes iguales de proyectos muy distintos serían indistinguibles.

Los campos filtrables (sector, tecnología, año, complejidad, horas) van en metadata y no se
embeben: sirven para filtrar en la búsqueda, no para medir parecido semántico.
"""

from functools import lru_cache

import tiktoken

from app.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk

EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_TEMPLATE = """[Project: {project_summary}]
[Client sector: {sector} | Year: {year} | Main tech: {main_technology}]

Component: {name}
Description: {description}
Tech stack: {tech_stack}
Complexity: {complexity}
Estimated hours: {hours}"""


@lru_cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


class JSONStructuralChunker:
    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        return [
            self._chunk(budget, component) for budget in budgets for component in budget.components
        ]

    def _chunk(self, budget: Budget, component: BudgetComponent) -> Chunk:
        text = CHUNK_TEMPLATE.format(
            project_summary=budget.project_summary,
            sector=budget.client_metadata.sector,
            year=budget.year,
            main_technology=budget.main_technology,
            name=component.name,
            description=component.description,
            tech_stack=", ".join(component.tech_stack),
            complexity=component.complexity,
            hours=component.estimated_hours,
        )
        return Chunk(
            chunk_id=f"{budget.budget_id}::{component.component_id}",
            text=text,
            metadata={
                "budget_id": budget.budget_id,
                "component_id": component.component_id,
                "client_sector": budget.client_metadata.sector,
                "main_technology": budget.main_technology,
                "year": budget.year,
                "complexity": component.complexity,
                "estimated_hours": component.estimated_hours,
            },
            token_count=count_tokens(text),
        )
