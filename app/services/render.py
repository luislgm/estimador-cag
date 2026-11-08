"""Convierte una estimación estructurada en markdown, según el formato pedido.

El LLM devuelve siempre la misma estructura (EstimationResult); el formato de salida es
cuestión de presentación y se resuelve aquí, de forma determinista:

- phases_table: tabla por fases.
- line_items:   lista numerada de fases con horas y coste.
- narrative:    un párrafo por fase, sin tablas ni listas.
"""

from app.schemas.estimation import (
    HOURLY_RATE_EUR,
    DetailLevel,
    EstimationRequest,
    EstimationResult,
    OutputFormat,
    Phase,
)


def fmt_int(value: int) -> str:
    """Número con punto como separador de miles: 13000 -> '13.000'."""
    return f"{value:,}".replace(",", ".")


def _weeks(n: int) -> str:
    return f"{n} semana" if n == 1 else f"{n} semanas"


def _assumptions(phase: Phase, indent: str = "") -> list[str]:
    return [f"{indent}- *Asunción:* {a}" for a in phase.assumptions]


def _phases_table(result: EstimationResult, show_assumptions: bool) -> list[str]:
    lines = ["| Fase | Semanas | Horas | Coste (€) |", "|---|---|---|---|"]
    lines += [
        f"| {p.name} | {p.duration_weeks} | {fmt_int(p.hours)} | {fmt_int(p.cost_eur)} |"
        for p in result.phases
    ]
    lines.append(
        f"| **Total** | **{result.total_duration_weeks}** | **{fmt_int(result.total_hours)}** "
        f"| **{fmt_int(result.total_cost_eur)}** |"
    )
    if show_assumptions and any(p.assumptions for p in result.phases):
        lines += ["", "### Asunciones por fase"]
        for p in result.phases:
            if p.assumptions:
                lines += ["", f"**{p.name}**", *_assumptions(p)]
    return lines


def _line_items(result: EstimationResult, show_assumptions: bool) -> list[str]:
    lines = []
    for i, p in enumerate(result.phases, start=1):
        lines.append(
            f"{i}. **{p.name}**: {fmt_int(p.hours)} horas — {fmt_int(p.cost_eur)} € "
            f"({_weeks(p.duration_weeks)}). {p.summary}"
        )
        if show_assumptions:
            lines += _assumptions(p, indent="   ")
    return lines


def _narrative(result: EstimationResult, show_assumptions: bool) -> list[str]:
    paragraphs = []
    for p in result.phases:
        text = (
            f"**{p.name}.** {p.summary} Esta fase supone {fmt_int(p.hours)} horas "
            f"({fmt_int(p.cost_eur)} €) y dura {_weeks(p.duration_weeks)}."
        )
        if show_assumptions and p.assumptions:
            text += " Se asume que " + "; que ".join(a.rstrip(".") for a in p.assumptions) + "."
        paragraphs += [text, ""]
    return paragraphs[:-1]


_RENDERERS = {
    OutputFormat.PHASES_TABLE: _phases_table,
    OutputFormat.LINE_ITEMS: _line_items,
    OutputFormat.NARRATIVE: _narrative,
}


def render_markdown(result: EstimationResult, request: EstimationRequest) -> str:
    """La estimación en markdown, con el formato y el nivel de detalle de la petición."""
    lines = [f"## Estimación: {result.project_name}", "", result.summary, ""]
    if result.out_of_scope:
        lines.append(f"*Confianza: {result.confidence_pct} %. No se ha estimado el proyecto.*")
        return "\n".join(lines)

    show_assumptions = request.detail_level == DetailLevel.DETAILED
    lines += _RENDERERS[request.output_format](result, show_assumptions)
    lines += [
        "",
        f"**Total estimado: {fmt_int(result.total_hours)} horas — "
        f"{fmt_int(result.total_cost_eur)} € (tarifa {HOURLY_RATE_EUR} €/h)**",
    ]
    if result.team:
        lines.append(f"**Equipo recomendado:** {', '.join(result.team)}")
    lines.append(f"**Duración estimada:** {_weeks(result.total_duration_weeks)}")
    lines.append(f"**Confianza:** {result.confidence_pct} %")
    if result.risks:
        lines += ["", "### Riesgos", *[f"- {r}" for r in result.risks]]
    return "\n".join(lines)
