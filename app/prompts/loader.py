"""Carga y renderiza las plantillas de prompt versionadas (Jinja2).

Estructura en disco: app/prompts/<caso_de_uso>/<versión>/<rol>.j2

    estimation/v1/system.j2    rol, instrucciones y bloques según el formulario
    estimation/v1/user.j2      la descripción del proyecto
    estimation/v1/examples.j2  estimaciones de ejemplo (few-shot), incluidas en system.j2

Para crear una versión nueva se copia la carpeta (v1 -> v2), se edita y se pasa
version="v2": el resto del código no cambia.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.schemas.estimation import EstimationRequest

PROMPT_VERSION = "v1"  # versión activa en el servicio

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent),
    undefined=StrictUndefined,  # una variable que falte es un error, no un texto vacío
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
)


def render_estimation_prompt(request: EstimationRequest, version: str = "v1") -> tuple[str, str]:
    """Devuelve (system, user) listos para enviar al modelo."""
    context = {
        "version": version,
        "description": request.description,
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
    }
    system = _env.get_template(f"estimation/{version}/system.j2").render(context)
    user = _env.get_template(f"estimation/{version}/user.j2").render(context)
    return system.strip(), user.strip()


def render_examples(version: str = "v1") -> str:
    """Las estimaciones de ejemplo (few-shot) de una versión, tal como se inyectan."""
    return _env.get_template(f"estimation/{version}/examples.j2").render().strip()
