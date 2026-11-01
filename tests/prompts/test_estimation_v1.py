import re

import pytest
from jinja2 import StrictUndefined, TemplateNotFound, UndefinedError

from app.prompts import loader
from app.prompts.loader import PROMPT_VERSION, render_estimation_prompt, render_examples
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType

DESCRIPTION = "Una gestoría quiere un portal para que sus clientes suban facturas."

# Una frase característica de cada bloque condicional de system.j2
OUTPUT_FORMAT_MARKERS = {
    OutputFormat.PHASES_TABLE: "tabla markdown por fases",
    OutputFormat.LINE_ITEMS: "lista numerada de tareas",
    OutputFormat.NARRATIVE: "texto en párrafos",
}
DETAIL_LEVEL_MARKERS = {
    DetailLevel.SUMMARY: "3-5 bloques de trabajo",
    DetailLevel.MEDIUM: "6-10 partes",
    DetailLevel.DETAILED: "10 o más partes",
}
PROJECT_TYPE_MARKERS = {
    ProjectType.MOBILE_APP: "Es una app móvil",
    ProjectType.WEB_SAAS: "Es una aplicación web multiusuario",
    ProjectType.INTERNAL_TOOL: "Es una herramienta interna",
    ProjectType.DATA_PIPELINE: "Es un pipeline de datos",
}


def make_request(**changes) -> EstimationRequest:
    fields = {
        "description": DESCRIPTION,
        "project_type": "web_saas",
        "detail_level": "medium",
        "output_format": "phases_table",
    }
    return EstimationRequest(**fields | changes)


# --- Los tres tests que pide la Parte 4 ---

# Palabra clave del formato phases_table: las columnas de la tabla que se piden al modelo
PHASES_TABLE_KEYWORD = "| Fase | Semanas | Horas | Coste (€) |"
# Instrucción extra del nivel detailed
ASSUMPTIONS_PER_PHASE = "Para cada fase, lista las asunciones de la fase"


def test_render_includes_description_inside_its_block():
    description = "Un gimnasio quiere una app para reservar clases y pagar la cuota mensual."

    _, user = render_estimation_prompt(make_request(description=description), version="v1")

    block = re.search(r"<descripcion_proyecto>\n(.*?)\n</descripcion_proyecto>", user, re.DOTALL)
    assert block is not None
    assert block.group(1) == description  # literal, sin cambios


def test_phases_table_keyword_only_with_phases_table_format():
    with_table, _ = render_estimation_prompt(make_request(output_format="phases_table"))
    narrative, _ = render_estimation_prompt(make_request(output_format="narrative"))

    assert PHASES_TABLE_KEYWORD in with_table
    assert PHASES_TABLE_KEYWORD not in narrative


def test_assumptions_per_phase_only_with_detailed_level():
    detailed, _ = render_estimation_prompt(make_request(detail_level="detailed"))
    summary, _ = render_estimation_prompt(make_request(detail_level="summary"))

    assert ASSUMPTIONS_PER_PHASE in detailed
    assert ASSUMPTIONS_PER_PHASE not in summary


# --- Resto de comprobaciones del loader y las plantillas ---


def test_environment_configuration():
    assert loader._env.undefined is StrictUndefined
    assert loader._env.trim_blocks is True
    assert loader._env.lstrip_blocks is True


def test_returns_system_and_user():
    system, user = render_estimation_prompt(make_request(), version="v1")

    assert system.startswith("Eres un estimador de proyectos de software experto")
    assert user.startswith("Estima el siguiente proyecto")
    assert PROMPT_VERSION == "v1"


@pytest.mark.parametrize("output_format", list(OutputFormat))
def test_only_the_chosen_output_format_block_is_included(output_format):
    system, _ = render_estimation_prompt(make_request(output_format=output_format))

    for option, marker in OUTPUT_FORMAT_MARKERS.items():
        assert (marker in system) == (option == output_format), option


@pytest.mark.parametrize("detail_level", list(DetailLevel))
def test_only_the_chosen_detail_level_block_is_included(detail_level):
    system, _ = render_estimation_prompt(make_request(detail_level=detail_level))

    for option, marker in DETAIL_LEVEL_MARKERS.items():
        assert (marker in system) == (option == detail_level), option


@pytest.mark.parametrize("project_type", list(ProjectType))
def test_only_the_chosen_project_type_block_is_included(project_type):
    system, _ = render_estimation_prompt(make_request(project_type=project_type))

    for option, marker in PROJECT_TYPE_MARKERS.items():
        assert (marker in system) == (option == project_type), option


def test_system_includes_examples_template():
    system, _ = render_estimation_prompt(make_request())

    assert render_examples("v1") in system  # {% include %} de examples.j2


def test_examples_are_well_formed():
    examples = render_examples("v1")
    blocks = re.findall(r"<estimacion>(.*?)</estimacion>", examples, re.DOTALL)

    assert 2 <= len(blocks) <= 3
    for block in blocks:
        hours = [int(h) for h in re.findall(r"^\d+\. .*?: (\d+) horas", block, re.MULTILINE)]
        costs = [
            int(c.replace(".", ""))
            for c in re.findall(r"^\d+\. .*?horas — ([\d.]+) €", block, re.MULTILINE)
        ]
        total_hours, total_cost = re.search(
            r"Total estimado: (\d+) horas — ([\d.]+) €", block
        ).groups()
        assert sum(hours) == int(total_hours)  # las horas suman el total
        assert sum(costs) == int(total_cost.replace(".", ""))  # y los costes también
        assert all(c == h * 50 for h, c in zip(hours, costs, strict=True))  # a 50 €/h
        assert "Equipo recomendado" in block and "Duración estimada" in block


def test_examples_do_not_copy_the_course_statement():
    assert "Gestión de Inventario" not in render_examples("v1")


def test_user_wraps_the_description():
    _, user = render_estimation_prompt(make_request())

    assert f"<descripcion_proyecto>\n{DESCRIPTION}\n</descripcion_proyecto>" in user


def test_no_leftover_template_syntax():
    system, user = render_estimation_prompt(make_request())

    for text in (system, user):
        assert "{{" not in text and "{%" not in text


def test_version_is_a_parameter(tmp_path, monkeypatch):
    """Cambiar de versión es solo un argumento: una v2 en disco se usa sin tocar código."""
    v2 = tmp_path / "estimation" / "v2"
    v2.mkdir(parents=True)
    (v2 / "system.j2").write_text("system v2: {{ output_format }}", encoding="utf-8")
    (v2 / "user.j2").write_text("user v2: {{ description }}", encoding="utf-8")
    monkeypatch.setattr(loader._env.loader, "searchpath", [str(tmp_path)])

    system, user = render_estimation_prompt(make_request(), version="v2")

    assert system == "system v2: phases_table"
    assert user == f"user v2: {DESCRIPTION}"


def test_unknown_version_fails():
    with pytest.raises(TemplateNotFound):
        render_estimation_prompt(make_request(), version="v99")


def test_strict_undefined_rejects_missing_variables(tmp_path, monkeypatch):
    v3 = tmp_path / "estimation" / "v3"
    v3.mkdir(parents=True)
    (v3 / "system.j2").write_text("{{ variable_que_no_existe }}", encoding="utf-8")
    (v3 / "user.j2").write_text("{{ description }}", encoding="utf-8")
    monkeypatch.setattr(loader._env.loader, "searchpath", [str(tmp_path)])

    with pytest.raises(UndefinedError):
        render_estimation_prompt(make_request(), version="v3")
