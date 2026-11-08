from app.services.render import fmt_int, render_markdown
from tests.factories import make_phase, make_request, make_result


def test_fmt_int():
    assert fmt_int(13000) == "13.000"


def test_phases_table():
    text = render_markdown(make_result(), make_request(output_format="phases_table"))

    assert text.startswith("## Estimación: Reservas para restaurante")
    assert "| Fase | Semanas | Horas | Coste (€) |" in text
    assert "| Diseño | 2 | 20 | 1.000 |" in text
    assert "| **Total** | **4** | **80** | **4.000** |" in text
    assert "**Total estimado: 80 horas — 4.000 € (tarifa 50 €/h)**" in text
    assert "**Equipo recomendado:** 1 desarrollador full-stack" in text
    assert "### Riesgos" in text


def test_line_items():
    text = render_markdown(make_result(), make_request(output_format="line_items"))

    assert "1. **Diseño**: 20 horas — 1.000 € (2 semanas)." in text
    assert "2. **Desarrollo**: 60 horas — 3.000 €" in text
    assert "| Fase |" not in text


def test_narrative_has_no_tables_or_lists():
    text = render_markdown(make_result(), make_request(output_format="narrative"))

    assert "**Diseño.** Trabajo de diseño. Esta fase supone 20 horas (1.000 €)" in text
    assert "| Fase |" not in text
    assert "\n1. " not in text


def test_assumptions_only_in_detailed():
    phases = [make_phase("Diseño", 20, assumptions=["El cliente aporta su manual de marca"])]
    result = make_result(phases=phases)

    detailed = render_markdown(result, make_request(detail_level="detailed"))
    medium = render_markdown(result, make_request(detail_level="medium"))

    assert "El cliente aporta su manual de marca" in detailed
    assert "El cliente aporta su manual de marca" not in medium


def test_assumptions_in_every_format():
    phases = [make_phase("Diseño", 20, assumptions=["Hay manual de marca"])]
    result = make_result(phases=phases)

    for output_format in ("phases_table", "line_items", "narrative"):
        request = make_request(detail_level="detailed", output_format=output_format)
        assert "Hay manual de marca" in render_markdown(result, request), output_format


def test_out_of_scope_shows_no_figures():
    result = make_result(
        confidence_pct=10,
        summary="Fuera de alcance: falta saber qué funcionalidades quieren.",
        phases=[make_phase("Sin estimar", 0)],
    )

    text = render_markdown(result, make_request())

    assert "Fuera de alcance:" in text
    assert "No se ha estimado el proyecto" in text
    assert "| Fase |" not in text
