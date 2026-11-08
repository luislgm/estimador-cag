"""Cliente Streamlit del estimador CAG: conversación por sesiones, con memoria y adjuntos.

Al cargar la página crea una sesión en el servicio IA (POST /api/v1/sessions). Cada envío del
formulario es un turno (POST /api/v1/sessions/{id}/estimate, multipart): la transcripción,
los adjuntos (PDF o Word) y las opciones. El servicio recuerda el proyecto entre turnos; el
panel lateral muestra esa memoria (project_metadata), separada del historial.

Streamlit no llama al LLM: solo habla con la API, cuya URL se lee de ESTIMATOR_API_URL.

Ejecutar con:  uv run --group ui streamlit run streamlit_app.py
"""

import os
import time

import httpx
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from app.prompts import PROMPT_VERSION, render_estimation_prompt, render_examples
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType
from app.sessions import ProjectMetadata

load_dotenv()
API_URL = os.getenv("ESTIMATOR_API_URL", "http://127.0.0.1:8000").rstrip("/")
SESSIONS_URL = f"{API_URL}/api/v1/sessions"

PROJECT_TYPE_LABELS = {
    ProjectType.MOBILE_APP: "App móvil",
    ProjectType.WEB_SAAS: "Web SaaS",
    ProjectType.INTERNAL_TOOL: "Herramienta interna",
    ProjectType.DATA_PIPELINE: "Pipeline de datos",
}
DETAIL_LEVEL_LABELS = {
    DetailLevel.SUMMARY: "Resumen",
    DetailLevel.MEDIUM: "Medio",
    DetailLevel.DETAILED: "Detallado",
}
OUTPUT_FORMAT_LABELS = {
    OutputFormat.PHASES_TABLE: "Tabla por fases",
    OutputFormat.LINE_ITEMS: "Desglose de tareas",
    OutputFormat.NARRATIVE: "Narrativo",
}
TIMEOUT = httpx.Timeout(180.0, connect=5.0)


def fmt_int(value: int) -> str:
    """Número entero con punto como separador de miles: 1746 -> '1.746'."""
    return f"{value:,}".replace(",", ".")


def fmt_seconds(value: float) -> str:
    """Segundos con coma decimal: 7.34 -> '7,3 s'."""
    return f"{value:.1f}".replace(".", ",") + " s"


def fmt_cost(value: float | None) -> str:
    """Coste en USD con coma decimal: 0.00527 -> '0,0053 $'. Sin precio conocido -> 'n/d'."""
    return "n/d" if value is None else f"{value:.4f}".replace(".", ",") + " $"


def params_label(request: dict) -> str:
    return " · ".join(
        [
            PROJECT_TYPE_LABELS[ProjectType(request["project_type"])],
            DETAIL_LEVEL_LABELS[DetailLevel(request["detail_level"])],
            OUTPUT_FORMAT_LABELS[OutputFormat(request["output_format"])],
        ]
    )


@st.cache_data(ttl=30, show_spinner=False)
def api_health() -> dict | None:
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).json()
    except httpx.HTTPError:
        return None


def new_session() -> None:
    """Crea una sesión en la API y reinicia la conversación en la página."""
    st.session_state.turns = []
    st.session_state.metadata = ProjectMetadata().model_dump()
    st.session_state.window = None
    try:
        response = httpx.post(SESSIONS_URL, timeout=TIMEOUT)
        st.session_state.session_id = response.json()["session_id"]
    except httpx.HTTPError:
        st.session_state.session_id = None


def send_turn(request: EstimationRequest, files) -> tuple[dict | None, str | None, float]:
    """Envía un turno. Devuelve (respuesta, mensaje de error, segundos)."""
    start = time.perf_counter()
    response = httpx.post(
        f"{SESSIONS_URL}/{st.session_state.session_id}/estimate",
        data={
            "transcript": request.description,
            "project_type": request.project_type.value,
            "detail_level": request.detail_level.value,
            "output_format": request.output_format.value,
        },
        files=[("attachments", (f.name, f.getvalue(), f.type)) for f in files] or None,
        timeout=TIMEOUT,
    )
    seconds = time.perf_counter() - start
    if response.status_code == 200:
        return response.json(), None, seconds
    detail = response.json().get("detail") if response.content else None
    if response.status_code == 404:
        return None, "session_lost", seconds
    if response.status_code == 400 and isinstance(detail, dict):
        return None, f"⛔ Descripción rechazada: {detail['message']}", seconds
    return None, f"La API respondió {response.status_code}: {detail or response.text}", seconds


def render_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.markdown(turn["request"]["description"])
        if turn["attachments"]:
            st.caption("📎 " + ", ".join(a["filename"] for a in turn["attachments"]))
        st.caption(params_label(turn["request"]))
    with st.chat_message("assistant"):
        r = turn["response"]
        if r["result"]["summary"].startswith("Fuera de alcance:"):
            st.warning(
                "El modelo no ha podido estimar con suficiente confianza: falta información. "
                "Añade más detalle en el siguiente mensaje."
            )
        st.markdown(r["text"])
        with st.expander("Ver estimación estructurada (JSON)"):
            st.json(r["result"], expanded=False)
        st.caption(
            f"Turno {r['session']['turns']} · {r['model']} · "
            f"{fmt_int(r['usage']['input_tokens'] + r['usage']['output_tokens'])} tokens · "
            f"{fmt_seconds(turn['seconds'])} · {fmt_cost(r['cost_usd'])}"
        )


# --- Página ---
st.set_page_config(page_title="Estimador CAG", page_icon="📋")
st.title("📋 Estimador CAG")
st.caption(
    "Describe el proyecto y ve refinándolo en varios mensajes: el estimador recuerda de qué "
    "proyecto habláis. Puedes adjuntar PDFs o documentos Word con especificaciones."
)

if "session_id" not in st.session_state:
    new_session()

with st.sidebar:
    st.header("Servicio IA")
    health = api_health()
    if health:
        st.caption(f"✅ {API_URL} · **{health['provider']}** · **{health['model']}**")
    else:
        st.error(f"No se puede conectar con la API en {API_URL}")
    if st.button("Nueva conversación", width="stretch"):
        new_session()
    if st.session_state.session_id:
        st.caption(f"Sesión `{st.session_state.session_id[:8]}`")

    st.subheader("🧠 Memoria del proyecto")
    memory_box = st.container()  # se rellena al final, con el estado tras el último turno

    st.subheader("📊 Última llamada")
    metrics_box = st.container()

    st.subheader(f"📝 System prompt ({PROMPT_VERSION})")
    prompt_box = st.container()

    examples = render_examples(PROMPT_VERSION)
    st.subheader(f"📚 Contexto inyectado ({examples.count('<ejemplo numero=')} estimaciones)")
    with st.expander("Ver estimaciones de ejemplo (examples.j2)"):
        st.code(examples, language="markdown", wrap_lines=True)

for turn in st.session_state.turns:
    render_turn(turn)

with st.form("turn_form", clear_on_submit=True):
    first = not st.session_state.turns
    description = st.text_area(
        "Descripción del proyecto" if first else "Siguiente mensaje",
        height=150,
        placeholder=(
            "Objetivos, funcionalidades, usuarios, integraciones, plazos…"
            if first
            else "Añade información, cambia el alcance o pide que ajuste la estimación…"
        ),
        help="Entre 20 y 2.000 caracteres.",
    )
    attachments = st.file_uploader(
        "Adjuntos (PDF o Word)", type=["pdf", "docx"], accept_multiple_files=True
    )
    col1, col2 = st.columns(2)
    project_type = col1.selectbox(
        "Tipo de proyecto", list(ProjectType), format_func=PROJECT_TYPE_LABELS.get
    )
    output_format = col2.selectbox(
        "Formato", list(OutputFormat), format_func=OUTPUT_FORMAT_LABELS.get
    )
    detail_level = st.radio(
        "Nivel de detalle",
        list(DetailLevel),
        index=1,
        format_func=DETAIL_LEVEL_LABELS.get,
        horizontal=True,
    )
    submitted = st.form_submit_button("Enviar", type="primary")

if submitted:
    try:
        # Mismas reglas de validación que la API
        request = EstimationRequest(
            description=description,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
        )
    except ValidationError as e:
        st.error("Revisa el formulario: " + "; ".join(err["msg"] for err in e.errors()), icon="⚠️")
    else:
        if not st.session_state.session_id:
            new_session()
        try:
            with st.spinner("Generando y validando la estimación…"):
                response, error, seconds = send_turn(request, attachments or [])
        except httpx.HTTPError as e:
            st.error(f"No se puede conectar con la API en {API_URL}: {e}")
        else:
            if error == "session_lost":
                new_session()
                st.warning(
                    "La sesión ya no existe (el servicio se ha reiniciado). He empezado una "
                    "conversación nueva: vuelve a enviar el mensaje."
                )
            elif error:
                st.error(error)
            else:
                st.session_state.turns.append(
                    {
                        "request": request.model_dump(mode="json"),
                        "attachments": response["attachments"],
                        "response": response,
                        "seconds": seconds,
                    }
                )
                st.session_state.metadata = response["session"]["project_metadata"]
                st.session_state.window = response["session"]
                st.rerun()  # para mostrar el turno en la conversación, encima del formulario

metadata = ProjectMetadata(**st.session_state.metadata)
with memory_box:
    if metadata.is_empty():
        st.caption("Vacía: se rellena con la primera estimación.")
    else:
        st.markdown(f"**Proyecto:** {metadata.project_name or '—'}")
        st.markdown(f"**Equipo asumido:** {metadata.assumed_team_size or '—'} personas")
        st.markdown("**Tecnologías:** " + (", ".join(metadata.mentioned_technologies) or "—"))
        with st.expander("Alcance acordado"):
            st.write(metadata.agreed_scope or "—")
    window = st.session_state.window
    if window:
        st.caption(
            f"Historial: {window['turns']} turno(s); se envían al LLM los últimos "
            f"{window['turns_in_window']} de un máximo de {window['max_turns']}."
        )

with metrics_box:
    if not st.session_state.turns:
        st.caption("Aún no se ha hecho ninguna estimación.")
    else:
        last = st.session_state.turns[-1]
        r = last["response"]
        c1, c2 = st.columns(2)
        c1.metric("Tokens entrada", fmt_int(r["usage"]["input_tokens"]))
        c2.metric("Tokens salida", fmt_int(r["usage"]["output_tokens"]))
        c1.metric("Tiempo", fmt_seconds(last["seconds"]))
        c2.metric("Confianza", f"{r['result']['confidence_pct']} %")
        c1.metric("Coste", fmt_cost(r["cost_usd"]))

with prompt_box:
    # El system prompt del siguiente turno, con la memoria actual del proyecto
    preview = EstimationRequest(
        description="(aquí irá el siguiente mensaje)",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.PHASES_TABLE,
    )
    system_prompt, _ = render_estimation_prompt(preview, PROMPT_VERSION, project_metadata=metadata)
    st.caption("El del siguiente turno, con la memoria actual (opciones por defecto).")
    with st.expander(f"Ver ({fmt_int(len(system_prompt))} caracteres)"):
        st.code(system_prompt, language="markdown", wrap_lines=True)
