"""Cliente Streamlit del estimador CAG.

Un formulario construye un EstimationRequest y lo envía al servicio IA
(POST /api/v1/estimate/stream). La estimación se muestra en streaming según llega.
Streamlit no llama al LLM: solo habla con la API, cuya URL se lee de ESTIMATOR_API_URL.

Ejecutar con:  uv run --group ui streamlit run streamlit_app.py
"""

import json
import os
import time
from collections.abc import Iterator
from datetime import datetime

import httpx
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from app.prompts import PROMPT_VERSION, render_estimation_prompt, render_examples
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType

load_dotenv()
API_URL = os.getenv("ESTIMATOR_API_URL", "http://127.0.0.1:8000").rstrip("/")
STREAM_ENDPOINT = f"{API_URL}/api/v1/estimate/stream"

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
FINISH_REASON_LABELS = {"stop": "Completa", "length": "Cortada", "content_filter": "Rechazada"}


def fmt_int(value: int) -> str:
    """Número entero con punto como separador de miles: 1746 -> '1.746'."""
    return f"{value:,}".replace(",", ".")


def fmt_seconds(value: float) -> str:
    """Segundos con coma decimal: 7.34 -> '7,3 s'."""
    return f"{value:.1f}".replace(".", ",") + " s"


def fmt_cost(value: float | None) -> str:
    """Coste en USD con coma decimal: 0.00527 -> '0,0053 $'. Sin precio conocido -> 'n/d'."""
    return "n/d" if value is None else f"{value:.4f}".replace(".", ",") + " $"


def fmt_finish_reason(value: str) -> str:
    """Traduce el motivo de fin técnico de la API a un texto comprensible."""
    return FINISH_REASON_LABELS.get(value, value)


class ApiStream:
    """Llama a POST /api/v1/estimate/stream y, al iterar, devuelve los trozos de texto.

    Al terminar, `response` tiene la respuesta completa (evento `done`) o `error` el mensaje
    de error (evento `error`, o un código HTTP de error de la API).
    """

    def __init__(self, request: EstimationRequest):
        self.payload = request.model_dump(mode="json")
        self.response: dict | None = None
        self.error: str | None = None

    def __iter__(self) -> Iterator[str]:
        with httpx.stream(
            "POST",
            STREAM_ENDPOINT,
            json=self.payload,
            timeout=httpx.Timeout(120.0, connect=5.0),
        ) as http_response:
            if http_response.status_code != 200:
                http_response.read()
                self.error = f"La API respondió {http_response.status_code}: {http_response.text}"
                return
            event = None
            for line in http_response.iter_lines():
                if line.startswith("event:"):
                    event = line.removeprefix("event:").strip()
                elif line.startswith("data:"):
                    data = json.loads(line.removeprefix("data:").strip())
                    if event == "token":
                        yield data["text"]
                    elif event == "done":
                        self.response = data
                    elif event == "error":
                        self.error = data["message"]


def timed(stream: ApiStream, timings: dict) -> Iterator[str]:
    """Pasa los trozos del stream tal cual y anota cuándo llega el primero y cuándo acaba."""
    start = time.perf_counter()
    for chunk in stream:
        timings.setdefault("first_token_seconds", time.perf_counter() - start)
        yield chunk
    timings["total_seconds"] = time.perf_counter() - start
    timings.setdefault("first_token_seconds", timings["total_seconds"])


@st.cache_data(ttl=30, show_spinner=False)
def api_health() -> dict | None:
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).json()
    except httpx.HTTPError:
        return None


def entry_title(entry: dict) -> str:
    first_line = entry["text"].strip().splitlines()[0] if entry["text"].strip() else ""
    if first_line.startswith("## Estimación:"):
        return first_line.removeprefix("## Estimación:").strip()
    return entry["request"]["description"][:60] + "…"


def params_label(request: dict) -> str:
    return " · ".join(
        [
            PROJECT_TYPE_LABELS[ProjectType(request["project_type"])],
            DETAIL_LEVEL_LABELS[DetailLevel(request["detail_level"])],
            OUTPUT_FORMAT_LABELS[OutputFormat(request["output_format"])],
        ]
    )


def entry_caption(entry: dict) -> str:
    m = entry["metrics"]
    caption = (
        f"{m['model']} · prompt {m['prompt_version']} · "
        f"{fmt_int(m['input_tokens'] + m['output_tokens'])} tokens "
        f"({fmt_int(m['input_tokens'])} entrada + {fmt_int(m['output_tokens'])} salida) · "
        f"{fmt_seconds(m['total_seconds'])} · {fmt_cost(m['cost_usd'])}"
    )
    return caption + (" · ⚡ desde caché" if m["cache_hit"] else "")


def render_entry(entry: dict) -> None:
    st.markdown(entry["text"])
    if entry["metrics"]["finish_reason"] == "length":
        st.warning(
            "La estimación se ha cortado porque alcanzó el límite de tokens de salida. "
            "Puede faltar el final (totales, equipo o duración)."
        )
    st.caption(entry_caption(entry))


def make_entry(request: EstimationRequest, response: dict, timings: dict) -> dict:
    return {
        "time": datetime.now().strftime("%H:%M"),
        "request": request.model_dump(mode="json"),
        "text": response["text"],
        "metrics": {
            "model": response["model"],
            "provider": response["provider"],
            "prompt_version": response["prompt_version"],
            "input_tokens": response["usage"]["input_tokens"],
            "output_tokens": response["usage"]["output_tokens"],
            "cost_usd": response["cost_usd"],
            "finish_reason": response["finish_reason"],
            "cache_hit": response["cache_hit"],
            **timings,
        },
    }


# --- Página ---
st.set_page_config(page_title="Estimador CAG", page_icon="📋")
st.title("📋 Estimador CAG")
st.caption(
    "Describe el proyecto (o pega la transcripción de la reunión), elige el tipo, el nivel "
    "de detalle y el formato, y obtendrás una estimación basada en estimaciones previas."
)

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Servicio IA")
    health = api_health()
    if health:
        st.caption(f"✅ {API_URL} · **{health['provider']}** · **{health['model']}**")
    else:
        st.error(f"No se puede conectar con la API en {API_URL}")
    if st.button("Nueva sesión", width="stretch"):
        st.session_state.history = []

    st.subheader("📊 Última llamada")
    metrics_box = st.container()  # se rellena al final, con la llamada que acaba de hacerse

    st.subheader(f"🧠 Prompts ({PROMPT_VERSION})")
    prompts_box = st.container()  # se rellena al final, con la última estimación

    examples = render_examples(PROMPT_VERSION)
    st.subheader(f"📚 Contexto inyectado ({examples.count('<ejemplo numero=')} estimaciones)")
    with st.expander("Ver estimaciones de ejemplo (examples.j2)"):
        st.code(examples, language="markdown", wrap_lines=True)

with st.form("estimation_form"):
    description = st.text_area(
        "Descripción del proyecto",
        height=180,
        placeholder="Objetivos, funcionalidades, usuarios, integraciones, plazos…",
        help="Entre 20 y 2.000 caracteres.",
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
    submitted = st.form_submit_button("Estimar", type="primary")

new_entry = False
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
        st.error(
            "Revisa el formulario: " + "; ".join(err["msg"] for err in e.errors()),
            icon="⚠️",
        )
    else:
        st.caption(params_label(request.model_dump(mode="json")))
        stream = ApiStream(request)
        timings: dict = {}
        try:
            st.write_stream(timed(stream, timings))
        except httpx.HTTPError as e:
            st.error(f"No se puede conectar con la API en {API_URL}: {e}")
        else:
            if stream.error:
                st.error(stream.error)
            elif stream.response:
                entry = make_entry(request, stream.response, timings)
                st.session_state.history.append(entry)
                new_entry = True
                if entry["metrics"]["finish_reason"] == "length":
                    st.warning("La estimación se ha cortado: puede faltar el final.")
                st.caption(entry_caption(entry))

history = st.session_state.history
if history and not new_entry and not submitted:
    st.caption(params_label(history[-1]["request"]))
    render_entry(history[-1])

previous = history[:-1] if history and (new_entry or not submitted) else history
if previous:
    st.divider()
    st.subheader("Estimaciones anteriores de esta sesión")
    for entry in reversed(previous):
        label = f"{entry['time']} · {entry_title(entry)} · {params_label(entry['request'])}"
        with st.expander(label):
            render_entry(entry)

with metrics_box:
    if not history:
        st.caption("Aún no se ha hecho ninguna estimación.")
    else:
        m = history[-1]["metrics"]
        st.caption(f"Modelo: **{m['model']}** · Prompt: **{m['prompt_version']}**")
        if m["cache_hit"]:
            st.caption("⚡ Respuesta servida desde la caché: sin llamada al LLM")
        c1, c2 = st.columns(2)
        c1.metric("Tokens entrada", fmt_int(m["input_tokens"]))
        c2.metric("Tokens salida", fmt_int(m["output_tokens"]))
        c1.metric("Tiempo total", fmt_seconds(m["total_seconds"]))
        c2.metric("Primer token", fmt_seconds(m["first_token_seconds"]))
        c1.metric("Coste", fmt_cost(m["cost_usd"]))
        c2.metric(
            "Respuesta",
            fmt_finish_reason(m["finish_reason"]),
            help=f"Motivo de fin devuelto por la API: {m['finish_reason']}",
        )

with prompts_box:
    if history:
        prompt_request = EstimationRequest(**history[-1]["request"])
        st.caption("Los de la última estimación.")
    else:
        # Antes de la primera estimación: con las opciones por defecto del formulario
        prompt_request = EstimationRequest(
            description="(aquí irá la descripción del proyecto)",
            project_type=ProjectType.MOBILE_APP,
            detail_level=DetailLevel.MEDIUM,
            output_format=OutputFormat.PHASES_TABLE,
        )
        st.caption("Con las opciones por defecto; tras estimar, los de la última estimación.")
    system_prompt, user_prompt = render_estimation_prompt(prompt_request, PROMPT_VERSION)
    with st.expander(f"System ({fmt_int(len(system_prompt))} caracteres)"):
        st.code(system_prompt, language="markdown", wrap_lines=True)
    with st.expander("User"):
        st.code(user_prompt, language="markdown", wrap_lines=True)
