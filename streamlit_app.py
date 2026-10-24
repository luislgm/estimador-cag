"""Interfaz de chat para el estimador CAG.

Reutiliza el mismo servicio que el endpoint POST /api/v1/estimate: mismo system prompt,
mismos ejemplos de contexto y misma configuración (.env).

Ejecutar con:  uv run --group ui streamlit run streamlit_app.py
"""

import time
from collections.abc import Iterator

import anthropic
import streamlit as st
from pydantic import ValidationError

from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES
from app.logging_config import setup_logging
from app.schemas.estimation import EstimationRequest
from app.services.llm_service import EstimationStream, build_system_prompt
from app.tracing import setup_tracing

st.set_page_config(page_title="Estimador CAG", page_icon="📋")
st.title("📋 Estimador CAG")
st.caption(
    "Pega la transcripción de una reunión con un cliente y obtendrás una estimación "
    "del proyecto basada en estimaciones previas."
)

# La API key se lee del .env / variables de entorno a través de la configuración
try:
    settings = get_settings()
except ValidationError as e:
    st.error(f"Configuración incompleta. Revisa tu archivo .env:\n\n{e}")
    st.stop()

setup_logging(settings.log_level, settings.app_env)
setup_tracing(settings)


def fmt_int(value: int) -> str:
    """Número entero con punto como separador de miles: 1746 -> '1.746'."""
    return f"{value:,}".replace(",", ".")


def fmt_seconds(value: float) -> str:
    """Segundos con coma decimal: 7.34 -> '7,3 s'."""
    return f"{value:.1f}".replace(".", ",") + " s"


def fmt_cost(value: float | None) -> str:
    """Coste en USD con coma decimal: 0.00527 -> '0,0053 $'. Sin precio conocido -> 'n/d'."""
    return "n/d" if value is None else f"{value:.4f}".replace(".", ",") + " $"


FINISH_REASON_LABELS = {
    "end_turn": "Completa",
    "max_tokens": "Cortada",
    "refusal": "Rechazada",
}


def fmt_finish_reason(value: str) -> str:
    """Traduce el stop_reason técnico de la API a un texto comprensible."""
    return FINISH_REASON_LABELS.get(value, value)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None


# --- Panel lateral: visibilidad del contexto CAG ---
with st.sidebar:
    st.header("Contexto CAG")
    st.caption(f"Proveedor: **{settings.llm_provider}** · Modelo: **{settings.llm_model}**")
    if st.button("Nueva conversación", width="stretch"):
        st.session_state.messages = []
        st.session_state.last_metrics = None

    st.subheader("📊 Última llamada")
    # Se rellena al final del script, para mostrar también la llamada que acaba de hacerse
    metrics_box = st.container()

    st.subheader("🧠 System prompt activo")
    system_prompt = build_system_prompt()
    with st.expander(f"Ver prompt completo ({fmt_int(len(system_prompt))} caracteres)"):
        st.code(system_prompt, language="markdown", wrap_lines=True)

    st.subheader(f"📚 Contexto inyectado ({len(ESTIMATION_EXAMPLES)} estimaciones)")
    for i, example in enumerate(ESTIMATION_EXAMPLES, start=1):
        title = example["estimation"].strip().splitlines()[0].removeprefix("## Estimación: ")
        with st.expander(f"{i}. {title}"):
            st.markdown("**Lo que pedía el cliente**")
            st.write(example["meeting_summary"])
            st.markdown("**Estimación**")
            st.markdown(example["estimation"])


def render_metrics(metrics: dict | None) -> None:
    with metrics_box:
        if metrics is None:
            st.caption("Aún no se ha hecho ninguna llamada al LLM.")
            return
        st.caption(f"Modelo: **{metrics['model']}**")
        col1, col2 = st.columns(2)
        col1.metric("Tokens entrada", fmt_int(metrics["input_tokens"]))
        col2.metric("Tokens salida", fmt_int(metrics["output_tokens"]))
        col1.metric("Tiempo total", fmt_seconds(metrics["total_seconds"]))
        col2.metric("Primer token", fmt_seconds(metrics["first_token_seconds"]))
        col1.metric("Coste", fmt_cost(metrics["cost_usd"]))
        col2.metric(
            "Respuesta",
            fmt_finish_reason(metrics["finish_reason"]),
            help=f"Motivo de fin devuelto por la API: {metrics['finish_reason']}",
        )


def timed(stream: EstimationStream, timings: dict) -> Iterator[str]:
    """Pasa los trozos del stream tal cual y anota cuándo llega el primero y cuándo acaba."""
    start = time.perf_counter()
    for chunk in stream:
        timings.setdefault("first_token_seconds", time.perf_counter() - start)
        yield chunk
    timings["total_seconds"] = time.perf_counter() - start
    timings.setdefault("first_token_seconds", timings["total_seconds"])


def answer(prompt: str) -> None:
    """Valida la transcripción, genera la estimación en streaming y la guarda en el historial."""
    try:
        # Mismas reglas de validación que el endpoint (mínimo 50 caracteres)
        request = EstimationRequest(transcription=prompt)
    except ValidationError:
        reply = (
            "La transcripción es demasiado corta. Escribe al menos 50 caracteres "
            "describiendo lo que necesita el cliente."
        )
        st.warning(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        return

    # Streaming: la estimación se va escribiendo a medida que llegan los tokens
    stream = EstimationStream(request.transcription)
    timings: dict = {}
    try:
        st.write_stream(timed(stream, timings))
    except anthropic.APIError as e:
        reply = f"Error al llamar al LLM: {e}"
        st.error(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        return

    result = stream.result
    st.session_state.last_metrics = {
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "finish_reason": result.finish_reason,
        **timings,
    }
    if result.truncated:
        st.warning(
            "La estimación se ha cortado porque alcanzó el límite de tokens de salida. "
            "Puede faltar el final (totales, equipo o duración)."
        )
    total_tokens = result.input_tokens + result.output_tokens
    caption = (
        f"{result.model} · {fmt_int(total_tokens)} tokens "
        f"({fmt_int(result.input_tokens)} entrada + {fmt_int(result.output_tokens)} salida) · "
        f"{fmt_seconds(timings['total_seconds'])} · {fmt_cost(result.cost_usd)}"
    )
    st.caption(caption)
    st.session_state.messages.append(
        {"role": "assistant", "content": result.estimation, "caption": caption}
    )


# --- Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("caption"):
            st.caption(message["caption"])

if prompt := st.chat_input("Escribe o pega la transcripción de la reunión..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        answer(prompt)

render_metrics(st.session_state.last_metrics)
