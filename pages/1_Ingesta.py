"""Página de Streamlit para la ingesta de datos: catálogo, ejecución, informe y supresión.

Como el resto del cliente, solo habla con la API (ESTIMATOR_API_URL): la ingesta se ejecuta
en el servicio, en segundo plano, y aquí se consulta el estado del trabajo hasta que acaba.
La API necesita base de datos (DATABASE_URL) y, para las fuentes con datos personales,
PII_SALT; si faltan, la página muestra el error que devuelve la API.
"""

import os
import time

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("ESTIMATOR_API_URL", "http://127.0.0.1:8000").rstrip("/")
INGESTION_URL = f"{API_URL}/api/v1/ingestion"
POLL_SECONDS = float(os.getenv("INGESTION_POLL_SECONDS", "1"))
POLL_LIMIT = 120  # la primera ingesta con datos personales carga el modelo de spaCy (~10 s)
PII_TYPES = {
    "PERSON": "Persona",
    "EMAIL_ADDRESS": "Email",
    "PHONE_NUMBER": "Teléfono",
    "ES_NIF": "DNI",
    "ES_NIE": "NIE",
    "IBAN_CODE": "IBAN",
    "CREDIT_CARD": "Tarjeta",
}
DECISIONS = {"include": "✅ include", "review": "⏸️ review", "exclude": "⛔ exclude"}


def error_detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail"))
    except ValueError:
        return response.text


def wait_for_job(job_id: str) -> dict:
    """Consulta el trabajo hasta que termina (o hasta POLL_LIMIT intentos)."""
    job: dict = {}
    for _ in range(POLL_LIMIT):
        job = httpx.get(f"{INGESTION_URL}/jobs/{job_id}", timeout=10).json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(POLL_SECONDS)
    return job


def render_report(job: dict) -> None:
    if job["status"] == "failed":
        st.error(f"La ingesta ha fallado: {job['error']}")
        return
    if job["status"] != "completed":
        st.info(f"El trabajo sigue en estado '{job['status']}'. Vuelve a consultarlo en un rato.")
        return
    report = job["report"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Archivos", report["files"])
    c2.metric("Documentos", report["documents"])
    c3.metric("Reparados", report["repaired"])
    c4.metric("Cuarentena", len(report["quarantined"]))
    c5.metric("Descartados", len(report["discarded"]))
    for title, items in (
        ("🟡 En cuarentena (recuperables con revisión)", report["quarantined"]),
        ("🔴 Descartados (erróneos)", report["discarded"]),
    ):
        if items:
            with st.expander(f"{title}: {len(items)}"):
                for key, reasons in items.items():
                    st.markdown(f"- **{key}**: {', '.join(reasons)}")
    if report["deduplicated"]:
        st.caption("Versiones antiguas retiradas: " + ", ".join(report["deduplicated"]))
    if report["pii_replacements"]:
        counts = ", ".join(
            f"{PII_TYPES.get(k, k)}: {v}" for k, v in report["pii_replacements"].items()
        )
        st.caption(f"🔒 Datos personales seudonimizados en el texto: {counts}")

    response = httpx.get(f"{INGESTION_URL}/jobs/{job['job_id']}/documents?limit=5", timeout=10)
    if response.status_code == 200:
        st.markdown("**Vista previa de los documentos** (ya limpios y seudonimizados)")
        for doc in response.json():
            with st.expander(f"{doc['id']} ({doc['kind']})"):
                st.text(doc["text"])
                st.json(doc["metadata"], expanded=False)


st.set_page_config(page_title="Ingesta · Estimador CAG", page_icon="📥", layout="wide")
st.title("📥 Ingesta de datos")
st.caption(
    "Del archivo en bruto al documento limpio y seudonimizado: catálogo -> carga -> "
    "reparación -> validación -> deduplicado -> seudonimización."
)

try:
    response = httpx.get(f"{INGESTION_URL}/sources", timeout=10)
except httpx.HTTPError as e:
    st.error(f"No se puede conectar con la API en {API_URL}: {e}")
    st.stop()
sources = response.json()

st.subheader("Catálogo de fuentes")
st.dataframe(
    pd.DataFrame(
        [
            {
                "Fuente": s["name"],
                "Decisión": DECISIONS[s["decision"]],
                "Calidad (1-5)": round(s["quality"], 2),
                "Datos personales": "sí" if s["contains_pii"] else "no",
                "Archivos": s["files"],
                "Responsable": s["owner"],
                "Motivo": s["decision_reason"] or "",
            }
            for s in sources
        ]
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("Ingerir una fuente")
names = [s["name"] for s in sources]
chosen = st.selectbox(
    "Fuente",
    names,
    format_func=lambda n: f"{n} ({next(s['decision'] for s in sources if s['name'] == n)})",
    help="Las fuentes en review o exclude se pueden elegir para ver cómo las rechaza la API.",
)
if st.button("Ingerir", type="primary"):
    response = httpx.post(f"{INGESTION_URL}/runs", json={"source": chosen}, timeout=30)
    if response.status_code != 202:
        st.error(
            f"La API ha rechazado la ingesta ({response.status_code}): {error_detail(response)}"
        )
    else:
        with st.spinner("Ingiriendo… (la primera vez carga el modelo de spaCy)"):
            st.session_state.last_job = wait_for_job(response.json()["job_id"])

job = st.session_state.get("last_job")
if job:
    st.markdown(f"**Último trabajo:** `{job['source']}` · `{job['job_id'][:8]}`")
    render_report(job)

st.subheader("Derecho de supresión")
st.caption(
    "Borra la relación entre un dato personal y su seudónimo (art. 17 del RGPD). El dato solo "
    "se usa para calcular su hash: no se guarda."
)
with st.form("forget"):
    col1, col2 = st.columns([1, 3])
    entity = col1.selectbox("Tipo", list(PII_TYPES), format_func=PII_TYPES.get)
    value = col2.text_input("Dato original", placeholder="María García López")
    if st.form_submit_button("Olvidar") and value.strip():
        response = httpx.request(
            "DELETE",
            f"{INGESTION_URL}/pseudonyms",
            json={"entity_type": entity, "value": value},
            timeout=10,
        )
        if response.status_code != 200:
            st.error(f"No se ha podido ({response.status_code}): {error_detail(response)}")
        elif response.json()["forgotten"]:
            st.success("Olvidado: ese dato ya no se puede relacionar con su seudónimo.")
        else:
            st.info("No había ningún seudónimo para ese dato.")
