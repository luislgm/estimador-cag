"""Tests de integración de las sesiones (Paso 7), con httpx.AsyncClient sobre la app."""

import json
from unittest.mock import patch

import httpx
import litellm
import pytest

from app.main import app
from app.schemas.estimation import EstimationResult
from app.services import llm_client
from app.sessions import store
from tests.factories import (
    make_docx,
    make_info,
    make_pdf,
    make_phase,
    make_result,
    tool_call_response,
)

FORM = {"project_type": "web_saas", "detail_level": "summary", "output_format": "phases_table"}
TRANSCRIPT = "Una distribuidora de bebidas quiere un portal B2B para que los bares hagan pedidos."


@pytest.fixture(autouse=True)
def clean_store():
    store.clear()
    yield
    store.clear()


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def create_session(client) -> str:
    response = await client.post("/api/v1/sessions")
    assert response.status_code == 201
    return response.json()["session_id"]


async def send(client, session_id, transcript=TRANSCRIPT, files=None, **form):
    return await client.post(
        f"/api/v1/sessions/{session_id}/estimate",
        data=FORM | form | {"transcript": transcript},
        files=files,
    )


class FakeLLM:
    """Sustituye a acomplete_structured y guarda con qué se le llamó en cada turno."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    async def __call__(self, messages, response_model, max_tokens):
        # messages = [system, *historial de la ventana, mensaje actual]
        assert messages[0]["role"] == "system" and messages[-1]["role"] == "user"
        user = messages[-1]["content"]
        self.calls.append(
            {"system": messages[0]["content"], "user": user, "history": messages[1:-1]}
        )
        result = self.results[min(len(self.calls), len(self.results)) - 1]
        return (result(user) if callable(result) else result), make_info()


# --- Los tres tests del Paso 7 ---


async def test_two_turns_update_the_project_metadata(client):
    llm = FakeLLM(
        make_result(project_name="Portal B2B Bebidas", team_size=2, technologies=["Django"]),
        make_result(
            project_name="Portal B2B Bebidas", team_size=3, technologies=["Django", "Flutter"]
        ),
    )
    session_id = await create_session(client)

    with patch.object(llm_client, "acomplete_structured", llm):
        first = (await send(client, session_id)).json()
        second = (
            await send(client, session_id, "Añade una app móvil para los repartidores, por favor.")
        ).json()

    assert first["session"]["project_metadata"]["assumed_team_size"] == 2
    assert first["session"]["project_metadata"]["mentioned_technologies"] == ["Django"]
    assert second["session"]["project_metadata"]["assumed_team_size"] == 3
    assert second["session"]["project_metadata"]["mentioned_technologies"] == ["Django", "Flutter"]
    assert second["session"]["turns"] == 2
    # El segundo turno recibe la memoria en el system prompt y el primer turno en el historial
    assert "<project_metadata>\n</project_metadata>" in llm.calls[0]["system"]
    assert "- Nombre del proyecto: Portal B2B Bebidas" in llm.calls[1]["system"]
    assert "- Tecnologías: Django" in llm.calls[1]["system"]
    assert [m["role"] for m in llm.calls[1]["history"]] == ["user", "assistant"]
    assert TRANSCRIPT in llm.calls[1]["history"][0]["content"]
    # El asistente queda en el historial como markdown, no como JSON (el JSON en el historial
    # hace que el modelo anide JSON como texto en la herramienta)
    assistant = llm.calls[1]["history"][1]["content"]
    assert assistant.startswith("## Estimación: Portal B2B Bebidas")
    assert not assistant.lstrip().startswith("{")


async def test_pdf_attachment_influences_the_estimation(client):
    def estimate(user):
        # Simula un modelo que solo menciona SAP si el documento llega en el prompt
        technologies = ["PostgreSQL", "SAP Business One"] if "SAP Business One" in user else []
        hours = 120 if technologies else 80
        phases = [make_phase("Desarrollo", hours)]
        return make_result(phases=phases, technologies=technologies)

    pdf = make_pdf("Requisitos tecnicos", "Hay que integrarse con el ERP SAP Business One.")
    with patch.object(llm_client, "acomplete_structured", FakeLLM(estimate)):
        without = (await send(client, await create_session(client))).json()
        with_pdf = (
            await send(
                client,
                await create_session(client),
                files=[("attachments", ("requisitos.pdf", pdf, "application/pdf"))],
            )
        ).json()

    expected_text = "Requisitos tecnicos\nHay que integrarse con el ERP SAP Business One."
    assert with_pdf["attachments"] == [
        {"filename": "requisitos.pdf", "chars": len(expected_text), "truncated": False}
    ]
    assert "SAP Business One" not in without["result"]["technologies"]
    assert "SAP Business One" in with_pdf["result"]["technologies"]
    assert with_pdf["result"]["total_hours"] > without["result"]["total_hours"]


async def test_history_sent_to_the_llm_never_exceeds_max_turns(client):
    """De extremo a extremo: lo que de verdad llega a LiteLLM en cada uno de 8 turnos."""
    sizes = []

    async def fake_acompletion(**params):
        sizes.append(len(params["messages"]))
        return tool_call_response(make_result().model_dump(mode="json"))

    session_id = await create_session(client)
    with patch.object(litellm, "acompletion", fake_acompletion):
        for turn in range(1, 9):
            response = await send(client, session_id, f"{TRANSCRIPT} Mensaje número {turn}.")
            assert response.status_code == 200

    max_turns = 6
    # system + 2 mensajes por turno de la ventana + mensaje actual
    assert sizes == [1 + 2 * min(t, max_turns) + 1 for t in range(8)]
    assert max(sizes) == 1 + 2 * max_turns + 1
    state = (await client.get(f"/api/v1/sessions/{session_id}")).json()
    assert (state["turns"], state["turns_in_window"]) == (8, max_turns)


# --- Resto del comportamiento del endpoint ---


async def test_create_and_get_session(client):
    session_id = await create_session(client)

    state = (await client.get(f"/api/v1/sessions/{session_id}")).json()

    assert state == {
        "session_id": session_id,
        "turns": 0,
        "turns_in_window": 0,
        "max_turns": 6,
        "project_metadata": {
            "project_name": None,
            "assumed_team_size": None,
            "mentioned_technologies": [],
            "agreed_scope": None,
        },
    }


async def test_unknown_session_returns_404(client):
    assert (await client.get("/api/v1/sessions/no-existe")).status_code == 404
    assert (await send(client, "no-existe")).status_code == 404


async def test_response_keeps_the_estimation_schema(client):
    with patch.object(llm_client, "acomplete_structured", FakeLLM(make_result())):
        body = (await send(client, await create_session(client))).json()

    EstimationResult.model_validate(body["result"])
    assert body["prompt_version"] == "v1"
    assert body["text"].startswith("## Estimación:")
    assert body["cache_hit"] is False


async def test_docx_attachment(client):
    llm = FakeLLM(make_result())
    document = make_docx("Propuesta anterior: 3 fases y 200 horas.")
    docx_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    with patch.object(llm_client, "acomplete_structured", llm):
        response = await send(
            client,
            await create_session(client),
            files=[("attachments", ("propuesta.docx", document, docx_type))],
        )

    assert response.status_code == 200
    assert "--- attachment: propuesta.docx ---" in llm.calls[0]["user"]
    assert "Propuesta anterior: 3 fases y 200 horas." in llm.calls[0]["user"]


async def test_several_attachments(client):
    llm = FakeLLM(make_result())
    files = [
        ("attachments", ("a.pdf", make_pdf("Documento A"), "application/pdf")),
        ("attachments", ("b.pdf", make_pdf("Documento B"), "application/pdf")),
    ]

    with patch.object(llm_client, "acomplete_structured", llm):
        body = (await send(client, await create_session(client), files=files)).json()

    assert [a["filename"] for a in body["attachments"]] == ["a.pdf", "b.pdf"]
    assert "Documento A" in llm.calls[0]["user"] and "Documento B" in llm.calls[0]["user"]


async def test_unsupported_attachment_returns_422(client):
    files = [("attachments", ("notas.txt", b"hola", "text/plain"))]

    response = await send(client, await create_session(client), files=files)

    assert response.status_code == 422
    assert "formato no admitido" in response.json()["detail"]


async def test_too_many_attachments_returns_422(client):
    files = [("attachments", (f"{i}.pdf", make_pdf("x"), "application/pdf")) for i in range(6)]

    response = await send(client, await create_session(client), files=files)

    assert response.status_code == 422


async def test_personal_data_in_an_attachment_is_rejected(client):
    pdf = make_pdf("Contacto del cliente: ana.garcia@empresa.es")
    files = [("attachments", ("contacto.pdf", pdf, "application/pdf"))]

    with patch.object(llm_client, "acomplete_structured", FakeLLM(make_result())) as llm:
        response = await send(client, await create_session(client), files=files)

    assert response.status_code == 400
    assert response.json()["detail"]["reason"] == "pii"
    assert "contacto.pdf" in response.json()["detail"]["message"]
    assert llm.calls == []


async def test_invalid_transcript_returns_422(client):
    response = await send(client, await create_session(client), transcript="corta")

    assert response.status_code == 422
    assert json.dumps(response.json()).count("description") >= 1


async def test_llm_failure_returns_502(client):
    async def failing(*args, **kwargs):
        raise llm_client.LLMError("conexión rechazada")

    with patch.object(llm_client, "acomplete_structured", failing):
        response = await send(client, await create_session(client))

    assert response.status_code == 502
