import pytest

from app.sessions import (
    ConversationHistory,
    ProjectMetadata,
    SessionNotFound,
    SessionStore,
)
from tests.factories import make_phase, make_result

# --- Ventana deslizante ---


def test_window_keeps_the_last_turns():
    history = ConversationHistory(max_turns=3)
    for i in range(1, 6):
        history.add_turn(f"usuario {i}", f"asistente {i}")

    assert len(history) == 3
    assert history.total_turns == 5
    assert [m["content"] for m in history.messages()] == [
        "usuario 3",
        "asistente 3",
        "usuario 4",
        "asistente 4",
        "usuario 5",
        "asistente 5",
    ]


def test_to_messages_list_always_starts_with_the_system_prompt():
    history = ConversationHistory(max_turns=2)
    for i in range(4):
        history.add_turn(f"u{i}", f"a{i}")

    messages = history.to_messages_list(ProjectMetadata(), lambda m: "SYSTEM")

    assert messages[0] == {"role": "system", "content": "SYSTEM"}
    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in messages[1:]] == ["u2", "a2", "u3", "a3"]


def test_to_messages_list_regenerates_the_system_prompt_from_the_current_metadata():
    history = ConversationHistory(max_turns=1)
    history.add_turn("u0", "a0")
    history.add_turn("u1", "a1")  # u0 ya ha salido de la ventana
    metadata = ProjectMetadata(project_name="Portal B2B", assumed_team_size=3)

    def build_system(m: ProjectMetadata) -> str:
        return f"Proyecto: {m.project_name}, equipo de {m.assumed_team_size}"

    first = history.to_messages_list(ProjectMetadata(), build_system)
    second = history.to_messages_list(metadata, build_system)

    assert first[0]["content"] == "Proyecto: None, equipo de None"
    # La memoria sobrevive a la ventana: el nombre sigue en el system prompt aunque el turno
    # donde apareció ya no se envíe
    assert second[0] == {"role": "system", "content": "Proyecto: Portal B2B, equipo de 3"}
    assert second[1:] == [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]


def test_empty_history():
    history = ConversationHistory(max_turns=6)

    assert history.to_messages_list(ProjectMetadata(), lambda m: "SYSTEM") == [
        {"role": "system", "content": "SYSTEM"}
    ]
    assert len(history) == 0


# --- Memoria del proyecto ---


def test_metadata_starts_empty():
    assert ProjectMetadata().is_empty()


def test_metadata_is_extracted_from_the_result():
    result = make_result(team_size=3, technologies=["Flutter", "PostgreSQL"])

    metadata = ProjectMetadata().updated_with(result)

    assert metadata.project_name == "Reservas para restaurante"
    assert metadata.assumed_team_size == 3
    assert metadata.mentioned_technologies == ["Flutter", "PostgreSQL"]
    assert metadata.agreed_scope == result.summary


def test_metadata_keeps_the_latest_refined_technologies():
    first = ProjectMetadata().updated_with(make_result(technologies=["PostgreSQL o MySQL"]))

    second = first.updated_with(make_result(technologies=["PostgreSQL", "SAP Business One"]))

    assert second.mentioned_technologies == ["PostgreSQL", "SAP Business One"]


def test_metadata_keeps_previous_values_when_the_result_omits_them():
    first = ProjectMetadata().updated_with(make_result(team_size=4, technologies=["Django"]))

    second = first.updated_with(make_result(team_size=0, technologies=[]))

    assert second.assumed_team_size == 4
    assert second.mentioned_technologies == ["Django"]


def test_out_of_scope_result_does_not_change_the_memory():
    memory = ProjectMetadata(project_name="Portal", assumed_team_size=2)
    out_of_scope = make_result(
        confidence_pct=10,
        summary="Fuera de alcance: no se entiende la petición.",
        phases=[make_phase("Sin estimar", 0)],
    )

    assert memory.updated_with(out_of_scope) == memory


# --- Almacén de sesiones ---


def test_store_creates_and_finds_sessions():
    store = SessionStore()

    session = store.create()

    assert store.get(session.id) is session
    assert len(session.id) == 36  # UUID v4
    assert session.history.max_turns == 6  # SESSION_MAX_TURNS por defecto


def test_store_unknown_session():
    with pytest.raises(SessionNotFound):
        SessionStore().get("no-existe")


def test_window_size_comes_from_settings(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("SESSION_MAX_TURNS", "3")
    get_settings.cache_clear()

    assert SessionStore().create().history.max_turns == 3
