"""Seudonimización: detección en español, consistencia, normalización y olvido."""

import pytest

from app.ingestion import pii
from app.ingestion.pii import InMemoryMappingStore, Pseudonymizer, hash_value

SALT = b"sal-de-test"


@pytest.fixture
def pseudonymizer() -> Pseudonymizer:
    return Pseudonymizer(SALT, InMemoryMappingStore())


def test_detects_personal_data_in_spanish(pseudonymizer):
    text = (
        "Reunión con María García López (maria.garcia@acme.es, tel. 612 345 678) de Acme "
        "en Valencia. DNI 12345678Z."
    )

    result = pseudonymizer.pseudonymize(text)

    assert result.counts == {"PERSON": 1, "EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1, "ES_NIF": 1}
    for real in ("María García López", "maria.garcia@acme.es", "612 345 678", "12345678Z"):
        assert real not in result.text
    # No son datos personales: la palabra inicial, la empresa y la ciudad se quedan
    assert result.text.startswith("Reunión con ")
    assert "de Acme en Valencia." in result.text


def test_person_spans_are_trimmed_to_the_name(pseudonymizer):
    text = "María García López: Necesitamos una app. Carlos Pérez Soto: Perfecto."

    result = pseudonymizer.pseudonymize(text)

    assert ": Necesitamos una app." in result.text and ": Perfecto." in result.text
    assert result.counts == {"PERSON": 2}


def test_a_name_seen_once_is_replaced_everywhere(pseudonymizer):
    # El modelo no detecta igual a la misma persona en todas las frases: basta con una
    text = "Participantes: Ana López Ferrer (preventa).\n\nAna López Ferrer: Entendido."

    result = pseudonymizer.pseudonymize(text)

    assert "Ana López" not in result.text
    pseudonym = result.replacements[0].pseudonym
    assert result.text.count(pseudonym) == 2


def test_known_values_are_replaced_even_if_not_detected(pseudonymizer):
    result = pseudonymizer.pseudonymize("Contacto: pedro", known={"pedro": "PERSON"})

    assert "pedro" not in result.text


def test_single_first_names_are_a_known_gap(pseudonymizer):
    # Concesión explícita: PERSON necesita nombre y apellido (ver pii.py)
    assert "pedro" in pseudonymizer.pseudonymize("hablamos con pedro de compras").text


def test_identifiers_in_capitals_are_not_names():
    assert pii._trim_person("Presupuesto BUDGET-2024-0002", 0, 18) == (0, len("Presupuesto"))
    assert pii._trim_person("Evaristo de Lladó dijo", 0, 22) == (0, len("Evaristo de Lladó"))
    assert pii._trim_person("Marta de", 0, 8) == (0, len("Marta"))
    assert pii._trim_person("123", 0, 3) == (0, 0)


def test_same_value_same_pseudonym_regardless_of_format(pseudonymizer):
    a = pseudonymizer.pseudonym("PHONE_NUMBER", "612 345 678")
    b = pseudonymizer.pseudonym("PHONE_NUMBER", "+34 612345678")
    c = pseudonymizer.pseudonym("PERSON", "  MARÍA garcía ")

    assert a == b
    assert c == pseudonymizer.pseudonym("PERSON", "María García")
    assert pii.normalize_value("IBAN_CODE", "es91 2100 0418") == "ES9121000418"


def test_pseudonyms_depend_on_the_salt():
    store = InMemoryMappingStore()
    one = Pseudonymizer(b"sal-1", store).pseudonym("PERSON", "María García")
    two = Pseudonymizer(b"sal-2", store).pseudonym("PERSON", "María García")

    assert one != two
    assert hash_value(b"sal-1", "PERSON", "x") != hash_value(b"sal-2", "PERSON", "x")


def test_every_entity_type_has_a_generator():
    assert set(pii._GENERATORS) == set(pii.PII_ENTITIES)
    for entity in pii.PII_ENTITIES:
        assert pii.fake_value(entity)


def test_forgetting_breaks_the_link(pseudonymizer):
    first = pseudonymizer.pseudonym("PERSON", "María García")
    original_hash = hash_value(SALT, "PERSON", "María García")

    assert pseudonymizer.store.forget("PERSON", original_hash)
    assert not pseudonymizer.store.forget("PERSON", original_hash)
    # El seudónimo no se deriva del dato: tras olvidarlo, nadie puede recalcular el anterior
    # (ni con la sal), y una nueva aparición recibe otro distinto
    assert pseudonymizer.pseudonym("PERSON", "María García") != first


def test_a_salt_is_required():
    with pytest.raises(ValueError, match="PII_SALT"):
        Pseudonymizer(b"", InMemoryMappingStore())
