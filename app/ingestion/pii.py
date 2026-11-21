"""Seudonimización consistente de datos personales (RGPD): el mismo dato, el mismo seudónimo.

    texto -> Presidio detecta entidades -> HMAC(sal, dato) -> seudónimo estable -> texto limpio

- Detección: Presidio con el modelo de spaCy en español (personas) y sus reconocedores de
  email, DNI/NIE, IBAN y tarjetas, más uno propio para teléfonos españoles (la misma
  expresión que el guardrail de entrada: el de Presidio no reconoce "612 345 678").
- Clave: HMAC-SHA256 del dato normalizado con una sal secreta (PII_SALT). Lo que se guarda
  es el hash, nunca el dato: leer la base de datos no permite recuperar el original.
- Seudónimo: Faker en español, al azar la primera vez que se ve el dato. La consistencia
  ("María García" es la misma persona ficticia en todos los documentos, así que las
  relaciones entre documentos se conservan) la da el almacén, no el generador.
- Almacén: guarda (tipo, hash) -> seudónimo, y es el único vínculo entre el dato y su
  seudónimo. Olvidar a alguien (art. 17, derecho de supresión) es borrar su fila: a partir
  de ahí nadie puede relacionar el dato con su seudónimo, ni siquiera con la sal. Por eso el
  seudónimo no se deriva del hash: si lo hiciera, se podría recalcular y borrar no serviría.

Decisiones sobre qué se seudonimiza:
- Solo datos personales. Las empresas (Acme, Beta Norte) no lo son y se necesitan para
  estimar; las ubicaciones tampoco se tratan (el modelo pequeño confunde empresas con
  lugares, y una ciudad no identifica a nadie).
- PERSON exige al menos dos palabras (nombre y apellido): el modelo pequeño marca como
  persona palabras sueltas en mayúscula al inicio de frase ("Reunión"). A cambio, un nombre
  suelto ("Pedro") no se detecta. Es una concesión explícita, medida en los tests.
"""

import hashlib
import hmac
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

from faker import Faker
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

from app.guardrails.input import SPANISH_PHONE_REGEX

SPACY_MODEL = "es_core_news_sm"
PII_ENTITIES = (
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "ES_NIF",
    "ES_NIE",
    "IBAN_CODE",
    "CREDIT_CARD",
)
MIN_SCORE = 0.5


class SpanishPhoneRecognizer(PatternRecognizer):
    def __init__(self) -> None:
        super().__init__(
            supported_entity="PHONE_NUMBER",
            name="SpanishPhoneRecognizer",
            patterns=[Pattern("telefono_es", SPANISH_PHONE_REGEX, 0.9)],
            supported_language="es",
        )


@lru_cache
def build_analyzer() -> AnalyzerEngine:
    """El analizador de Presidio en español. Cargar el modelo de spaCy tarda unos segundos,
    así que se crea una sola vez por proceso."""
    nlp = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "es", "model_name": SPACY_MODEL}],
        }
    ).create_engine()
    registry = RecognizerRegistry(supported_languages=["es"])
    registry.load_predefined_recognizers(languages=["es"], nlp_engine=nlp)
    registry.add_recognizer(SpanishPhoneRecognizer())
    return AnalyzerEngine(nlp_engine=nlp, registry=registry, supported_languages=["es"])


def normalize_value(entity_type: str, value: str) -> str:
    """Forma canónica del dato, para que '612 345 678' y '612345678' den el mismo hash."""
    if entity_type in ("PHONE_NUMBER", "IBAN_CODE", "CREDIT_CARD"):
        digits = re.sub(r"[\s.-]", "", value)
        return digits.removeprefix("+34") if entity_type == "PHONE_NUMBER" else digits.upper()
    return " ".join(value.lower().split())


def hash_value(salt: bytes, entity_type: str, value: str) -> str:
    message = f"{entity_type}:{normalize_value(entity_type, value)}".encode()
    return hmac.new(salt, message, hashlib.sha256).hexdigest()


_GENERATORS = {
    "PERSON": lambda f: f.name(),
    "EMAIL_ADDRESS": lambda f: f.email(),
    "PHONE_NUMBER": lambda f: f.phone_number(),
    "ES_NIF": lambda f: f.nif(),
    "ES_NIE": lambda f: f.nie(),
    "IBAN_CODE": lambda f: f.iban(),
    "CREDIT_CARD": lambda f: f.credit_card_number(),
}


_FAKER = Faker("es_ES")


def fake_value(entity_type: str) -> str:
    """Un seudónimo nuevo al azar (ver la nota del módulo: no se deriva del dato)."""
    return _GENERATORS[entity_type](_FAKER)


class MappingStore(Protocol):
    def get_or_create(self, entity_type: str, original_hash: str, candidate: str) -> str: ...

    def forget(self, entity_type: str, original_hash: str) -> bool: ...


class InMemoryMappingStore:
    """Almacén en memoria (CLI y tests). En el servicio se usa el de la base de datos."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], str] = {}

    def get_or_create(self, entity_type: str, original_hash: str, candidate: str) -> str:
        return self.rows.setdefault((entity_type, original_hash), candidate)

    def forget(self, entity_type: str, original_hash: str) -> bool:
        return self.rows.pop((entity_type, original_hash), None) is not None


_NAME_RE = re.compile(r"[^\W\d_]+(?:[ '-][^\W\d_]+)*")
_PARTICLES = {"de", "del", "la", "las", "los", "y", "i"}


def _is_name_word(word: str) -> bool:
    """Empieza en mayúscula y no está entera en mayúsculas (siglas e identificadores, como
    BUDGET en 'BUDGET-2024-0002', no son nombres)."""
    return word[:1].isupper() and (len(word) == 1 or not word.isupper())


def _trim_person(text: str, start: int, end: int) -> tuple[int, int]:
    """Recorta un PERSON al nombre: hasta el primer signo o palabra que no puede ser parte de un
    nombre, y sin partículas al final ("María García López: Necesitamos" -> "María García
    López"; "Presupuesto BUDGET-2024" -> "Presupuesto", que luego no llega a dos palabras).
    Las partículas como "de" se aceptan en medio: "Evaristo de Lladó"."""
    match = _NAME_RE.match(text, start, end)
    if not match:
        return start, start
    words = []
    for word in match.group().split(" "):
        if not (_is_name_word(word) or word.lower() in _PARTICLES):
            break
        words.append(word)
    while words and words[-1].lower() in _PARTICLES:
        words.pop()
    return start, start + len(" ".join(words))


@dataclass(frozen=True)
class Replacement:
    entity_type: str
    pseudonym: str
    start: int
    end: int


@dataclass
class PseudonymizationResult:
    text: str
    replacements: list[Replacement] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for r in self.replacements:
            totals[r.entity_type] = totals.get(r.entity_type, 0) + 1
        return totals


class Pseudonymizer:
    def __init__(
        self,
        salt: bytes,
        store: MappingStore,
        analyzer: AnalyzerEngine | None = None,
        entities: tuple[str, ...] = PII_ENTITIES,
    ):
        if not salt:
            raise ValueError("la seudonimización necesita una sal (PII_SALT)")
        self.salt = salt
        self.store = store
        self.analyzer = analyzer or build_analyzer()
        self.entities = entities

    def pseudonym(self, entity_type: str, value: str) -> str:
        """El seudónimo de un dato cuyo tipo ya se conoce (p. ej., el email de un contacto)."""
        original_hash = hash_value(self.salt, entity_type, value)
        return self.store.get_or_create(entity_type, original_hash, fake_value(entity_type))

    def detect(self, text: str, known: dict[str, str] | None = None) -> list[tuple[str, int, int]]:
        """Entidades (tipo, inicio, fin) sin solapes.

        1. Presidio detecta; los PERSON se recortan al nombre (el modelo pequeño a veces se
           come lo que sigue: "María García López: Necesitamos").
        2. Propagación: cada valor detectado, y los conocidos de antemano (`known`, valor ->
           tipo, p. ej. el contacto del presupuesto), se buscan en todo el texto. El modelo
           no detecta igual a la misma persona en todas las frases; basta con una.
        3. Sin solapes: gana la de más puntuación y, a igualdad, la más larga.
        """
        results = self.analyzer.analyze(
            text=text, language="es", entities=list(self.entities), score_threshold=MIN_SCORE
        )
        found: list[tuple[float, str, int, int]] = []
        for r in results:
            start, end = r.start, r.end
            if r.entity_type == "PERSON":
                start, end = _trim_person(text, start, end)
                if len(text[start:end].split()) < 2:
                    continue  # ver la nota del módulo: nombre y apellido como mínimo
            found.append((r.score, r.entity_type, start, end))

        values = {text[s:e]: t for _, t, s, e in found} | (known or {})
        for value, entity_type in values.items():
            for match in re.finditer(re.escape(value), text, re.IGNORECASE):
                found.append((1.0, entity_type, match.start(), match.end()))

        chosen: list[tuple[str, int, int]] = []
        for _, entity_type, start, end in sorted(found, key=lambda f: (-f[0], f[2] - f[3])):
            if all(end <= s or start >= e for _, s, e in chosen):
                chosen.append((entity_type, start, end))
        return sorted(chosen, key=lambda c: c[1])

    def pseudonymize(
        self, text: str, known: dict[str, str] | None = None
    ) -> PseudonymizationResult:
        replacements = []
        result = text
        # De derecha a izquierda, para que los offsets de lo que queda sigan siendo válidos
        for entity_type, start, end in reversed(self.detect(text, known)):
            fake = self.pseudonym(entity_type, text[start:end])
            result = result[:start] + fake + result[end:]
            replacements.append(Replacement(entity_type, fake, start, end))
        return PseudonymizationResult(result, list(reversed(replacements)))
