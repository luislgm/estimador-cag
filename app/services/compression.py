"""Política de compresión del historial: qué pasa con los turnos que salen de la ventana.

Sin compresión, la ventana deslizante descarta los turnos antiguos y se pierde lo que
contenían. Con ella, cada turno expulsado sigue uno de dos caminos:

1. Ancla: si el mensaje del usuario contiene un compromiso duradero (NDA, contrato firmado,
   alcance cerrado, presupuesto máximo, normativa, fecha límite obligatoria...), el turno se
   conserva literal, fuera de la ventana. Como mucho SESSION_MAX_ANCHORS: si no caben, las
   anclas más antiguas pasan al resumen.
2. Resumen: el resto se integra, con una llamada al LLM, en un único resumen acumulado que
   se inyecta en el system prompt.

La detección de anclas es heurística (expresiones regulares en español e inglés): no cuesta
nada, es determinista y explicable (se registra qué regla saltó). Es conservadora a
propósito: un ancla de más ocupa prompt para siempre, mientras que una de menos solo acaba
en el resumen.

El resumen es best-effort: si la llamada falla, se conserva el resumen anterior y la
conversación sigue (se pierde ese tramo, pero nunca se rompe el turno del usuario).
"""

import re
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field

from app.prompts.loader import render_summary_prompt
from app.services import llm_client
from app.sessions import ConversationHistory, Turn

log = structlog.get_logger(module=__name__)

SUMMARY_MAX_TOKENS = 1000

# Reglas de anclaje: nombre (para los logs) y patrón. La lista es corta a propósito.
ANCHOR_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "confidencialidad",
        re.compile(r"\b(nda|non[- ]?disclosure|acuerdo de confidencialidad|bajo embargo)\b", re.I),
    ),
    (
        "contrato_firmado",
        re.compile(
            r"\b(firmad[oa]s?|hemos firmado|ha firmado|han firmado)\b.{0,40}"
            r"\b(contrato|acuerdo|propuesta|presupuesto|sow)\b"
            r"|\b(contrato|acuerdo|propuesta|presupuesto|sow)\b.{0,20}\bfirmad[oa]\b"
            r"|\b(signed|countersigned)\s+(the\s+)?(contract|sow|msa|agreement)\b",
            re.I,
        ),
    ),
    (
        "alcance_cerrado",
        re.compile(
            r"\b(alcance|backlog)\b.{0,20}\b(cerrad[oa]|congelad[oa]|definitiv[oa])\b"
            r"|\b(scope|backlog)\s+(is\s+)?(frozen|locked|final)\b",
            re.I,
        ),
    ),
    (
        "presupuesto_maximo",
        re.compile(
            r"\b(presupuesto|budget)\b.{0,25}"
            r"\b(máximo|maximo|tope|cerrado|aprobado|no puede superar|capped|locked)\b"
            r"|\bno (puede|podemos|debe) superar\b.{0,30}(€|euros|eur\b)",
            re.I,
        ),
    ),
    (
        "normativa",
        re.compile(r"\b(rgpd|gdpr|lopd|hipaa|pci[- ]?dss|iso[- ]?27001|ens|sox)\b", re.I),
    ),
    (
        "fecha_limite",
        re.compile(
            r"\b(fecha l[ií]mite|deadline)\b.{0,30}\b(innegociable|obligatori[oa]|fij[oa]|hard)\b"
            r"|\btiene que estar en producci[oó]n (antes del?|para)\b"
            r"|\bhard deadline\b|\bmust go live by\b",
            re.I,
        ),
    ),
    (
        "compromiso_explicito",
        re.compile(
            r"\b(el cliente|hemos|nos hemos)\s+(ha\s+)?(acordado|aceptado|comprometido)\b"
            r"|\b(we|the client)\s+(agreed|committed)\s+to\b",
            re.I,
        ),
    ),
)


def detect_anchor(text: str) -> list[str]:
    """Nombres de las reglas de anclaje que cumple el texto (vacío si no es un ancla)."""
    return [name for name, pattern in ANCHOR_RULES if pattern.search(text)]


class ConversationSummary(BaseModel):
    """Salida estructurada del resumidor."""

    summary: str = Field(..., min_length=1, max_length=4000, description="Resumen acumulado")


@dataclass
class CompressionReport:
    anchored: list[list[str]] = field(default_factory=list)  # reglas de cada ancla nueva
    summarized: int = 0  # turnos integrados en el resumen
    cost_usd: float = 0.0
    summary_failed: bool = False


async def compress(history: ConversationHistory, evicted: list[Turn]) -> CompressionReport:
    """Aplica la política a los turnos que acaban de salir de la ventana (modifica `history`)."""
    report = CompressionReport()
    to_summarize: list[Turn] = []
    for turn in evicted:
        rules = detect_anchor(turn.user)
        if rules and history.max_anchors > 0:
            to_summarize.extend(history.add_anchor(turn))
            report.anchored.append(rules)
        else:
            to_summarize.append(turn)

    if to_summarize:
        system, user = render_summary_prompt(history.summary, to_summarize)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            result, info = await llm_client.acomplete_structured(
                messages, ConversationSummary, max_tokens=SUMMARY_MAX_TOKENS
            )
        except llm_client.LLMError as e:
            report.summary_failed = True
            log.warning("summary_failed", turns=len(to_summarize), error=str(e))
        else:
            history.summary = result.summary
            history.summarized_turns += len(to_summarize)
            report.summarized = len(to_summarize)
            report.cost_usd = info.cost_usd or 0.0

    if evicted:
        log.info(
            "history_compressed",
            evicted=len(evicted),
            anchored=report.anchored,
            summarized=report.summarized,
            anchors_kept=len(history.anchors),
            summary_failed=report.summary_failed,
            cost_usd=report.cost_usd,
        )
    return report
