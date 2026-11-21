"""¿CAG, RAG o híbrido? La decisión de arquitectura, a partir del corpus.

    uv run python -m app.ingestion.architecture                     # el corpus actual
    uv run python -m app.ingestion.architecture --scale 50          # si creciera ×50

CAG es viable solo si se cumplen las cuatro restricciones a la vez (un AND, no un OR: con
una que falle, CAG no sirve):

1. Ventana de contexto: el corpus cabe dejando margen (USABLE_WINDOW_FRACTION).
2. Coste: enviar el corpus en cada turno cabe en el presupuesto por turno.
3. Latencia: con el corpus en el prompt, el turno cabe en el presupuesto de latencia.
4. Lost in the middle: por encima de CENTER_RECALL_TOKENS, el modelo deja de usar bien lo
   que está en el centro del contexto.

La latencia y el coste salen de lo medido en evals/stress/REPORT.md con gpt-4o-mini (la
latencia crece ~1,4 s por cada 1.000 tokens de entrada) y los presupuestos son los mismos
del stress test: 10 s y 0,002 USD por turno.

Si CAG es viable, los ejes de decisión dicen si basta o conviene un híbrido: datos que se
renuevan más a menudo que la caché de prefijos (una semana), necesidad de citar la fuente
de cada dato (trazabilidad) o permisos distintos por usuario sobre los documentos. El
fine-tuning no aparece a propósito: enseña a responder, pero no añade datos nuevos ni da
trazabilidad.
"""

import argparse
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from app.config import get_settings
from app.ingestion.catalog import CatalogDecision, load_catalog
from app.ingestion.loaders import FileSystemLoader
from app.ingestion.parsers import ParseContext, default_registry

CONTEXT_WINDOW_TOKENS = 128_000  # gpt-4o-mini
USABLE_WINDOW_FRACTION = 0.7  # margen para system prompt, historial y respuesta
CENTER_RECALL_TOKENS = 100_000
INPUT_USD_PER_MTOK = 0.15  # gpt-4o-mini
BASE_TURN_TOKENS = 5_000  # system prompt + historial típico (stress test, turnos 1-6)
BASE_LATENCY_MS = 7_000  # P50 por debajo de 5.000 tokens (stress test)
LATENCY_MS_PER_KTOK = 1_400  # pendiente medida en el stress test
LATENCY_BUDGET_MS = 10_000
COST_BUDGET_USD = 0.002
CACHE_FRIENDLY_REFRESH_DAYS = 7
CHARS_PER_TOKEN = 4  # aproximación para español

Architecture = Literal["CAG", "Hybrid", "RAG"]


@dataclass
class CAGViability:
    corpus_tokens: int
    context_window_ok: bool
    cost_ok: bool
    latency_ok: bool
    lost_in_the_middle_ok: bool
    estimated_cost_usd: float
    estimated_latency_ms: int

    @property
    def viable(self) -> bool:
        return all(
            (self.context_window_ok, self.cost_ok, self.latency_ok, self.lost_in_the_middle_ok)
        )


def evaluate_cag(corpus_tokens: int) -> CAGViability:
    turn_tokens = corpus_tokens + BASE_TURN_TOKENS
    cost = turn_tokens * INPUT_USD_PER_MTOK / 1_000_000
    latency = BASE_LATENCY_MS + max(0, turn_tokens - BASE_TURN_TOKENS) * LATENCY_MS_PER_KTOK // 1000
    return CAGViability(
        corpus_tokens=corpus_tokens,
        context_window_ok=turn_tokens <= CONTEXT_WINDOW_TOKENS * USABLE_WINDOW_FRACTION,
        cost_ok=cost <= COST_BUDGET_USD,
        latency_ok=latency <= LATENCY_BUDGET_MS,
        lost_in_the_middle_ok=corpus_tokens <= CENTER_RECALL_TOKENS,
        estimated_cost_usd=round(cost, 6),
        estimated_latency_ms=latency,
    )


@dataclass
class Recommendation:
    architecture: Architecture
    reasons: list[str] = field(default_factory=list)


def recommend_architecture(
    viability: CAGViability,
    *,
    refresh_days: int,
    needs_traceability: bool,
    needs_access_control: bool,
) -> Recommendation:
    if not viability.viable:
        failed = [
            name
            for name, ok in (
                ("no cabe en la ventana", viability.context_window_ok),
                (f"coste {viability.estimated_cost_usd} USD/turno", viability.cost_ok),
                (f"latencia {viability.estimated_latency_ms} ms", viability.latency_ok),
                ("lost in the middle", viability.lost_in_the_middle_ok),
            )
            if not ok
        ]
        return Recommendation("RAG", [f"CAG no es viable: {', '.join(failed)}"])
    reasons = []
    if refresh_days < CACHE_FRIENDLY_REFRESH_DAYS:
        reasons.append(f"los datos se renuevan cada {refresh_days} días: la caché no compensa")
    if needs_traceability:
        reasons.append("hay que citar la fuente de cada dato")
    if needs_access_control:
        reasons.append("hay permisos distintos por usuario sobre los documentos")
    if reasons:
        return Recommendation("Hybrid", reasons)
    return Recommendation("CAG", ["el corpus cabe y cumple las cuatro restricciones"])


def corpus_tokens() -> int:
    """Tokens aproximados del texto de las fuentes incluidas en el catálogo."""
    settings, catalog, registry = get_settings(), load_catalog(), default_registry()
    loader = FileSystemLoader(settings.data_dir)
    chars = 0
    for source in catalog.sources:
        if source.decision != CatalogDecision.INCLUDE:
            continue
        context = ParseContext(source, catalog.version, datetime.now(UTC))
        parser = registry.for_format(source.format)
        for blob in loader.load(source):
            chars += sum(len(d.text) for d in parser.parse(blob, context))
    return chars // CHARS_PER_TOKEN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="¿CAG, RAG o híbrido para este corpus?")
    parser.add_argument("--scale", type=float, default=1.0, help="Simula un corpus N veces mayor")
    parser.add_argument("--refresh-days", type=int, default=30)
    parser.add_argument("--traceability", action="store_true", help="Hay que citar la fuente")
    parser.add_argument("--access-control", action="store_true", help="Permisos por usuario")
    args = parser.parse_args(argv)

    tokens = int(corpus_tokens() * args.scale)
    viability = evaluate_cag(tokens)
    rec = recommend_architecture(
        viability,
        refresh_days=args.refresh_days,
        needs_traceability=args.traceability,
        needs_access_control=args.access_control,
    )
    print(f"Corpus: ~{tokens:,} tokens (×{args.scale:g})".replace(",", "."))
    for name, ok in (
        ("Ventana de contexto", viability.context_window_ok),
        (f"Coste ({viability.estimated_cost_usd} USD/turno)", viability.cost_ok),
        (f"Latencia (~{viability.estimated_latency_ms} ms/turno)", viability.latency_ok),
        ("Lost in the middle", viability.lost_in_the_middle_ok),
    ):
        print(f"  {'OK   ' if ok else 'FALLA'} {name}")
    print(f"Recomendación: {rec.architecture}. " + "; ".join(rec.reasons))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
