"""Tablas del informe de la prueba de carga a partir de results.csv (Markdown).

    uv run python -m evals.stress.report [evals/stress/results.csv]

Imprime las tablas que van en REPORT.md: el resumen por escenario y tamaño de adjunto, las
tres curvas (latencia frente a tokens, coste acumulado frente a turno y recall frente a N),
el efecto de los adjuntos, los hechos obsoletos y la ráfaga. La lectura (los dos párrafos)
se escribe a mano a partir de estas tablas.
"""

import csv
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from evals.stress.run import DEFAULT_OUTPUT
from evals.stress.scenarios import SCENARIOS

TURN_POINTS = (1, 3, 6, 7, 10, 15, 20)
TOKEN_BUCKETS = ((0, 5_000), (5_000, 7_500), (7_500, 10_000), (10_000, 15_000), (15_000, 10**9))


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(value: str) -> float | None:
    return float(value) if value not in ("", None) else None


def values(rows: Iterable[dict], column: str) -> list[float]:
    return [v for v in (num(r[column]) for r in rows) if v is not None]


def percentile(data: list[float], pct: float) -> float | None:
    if not data:
        return None
    ordered = sorted(data)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def mean(data: list[float]) -> float | None:
    return statistics.fmean(data) if data else None


def fmt(value: float | None, pattern: str = "{:.0f}") -> str:
    return "—" if value is None else pattern.format(value)


def pct(value: float | None) -> str:
    return fmt(None if value is None else value * 100, "{:.0f} %")


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def ok(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["status_code"] == "200"]


def conversations(rows: list[dict]) -> dict[tuple, list[dict]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["scenario"], r["attachment_kb"], r["repeat"])].append(r)
    return grouped


def summary_table(rows: list[dict]) -> str:
    groups = defaultdict(list)
    for r in rows:
        if not r["scenario"].startswith("burst"):
            groups[(r["scenario"], int(r["attachment_kb"]))].append(r)
    out = []
    for (scenario, kb), group in sorted(groups.items()):
        good = ok(group)
        convs = conversations(good)
        cost = mean([sum(values(c, "cost_usd")) for c in convs.values()])
        latency = values(good, "latency_ms")
        out.append(
            [
                scenario,
                f"{kb} KB",
                f"{len(good)}/{len(group)}",
                fmt(percentile(latency, 50)),
                fmt(percentile(latency, 95)),
                fmt(cost, "{:.4f}"),
                pct(mean([float(r["cache_hit_kind"] == "exact") for r in good])),
                pct(mean([float(r["cache_hit_kind"] == "semantic") for r in good])),
                pct(mean(values(good, "recall_long_term"))),
                pct(mean(values(good, "recall_context"))),
                pct(mean(values(good, "latency_budget"))),
                pct(mean(values(good, "cost_budget"))),
            ]
        )
    header = [
        "Escenario",
        "Adjunto",
        "Turnos OK",
        "Latencia P50 (ms)",
        "Latencia P95 (ms)",
        "Coste medio por conversación (USD)",
        "Hit caché exacta",
        "Hit caché semántica",
        "Recall memoria largo plazo",
        "Recall contexto completo",
        "Dentro de presupuesto de latencia",
        "Dentro de presupuesto de coste",
    ]
    return table(header, out)


def latency_vs_tokens(rows: list[dict]) -> str:
    out = []
    for low, high in TOKEN_BUCKETS:
        bucket = [r for r in ok(rows) if low <= float(r["tokens_in"]) < high]
        latency = values(bucket, "latency_ms")
        label = f"≥ {low:,}".replace(",", ".") if high == 10**9 else f"{low:,}–{high:,}"
        out.append(
            [
                label.replace(",", "."),
                str(len(bucket)),
                fmt(percentile(latency, 50)),
                fmt(percentile(latency, 95)),
                fmt(mean(values(bucket, "cost_usd")), "{:.5f}"),
            ]
        )
    return table(
        ["tokens_in", "Turnos", "Latencia P50 (ms)", "Latencia P95 (ms)", "Coste medio (USD)"], out
    )


def _by_scenario(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for r in ok(rows):
        if not r["scenario"].startswith("burst"):
            grouped[r["scenario"]].append(r)
    return dict(sorted(grouped.items()))


def cost_vs_turn(rows: list[dict]) -> str:
    scenarios = _by_scenario(rows)
    out = []
    for turn in TURN_POINTS:
        line = [str(turn)]
        for group in scenarios.values():
            per_turn = values([r for r in group if int(r["turn_index"]) == turn], "cost_usd")
            cumulative = []
            for conv in conversations(group).values():
                upto = [r for r in conv if int(r["turn_index"]) <= turn]
                if len(upto) == turn:
                    cumulative.append(sum(values(upto, "cost_usd")))
            line += [fmt(mean(per_turn), "{:.5f}"), fmt(mean(cumulative), "{:.4f}")]
        out.append(line)
    header = ["Turno"]
    for name in scenarios:
        header += [f"{name}: coste del turno", f"{name}: acumulado"]
    return table(header, out)


def turn_cost_ratio(rows: list[dict], first: int = 1, last: int = 20) -> dict[str, float | None]:
    ratios = {}
    for name, group in _by_scenario(rows).items():
        a = mean(values([r for r in group if int(r["turn_index"]) == first], "cost_usd"))
        b = mean(values([r for r in group if int(r["turn_index"]) == last], "cost_usd"))
        ratios[name] = b / a if a and b else None
    return ratios


def recall_vs_n(rows: list[dict]) -> str:
    scenarios = _by_scenario(rows)
    out = []
    for turn in TURN_POINTS[1:]:
        line = [str(turn)]
        for group in scenarios.values():
            at = [r for r in group if int(r["turn_index"]) == turn]
            line += [
                pct(mean(values(at, "recall_long_term"))),
                pct(mean(values(at, "recall_context"))),
                pct(mean(values(at, "project_name_long_term"))),
            ]
        out.append(line)
    header = ["N"]
    for name in scenarios:
        header += [f"{name}: largo plazo", f"{name}: contexto", f"{name}: nombre"]
    return table(header, out)


def attachments_table(rows: list[dict]) -> str:
    sizes = sorted(
        {int(r["attachment_kb"]) for r in ok(rows) if not r["scenario"].startswith("burst")}
    )
    out = []
    for kb in sizes:
        group = [r for r in ok(rows) if int(r["attachment_kb"]) == kb]
        first = [r for r in group if r["turn_index"] == "1"]
        window = [r for r in group if 2 <= int(r["turn_index"]) <= 6]
        after = [r for r in group if int(r["turn_index"]) >= 8]
        out.append(
            [
                f"{kb} KB",
                fmt(mean(values(first, "attachments_total_chars"))),
                fmt(mean(values(first, "tokens_in"))),
                fmt(percentile(values(first, "latency_ms"), 50)),
                fmt(mean(values(first, "cost_usd")), "{:.5f}"),
                pct(mean(values(first, "attachment_head_recall"))),
                pct(mean(values(first, "attachment_tail_recall"))),
                fmt(mean(values(window, "tokens_in"))),
                fmt(mean(values(after, "tokens_in"))),
            ]
        )
    header = [
        "Adjunto",
        "Caracteres que llegan",
        "tokens_in turno 1",
        "Latencia P50 turno 1 (ms)",
        "Coste turno 1 (USD)",
        "Recall inicio del PDF",
        "Recall final del PDF",
        "tokens_in turnos 2-6",
        "tokens_in turnos 8+",
    ]
    return table(header, out)


def stale_table(rows: list[dict]) -> str:
    out = []
    for name, group in _by_scenario(rows).items():
        if not SCENARIOS[name].stale_facts_before(len(SCENARIOS[name].turns) + 1):
            continue  # el escenario no deja ningún hecho obsoleto
        later = [r for r in group if int(r["turn_index"]) >= 10]
        out.append(
            [
                name,
                pct(mean([float(float(r["stale_facts_in_metadata"]) > 0) for r in later])),
                pct(mean([float(float(r["stale_facts_in_long_term"]) > 0) for r in later])),
            ]
        )
    return table(
        ["Escenario", "Hecho obsoleto en metadata (turnos ≥ 10)", "… en memoria a largo plazo"],
        out,
    )


def burst_table(rows: list[dict]) -> str:
    groups = defaultdict(list)
    for r in rows:
        if r["scenario"].startswith("burst"):
            groups[int(r["scenario"].split("_")[1])].append(r)
    out = []
    for level, group in sorted(groups.items()):
        latency = values(ok(group), "latency_ms")
        out.append(
            [
                str(level),
                f"{len(ok(group))}/{len(group)}",
                fmt(percentile(latency, 50)),
                fmt(percentile(latency, 95)),
                pct(mean(values(ok(group), "latency_budget"))),
            ]
        )
    return table(
        [
            "Peticiones simultáneas",
            "OK",
            "Latencia P50 (ms)",
            "Latencia P95 (ms)",
            "Dentro de presupuesto",
        ],
        out,
    )


def render(rows: list[dict]) -> str:
    ratios = turn_cost_ratio(rows)
    ratio_text = ", ".join(f"{k}: ×{fmt(v, '{:.1f}')}" for k, v in ratios.items())
    sections = [
        ("Resumen por escenario y adjunto", summary_table(rows)),
        ("Curva 1: latencia frente a tokens de entrada", latency_vs_tokens(rows)),
        (
            "Curva 2: coste frente a turno",
            cost_vs_turn(rows) + f"\n\nCoste del turno 20 / turno 1: {ratio_text}",
        ),
        ("Curva 3: recall de hechos frente a N", recall_vs_n(rows)),
        ("Adjuntos", attachments_table(rows)),
        ("Hechos obsoletos", stale_table(rows)),
        ("Ráfaga", burst_table(rows)),
    ]
    total = len(rows)
    failed = total - len(ok(rows))
    header = f"{total} filas, {failed} con error.\n"
    return header + "\n".join(f"\n### {title}\n\n{body}" for title, body in sections)


def main(argv: list[str] | None = None) -> int:
    path = Path(argv[0]) if argv else DEFAULT_OUTPUT
    print(render(load(path)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
