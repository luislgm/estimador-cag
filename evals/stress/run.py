"""Prueba de carga del CAG: conversaciones largas, adjuntos grandes y varias en paralelo.

    uv run python -m evals.stress.run                       # app en proceso, todo por defecto
    uv run python -m evals.stress.run --http http://localhost:8000 \\
        --scenarios growing,pivot,contradiction --attachment-sizes 0,5,20,50,100 \\
        --repeats 3 --output evals/stress/results.csv

Para cada escenario × tamaño de adjunto × repetición abre una sesión y envía los turnos del
escenario (el adjunto va en el turno 1). Tras cada turno:

- lee la observación del turno (session.last_turn en la respuesta: lo mismo que el evento
  turn_observed) en vez de parsear los logs, porque funciona igual en proceso y por HTTP;
- lee la memoria completa con GET /sessions/{id}/snapshot y mide qué hechos de los turnos
  anteriores siguen en ella (MemoryDriftMetric);
- evalúa los presupuestos de latencia y coste, y escribe una fila en el CSV.

Las conversaciones se lanzan en paralelo (--concurrency). Con --burst 1,5,10,20 se mide
además la ráfaga: N primeros turnos simultáneos en sesiones distintas, para ver cómo crece la
latencia con la concurrencia (filas con scenario = burst_N).

Cuesta dinero: unas 900 llamadas con la configuración por defecto (~1 $ con gpt-4o-mini).
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from evals.metrics import normalize
from evals.stress.fixtures.build_pdfs import HEAD_FACT, TAIL_FACT, build_all, fixture_path
from evals.stress.metrics import (
    LOCATIONS,
    LONG_TERM,
    CostBudgetMetric,
    LatencyBudgetMetric,
    MemoryDriftMetric,
)
from evals.stress.scenarios import SCENARIOS, Scenario

STRESS_DIR = Path(__file__).parent
DEFAULT_OUTPUT = STRESS_DIR / "results.csv"
TIMEOUT = 300
FORM = {"project_type": "web_saas", "detail_level": "medium", "output_format": "phases_table"}

# Presupuestos: contratos de diseño, fijados antes de medir.
# - 10 s por turno: más allá, una herramienta interactiva se percibe como colgada.
# - 0,002 USD por turno: con 1.000 turnos al día, 2 USD diarios por cliente.
LATENCY_BUDGET_MS = 10_000
COST_BUDGET_USD = 0.002

OBSERVATION_FIELDS = [
    "turn_index",
    "session_id",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
    "llm_calls",
]
FIELDS = [
    "scenario",
    "attachment_kb",
    "repeat",
    "status_code",
    *OBSERVATION_FIELDS,
    "total_hours",
    "latency_budget",
    "cost_budget",
    "facts_checked",
    "recall_long_term",
    "recall_context",
    "project_name_long_term",
    "project_name_context",
    "stale_facts_in_metadata",
    "stale_facts_in_long_term",
    "attachment_head_recall",
    "attachment_tail_recall",
    "error",
]


def _client(base_url: str | None) -> httpx.AsyncClient:
    if base_url:
        return httpx.AsyncClient(base_url=base_url, timeout=TIMEOUT)
    from app.main import app  # solo en proceso

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://stress", timeout=TIMEOUT
    )


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def memory_columns(scenario: Scenario, turn_index: int, snapshot: dict) -> dict[str, Any]:
    """Recall de los hechos de los turnos anteriores, y hechos obsoletos que siguen ahí."""
    facts = scenario.facts_before(turn_index)
    long_term = [MemoryDriftMetric(f, LONG_TERM).evaluate(snapshot).score for f in facts]
    context = [MemoryDriftMetric(f, LOCATIONS).evaluate(snapshot).score for f in facts]
    stale = scenario.stale_facts_before(turn_index)
    project = scenario.turns[0].fact
    return {
        "facts_checked": len(facts),
        "recall_long_term": _mean(long_term),
        "recall_context": _mean(context),
        "project_name_long_term": (
            MemoryDriftMetric(project, LONG_TERM).evaluate(snapshot).score
            if turn_index > 1
            else None
        ),
        "project_name_context": (
            MemoryDriftMetric(project, LOCATIONS).evaluate(snapshot).score
            if turn_index > 1
            else None
        ),
        "stale_facts_in_metadata": sum(
            bool(MemoryDriftMetric(f, ("metadata",)).found_in(snapshot)) for f in stale
        ),
        "stale_facts_in_long_term": sum(
            bool(MemoryDriftMetric(f, LONG_TERM).found_in(snapshot)) for f in stale
        ),
    }


def attachment_columns(body: dict, attachment_kb: int, turn_index: int) -> dict[str, Any]:
    """¿La respuesta del turno con el adjunto recoge el hecho inicial y el final del PDF?"""
    if attachment_kb == 0 or turn_index != 1:
        return {"attachment_head_recall": None, "attachment_tail_recall": None}
    answer = normalize(json.dumps(body["result"], ensure_ascii=False) + body["text"])
    return {
        "attachment_head_recall": float(normalize(HEAD_FACT) in answer),
        "attachment_tail_recall": float(normalize(TAIL_FACT) in answer),
    }


RATE_LIMIT_RETRIES = 3
RATE_LIMIT_WAIT_S = 20.0


async def post_turn(
    client: httpx.AsyncClient, session_id: str, transcript: str, files
) -> tuple[httpx.Response, dict]:
    """Envía un turno. Si el proveedor responde con límite de uso (rate limit), espera y
    reintenta: un turno fallido no cambia la sesión, así que reintentar es seguro."""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        response = await client.post(
            f"/api/v1/sessions/{session_id}/estimate",
            data=FORM | {"transcript": transcript},
            files=files,
        )
        body = response.json()
        limited = response.status_code == 502 and "ratelimit" in str(body).lower()
        if not limited or attempt == RATE_LIMIT_RETRIES:
            return response, body
        print(f"rate limit en {session_id[:8]}, reintento {attempt + 1}", file=sys.stderr)
        await asyncio.sleep(RATE_LIMIT_WAIT_S * (attempt + 1))
    raise AssertionError("inalcanzable")  # pragma: no cover


async def run_conversation(
    client: httpx.AsyncClient,
    scenario: Scenario,
    attachment_kb: int,
    repeat: int,
    max_turns: int,
    latency: LatencyBudgetMetric,
    cost: CostBudgetMetric,
) -> list[dict[str, Any]]:
    session_id = (await client.post("/api/v1/sessions")).json()["session_id"]
    rows = []
    for index, turn in enumerate(scenario.turns[:max_turns], start=1):
        base = {"scenario": scenario.name, "attachment_kb": attachment_kb, "repeat": repeat}
        files = None
        if index == 1 and attachment_kb:
            pdf = fixture_path(attachment_kb).read_bytes()
            files = [("attachments", (f"requisitos_{attachment_kb}kb.pdf", pdf, "application/pdf"))]
        try:
            response, body = await post_turn(client, session_id, turn.transcript, files)
            if response.status_code == 200:
                snapshot_response = await client.get(f"/api/v1/sessions/{session_id}/snapshot")
                snapshot = snapshot_response.json()
        except Exception as e:  # noqa: BLE001 - un fallo inesperado no debe tumbar la prueba
            # status 0: no hay respuesta HTTP (red caída, o una excepción de la app que el
            # transporte en proceso propaga en vez de convertirla en un 500)
            rows.append(base | {"turn_index": index, "status_code": 0, "error": repr(e)[:300]})
            break
        if response.status_code != 200:
            error = str(body.get("detail"))[:300]
            rows.append(
                base | {"turn_index": index, "status_code": response.status_code, "error": error}
            )
            break
        observation = body["session"]["last_turn"]
        rows.append(
            base
            | {"status_code": 200, **observation}
            | {
                "total_hours": body["result"]["total_hours"],
                "latency_budget": latency.evaluate(observation).score,
                "cost_budget": cost.evaluate(observation).score,
            }
            | memory_columns(scenario, index, snapshot)
            | attachment_columns(body, attachment_kb, index)
        )
    return rows


async def run_stress(
    scenarios: list[Scenario],
    attachment_sizes: list[int],
    repeats: int,
    *,
    max_turns: int = 20,
    base_url: str | None = None,
    concurrency: int = 4,
    latency_budget_ms: int = LATENCY_BUDGET_MS,
    cost_budget_usd: float = COST_BUDGET_USD,
    on_rows: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Todas las conversaciones; `on_rows` recibe las filas de cada una al terminar (para ir
    guardando el CSV y no perder lo medido si algo falla a mitad)."""
    build_all([kb for kb in attachment_sizes if kb])
    latency, cost = LatencyBudgetMetric(latency_budget_ms), CostBudgetMetric(cost_budget_usd)
    semaphore = asyncio.Semaphore(concurrency)
    jobs = [(s, kb, r) for s in scenarios for kb in attachment_sizes for r in range(1, repeats + 1)]
    done = 0
    started = time.perf_counter()

    async with _client(base_url) as client:

        async def job(scenario: Scenario, kb: int, repeat: int) -> list[dict[str, Any]]:
            nonlocal done
            async with semaphore:
                rows = await run_conversation(
                    client, scenario, kb, repeat, max_turns, latency, cost
                )
            done += 1
            if on_rows:
                on_rows(rows)
            failed = [r for r in rows if r["status_code"] != 200]
            print(
                f"[{done}/{len(jobs)}] {scenario.name} {kb} KB #{repeat}: {len(rows)} turnos"
                + (f", error {failed[0]['status_code']}" if failed else "")
                + f" ({time.perf_counter() - started:.0f} s)",
                file=sys.stderr,
                flush=True,
            )
            return rows

        results = await asyncio.gather(*(job(*j) for j in jobs))
    return [row for rows in results for row in rows]


async def run_burst(
    levels: list[int],
    *,
    base_url: str | None = None,
    latency_budget_ms: int = LATENCY_BUDGET_MS,
    cost_budget_usd: float = COST_BUDGET_USD,
) -> list[dict[str, Any]]:
    """Para cada nivel N, N primeros turnos a la vez (cada uno en su sesión)."""
    latency, cost = LatencyBudgetMetric(latency_budget_ms), CostBudgetMetric(cost_budget_usd)
    first_turn = SCENARIOS["growing"].turns[0].transcript
    rows: list[dict[str, Any]] = []
    async with _client(base_url) as client:

        async def one(level: int, repeat: int) -> dict[str, Any]:
            session_id = (await client.post("/api/v1/sessions")).json()["session_id"]
            base = {"scenario": f"burst_{level}", "attachment_kb": 0, "repeat": repeat}
            try:
                response, body = await post_turn(client, session_id, first_turn, None)
            except Exception as e:  # noqa: BLE001 - igual que en run_conversation
                return base | {"turn_index": 1, "status_code": 0, "error": repr(e)[:300]}
            if response.status_code != 200:
                error = str(body.get("detail"))[:300]
                return base | {"turn_index": 1, "status_code": response.status_code, "error": error}
            observation = body["session"]["last_turn"]
            return (
                base
                | {"status_code": 200, **observation}
                | {
                    "total_hours": body["result"]["total_hours"],
                    "latency_budget": latency.evaluate(observation).score,
                    "cost_budget": cost.evaluate(observation).score,
                }
            )

        for level in levels:
            rows += await asyncio.gather(*(one(level, i) for i in range(1, level + 1)))
            print(f"ráfaga de {level}: hecha", file=sys.stderr, flush=True)
    return rows


def write_csv(rows: list[dict[str, Any]], output: Path, append: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not (append and output.exists())
    with output.open("w" if new_file else "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def _int_list(value: str) -> list[int]:
    return [int(v) for v in value.split(",")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prueba de carga del CAG")
    parser.add_argument("--http", metavar="URL", help="API levantada en vez de la app en proceso")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--attachment-sizes", type=_int_list, default=[0, 5, 20, 50, 100])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--turns", type=int, default=20, help="Turnos por conversación (máx.)")
    parser.add_argument("--concurrency", type=int, default=4, help="Conversaciones en paralelo")
    parser.add_argument("--latency-budget-ms", type=int, default=LATENCY_BUDGET_MS)
    parser.add_argument("--cost-budget-usd", type=float, default=COST_BUDGET_USD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--burst", type=_int_list, default=[], help="Niveles de la ráfaga")
    parser.add_argument("--skip-scenarios", action="store_true", help="Solo la ráfaga")
    parser.add_argument("--append", action="store_true", help="Añadir al CSV existente")
    args = parser.parse_args(argv)

    names = args.scenarios.split(",")
    unknown = set(names) - set(SCENARIOS)
    if unknown:
        parser.error(f"escenarios desconocidos: {sorted(unknown)}")
    budgets = {
        "latency_budget_ms": args.latency_budget_ms,
        "cost_budget_usd": args.cost_budget_usd,
    }
    if not args.append:
        write_csv([], args.output)  # cabecera; las filas se añaden según terminan

    def save(new_rows: list[dict[str, Any]]) -> None:
        write_csv(new_rows, args.output, append=True)

    rows = []
    if not args.skip_scenarios:
        rows += asyncio.run(
            run_stress(
                [SCENARIOS[n] for n in names],
                args.attachment_sizes,
                args.repeats,
                max_turns=args.turns,
                base_url=args.http,
                concurrency=args.concurrency,
                on_rows=save,
                **budgets,
            )
        )
    if args.burst:
        burst_rows = asyncio.run(run_burst(args.burst, base_url=args.http, **budgets))
        save(burst_rows)
        rows += burst_rows
    failed = sum(r["status_code"] != 200 for r in rows)
    print(f"{len(rows)} filas ({failed} con error) -> {args.output}")
    return 0 if rows and not failed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
