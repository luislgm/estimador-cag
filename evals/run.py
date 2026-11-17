"""Lanza el golden dataset contra la API con el LLM real y puntúa las respuestas.

    uv run python -m evals.run                          # todos los casos, app en proceso
    uv run python -m evals.run --case tier_tecnico_flotas --case web_corporativa_simple
    uv run python -m evals.run --http http://localhost:8000   # contra la API levantada
    uv run python -m evals.run --output informe.json
    uv run python -m evals.run --mode acb                 # con el ciclo Actor-Critic-Boss

Por defecto usa la app en proceso (httpx sobre ASGI, sin levantar servidor) con el proveedor
y el modelo de la configuración (.env o variables de entorno). Cada caso se envía a una
sesión nueva, turno a turno, como haría un usuario.

Imprime una tabla por caso y el resumen por métrica, y guarda el informe completo en JSON
(por defecto en evals/reports/) para comparar ejecuciones: un cambio de prompt o de modelo
que empeore alguna métrica se ve al momento. Sale con código 1 si alguna métrica queda por
debajo de su umbral, así que sirve como puerta de regresión.

Con --mode acb, cada turno pasa por /estimate-acb (crítico y, si hace falta, otra versión del
actor): sirve para medir si la revisión mejora las métricas o solo multiplica el coste.

Cuesta dinero: unos 0,01-0,03 $ por caso según el modelo (de 2 a 4 veces más en modo acb).
Por eso no va en la CI normal.
"""

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from evals.dataset import EvalCase, load_dataset
from evals.metrics import MetricResult, TurnOutcome, aggregate, score_case

REPORTS_DIR = Path(__file__).parent / "reports"
TIMEOUT = 180
Mode = Literal["actor", "acb"]
ENDPOINTS: dict[str, str] = {"actor": "estimate", "acb": "estimate-acb"}


@dataclass
class CaseReport:
    id: str
    seconds: float
    cost_usd: float
    metrics: list[MetricResult]
    turns: list[dict[str, Any]] = field(default_factory=list)  # respuesta resumida por turno


@dataclass
class SuiteReport:
    started_at: str
    mode: str
    provider: str
    model: str
    prompt_version: str | None
    seconds: float
    cost_usd: float
    metrics: dict[str, dict[str, Any]]
    cases: list[CaseReport]

    @property
    def passed(self) -> bool:
        return all(m["passed"] for m in self.metrics.values())


def _client(base_url: str | None) -> httpx.AsyncClient:
    if base_url:
        return httpx.AsyncClient(base_url=base_url, timeout=TIMEOUT)
    from app.main import app  # solo en proceso: con --http no hace falta importar la app

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://evals", timeout=TIMEOUT
    )


def _turn_summary(outcome: TurnOutcome) -> dict[str, Any]:
    if not outcome.ok:
        return {"status_code": outcome.status_code, "detail": outcome.body.get("detail")}
    result = outcome.body["result"]
    return {
        "status_code": 200,
        "project_name": result["project_name"],
        "total_hours": result["total_hours"],
        "confidence_pct": result["confidence_pct"],
        "tier": outcome.body["session"].get("tier"),
        "cost_usd": outcome.body.get("cost_usd"),
        "prompt_version": outcome.body.get("prompt_version"),
        **_acb_summary(outcome.body.get("acb")),
    }


def _acb_summary(acb: dict[str, Any] | None) -> dict[str, Any]:
    if acb is None:
        return {}
    return {
        "acb_outcome": acb["outcome"],
        "acb_hours": [i["total_hours"] for i in acb["iterations"]],
        "acb_llm_calls": acb["llm_calls"],
    }


async def run_case(client: httpx.AsyncClient, case: EvalCase, mode: Mode = "actor") -> CaseReport:
    started = time.perf_counter()
    session_id = (await client.post("/api/v1/sessions")).json()["session_id"]
    form = {
        "project_type": case.project_type.value,
        "detail_level": case.detail_level.value,
        "output_format": case.output_format.value,
    }
    outcomes: list[TurnOutcome] = []
    for message in case.turns:
        try:
            response = await client.post(
                f"/api/v1/sessions/{session_id}/{ENDPOINTS[mode]}",
                data=form | {"transcript": message},
            )
            outcome = TurnOutcome(response.status_code, response.json())
        except httpx.HTTPError as e:
            outcome = TurnOutcome(0, {"detail": f"{type(e).__name__}: {e}"})
        outcomes.append(outcome)
        if not outcome.ok:
            break
    return CaseReport(
        id=case.id,
        seconds=round(time.perf_counter() - started, 1),
        cost_usd=round(sum(o.body.get("cost_usd") or 0 for o in outcomes if o.ok), 6),
        metrics=score_case(case, outcomes),
        turns=[_turn_summary(o) for o in outcomes],
    )


async def run_suite(
    cases: list[EvalCase],
    *,
    base_url: str | None = None,
    concurrency: int = 4,
    mode: Mode = "actor",
) -> SuiteReport:
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(concurrency)
    async with _client(base_url) as client:
        health = (await client.get("/health")).json()

        async def limited(case: EvalCase) -> CaseReport:
            async with semaphore:
                return await run_case(client, case, mode)

        reports = await asyncio.gather(*(limited(case) for case in cases))
    prompt_versions = {
        t.get("prompt_version") for r in reports for t in r.turns if "prompt_version" in t
    }
    return SuiteReport(
        started_at=started_at,
        mode=mode,
        provider=health.get("provider", "?"),
        model=health.get("model", "?"),
        prompt_version=prompt_versions.pop() if len(prompt_versions) == 1 else None,
        seconds=round(time.perf_counter() - started, 1),
        cost_usd=round(sum(r.cost_usd for r in reports), 6),
        metrics=aggregate([r.metrics for r in reports]),
        cases=list(reports),
    )


def format_report(report: SuiteReport) -> str:
    lines = [
        f"Evals ({report.mode}) · {report.provider} · {report.model} · {len(report.cases)} casos · "
        f"{report.seconds} s · {report.cost_usd:.4f} USD",
        "",
    ]
    width = max(len(c.id) for c in report.cases)
    for case in report.cases:
        failed = [m for m in case.metrics if m.score < 1]
        status = "OK   " if not failed else "FALLO"
        acb = case.turns[-1].get("acb_hours") if case.turns else None
        extra = f"  revisión: {' -> '.join(map(str, acb))} h" if acb and len(acb) > 1 else ""
        lines.append(
            f"{status} {case.id:<{width}}  {case.seconds:>5} s  {case.cost_usd:.4f} USD{extra}"
        )
        for m in failed:
            lines.append(f"        {m.name} = {m.score:.2f}  ({m.detail})")
    lines += ["", "Métrica          Media  Umbral  Casos"]
    for name, m in report.metrics.items():
        mark = "OK   " if m["passed"] else "FALLO"
        lines.append(
            f"{mark} {name:<15} {m['mean']:>5.2f}  {m['threshold']:>6.2f}  {m['cases']:>5}"
        )
    lines += ["", "RESULTADO: " + ("pasa" if report.passed else "NO pasa")]
    return "\n".join(lines)


def save_report(report: SuiteReport, output: Path | None = None) -> Path:
    if output is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        output = REPORTS_DIR / f"{stamp}-{report.mode}-{report.model.replace('/', '_')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(report) | {"passed": report.passed}
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evals del estimador con el golden dataset")
    parser.add_argument("--case", action="append", help="Solo estos casos (se puede repetir)")
    parser.add_argument("--http", metavar="URL", help="API levantada en vez de la app en proceso")
    parser.add_argument("--output", type=Path, help="Ruta del informe JSON")
    parser.add_argument("--concurrency", type=int, default=4, help="Casos en paralelo")
    parser.add_argument(
        "--mode", choices=list(ENDPOINTS), default="actor", help="acb: con el crítico"
    )
    args = parser.parse_args(argv)

    cases = load_dataset()
    if args.case:
        unknown = set(args.case) - {c.id for c in cases}
        if unknown:
            parser.error(f"casos desconocidos: {sorted(unknown)}")
        cases = [c for c in cases if c.id in args.case]

    report = asyncio.run(
        run_suite(cases, base_url=args.http, concurrency=args.concurrency, mode=args.mode)
    )
    print(format_report(report))
    print(f"\nInforme: {save_report(report, args.output)}")
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
