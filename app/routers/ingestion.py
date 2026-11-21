"""Ingesta de datos por HTTP.

GET    /api/v1/ingestion/sources         el catálogo: qué fuentes hay y cuáles se ingieren
POST   /api/v1/ingestion/runs            lanza la ingesta de una fuente (202 + job_id)
GET    /api/v1/ingestion/jobs/{job_id}   estado e informe de una ejecución
GET    /api/v1/ingestion/jobs/{job_id}/documents   documentos que ha producido (vista previa)
DELETE /api/v1/ingestion/pseudonyms      derecho de supresión: olvida el seudónimo de un dato

El router es fino: valida la fuente contra el catálogo antes de aceptar el trabajo (así una
fuente excluida da un 400 al momento, no un trabajo fallido), registra el trabajo y lanza el
orquestador en segundo plano. Sin DATABASE_URL, los endpoints que necesitan la base de datos
responden 503 (la ingesta sigue disponible por CLI).
"""

import json
from pathlib import Path

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from app.config import get_settings
from app.ingestion.catalog import load_catalog
from app.ingestion.loaders import FileSystemLoader
from app.ingestion.orchestrator import IngestionRejected, check_source, ingest_source
from app.ingestion.parsers import default_registry
from app.ingestion.pii import Pseudonymizer, hash_value
from app.persistence import JobRepository, SqlMappingStore, session_factory
from app.schemas.ingestion import (
    DocumentView,
    ForgetRequest,
    ForgetResponse,
    IngestionJobView,
    IngestionRunRequest,
    SourceInfo,
)

log = structlog.get_logger(module=__name__)

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


def _sessions():
    sessions = session_factory()
    if sessions is None:
        raise HTTPException(
            status_code=503, detail="La ingesta por API necesita DATABASE_URL configurada"
        )
    return sessions


def _salt() -> bytes:
    salt = get_settings().pii_salt
    return salt.get_secret_value().encode() if salt else b""


def run_job(job_id: str, source: str) -> None:
    """La ejecución en segundo plano: actualiza el trabajo según avanza."""
    settings, sessions = get_settings(), session_factory()
    jobs = JobRepository(sessions)
    jobs.mark_running(job_id)
    try:
        salt = _salt()
        pseudonymizer = Pseudonymizer(salt, SqlMappingStore(sessions)) if salt else None
        _, report = ingest_source(
            load_catalog(),
            source,
            loader=FileSystemLoader(settings.data_dir),
            registry=default_registry(),
            pseudonymizer=pseudonymizer,
            output_dir=settings.processed_dir,
        )
    except Exception as e:  # noqa: BLE001 - cualquier fallo queda en el trabajo, no se pierde
        log.error("ingestion_failed", job_id=job_id, source=source, error=str(e))
        jobs.fail(job_id, f"{type(e).__name__}: {e}")
        return
    log.info("ingestion_completed", job_id=job_id, source=source, documents=report.documents)
    jobs.complete(job_id, report.to_dict())


@router.get("/sources", response_model=list[SourceInfo])
async def list_sources() -> list[SourceInfo]:
    data_dir = get_settings().data_dir
    return [
        SourceInfo(
            name=s.name,
            description=s.description,
            owner=s.owner,
            format=s.format,
            decision=s.decision.value,
            decision_reason=s.decision_reason,
            quality=s.quality.average,
            contains_pii=s.sensitivity.contains_pii,
            files=len(list(data_dir.glob(s.path))),
        )
        for s in load_catalog().sources
    ]


@router.post(
    "/runs",
    response_model=IngestionJobView,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"description": "La fuente no se puede ingerir (no existe, excluida, sin sal...)"},
        503: {"description": "No hay base de datos configurada"},
    },
)
async def start_run(body: IngestionRunRequest, background: BackgroundTasks) -> IngestionJobView:
    sessions = _sessions()
    try:
        check_source(load_catalog(), body.source, default_registry(), bool(_salt()))
    except IngestionRejected as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    jobs = JobRepository(sessions)
    job_id = jobs.create(body.source)
    background.add_task(run_job, job_id, body.source)
    log.info("ingestion_job_created", job_id=job_id, source=body.source)
    return _view(jobs.get(job_id))


@router.get("/jobs/{job_id}", response_model=IngestionJobView)
async def get_job(job_id: str) -> IngestionJobView:
    job = JobRepository(_sessions()).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    return _view(job)


@router.get("/jobs/{job_id}/documents", response_model=list[DocumentView])
async def get_job_documents(
    job_id: str, limit: int = Query(20, ge=1, le=200)
) -> list[DocumentView]:
    """Los primeros documentos que produjo un trabajo terminado, leídos de su JSONL."""
    job = JobRepository(_sessions()).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    output = (job.report or {}).get("output")
    if job.status != "completed" or not output or not Path(output).exists():
        raise HTTPException(status_code=409, detail="El trabajo no tiene documentos (todavía)")
    documents = []
    with Path(output).open(encoding="utf-8") as lines:
        for line in lines:
            if len(documents) == limit:
                break
            data = json.loads(line)
            documents.append(
                DocumentView(
                    id=data["id"], kind=data["kind"], text=data["text"], metadata=data["metadata"]
                )
            )
    return documents


@router.delete("/pseudonyms", response_model=ForgetResponse)
async def forget_pseudonym(body: ForgetRequest) -> ForgetResponse:
    """Derecho de supresión: borra la relación entre el dato y su seudónimo. El dato llega en
    la petición solo para calcular su hash; no se guarda ni se registra en los logs."""
    sessions, salt = _sessions(), _salt()
    if not salt:
        raise HTTPException(status_code=503, detail="No hay PII_SALT configurada")
    original_hash = hash_value(salt, body.entity_type, body.value)
    forgotten = SqlMappingStore(sessions).forget(body.entity_type, original_hash)
    log.info("pseudonym_forgotten", entity_type=body.entity_type, forgotten=forgotten)
    return ForgetResponse(forgotten=forgotten)


def _view(job) -> IngestionJobView:
    return IngestionJobView(
        job_id=job.id,
        source=job.source,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        report=job.report,
        error=job.error,
    )
