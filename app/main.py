import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request

from app.config import get_settings
from app.logging_config import setup_logging
from app.routers import estimations
from app.schemas.health import HealthResponse
from app.tracing import setup_tracing

log = structlog.get_logger(module=__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level, settings.app_env)
    setup_tracing(settings)
    log.info(
        "application_started",
        environment=settings.app_env,
        provider=settings.llm_provider,
        model=settings.llm_model,
    )
    yield
    log.info("application_shutdown")


app = FastAPI(
    title="Estimador CAG",
    description=(
        "API que genera estimaciones de proyectos de software a partir de la transcripción "
        "de una reunión con el cliente. Usa una arquitectura CAG (Cache-Augmented Generation): "
        "las estimaciones históricas viajan como contexto en el prompt del LLM."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Asigna un request_id a cada petición y lo añade a todos sus logs y a la respuesta."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id, method=request.method, path=request.url.path
    )
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    # El healthcheck de Docker llama a /health cada 30 s: a nivel debug para no llenar el log
    log_method = log.debug if request.url.path == "/health" else log.info
    log_method(
        "request_completed",
        status_code=response.status_code,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    return response


app.include_router(estimations.router, prefix="/api/v1")


@app.get("/health", tags=["health"], response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        environment=settings.app_env,
        provider=settings.llm_provider,
        model=settings.llm_model,
    )
