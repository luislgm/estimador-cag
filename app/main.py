from fastapi import FastAPI

from app.config import get_settings
from app.routers import estimations

app = FastAPI(
    title="Estimador CAG",
    description=(
        "API que genera estimaciones de proyectos de software a partir de la transcripción "
        "de una reunión con el cliente. Usa una arquitectura CAG (Cache-Augmented Generation): "
        "las estimaciones históricas viajan como contexto en el prompt del LLM."
    ),
    version="0.1.0",
)

app.include_router(estimations.router, prefix="/api/v1")


@app.get("/health", tags=["health"])
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "environment": settings.app_env,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
    }
