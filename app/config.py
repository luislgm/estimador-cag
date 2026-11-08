from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración de la aplicación cargada desde variables de entorno / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API keys (sin default: requeridas según el proveedor elegido)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # LLM
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o-mini"

    # Aplicación
    app_env: str = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    # Guardrail de moderación (API de OpenAI). Necesita OPENAI_API_KEY aunque el LLM sea otro
    moderation_enabled: bool = False

    # Caché exact-match de respuestas del LLM en Redis (opcional: sin URL no se activa)
    redis_url: str | None = None
    cache_ttl: int = 86400  # segundos que dura cada respuesta en la caché (1 día)

    # Caché semántica (Redis Stack + embeddings de OpenAI). off | log_only | on
    semantic_cache_mode: Literal["off", "log_only", "on"] = "off"
    semantic_cache_threshold: float = Field(default=0.95, ge=0, le=1)
    embedding_model: str = "openai/text-embedding-3-small"

    # Sesiones conversacionales: turnos (par user + assistant) que se envían al LLM
    session_max_turns: int = Field(default=6, ge=1, le=50)

    # Trazas de las llamadas al LLM en Arize Phoenix (opcional: sin endpoint no se activa)
    phoenix_collector_endpoint: str | None = None
    phoenix_project_name: str = "estimador-cag"

    @model_validator(mode="after")
    def check_api_key_for_provider(self) -> "Settings":
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY es obligatoria cuando LLM_PROVIDER=openai")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY es obligatoria cuando LLM_PROVIDER=anthropic")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
