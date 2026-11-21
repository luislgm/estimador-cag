from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
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
    # Compresión de los turnos que salen de la ventana: los que contienen un compromiso (NDA,
    # contrato, alcance cerrado...) se conservan literales como anclas, como mucho
    # SESSION_MAX_ANCHORS; el resto se resume con el LLM. Desactivada, solo hay ventana.
    session_compression_enabled: bool = True
    session_max_anchors: int = Field(default=4, ge=0, le=20)
    # Ciclo Actor-Critic-Boss (POST /sessions/{id}/estimate-acb): vueltas como mucho, cada una
    # con una estimación del actor y una revisión del crítico
    acb_max_iterations: int = Field(default=2, ge=1, le=4)

    # Ingesta de datos (app/ingestion): catálogo y datos de origen, y salida procesada
    data_dir: Path = Path("data")
    processed_dir: Path = Path("data/processed")
    # Base de datos de la ingesta (trabajos y seudónimos). Sin URL, el endpoint no se activa
    # (el CLI funciona igual, con los seudónimos en memoria). Ej.: postgresql+psycopg://...
    database_url: str | None = None
    # Sal secreta del HMAC de la seudonimización. Sin ella no se ingieren fuentes con datos
    # personales: con una sal conocida, los hashes se podrían revertir por fuerza bruta.
    pii_salt: SecretStr | None = None

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
