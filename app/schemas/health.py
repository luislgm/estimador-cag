from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado del servicio", examples=["ok"])
    environment: str = Field(..., description="Entorno de ejecución")
    provider: str = Field(..., description="Proveedor del LLM configurado")
    model: str = Field(..., description="Modelo del LLM configurado")
