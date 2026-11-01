from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class EstimationRequest(BaseModel):
    # Quita espacios al principio y al final antes de validar la longitud
    model_config = ConfigDict(str_strip_whitespace=True)

    transcription: str = Field(
        ...,
        min_length=50,
        description="Transcripción de la reunión con el cliente (mínimo 50 caracteres)",
        examples=[
            "En la reunión con el cliente se discutió la necesidad de una web para "
            "reservar mesas en su restaurante, con pago online y un panel para el personal."
        ],
    )


class TokenUsage(BaseModel):
    input_tokens: int = Field(..., description="Tokens enviados al modelo")
    output_tokens: int = Field(..., description="Tokens generados por el modelo")
    total_tokens: int = Field(..., description="Suma de tokens de entrada y salida")


class EstimationResponse(BaseModel):
    estimation: str = Field(..., description="Estimación generada en formato markdown")
    model: str = Field(..., description="Modelo que ha generado la estimación")
    provider: str = Field(..., description="Proveedor del LLM")
    usage: TokenUsage = Field(..., description="Consumo de tokens de la llamada")
    cost_usd: float | None = Field(
        ..., description="Coste estimado en USD (null si el modelo no tiene precio configurado)"
    )
    finish_reason: str = Field(
        ...,
        description="Motivo de fin de la generación. 'length' indica estimación cortada",
        examples=["stop"],
    )
    created_at: datetime = Field(..., description="Fecha y hora de la estimación (UTC)")
