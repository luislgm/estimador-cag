# Estimador CAG

API que genera **estimaciones de proyectos de software** a partir de la transcripción de una reunión con un cliente, usando un LLM (Claude de Anthropic u OpenAI).

## ¿Qué es CAG?

**CAG (Cache-Augmented Generation)** es una arquitectura en la que el conocimiento que necesita el modelo viaja **directamente en el prompt** de cada llamada, en lugar de recuperarse de una base de datos vectorial como en RAG.

En este proyecto, ese conocimiento son **estimaciones históricas** de proyectos anteriores. Se incluyen en el system prompt como *few-shot examples* para que el modelo genere nuevas estimaciones con el mismo criterio: granularidad de tareas, horas, tarifa, equipo y duración.

```
[system]    → Instrucciones + ejemplos de estimaciones previas
[user]      → Transcripción de la reunión a estimar
[assistant] → Estimación generada por el modelo
```

## Estructura del proyecto

```
estimador-cag/
├── app/
│   ├── main.py                 # App FastAPI: router, /health y metadatos de Swagger
│   ├── config.py               # Configuración con Pydantic BaseSettings (.env)
│   ├── context/
│   │   └── examples.py         # Estimaciones históricas (few-shot examples)
│   ├── routers/
│   │   └── estimations.py      # Endpoint POST /api/v1/estimate y schemas
│   └── services/
│       └── llm_service.py      # Construcción del prompt y llamada al LLM
├── .env.example                # Plantilla de variables de entorno
└── pyproject.toml              # Dependencias (gestionadas con uv)
```

## Requisitos

- Python 3.11 o superior
- [uv](https://docs.astral.sh/uv/)
- Una API key de [Anthropic](https://console.anthropic.com/)

## Instalación

```bash
git clone <url-del-repositorio>
cd estimador-cag
uv sync
```

## Configuración

Copia la plantilla y rellena tu API key:

```bash
cp .env.example .env
```

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `OPENAI_API_KEY` | API key de OpenAI | Sin default (requerida si se usa OpenAI) |
| `ANTHROPIC_API_KEY` | API key de Anthropic | Sin default (requerida si se usa Anthropic) |
| `LLM_PROVIDER` | Proveedor a utilizar: `openai` o `anthropic` | `openai` |
| `LLM_MODEL` | Modelo a utilizar | `gpt-4o-mini` |
| `APP_ENV` | Entorno de ejecución | `development` |
| `LOG_LEVEL` | Nivel de logging | `DEBUG` |

Configuración usada en este proyecto:

```env
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5
```

> La aplicación valida al arrancar que exista la API key del proveedor elegido. Si falta, falla con un error claro.
>
> Actualmente el servicio LLM solo implementa la llamada a **Anthropic**.

## Ejecución

```bash
uv run uvicorn app.main:app --reload
```

La API queda disponible en http://127.0.0.1:8000.

## Documentación de la API

FastAPI genera la documentación automáticamente:

- **Swagger UI:** http://127.0.0.1:8000/docs (permite probar los endpoints)
- **ReDoc:** http://127.0.0.1:8000/redoc
- **Esquema OpenAPI:** http://127.0.0.1:8000/openapi.json

## Endpoints

### `POST /api/v1/estimate`

Genera una estimación a partir de la transcripción de una reunión.

**Request:**

```json
{
  "transcription": "En la reunión con el cliente se discutió la necesidad de..."
}
```

**Response `200`:**

```json
{
  "estimation": "## Estimación: ...\n\n### Desglose de tareas:\n...",
  "model": "claude-haiku-4-5",
  "provider": "anthropic",
  "usage": {
    "input_tokens": 1200,
    "output_tokens": 300,
    "total_tokens": 1500
  },
  "created_at": "2026-10-17T16:28:36Z"
}
```

**Errores:**

| Código | Motivo |
|---|---|
| `422` | Falta el campo `transcription` o está vacío |
| `502` | Error al llamar a la API del LLM (key inválida, límite de uso, etc.) |

**Ejemplo con curl:**

```bash
curl -X POST http://127.0.0.1:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{"transcription": "El cliente quiere una web para reservar mesas en su restaurante con pago online"}'
```

### `GET /health`

Devuelve el estado del servicio y la configuración activa.

```json
{
  "status": "ok",
  "environment": "development",
  "provider": "anthropic",
  "model": "claude-haiku-4-5"
}
```

## Ejemplos de contexto

Las estimaciones históricas están en [`app/context/examples.py`](app/context/examples.py). Cada ejemplo incluye:

- `meeting_summary`: resumen de lo que pedía el cliente.
- `estimation`: desglose de tareas con horas y costes (tarifa de 50 €/h), equipo recomendado y duración.

Los datos son ficticios. Para mejorar la calidad de las estimaciones, sustitúyelos por proyectos reales y representativos del tipo de trabajo que se quiere estimar.
