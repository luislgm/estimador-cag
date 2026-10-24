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
│   │   └── estimations.py      # Endpoint POST /api/v1/estimate
│   ├── schemas/                # Contratos de datos (Pydantic) de request y response
│   │   ├── estimation.py
│   │   └── health.py
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
| `LOG_LEVEL` | Nivel de logging (structlog; en `production` los logs salen en JSON) | `DEBUG` |
| `PHOENIX_COLLECTOR_ENDPOINT` | Endpoint de Arize Phoenix para las trazas del LLM (opcional) | Sin default (desactivado) |
| `PHOENIX_PROJECT_NAME` | Proyecto de Phoenix donde se guardan las trazas | `estimador-cag` |

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

### Con Docker

Con el `.env` configurado:

```bash
docker compose up --build
```

O sin Compose:

```bash
docker build -t estimador-cag .
docker run --rm -p 8000:8000 --env-file .env estimador-cag
```

La imagen se publica automáticamente en GitHub Container Registry con cada push a `main`:

```bash
docker run --rm -p 8000:8000 --env-file .env ghcr.io/luislgm/estimador-cag:latest
```

> El `.env` nunca se copia dentro de la imagen (está en `.dockerignore`): la configuración se pasa al arrancar el contenedor.

### Interfaz de chat (Streamlit)

Además de la API, hay una interfaz de chat en [`streamlit_app.py`](streamlit_app.py) para pegar una transcripción y ver la estimación:

```bash
uv run --group ui streamlit run streamlit_app.py
```

Se abre en http://localhost:8501. La estimación se muestra **en streaming**, escribiéndose token a token con `st.write_stream`. El panel lateral muestra el **contexto CAG**: el system prompt activo, las estimaciones de ejemplo inyectadas y las métricas de la última llamada (modelo, tokens de entrada y salida, tiempo total y tiempo hasta el primer token). Usa el mismo servicio que el endpoint (mismo system prompt, ejemplos y `.env`), mantiene el historial durante la sesión y aplica la misma validación de 50 caracteres mínimos. Streamlit está en un grupo de dependencias aparte (`ui`) para no incluirlo en la imagen Docker de la API.

## Logs

Los logs son estructurados ([structlog](https://www.structlog.org/)): texto legible en desarrollo y una línea JSON por evento con `APP_ENV=production`.

- Cada petición recibe un `request_id` que aparece en **todos** sus logs y se devuelve en la cabecera `X-Request-ID`. Si el cliente envía esa cabecera, se reutiliza su valor.
- Cada llamada al LLM registra `llm_call_completed` con modelo, tokens, coste, `finish_reason` y latencia (o `llm_call_failed` si falla).
- Las peticiones a `/health` se registran a nivel `debug` para no llenar el log con el healthcheck de Docker.

```
19:20:41 [info     ] request_completed   duration_ms=5 method=POST path=/api/v1/estimate request_id=prueba-1 status_code=422
```

## Trazas de las llamadas al LLM (Arize Phoenix)

[Arize Phoenix](https://phoenix.arize.com/) guarda cada llamada a Claude con el **system prompt** (instrucciones + ejemplos CAG), la transcripción, la respuesta, los tokens y la latencia, y lo muestra en un panel web. Todo se ejecuta **en local**: los datos no salen de tu máquina.

- `docker compose up -d` levanta Phoenix junto a la API. El panel está en http://localhost:6006 (proyecto `estimador-cag`).
- Se activa con `PHOENIX_COLLECTOR_ENDPOINT` en el `.env` (`http://localhost:6006/v1/traces` para la API o Streamlit en local; dentro de Docker Compose la API usa `http://phoenix:6006/v1/traces`). **Si no está definida, no se activa nada** y la app funciona igual.
- Cubre las dos vías: la API (`messages.create`) y el streaming de Streamlit (`messages.stream`).
- La instrumentación ([`app/tracing.py`](app/tracing.py)) usa OpenInference y no requiere cambios en el servicio. Las trazas propias del SDK de Anthropic se desactivan (`open_telemetry=False`) para que no salgan duplicadas.

## Documentación de la API

FastAPI genera la documentación automáticamente:

- **Swagger UI:** http://127.0.0.1:8000/docs (permite probar los endpoints)
- **ReDoc:** http://127.0.0.1:8000/redoc
- **Esquema OpenAPI:** http://127.0.0.1:8000/openapi.json

## Endpoints

### `POST /api/v1/estimate`

Genera una estimación a partir de la transcripción de una reunión. La transcripción debe tener **al menos 50 caracteres** (sin contar espacios al principio y al final), para evitar llamadas inútiles al LLM.

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
  "cost_usd": 0.0027,
  "finish_reason": "end_turn",
  "created_at": "2026-10-17T16:28:36Z"
}
```

- `cost_usd`: coste estimado según la tabla de precios de [`pricing.py`](app/services/pricing.py); `null` si el modelo no está en la tabla.
- `finish_reason`: `end_turn` si la estimación terminó con normalidad; `max_tokens` si se cortó por el límite de tokens de salida.

**Errores:**

| Código | Motivo |
|---|---|
| `422` | Falta el campo `transcription` o tiene menos de 50 caracteres (no se llama al LLM) |
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

## Tests y calidad de código

Los tests simulan la llamada al LLM, así que no necesitan API key ni consumen tokens.

```bash
uv run pytest --cov          # tests con cobertura (mínimo 80 %)
uv run ruff check .          # lint
uv run ruff format .         # formatear el código
```

## Integración continua

El workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) se ejecuta en cada push y pull request a `main`:

| Job | Qué comprueba |
|---|---|
| **Lint y formato** | `ruff check` y `ruff format --check` |
| **Tests** | Instala desde `uv.lock`, compila el código y pasa los tests con cobertura en Python 3.11, 3.12 y 3.13 |
| **Auditoría de dependencias** | `pip-audit` busca vulnerabilidades conocidas en las dependencias |
| **Imagen Docker** | Construye la imagen, arranca el contenedor y comprueba que `/health` y `/docs` responden. En push a `main` la publica en `ghcr.io` con las etiquetas `latest` y `sha-<commit>` |

## Ejemplos de contexto

Las estimaciones históricas están en [`app/context/examples.py`](app/context/examples.py). Cada ejemplo incluye:

- `meeting_summary`: resumen de lo que pedía el cliente.
- `estimation`: desglose de tareas con horas y costes (tarifa de 50 €/h), equipo recomendado y duración.

Los datos son ficticios. Para mejorar la calidad de las estimaciones, sustitúyelos por proyectos reales y representativos del tipo de trabajo que se quiere estimar.
