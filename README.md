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
│   ├── prompts/                # Plantillas de prompt versionadas (Jinja2)
│   │   ├── loader.py           # render_estimation_prompt(request, version)
│   │   └── estimation/v1/      # system.j2, user.j2 y examples.j2
│   ├── routers/
│   │   └── estimations.py      # Endpoint POST /api/v1/estimate
│   ├── schemas/                # Contratos de datos (Pydantic) de request y response
│   │   ├── estimation.py
│   │   └── health.py
│   ├── services/
│   │   ├── llm_service.py      # Qué se pide: system prompt CAG + estimación
│   │   ├── llm_client.py       # Cómo se pide: wrapper sobre LiteLLM (OpenAI / Anthropic)
│   │   └── cache.py            # Caché exact-match en Redis
│   ├── logging_config.py       # Logs estructurados (structlog)
│   └── tracing.py              # Trazas del LLM en Arize Phoenix (opcional)
├── streamlit_app.py            # Cliente Streamlit (formulario)
├── .env.example                # Plantilla de variables de entorno
└── pyproject.toml              # Dependencias (gestionadas con uv)
```

## Requisitos

- Python 3.11 o superior
- [uv](https://docs.astral.sh/uv/)
- Una API key de [Anthropic](https://console.anthropic.com/) o de [OpenAI](https://platform.openai.com/)

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
| `REDIS_URL` | URL de Redis para la caché exact-match (opcional) | Sin default (desactivada) |
| `CACHE_TTL` | Segundos que dura cada respuesta en la caché | `86400` (1 día) |
| `ESTIMATOR_API_URL` | URL de la API que usa el cliente Streamlit | `http://127.0.0.1:8000` |

Ejemplos de configuración:

```env
# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5

# OpenAI
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

> La aplicación valida al arrancar que exista la API key del proveedor elegido. Si falta, falla con un error claro.

### Proveedores de LLM (LiteLLM)

Las llamadas al LLM pasan por [`app/services/llm_client.py`](app/services/llm_client.py), un wrapper sobre [LiteLLM](https://docs.litellm.ai/) que ofrece la misma interfaz para OpenAI y Anthropic. Para cambiar de proveedor basta con cambiar `LLM_PROVIDER` y `LLM_MODEL`: el resto del código no cambia.

- El modelo se envía a LiteLLM como `<proveedor>/<modelo>` (p. ej. `anthropic/claude-haiku-4-5`).
- Los tokens, el coste (tabla de precios de LiteLLM) y el motivo de fin (`stop` / `length`) salen igual para ambos proveedores, también en streaming.
- Los errores de cualquier proveedor se convierten en `LLMError`, que la API devuelve como `502`.
- La tabla de precios es la que trae el paquete de LiteLLM (`LITELLM_LOCAL_MODEL_COST_MAP=True`): se actualiza al actualizar LiteLLM.

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

### Cliente Streamlit

Un formulario para pedir estimaciones al servicio IA, en [`streamlit_app.py`](streamlit_app.py). Con la API en marcha:

```bash
uv run --group ui streamlit run streamlit_app.py
```

Se abre en http://localhost:8501.

- **Es un cliente de la API:** el formulario (`st.form`) construye un `EstimationRequest`, lo valida con el mismo schema de Pydantic que la API y lo envía a `POST /api/v1/estimate/stream`. Streamlit no llama al LLM ni necesita sus API keys; solo la URL de la API (`ESTIMATOR_API_URL`).
- **Streaming:** la estimación se escribe en tiempo real con `st.write_stream`, a partir de los eventos SSE.
- **Historial de la sesión:** la última estimación se muestra completa y las anteriores quedan plegadas debajo (con hora, título, tipo, detalle y formato), para comparar combinaciones. Se borra con "Nueva sesión" o al recargar la página.
- **Panel lateral:** estado de la API (proveedor y modelo), métricas de la última llamada (tokens, tiempos, coste, si vino de la caché), los prompts (system y user) de la última estimación y las estimaciones de ejemplo (contexto CAG).
- Streamlit está en un grupo de dependencias aparte (`ui`) para no incluirlo en la imagen Docker de la API.

## Prompts versionados

Los prompts son plantillas Jinja2 en [`app/prompts/`](app/prompts/), con la estructura `<caso_de_uso>/<versión>/<rol>.j2`:

```
app/prompts/
├── loader.py
└── estimation/
    └── v1/
        ├── system.j2     # rol, instrucciones y bloques condicionales; incluye examples.j2
        ├── user.j2       # envuelve la descripción del proyecto
        └── examples.j2   # 3 estimaciones de ejemplo (few-shot): el contexto CAG
```

- **`system.j2`**: rol del modelo e instrucciones generales, y **bloques condicionales** (`{% if %}`) según `project_type`, `output_format` y `detail_level`: solo se incluyen las instrucciones de las opciones elegidas. Añade los ejemplos con `{% include %}`.
- **`user.j2`**: envuelve la descripción del proyecto en `<descripcion_proyecto>`.
- **`examples.j2`**: tres proyectos inventados (app de fidelización, portal B2B y pipeline de datos), con el desglose de tareas, el total, el equipo y la duración. Sus horas y costes cuadran (hay un test que lo comprueba).

Los tests de las plantillas están en [`tests/prompts/test_estimation_v1.py`](tests/prompts/test_estimation_v1.py): renderizan los prompts sin llamar a ningún modelo (milisegundos) y comprueban, entre otras cosas, que la descripción aparece literal en su bloque, que cada bloque condicional solo aparece con su opción y que los ejemplos cuadran.

[`loader.py`](app/prompts/loader.py) expone `render_estimation_prompt(request, version="v1") -> tuple[str, str]`, que devuelve `(system, user)` listos para el modelo. Usa un `Environment` de Jinja2 con `StrictUndefined` (una variable que falte es un error, no un texto vacío), `trim_blocks=True` y `lstrip_blocks=True`.

La versión activa es `PROMPT_VERSION` en `loader.py` y se devuelve en cada respuesta (`prompt_version`). Para probar un prompt nuevo: copiar `v1/` como `v2/`, editarlo y cambiar `PROMPT_VERSION`. Como la caché incluye el system prompt en su clave, cambiar de versión (o de opciones del formulario) no sirve respuestas antiguas.

## Logs

Los logs son estructurados ([structlog](https://www.structlog.org/)): texto legible en desarrollo y una línea JSON por evento con `APP_ENV=production`.

- Cada petición recibe un `request_id` que aparece en **todos** sus logs y se devuelve en la cabecera `X-Request-ID`. Si el cliente envía esa cabecera, se reutiliza su valor.
- Cada llamada al LLM registra `llm_call_completed` con modelo, tokens, coste, `finish_reason` y latencia (o `llm_call_failed` si falla).
- Las peticiones a `/health` se registran a nivel `debug` para no llenar el log con el healthcheck de Docker.

```
19:20:41 [info     ] request_completed   duration_ms=5 method=POST path=/api/v1/estimate request_id=prueba-1 status_code=422
```

## Caché exact-match (Redis)

Si llega una petición **idéntica** a una anterior (mismo modelo, mismo system prompt, misma transcripción y mismo `max_tokens`), se devuelve la respuesta guardada **sin llamar al LLM**: gratis y en milisegundos.

- `docker compose up -d` levanta Redis junto a la API, con persistencia en disco (volumen `redis_data`). La API y Streamlit **comparten la misma caché**.
- Se activa con `REDIS_URL` (`redis://127.0.0.1:6379/0` fuera de Docker; dentro de Docker Compose la API usa `redis://redis:6379/0`). **Sin `REDIS_URL` no hay caché** y todo funciona igual.
- La clave incluye el system prompt completo: si cambias el prompt o los ejemplos CAG, la caché se invalida sola.
- En un acierto: `cache_hit=true` y `cost_usd=0` (no se ha gastado nada); los tokens son los de la respuesta original. En streaming, la respuesta guardada se entrega de una vez.
- Las respuestas cortadas (`finish_reason=length`) **no se guardan**, para no volver a servirlas incompletas.
- Si Redis no responde, la app sigue funcionando sin caché: espera como máximo 1 s, registra `cache_unavailable` y no vuelve a intentarlo durante 30 s.
- Los aciertos se registran en los logs como `llm_cache_hit` (con `saved_usd`). En Phoenix no aparecen, porque no hay llamada al LLM.

## Trazas de las llamadas al LLM (Arize Phoenix)

[Arize Phoenix](https://phoenix.arize.com/) guarda cada llamada a Claude con el **system prompt** (instrucciones + ejemplos CAG), la transcripción, la respuesta, los tokens y la latencia, y lo muestra en un panel web. Todo se ejecuta **en local**: los datos no salen de tu máquina.

- `docker compose up -d` levanta Phoenix junto a la API. El panel está en http://localhost:6006 (proyecto `estimador-cag`).
- Se activa con `PHOENIX_COLLECTOR_ENDPOINT` en el `.env` (`http://localhost:6006/v1/traces` para la API o Streamlit en local; dentro de Docker Compose la API usa `http://phoenix:6006/v1/traces`). **Si no está definida, no se activa nada** y la app funciona igual.
- Cubre las dos vías (la API y el streaming de Streamlit) y los dos proveedores, porque instrumenta LiteLLM.
- La instrumentación ([`app/tracing.py`](app/tracing.py)) usa OpenInference y no requiere cambios en el servicio.

## Documentación de la API

FastAPI genera la documentación automáticamente:

- **Swagger UI:** http://127.0.0.1:8000/docs (permite probar los endpoints)
- **ReDoc:** http://127.0.0.1:8000/redoc
- **Esquema OpenAPI:** http://127.0.0.1:8000/openapi.json

## Endpoints

### `POST /api/v1/estimate`

Genera una estimación y la devuelve completa en una sola respuesta JSON.

**Request** (`EstimationRequest`):

```json
{
  "description": "Una clínica veterinaria quiere una web para pedir cita y ver el historial de vacunas.",
  "project_type": "web_saas",
  "detail_level": "summary",
  "output_format": "phases_table"
}
```

| Campo | Valores |
|---|---|
| `description` | Entre 20 y 2.000 caracteres (sin contar espacios al principio y al final) |
| `project_type` | `mobile_app`, `web_saas`, `internal_tool`, `data_pipeline` |
| `detail_level` | `summary`, `medium`, `detailed` |
| `output_format` | `phases_table`, `line_items`, `narrative` |

**Response `200`** (`EstimationResponse`):

```json
{
  "text": "## Estimación: ...

| Fase | Semanas | Horas | Coste (€) |
...",
  "prompt_version": "v1",
  "model": "claude-haiku-4-5",
  "provider": "anthropic",
  "usage": {"input_tokens": 1836, "output_tokens": 444, "total_tokens": 2280},
  "cost_usd": 0.004056,
  "finish_reason": "stop",
  "cache_hit": false,
  "created_at": "2026-11-01T19:50:19Z"
}
```

- `text` y `prompt_version` son los campos del contrato. El resto son metadatos de la llamada que usa el cliente.
- `prompt_version`: versión de la plantilla de prompt que generó la estimación (ver [Prompts versionados](#prompts-versionados)).
- `cost_usd`: coste estimado según la tabla de precios de LiteLLM; `null` si el modelo no está en la tabla.
- `finish_reason`: `stop` si la estimación terminó con normalidad; `length` si se cortó por el límite de tokens de salida.
- `cache_hit`: `true` si la respuesta viene de la caché (sin llamada al LLM y con `cost_usd = 0`).

**Errores:**

| Código | Motivo |
|---|---|
| `422` | Falta algún campo, la descripción no tiene entre 20 y 2.000 caracteres o un valor no es válido (no se llama al LLM) |
| `502` | Error al llamar a la API del LLM (key inválida, límite de uso, etc.) |

### `POST /api/v1/estimate/stream`

El mismo request, pero la estimación llega **en streaming** como eventos [Server-Sent Events](https://developer.mozilla.org/es/docs/Web/API/Server-sent_events). Es el endpoint que usa Streamlit.

| Evento | Datos | Cuándo |
|---|---|---|
| `token` | `{"text": "..."}` | Cada trozo de texto, según lo genera el LLM |
| `done` | La respuesta completa, igual que en `POST /api/v1/estimate` | Al terminar |
| `error` | `{"message": "..."}` | Si falla la llamada al LLM (no llega `done`) |

```
event: token
data: {"text": "## Estimación: Plataforma"}

event: done
data: {"text": "## Estimación: ...", "prompt_version": "v1", "usage": {...}, ...}
```

Los errores de validación devuelven `422` igual que el endpoint normal, antes de empezar el stream.

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

Las estimaciones de ejemplo (el contexto CAG) están en [`app/prompts/estimation/v1/examples.j2`](app/prompts/estimation/v1/examples.j2). Cada ejemplo incluye la descripción del proyecto y su estimación: desglose de tareas con horas y costes (tarifa de 50 €/h), equipo recomendado y duración.

Los datos son ficticios. Para mejorar la calidad de las estimaciones, sustitúyelos por proyectos reales y representativos del tipo de trabajo que se quiere estimar (en una versión nueva, `v2/`, para poder comparar).
