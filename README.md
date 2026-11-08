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
│   ├── guardrails/
│   │   ├── input.py            # Moderación, prompt injection y datos personales
│   │   └── output.py           # Deja sin cifras las estimaciones fuera de alcance
│   ├── routers/
│   │   └── estimations.py      # Endpoint POST /api/v1/estimate (traduce errores a HTTP)
│   ├── schemas/                # Contratos de datos (Pydantic): request, resultado y response
│   │   ├── estimation.py
│   │   └── health.py
│   ├── services/
│   │   ├── estimation.py       # Orquestador del pipeline
│   │   ├── llm_client.py       # Wrapper sobre LiteLLM + Instructor (salida estructurada)
│   │   ├── render.py           # Resultado estructurado -> markdown según el formato
│   │   ├── cache.py            # Caché exacta en Redis
│   │   └── semantic_cache.py   # Caché semántica (Redis Stack + embeddings)
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
| `SEMANTIC_CACHE_MODE` | Caché semántica: `off`, `log_only` u `on` | `off` |
| `SEMANTIC_CACHE_THRESHOLD` | Similitud mínima para servir desde la caché semántica | `0.95` |
| `MODERATION_ENABLED` | Guardrail de moderación de OpenAI (necesita `OPENAI_API_KEY`) | `false` |

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
- La salida estructurada se pide con [Instructor](https://python.useinstructor.com/) sobre LiteLLM (ver [Salida estructurada](#salida-estructurada-y-validadores)).
- Los tokens (de todos los intentos), el coste (tabla de precios de LiteLLM) y el motivo de fin salen igual para ambos proveedores.
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

- **Es un cliente de la API:** el formulario (`st.form`) construye un `EstimationRequest`, lo valida con el mismo schema de Pydantic que la API y lo envía a `POST /api/v1/estimate`. Streamlit no llama al LLM ni necesita sus API keys; solo la URL de la API (`ESTIMATOR_API_URL`).
- **Resultado:** muestra el markdown de la estimación y, en un desplegable, el **objeto estructurado** (JSON). Si un guardrail rechaza la descripción, muestra el motivo; si la confianza es baja, un aviso.
- **Sin streaming:** la respuesta estructurada solo es válida cuando está completa (y los validadores pueden pedir un reintento), así que se muestra al terminar.
- **Historial de la sesión:** la última estimación se muestra completa y las anteriores quedan plegadas debajo (con hora, título, tipo, detalle y formato), para comparar combinaciones. Se borra con "Nueva sesión" o al recargar la página.
- **Panel lateral:** estado de la API (proveedor y modelo), métricas de la última llamada (tokens, tiempo, coste, confianza y si vino de la caché exacta o semántica), los prompts (system y user) de la última estimación y las estimaciones de ejemplo (contexto CAG).
- Streamlit está en un grupo de dependencias aparte (`ui`) para no incluirlo en la imagen Docker de la API.

## Pipeline de estimación

[`app/services/estimation.py`](app/services/estimation.py) orquesta cada petición; el router solo traduce los errores a códigos HTTP:

```
1. Guardrails de entrada  -- rechaza --> 400
2. Prompt versionado (system + user)
3. Caché exacta           -- acierta --> respuesta (coste 0)
4. Caché semántica        -- acierta --> respuesta (coste 0)   [si está activada]
5. LLM con salida estructurada (Instructor: valida y reintenta)  -- falla --> 502
6. Guardrail de salida
7. Markdown según el formato pedido
8. Guardar en las cachés
```

Los guardrails van antes que las cachés: una descripción con datos personales o con un intento de prompt injection no se sirve nunca desde una caché. Solo se guardan resultados que han pasado los validadores y el guardrail de salida.

## Salida estructurada y validadores

El LLM no devuelve texto libre sino un objeto `EstimationResult` ([`schemas/estimation.py`](app/schemas/estimation.py)): `project_name`, `summary`, `confidence_pct`, `phases` (nombre, horas, coste, semanas, resumen y asunciones), `team`, `risks` y los totales.

Pydantic valida **reglas de negocio** que el modelo no puede saltarse:

- En cada fase, `cost_eur = hours × 50` (tarifa de 50 €/h).
- `total_hours` y `total_cost_eur` son la suma de las fases.
- Si `confidence_pct < 30`, el `summary` debe empezar por `"Fuera de alcance:"`.

[Instructor](https://python.useinstructor.com/) envía el esquema al modelo como herramienta y, si un validador falla, le devuelve el error para que lo corrija (hasta 2 reintentos; se registra `llm_response_retried`). Las fases van antes que los totales en el esquema para que el modelo fije primero las cifras de cada fase y luego las sume.

El campo `text` de la respuesta es el markdown generado a partir del objeto según el `output_format` ([`render.py`](app/services/render.py)): tabla por fases, lista de tareas o texto narrativo.

> El modo de Instructor es `TOOLS` (herramienta obligatoria). Funciona con `claude-haiku-4-5` y `gpt-4o-mini`; los modelos de Claude más recientes no admiten forzar una herramienta y necesitarían otro modo (p. ej. `JSON`).

## Guardrails

**De entrada** ([`guardrails/input.py`](app/guardrails/input.py)), antes de llamar al LLM. Si rechazan la descripción, la API responde `400` con el motivo:

| Capa | Qué detecta | Motivo |
|---|---|---|
| Moderación (opcional, `MODERATION_ENABLED`) | Contenido de odio, violencia, sexual… (API de moderación de OpenAI). Si la API no responde, no bloquea | `moderation` |
| Prompt injection | "Ignora las instrucciones anteriores", "ahora eres…", "muestra tu prompt", etiquetas `<system>`… (en español e inglés) | `prompt_injection` |
| Datos personales | Emails, teléfonos españoles, IBAN y DNI/NIE. No se registra el dato, solo su tipo | `pii` |

Las capas de prompt injection y datos personales son heurísticas con expresiones regulares: muestran el patrón, no sustituyen a una herramienta de cumplimiento normativo.

**De salida** ([`guardrails/output.py`](app/guardrails/output.py)): si una estimación es de baja confianza, la deja sin cifras (una única fase "Sin estimar"), para no dar números que parezcan fiables.

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
- Cada llamada al LLM registra `llm_call_completed` con modelo, tokens, coste, número de intentos y latencia (o `llm_call_failed` si falla); `estimation_generated` resume el resultado e `input_rejected` indica por qué un guardrail rechazó una descripción.
- Las peticiones a `/health` se registran a nivel `debug` para no llenar el log con el healthcheck de Docker.

```
19:20:41 [info     ] request_completed   duration_ms=5 method=POST path=/api/v1/estimate request_id=prueba-1 status_code=422
```

## Cachés (Redis)

`docker compose up -d` levanta **Redis Stack** junto a la API, con persistencia en disco (volumen `redis_data`). La API y Streamlit comparten las cachés. Se activan con `REDIS_URL` (`redis://127.0.0.1:6379/0` fuera de Docker; dentro de Docker Compose la API usa `redis://redis:6379/0`). **Sin `REDIS_URL` no hay caché** y todo funciona igual.

En un acierto: `cache_hit=true`, `cache_kind` (`exact` o `semantic`) y `cost_usd=0`; los tokens son los de la respuesta original. Si Redis no responde, la app sigue funcionando sin caché: espera como máximo 1 s, registra el aviso y no vuelve a intentarlo durante 30 s.

### Caché exacta

Si llega una petición **idéntica** a una anterior (mismo modelo, mismo system prompt, misma descripción y mismo `max_tokens`), se devuelve el resultado guardado **sin llamar al LLM**. La clave incluye el system prompt completo: si cambias el prompt, los ejemplos o las opciones del formulario, no se sirven respuestas antiguas.

### Caché semántica

Reutiliza la estimación de una descripción **parecida**: compara los embeddings de las descripciones (`text-embedding-3-small` de OpenAI; necesita `OPENAI_API_KEY`) y solo dentro del mismo "cubo" (versión del prompt, modelo, tipo, nivel de detalle y formato).

**Cuidado en un estimador:** la similitud no distingue bien un cambio de alcance. Medido con este proyecto:

| Comparada con "una web para alquilar bicis por horas y pagar online" | Similitud | Horas estimadas |
|---|---|---|
| La misma con un par de palabras distintas | 0,986 | 200 (igual) |
| La misma **con app móvil y GPS** | 0,885 | **370** |

Con un umbral de 0,85, la segunda recibiría la estimación de 200 h. Por eso:

- `SEMANTIC_CACHE_MODE=off` (por defecto): desactivada.
- `log_only`: calcula y registra la similitud (`semantic_cache_lookup`, con `would_hit`) pero **nunca sirve** la respuesta. Úsalo para calibrar el umbral con tráfico real.
- `on`: sirve la respuesta si la similitud supera `SEMANTIC_CACHE_THRESHOLD` (por defecto **0,95**, conservador).

## Trazas de las llamadas al LLM (Arize Phoenix)

[Arize Phoenix](https://phoenix.arize.com/) guarda cada llamada a Claude con el **system prompt** (instrucciones + ejemplos CAG), la transcripción, la respuesta, los tokens y la latencia, y lo muestra en un panel web. Todo se ejecuta **en local**: los datos no salen de tu máquina.

- `docker compose up -d` levanta Phoenix junto a la API. El panel está en http://localhost:6006 (proyecto `estimador-cag`).
- Se activa con `PHOENIX_COLLECTOR_ENDPOINT` en el `.env` (`http://localhost:6006/v1/traces` para la API o Streamlit en local; dentro de Docker Compose la API usa `http://phoenix:6006/v1/traces`). **Si no está definida, no se activa nada** y la app funciona igual.
- Cubre los dos proveedores porque instrumenta LiteLLM, incluidas las llamadas estructuradas de Instructor y sus reintentos. Los aciertos de caché y las peticiones rechazadas por los guardrails no generan traza (no llaman al LLM).
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
  "text": "## Estimación: Web de Citas Veterinarias\n\n...| Fase | Semanas | Horas | Coste (€) |...",
  "prompt_version": "v1",
  "result": {
    "project_name": "Web de Citas Veterinarias",
    "summary": "Plataforma web para pedir cita, consultar vacunas y recibir recordatorios...",
    "confidence_pct": 70,
    "phases": [
      {"name": "Descubrimiento y diseño UX/UI", "hours": 28, "cost_eur": 1400,
       "duration_weeks": 1, "summary": "...", "assumptions": []}
    ],
    "team": ["1 desarrollador full-stack", "1 diseñador UX/UI (part-time)"],
    "risks": ["..."],
    "total_hours": 220,
    "total_cost_eur": 11000,
    "total_duration_weeks": 13
  },
  "model": "claude-haiku-4-5-20251001",
  "provider": "anthropic",
  "usage": {"input_tokens": 3549, "output_tokens": 1247, "total_tokens": 4796},
  "cost_usd": 0.009784,
  "finish_reason": "tool_calls",
  "cache_hit": false,
  "cache_kind": null,
  "created_at": "2026-11-07T18:17:16Z"
}
```

- `text` y `prompt_version` son los campos del contrato; `result` es la estimación estructurada y validada; el resto son metadatos de la llamada.
- `cost_usd`: coste estimado (de todos los intentos) según la tabla de precios de LiteLLM; `null` si el modelo no está en la tabla.
- `cache_hit` / `cache_kind`: si la respuesta viene de una caché (`exact` o `semantic`), sin llamada al LLM y con `cost_usd = 0`.

**Errores:**

| Código | Motivo |
|---|---|
| `400` | Un guardrail de entrada ha rechazado la descripción. `detail` incluye `reason` (`moderation`, `prompt_injection` o `pii`) y `message` |
| `422` | Falta algún campo, la descripción no tiene entre 20 y 2.000 caracteres o un valor no es válido (no se llama al LLM) |
| `502` | Error al llamar al LLM (key inválida, límite de uso…) o respuesta que no pasa los validadores tras los reintentos |

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
