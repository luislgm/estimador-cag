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
│   │   ├── estimation/v1/      # system.j2, user.j2 y examples.j2
│   │   ├── conversation_summary/v1/  # Resumen de los turnos que salen de la ventana
│   │   └── critic/v1/          # El crítico del ciclo Actor-Critic-Boss
│   ├── guardrails/
│   │   ├── input.py            # Moderación, prompt injection y datos personales
│   │   └── output.py           # Deja sin cifras las estimaciones fuera de alcance
│   ├── routers/
│   │   ├── estimations.py      # POST /api/v1/estimate (sin sesión)
│   │   └── sessions.py         # Sesiones: POST /sessions y POST /sessions/{id}/estimate
│   ├── schemas/                # Contratos de datos (Pydantic): request, resultado y response
│   │   ├── estimation.py
│   │   └── health.py
│   ├── sessions.py             # Estado de las sesiones: historial y memoria del proyecto
│   ├── tiers.py                # Tier de audiencia (dirección, PM, técnico) de cada turno
│   ├── services/
│   │   ├── estimation.py       # Orquestador del pipeline
│   │   ├── conversation.py     # Un turno de una sesión (memoria + historial + adjuntos)
│   │   ├── compression.py      # Compresión del historial: anclas y resumen acumulado
│   │   ├── critic.py           # Crítico: revisa una estimación (Actor-Critic-Boss)
│   │   ├── boss.py             # Boss: acepta, pide otra versión o se queda con la mejor
│   │   ├── attachments.py      # Extracción de texto de PDFs y Word
│   │   ├── llm_client.py       # Wrapper sobre LiteLLM + Instructor (salida estructurada)
│   │   ├── render.py           # Resultado estructurado -> markdown según el formato
│   │   ├── cache.py            # Caché exacta en Redis
│   │   └── semantic_cache.py   # Caché semántica (Redis Stack + embeddings)
│   ├── ingestion/              # Ingesta de datos: catálogo, parsers, limpieza, seudonimización
│   ├── persistence.py          # Base de datos de la ingesta (SQLAlchemy): seudónimos y trabajos
│   ├── logging_config.py       # Logs estructurados (structlog)
│   └── tracing.py              # Trazas del LLM en Arize Phoenix (opcional)
├── alembic/                    # Migraciones de la base de datos de la ingesta
├── data/                       # Catálogo de fuentes (catalog.yaml) y datos de ejemplo (seed/)
├── evals/                      # Evals con el LLM real: golden dataset, métricas y runner
│   ├── golden_dataset.json     # Casos de referencia y lo que se espera de cada uno
│   ├── dataset.py              # Carga y valida el dataset
│   ├── metrics.py              # Métricas y umbrales
│   ├── run.py                  # python -m evals.run
│   └── stress/                 # Prueba de carga del CAG (ver evals/stress/REPORT.md)
├── streamlit_app.py            # Cliente Streamlit (formulario)
├── pages/1_Ingesta.py          # Página de Streamlit para la ingesta de datos
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
| `SESSION_MAX_TURNS` | Turnos (par user + assistant) del historial que se envían al LLM | `6` |
| `SESSION_COMPRESSION_ENABLED` | Anclas y resumen de los turnos que salen de la ventana (si no, se descartan) | `true` |
| `SESSION_MAX_ANCHORS` | Turnos antiguos que se conservan literales como anclas (0 = todo al resumen) | `4` |
| `ACB_MAX_ITERATIONS` | Vueltas como mucho del ciclo Actor-Critic-Boss (cada una: actor + crítico) | `2` |

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

Una conversación para estimar un proyecto en varios mensajes, en [`streamlit_app.py`](streamlit_app.py). Con la API en marcha:

```bash
uv run --group ui streamlit run streamlit_app.py
```

Se abre en http://localhost:8501.

- **Es un cliente de la API:** al cargar la página crea una sesión (`POST /api/v1/sessions`) y guarda el `session_id` en `st.session_state`. Cada envío del formulario es un turno (`POST /api/v1/sessions/{id}/estimate`), con la transcripción, los **adjuntos** (selector múltiple de PDF y Word) y las opciones. Streamlit no llama al LLM ni necesita sus API keys; solo la URL de la API (`ESTIMATOR_API_URL`).
- **Conversación:** cada turno muestra el mensaje (con sus adjuntos) y la estimación, con su versión estructurada (JSON) en un desplegable.
- **Panel lateral:** la **memoria del proyecto** (`project_metadata`) y cuántos turnos del historial se envían al LLM, para ver la separación entre memoria e historial; las métricas de la última llamada; el system prompt del siguiente turno, con la memoria actual; y las estimaciones de ejemplo.
- **"Nueva conversación"** crea otra sesión. Si la API se ha reiniciado y la sesión ya no existe, se crea una nueva automáticamente y se avisa.
- Streamlit está en un grupo de dependencias aparte (`ui`) para no incluirlo en la imagen Docker de la API.

## Sesiones conversacionales: memoria, historial y adjuntos

Una conversación iterativa sobre un mismo proyecto: el cliente refina el alcance, añade información o sube documentos, y el estimador recuerda de qué proyecto se habla.

```
POST /api/v1/sessions                         -> {"session_id": "..."}
POST /api/v1/sessions/{session_id}/estimate   (multipart: transcript + attachments + opciones)
POST /api/v1/sessions/{session_id}/estimate-acb   (lo mismo, revisado por un crítico)
GET  /api/v1/sessions/{session_id}            -> memoria y turnos de la sesión
```

### Historial y memoria, por separado

[`app/sessions.py`](app/sessions.py) separa dos cosas que a menudo se confunden:

| | **Historial** (`ConversationHistory`) | **Memoria** (`ProjectMetadata`) |
|---|---|---|
| Qué es | Los mensajes user/assistant que viajan al LLM en cada llamada | Los hechos del proyecto: nombre, tamaño del equipo, tecnologías y alcance acordado |
| Cómo se usa | Se envían los últimos `SESSION_MAX_TURNS` turnos (ventana deslizante, 6 por defecto) | Se inyecta en el system prompt, en el bloque `<project_metadata>` (vacío en el primer turno) |
| Qué pasa con lo antiguo | Los turnos más viejos se descartan | Se conserva: el nombre del proyecto no se pierde aunque el turno donde apareció ya no esté en la ventana |

- **Ventana deslizante:** cuando hay más de `SESSION_MAX_TURNS` turnos, se descartan los pares más antiguos. El system prompt no se guarda en el historial: `ConversationHistory.to_messages_list()` lo regenera en cada turno a partir del `project_metadata` actual y devuelve el array `messages` (system + turnos de la ventana), al que el turno añade el mensaje nuevo antes de llamar al LLM. Es la estrategia razonable para empezar: simple, predecible y con un coste por llamada acotado. Lo que sale de la ventana no se pierde: pasa por la compresión (ver más abajo).
- **Mensajes de continuación:** a partir del segundo turno, el mensaje del usuario no empieza por "Estima el siguiente proyecto:" sino por "Nuevo mensaje sobre el mismo proyecto. Actualiza la estimación completa…", y el system prompt pide mantener el `project_name`. Con el encabezado de siempre, "Añade también una app móvil" se leía como un proyecto nuevo (lo detectaron las evals).
- **Respuesta del asistente en el historial:** se guarda la estimación en **markdown**, no en JSON. Con respuestas JSON en el historial, el modelo las imita y acaba anidando JSON como texto dentro de la herramienta, y la respuesta deja de validar (lo vimos en las pruebas reales con 3 turnos). El markdown además ocupa la mitad de tokens.
- **Sin caché:** el endpoint de sesión no usa las cachés, porque la respuesta depende del historial y de la memoria.
- **Volatilidad aceptada:** las sesiones viven en un diccionario en memoria del proceso. Se pierden al reiniciar el servicio y no se comparten entre réplicas; para producción irían en Redis o en una base de datos.

### Compresión del historial: anclas y resumen acumulado

Con solo una ventana, lo que sale de ella se olvida: si en el turno 1 el cliente dijo que había firmado un NDA o que el presupuesto máximo era de 30.000 €, en el turno 8 el modelo ya no lo sabe (la memoria guarda nombre, equipo, tecnologías y alcance, pero no ese tipo de compromisos). [`app/services/compression.py`](app/services/compression.py) decide qué pasa con cada turno que expulsa la ventana:

1. **Ancla:** si el mensaje del usuario contiene un compromiso duradero, el turno se conserva **literal**, fuera de la ventana. Se detecta con reglas de expresiones regulares en español e inglés: confidencialidad (NDA), contrato o propuesta firmados, alcance cerrado, presupuesto máximo, normativa (RGPD, PCI-DSS, ISO 27001...), fecha límite obligatoria y compromisos explícitos ("hemos acordado", "el cliente ha aceptado"). Cuesta cero, es determinista y explicable (el log `history_compressed` dice qué regla saltó). Es conservadora a propósito: un ancla de más ocupa prompt para siempre; una de menos solo acaba en el resumen.
2. **Resumen:** el resto de turnos se integra, con una llamada al LLM (prompt versionado [`conversation_summary/v1`](app/prompts/conversation_summary/v1/)), en un **único resumen acumulado**: el resumen anterior más los turnos que salen, reescritos en como mucho 250 palabras.

Lo que recibe el LLM en cada turno:

```
system  (regenerado: instrucciones + <project_metadata> + <resumen_conversacion>)
anclas  (turnos antiguos literales, en orden)
ventana (últimos SESSION_MAX_TURNS turnos)
user    (mensaje actual)
```

- **El resumen va en el system prompt**, no como un mensaje más: así se mantiene la alternancia user/assistant, que Anthropic exige, y el system prompt ya se regenera en cada turno.
- **Límite de anclas** (`SESSION_MAX_ANCHORS`, 4 por defecto): si no caben, la más antigua pasa al resumen, para que el prompt no crezca sin límite.
- **El resumen es best-effort:** si su llamada falla, se conserva el resumen anterior y el turno del usuario no se ve afectado.
- **Coste aparte:** la compresión solo llama al LLM cuando un turno normal sale de la ventana (a partir del turno 7 con la configuración por defecto), y con el mismo modelo. Su coste acumulado se devuelve en `session.compression_cost_usd`, separado del `cost_usd` del turno.
- La detección mira el mensaje completo del usuario, incluidos los adjuntos: un documento que mencione el RGPD convierte el turno en ancla. Es intencionado (es una restricción del proyecto), y el límite de anclas acota lo que eso cuesta.
- Con `SESSION_COMPRESSION_ENABLED=false` se vuelve a la ventana simple.

El panel lateral de Streamlit muestra cuántos turnos hay como anclas y en el resumen, el resumen actual y su coste.

### Tier de audiencia dinámico

La misma estimación no se cuenta igual a la dirección que al equipo técnico. [`app/tiers.py`](app/tiers.py) decide en cada turno para quién es la respuesta:

| Tier | Para quién | Qué cambia en la respuesta |
|---|---|---|
| `executive` | Dirección | El summary empieza por coste, plazo y la decisión; fases sin jerga; riesgos de negocio primero |
| `pm` | Gestión del proyecto | Plazo, fases en orden, dependencias e hitos; riesgos de planificación primero |
| `developer` | Equipo técnico | Enfoque técnico y arquitectura; tecnologías concretas; riesgos técnicos primero |
| `default` | — | Nada: el prompt es el mismo que sin tier |

Cómo se decide, por orden:

1. **Override:** el campo `tier` del formulario (opcional) manda siempre.
2. **Reglas** sobre el mensaje del usuario, por prioridad: primero las menciones a la audiencia ("para el comité de dirección", "el CTO", "el jefe de proyecto") y después las señales del tema (retorno de la inversión → executive; hitos, sprints, cronograma → pm; Kubernetes, microservicios, endpoints → developer). Así, "para el comité de dirección: ¿cuánto cuesta migrar a Kubernetes?" es executive.
3. **Herencia:** si el mensaje no da pistas ("quita la descarga de facturas"), se mantiene el tier del turno anterior, para que no cambie de un turno a otro sin motivo.
4. Si no hay nada: `default`.

- **Sin llamadas al LLM:** cuesta cero, es determinista y **explicable**: el estado de la sesión devuelve `tier`, `tier_rule` (la regla que lo decidió) y `tier_inherited`, y el log `session_turn_completed` también los registra.
- **Enfoque, no formato:** el bloque `<audiencia>` del system prompt cambia el énfasis y el lenguaje; `output_format` y `detail_level` siguen decidiendo la forma, y las reglas de horas y costes no cambian.
- **Solo mira el mensaje del usuario:** ni los adjuntos (un documento técnico adjunto no convierte en técnica una conversación con dirección) ni la memoria (siempre tiene tecnologías, así que la regla técnica saltaría siempre).
- El tier solo se guarda si el turno sale bien. El endpoint sin sesión no lo usa.

En Streamlit hay un selector **Audiencia** (automática por defecto) y el panel lateral muestra el tier y por qué ("Gestión del proyecto (PM) (regla `planificacion`, mantenida del turno anterior)").

### Revisión Actor-Critic-Boss

`POST /api/v1/sessions/{id}/estimate-acb` hace el mismo turno que `/estimate`, pero la estimación pasa por un ciclo de revisión antes de llegar al usuario:

```
actor (estimador) -> crítico (revisa) -> boss (decide) ─┬─ sin problemas graves       -> acepta
                                                         ├─ problemas y quedan vueltas -> el actor rehace
                                                         └─ sin vueltas                -> la mejor versión
```

- **Actor:** el estimador de siempre. Cuando rehace, recibe en el system prompt (bloque `<revision_critica>`) su estimación anterior y los problemas que encontró el crítico.
- **Crítico** ([`critic.py`](app/services/critic.py), prompt [`critic/v1`](app/prompts/critic/v1/)): otra llamada al LLM que revisa sin modificar. Ve lo mismo que un revisor humano: el mensaje del usuario con sus adjuntos, la memoria del proyecto, la audiencia y la estimación. Devuelve una `CriticReview`: veredicto (`accept` o `revise`) y problemas concretos (alcance, horas, fases, equipo, riesgos o enfoque) con su gravedad y una sugerencia.
- **Boss** ([`boss.py`](app/services/boss.py)): decide sin llamar al LLM, así que es determinista y se prueba sin modelo. Solo pide otra versión si el crítico pide revisar y hay problemas de gravedad alta o media (los menores no justifican otra llamada), y como mucho `ACB_MAX_ITERATIONS` vueltas (2). Si se agotan, devuelve la versión con menos problemas graves.
- **Degradación:** si el crítico falla, se devuelve la estimación sin revisar; si falla la nueva versión del actor, la mejor anterior. El ciclo no rompe el turno.
- **Respuesta:** la de `/estimate` más `acb`, la traza: cada versión con sus horas, lo que dijo el crítico y lo que decidió el boss. `usage` y `cost_usd` suman todas las llamadas del turno. En el historial solo queda la versión final.

**¿Merece la pena?** Lo dicen las evals ([`--mode acb`](#evals-el-golden-dataset)). Con `gpt-4o-mini` como actor y crítico:

| | Normal | ACB |
|---|---|---|
| `hours_in_range` | 0,82 | 1,00 |
| `content_recall` | 0,95 | 1,00 |
| Coste de la suite | 0,011 $ | 0,044 $ (×4) |
| Tiempo | 41 s | 131 s (×3) |

El crítico corrige sobre todo las estimaciones cortas de los proyectos grandes (el marketplace pasa de 260 a 540 h). Pero tiende a subir las horas casi siempre: con la primera versión de su prompt inflaba los proyectos pequeños (la web corporativa, de 100 a 170 h) y marcaba como error que una receta de paella saliera "fuera de alcance"; el prompt actual se lo aclara. Son una o dos ejecuciones por modo, así que conviene repetirlas antes de dar la mejora por buena, y probar con otro modelo como crítico. Por coste y latencia es opcional: en Streamlit, casilla **Revisar con un crítico**, y la traza en un desplegable bajo la respuesta.

### Cómo se extrae el `project_metadata`

**A partir de la propia salida estructurada, sin una segunda llamada al LLM.** El `EstimationResult` ya incluye `project_name`, `summary` (alcance), `team_size` y `technologies`, que el modelo rellena dentro del esquema validado. Después de cada turno, `ProjectMetadata.updated_with(result)` copia esos campos:

- Es **tan barato como la heurística** (no hay llamada extra ni latencia añadida) y **tan fiable como el extractor**, porque los datos vienen del esquema validado, no de buscar patrones en un texto libre.
- Cada campo guarda **el valor más reciente**. El modelo recibe la memoria anterior y la refina; sumar las listas de tecnologías de todos los turnos dejaría suposiciones superadas (por ejemplo, "PostgreSQL o MySQL" junto a "PostgreSQL" una vez confirmado).
- Una estimación "fuera de alcance" no cambia la memoria.

### Adjuntos: camino B (extracción local)

Se admiten **PDF y Word (`.docx`)**, como mucho 5 archivos de 5 MB. [`app/services/attachments.py`](app/services/attachments.py) extrae el texto en el servicio (`pypdf` y `python-docx`, tablas incluidas) y lo añade al mensaje del usuario, separado con `--- attachment: nombre.pdf ---`. Cada documento se recorta a 15.000 caracteres para acotar los tokens.

Por qué el camino B y no subir el archivo al LLM (camino A):

- **Funciona igual con cualquier proveedor.** Todo pasa por LiteLLM; la Files API ataría el proyecto a OpenAI o a Anthropic.
- **Los guardrails revisan también los documentos:** un PDF con datos personales o con instrucciones dirigidas al modelo se rechaza igual que la transcripción.
- **Prepara el terreno para RAG** (trocear documentos en el módulo 3).
- A cambio, se pierden las imágenes y los diagramas, y un PDF escaneado no aporta texto (se indica en el prompt).

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

**La aritmética la hace el código, no el modelo.** El coste de cada fase (`cost_eur = hours × 50`) y los totales (`total_hours`, `total_cost_eur`) son **campos calculados** (`computed_field`): salen en el JSON de la respuesta, pero no están en el esquema que recibe el modelo, así que no tiene que calcularlos. Antes se le pedía sumar y un validador lo comprobaba, pero las [evals](#evals-el-golden-dataset) mostraron que `gpt-4o-mini` fallaba la suma en 2 de 13 casos incluso tras los reintentos, y el turno acababa en error. La duración total sí la da el modelo: las fases pueden solaparse.

Pydantic valida las **reglas de negocio** que dependen del criterio del modelo:

- Si `confidence_pct < 30`, el `summary` debe empezar por `"Fuera de alcance:"`.
- Una estimación dentro de alcance tiene al menos 2 fases. Las evals detectaron que `gpt-4o-mini` a veces corta la lista y devuelve solo la primera fase (32 h para un proyecto de 300); con el validador se le pide que la complete.
- Los límites de cada campo (horas por fase, número de fases, longitud de los textos...).

[Instructor](https://python.useinstructor.com/) envía el esquema al modelo como herramienta y, si un validador falla, le devuelve el error para que lo corrija (hasta 2 reintentos; se registra `llm_response_retried`).

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
├── estimation/
│   └── v1/
│       ├── system.j2     # rol, instrucciones y bloques condicionales; incluye examples.j2
│       ├── user.j2       # envuelve la descripción del proyecto
│       └── examples.j2   # 3 estimaciones de ejemplo (few-shot): el contexto CAG
└── conversation_summary/
    └── v1/               # resumen acumulado de los turnos que salen de la ventana
        ├── system.j2
        └── user.j2
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

Los tests de las sesiones (Paso 7 del ejercicio de la sesión 05) están en [`tests/test_sessions_api.py`](tests/test_sessions_api.py), con `httpx.AsyncClient` sobre la app: dos turnos que actualizan el `project_metadata`, un PDF adjunto que cambia la estimación y 8 turnos en los que lo enviado al LLM nunca supera la ventana de `SESSION_MAX_TURNS`. Los de la compresión están en [`tests/test_compression.py`](tests/test_compression.py): reglas de anclaje (positivos y negativos), el límite de anclas, el resumen (y qué pasa si falla) y una conversación de 9 turnos en la que el ancla y el resumen llegan al LLM. Los del tier, en [`tests/test_tiers.py`](tests/test_tiers.py): cada regla, la prioridad, el override, la herencia, el bloque del prompt y una conversación en la que el tier se deduce, se hereda y se fuerza. Los del ciclo Actor-Critic-Boss, en [`tests/test_acb.py`](tests/test_acb.py): cada decisión del boss (aceptar, rehacer, agotar vueltas, empates, fallos del crítico y del actor), el crítico, los prompts y el endpoint.

```bash
uv run pytest tests/test_sessions_api.py -v   # solo los de sesiones
```

```bash
uv run pytest --cov          # tests con cobertura (mínimo 80 %)
uv run ruff check .          # lint
uv run ruff format .         # formatear el código
```

## Evals: el golden dataset

Los tests comprueban que el código hace lo que debe, pero simulan el LLM: no dicen si el modelo estima bien. Las evals sí: lanzan un **golden dataset** de casos de referencia contra la API con el **LLM real** y puntúan las respuestas.

**El dataset** ([`evals/golden_dataset.json`](evals/golden_dataset.json)) tiene 13 casos: los cuatro tipos de proyecto, dos peticiones fuera de alcance (una receta, una reforma de cocina), tres con señales de audiencia (dirección, PM, técnico) y dos conversaciones de dos turnos (una que amplía el alcance y otra que lo reduce). Cada caso se envía a una sesión nueva, turno a turno, y define lo que se espera del último:

- `hours`: rango de horas razonable. Es amplio a propósito: busca estimaciones disparatadas, no la cifra exacta.
- `must_mention`: conceptos que deben aparecer, cada uno con sus sinónimos (sin distinguir mayúsculas ni tildes).
- `out_of_scope`, `tier` y `change` (si las horas deben subir o bajar).

**Métricas y umbrales** ([`evals/metrics.py`](evals/metrics.py)):

| Métrica | Qué mide | Umbral |
|---|---|---|
| `schema` | Todos los turnos devuelven un `EstimationResult` válido | 100 % |
| `scope` | Fuera de alcance cuando toca (y sin cifras), y solo entonces | 100 % |
| `tier` | La audiencia deducida es la esperada | 100 % |
| `memory` | El `project_name` se mantiene entre turnos | 100 % |
| `hours_in_range` | Las horas caen en el rango esperado | 70 % |
| `content_recall` | Proporción de conceptos esperados que aparecen | 70 % |
| `direction` | Las horas suben o bajan como se pidió | 50 % |

Lo que depende del contrato exige el 100 %; lo que depende del criterio del modelo tiene margen. La coherencia de los números no se mide: la calcula el código, no el modelo.

**Cómo se lanza** (cuesta dinero: con `gpt-4o-mini`, unos 0,01 $ la suite completa):

```bash
uv run python -m evals.run                                  # todos los casos, app en proceso
uv run python -m evals.run --case tier_tecnico_flotas       # solo algunos (se puede repetir)
uv run python -m evals.run --http http://localhost:8000     # contra la API levantada
uv run python -m evals.run --mode acb                       # con el ciclo Actor-Critic-Boss
LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini uv run python -m evals.run   # otro modelo

uv run pytest -m eval -s                                    # lo mismo, como test de pytest
```

Imprime una tabla por caso (con el detalle de lo que falla) y el resumen por métrica, y guarda el informe completo en `evals/reports/` (ignorado por git) para comparar ejecuciones. Sale con código 1 si alguna métrica queda por debajo de su umbral, así que sirve como **puerta de regresión** antes de cambiar un prompt o un modelo.

En pytest está marcado como `eval` y **excluido por defecto** (`addopts = "-m 'not eval'"`): `uv run pytest` y la CI no llaman al LLM. El código de las evals (dataset, métricas y runner) sí tiene sus tests normales, con el LLM simulado, en [`tests/test_evals.py`](tests/test_evals.py).

## Prueba de carga del CAG

[`evals/stress/`](evals/stress/) mide dónde empieza a romperse el CAG: conversaciones largas, adjuntos grandes y ráfagas. El resultado, con las curvas y la lectura, está en [`evals/stress/REPORT.md`](evals/stress/REPORT.md); los datos, en [`evals/stress/results.csv`](evals/stress/results.csv).

- **Observación por turno:** cada turno emite un único evento `turn_observed` (y lo guarda en `session.last_turn`) con todo lo medible: tamaño de la entrada y de los adjuntos, mensajes en la ventana, anclas, longitud del resumen, tokens, coste, latencia, caché y tier. `GET /api/v1/sessions/{id}/snapshot` devuelve además el texto de las anclas y de la ventana.
- **Escenarios** ([`scenarios.py`](evals/stress/scenarios.py)): tres conversaciones de 20 turnos (un proyecto que crece, uno que pivota de React a Flutter y uno que se contradice con el presupuesto), cada turno con el hecho que deberían recordar los siguientes.
- **Adjuntos** ([`fixtures/build_pdfs.py`](evals/stress/fixtures/build_pdfs.py)): PDFs deterministas de 5, 20, 50 y 100 KB de texto, con un hecho al principio y otro al final (no se versionan; el runner los regenera).
- **Métricas** ([`stress/metrics.py`](evals/stress/metrics.py)): `LatencyBudgetMetric`, `CostBudgetMetric` y `MemoryDriftMetric` (búsqueda literal del hecho en el resumen, las anclas, la metadata y, opcionalmente, la ventana).

```bash
uv run python -m evals.stress.run                          # 3 escenarios × 5 adjuntos × 3 repeticiones × 20 turnos
uv run python -m evals.stress.run --burst 1,5,10,20 --skip-scenarios --append   # ráfagas
uv run python -m evals.stress.run --http http://localhost:8000   # contra la API levantada
uv run python -m evals.stress.report                       # tablas del informe a partir del CSV
```

## Ingesta de datos (preparación de RAG)

[`app/ingestion/`](app/ingestion/) convierte los datos de la empresa (presupuestos, transcripciones de reuniones, tarifas) en documentos limpios y seudonimizados, listos para indexar:

```
catálogo -> carga -> parseo (+ reparación) -> validación -> deduplicado -> seudonimización -> JSONL
```

- **Catálogo** ([`data/catalog.yaml`](data/catalog.yaml)): cada fuente con su responsable, su calidad (completitud, consistencia, actualidad y fiabilidad, de 1 a 5), si tiene datos personales y una decisión: `include`, `review` (problemas conocidos, pendiente de su responsable) o `exclude`. Excluir o dejar en revisión exige un motivo. `uv run python -m app.ingestion.catalog` muestra el inventario.
- **Parsers** ([`parsers.py`](app/ingestion/parsers.py)): uno por formato (JSON de presupuestos, TXT de transcripciones con cabecera, Excel de tarifas), todos devuelven `Document`, el contrato común.
- **Limpieza** ([`cleaning.py`](app/ingestion/cleaning.py)): los presupuestos se **reparan** (espacios, moneda en minúsculas, importes como "12.500,00", fechas dd/mm/aaaa), se ponen en **cuarentena** si les falta un dato recuperable (el nombre del cliente) o se **descartan** si están mal de raíz (importe negativo, identificador inválido). Pandera con `lazy=True` recoge todos los fallos, no solo el primero. De varias versiones de un presupuesto queda la última.
- **Seudonimización** ([`pii.py`](app/ingestion/pii.py)): Presidio con spaCy en español detecta personas, emails, teléfonos, DNI/NIE, IBAN y tarjetas, y los sustituye por datos ficticios de Faker. El almacén guarda `(tipo, HMAC(sal, dato)) -> seudónimo`: la misma persona es la misma persona ficticia en todos los documentos, y la base de datos no contiene los datos originales. `DELETE /api/v1/ingestion/pseudonyms` borra la relación (derecho de supresión, art. 17); como el seudónimo no se deriva del dato, después ya nadie puede reconstruirla.
  - Las empresas y las ciudades no se tratan: no son datos personales y se necesitan para estimar.
  - Para evitar falsos positivos del modelo pequeño ("Reunión", "Presupuesto BUDGET"), una persona necesita nombre y apellido; a cambio, un nombre suelto ("Pedro") no se detecta. Cada nombre detectado se busca en todo el documento, porque el modelo no detecta igual a la misma persona en todas las frases.
- **Decisión de arquitectura** ([`architecture.py`](app/ingestion/architecture.py)): `uv run python -m app.ingestion.architecture [--scale N]` evalúa si el corpus cabe en CAG (ventana, coste, latencia y *lost in the middle*, con los datos del [stress test](evals/stress/REPORT.md)) y recomienda CAG, híbrido o RAG.

Por CLI (sin base de datos; los seudónimos, en memoria):

```bash
PII_SALT=... uv run python -m app.ingestion.orchestrator --all      # escribe data/processed/*.jsonl
```

Por API (necesita `DATABASE_URL`; en Docker, Postgres ya va en `docker-compose.yml` y las migraciones de Alembic se aplican al arrancar):

```
GET    /api/v1/ingestion/sources         el catálogo
POST   /api/v1/ingestion/runs            {"source": "budgets_2024"} -> 202 + job_id (en segundo plano)
GET    /api/v1/ingestion/jobs/{job_id}   estado e informe: reparados, cuarentena, descartados, seudónimos
DELETE /api/v1/ingestion/pseudonyms      {"entity_type": "PERSON", "value": "..."} -> derecho de supresión
```

En Streamlit, la página **Ingesta** ([`pages/1_Ingesta.py`](pages/1_Ingesta.py), en el menú lateral) hace todo esto sin Swagger: el catálogo, un botón para ingerir cada fuente (espera a que el trabajo termine), el informe con la cuarentena y los descartes, una vista previa de los documentos seudonimizados (`GET /api/v1/ingestion/jobs/{id}/documents`) y el formulario del derecho de supresión.

Para ingerir fuentes con datos personales hace falta `PII_SALT` en el `.env` (una cadena aleatoria y secreta). spaCy, Presidio y pandas hacen la imagen Docker bastante más pesada (~1,5 GB).

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
