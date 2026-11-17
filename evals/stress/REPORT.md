# Stress test del CAG: dónde rompe

Baseline cuantitativo del estimador CAG antes de RAG. Datos: [`results.csv`](results.csv) (una fila por turno). Cómo se reproduce: `uv run python -m evals.stress.run` y `uv run python -m evals.stress.report`.

## Montaje

- **Modelo:** `gpt-4o-mini` (OpenAI) para todo, también el resumidor. Un solo proveedor para que las curvas sean comparables.
- **Configuración sin tocar:** ventana de 6 turnos, 4 anclas como máximo, resumen acumulado activo, adjuntos recortados a 15.000 caracteres (`MAX_CHARS_PER_FILE`), sin revisión Actor-Critic-Boss.
- **Carga:** 3 escenarios de 20 turnos (`growing`, `pivot`, `contradiction`) × 5 adjuntos en el turno 1 (0, 5, 20, 50 y 100 KB de texto) × 3 repeticiones = 45 conversaciones, 8 en paralelo, con la app en proceso. Más una ráfaga de 1, 5, 10 y 20 primeros turnos simultáneos.
- **Volumen:** 763 filas (727 de conversaciones y 36 de ráfaga), 1.160 llamadas al LLM, 1,55 USD.
- **Presupuestos** (fijados antes de medir): 10 s de latencia por turno, porque más allá una herramienta interactiva se percibe colgada; 0,002 USD por turno, que con 1.000 turnos al día son 2 USD diarios por cliente.

### Decisiones

- **Observación:** cada turno emite un único evento `turn_observed` y lo guarda en `session.last_turn`. El runner lo lee de la respuesta en vez de parsear logs, porque funciona igual en proceso y por HTTP y no depende de dónde vayan los logs. La memoria completa sale de `GET /sessions/{id}/snapshot`, un endpoint de depuración nuevo con el texto de las anclas y de la ventana.
- **Métricas en `evals/stress/metrics.py`,** no en `evals/metrics.py`: aquellas puntúan un `EstimationResult` frente al golden dataset; estas, la observación del turno y el snapshot. Reutilizan `MetricResult`.
- **`MemoryDriftMetric` con dos alcances:** *largo plazo* (resumen + anclas + metadata, lo que sobrevive cuando un turno sale de la ventana) y *contexto* (eso más la ventana, todo lo que ve el LLM). Los hechos son nombres propios o cifras (Stripe, Auth0, 30.000…), para que la búsqueda literal sea fiable.
- **Caché:** 0 % en todas las filas, por diseño. Las sesiones no usan caché, porque la respuesta depende del historial: dos turnos con el mismo texto no son equivalentes. La caché solo existe en `POST /estimate` sin sesión.

## Tabla resumen

| Escenario | Adjunto | Turnos OK | Latencia P50 (ms) | Latencia P95 (ms) | Coste medio por conversación (USD) | Hit caché exacta | Hit caché semántica | Recall memoria largo plazo | Recall contexto completo | Dentro de presupuesto de latencia | Dentro de presupuesto de coste |
|---|---|---|---|---|---|---|---|---|---|---|---|
| contradiction | 0 KB | 0/3 | — | — | — | — | — | — | — | — | — |
| contradiction | 5 KB | 49/51 | 16063 | 26971 | 0.0321 | 0 % | 0 % | 89 % | 100 % | 27 % | 59 % |
| contradiction | 20 KB | 51/54 | 17166 | 35985 | 0.0431 | 0 % | 0 % | 90 % | 100 % | 18 % | 37 % |
| contradiction | 50 KB | 56/57 | 17977 | 37725 | 0.0439 | 0 % | 0 % | 91 % | 100 % | 25 % | 46 % |
| contradiction | 100 KB | 56/57 | 16205 | 37384 | 0.0430 | 0 % | 0 % | 91 % | 100 % | 23 % | 52 % |
| growing | 0 KB | 12/15 | 13572 | 22353 | 0.0188 | 0 % | 0 % | 79 % | 100 % | 42 % | 83 % |
| growing | 5 KB | 60/60 | 17926 | 31604 | 0.0414 | 0 % | 0 % | 100 % | 100 % | 13 % | 53 % |
| growing | 20 KB | 53/54 | 19419 | 43920 | 0.0431 | 0 % | 0 % | 100 % | 100 % | 17 % | 40 % |
| growing | 50 KB | 58/59 | 18953 | 31845 | 0.0431 | 0 % | 0 % | 100 % | 100 % | 22 % | 50 % |
| growing | 100 KB | 51/53 | 16778 | 34574 | 0.0350 | 0 % | 0 % | 100 % | 100 % | 24 % | 65 % |
| pivot | 0 KB | 53/55 | 13837 | 26522 | 0.0314 | 0 % | 0 % | 98 % | 100 % | 32 % | 74 % |
| pivot | 5 KB | 55/57 | 16858 | 28713 | 0.0369 | 0 % | 0 % | 98 % | 100 % | 20 % | 64 % |
| pivot | 20 KB | 43/45 | 15211 | 25860 | 0.0301 | 0 % | 0 % | 98 % | 100 % | 21 % | 60 % |
| pivot | 50 KB | 55/56 | 17276 | 32321 | 0.0415 | 0 % | 0 % | 99 % | 100 % | 20 % | 49 % |
| pivot | 100 KB | 48/51 | 18103 | 35611 | 0.0352 | 0 % | 0 % | 98 % | 100 % | 29 % | 54 % |

Las filas de 0 KB son pocas porque casi todas las conversaciones sin adjunto murieron en el turno 1 (ver *Fallos duros*).

## Curva 1: latencia frente a tokens de entrada

| tokens_in | Turnos | Latencia P50 (ms) | Latencia P95 (ms) | Coste medio (USD) |
|---|---|---|---|---|
| 0–5.000 | 58 | 6951 | 14667 | 0.00087 |
| 5.000–7.500 | 220 | 11619 | 22042 | 0.00147 |
| 7.500–10.000 | 233 | 19045 | 29838 | 0.00198 |
| 10.000–15.000 | 106 | 19705 | 27930 | 0.00275 |
| ≥ 15.000 | 107 | 28332 | 54314 | 0.00394 |

## Curva 2: coste frente a turno

| Turno | contradiction: coste del turno | contradiction: acumulado | growing: coste del turno | growing: acumulado | pivot: coste del turno | pivot: acumulado |
|---|---|---|---|---|---|---|
| 1 | 0.00158 | 0.0016 | 0.00146 | 0.0015 | 0.00116 | 0.0012 |
| 3 | 0.00171 | 0.0046 | 0.00135 | 0.0042 | 0.00126 | 0.0036 |
| 6 | 0.00213 | 0.0098 | 0.00178 | 0.0094 | 0.00179 | 0.0086 |
| 7 | 0.00289 | 0.0127 | 0.00300 | 0.0124 | 0.00268 | 0.0111 |
| 10 | 0.00238 | 0.0191 | 0.00212 | 0.0190 | 0.00221 | 0.0177 |
| 15 | 0.00275 | 0.0322 | 0.00226 | 0.0298 | 0.00262 | 0.0290 |
| 20 | 0.00313 | 0.0443 | 0.00297 | 0.0434 | 0.00278 | 0.0406 |

Coste del turno 20 / turno 1: contradiction: ×2.0, growing: ×2.0, pivot: ×2.4

## Curva 3: recall de hechos frente a N

*Largo plazo*: hechos de los turnos anteriores presentes en resumen, anclas o metadata. *Contexto*: además, en la ventana. *Nombre*: el nombre del proyecto del turno 1 en la memoria a largo plazo.

| N | contradiction: largo plazo | contradiction: contexto | contradiction: nombre | growing: largo plazo | growing: contexto | growing: nombre | pivot: largo plazo | pivot: contexto | pivot: nombre |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 100 % | 100 % | 100 % | 96 % | 100 % | 92 % | 100 % | 100 % | 100 % |
| 6 | 80 % | 100 % | 100 % | 98 % | 100 % | 92 % | 100 % | 100 % | 100 % |
| 7 | 83 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| 10 | 88 % | 100 % | 100 % | 100 % | 100 % | 100 % | 99 % | 100 % | 100 % |
| 15 | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 99 % | 100 % | 100 % |
| 20 | 100 % | 100 % | 100 % | 99 % | 100 % | 100 % | 100 % | 100 % | 100 % |

## Adjuntos

| Adjunto | Caracteres que llegan | tokens_in turno 1 | Latencia P50 turno 1 (ms) | Coste turno 1 (USD) | Recall inicio del PDF | Recall final del PDF | tokens_in turnos 2-6 | tokens_in turnos 8+ |
|---|---|---|---|---|---|---|---|---|
| 0 KB | 0 | 5394 | 8581 | 0.00116 | — | — | 3854 | 10049 |
| 5 KB | 5070 | 4393 | 10247 | 0.00098 | 100 % | 100 % | 6286 | 10320 |
| 20 KB | 15030 | 9962 | 10704 | 0.00189 | 100 % | 0 % | 9660 | 10985 |
| 50 KB | 15030 | 7183 | 7368 | 0.00141 | 100 % | 0 % | 9119 | 10621 |
| 100 KB | 15030 | 7252 | 6708 | 0.00144 | 100 % | 0 % | 8014 | 10444 |

`tokens_in` incluye los reintentos de validación de Instructor, así que entre tamaños tiene ruido; lo robusto es la columna de caracteres y el recall.

## Hechos obsoletos

Presencia del hecho superado (React tras el pivote, 30.000 € tras la corrección a 80.000 €) a partir del turno 10.

| Escenario | Hecho obsoleto en metadata (turnos ≥ 10) | … en memoria a largo plazo |
|---|---|---|
| contradiction | 0 % | 100 % |
| pivot | 0 % | 34 % |

## Ráfaga

| Peticiones simultáneas | OK | Latencia P50 (ms) | Latencia P95 (ms) | Dentro de presupuesto |
|---|---|---|---|---|
| 1 | 0/1 | — | — | — |
| 5 | 5/5 | 5755 | 12403 | 60 % |
| 10 | 8/10 | 8998 | 9575 | 100 % |
| 20 | 11/20 | 9331 | 12615 | 73 % |

Los fallos de la ráfaga son el mismo fallo de validación del turno 1 (la ráfaga usa ese mensaje), no saturación: ninguno es por límite de uso del proveedor. La latencia de los que terminan crece poco con la concurrencia en este rango (P50 de 5,8 s con 5 simultáneas a 9,3 s con 20).

## Fallos duros

De 45 conversaciones, **solo 18 completaron los 20 turnos**. Fallaron 27 de 727 turnos (3,7 %), pero cada fallo mata la conversación:

| Causa (validación tras 3 intentos) | Turnos | Cuándo |
|---|---|---|
| Más de 15 fases (`phases` tiene `max_length=15`) | 19 de los fallos registrados | Turnos 13–20 de proyectos que crecen |
| `confidence_pct < 30` sin declararlo "Fuera de alcance" | 12 | Turno 1 sin adjunto (mensaje inicial corto) y algún turno largo |
| Una sola fase | 2 | Esporádico |

Además, la prueba destapó un bug: cuando el modelo agota `max_tokens`, Instructor lanza `IncompleteOutputException`, que el cliente no convertía en `LLMError` y salía como un error 500. Está corregido (ahora es un 502 con mensaje claro). Es manejo de errores, no una optimización del CAG, así que no cambia lo medido.

## Lectura

**Dónde empieza a romperse mi CAG y por qué.** No rompe por memoria: en 20 turnos el resumen acumulado y las anclas conservan prácticamente todo. El recall a largo plazo vuelve al 100 % en cuanto actúa el resumen (turno 7), el nombre del proyecto sobrevive en el 100 % de los turnos ≥ 8 y el recall de contexto es del 100 % siempre. Rompe antes por dos lados. El primero es la **latencia**: crece con los tokens de entrada, de 7,0 s de P50 por debajo de 5.000 tokens a 28,3 s (P95 de 54 s) por encima de 15.000. Solo el 23 % de los turnos cumple el presupuesto de 10 s, y la mediana pasa de 7,6 s en el turno 1 a 25,2 s en el turno 20. El **coste** se mantiene acotado por la ventana: el turno 20 cuesta ×2,0–2,4 lo que el turno 1, unos 0,044 USD por conversación de 20 turnos, pero el turno que expulsa el primer turno de la ventana se dispara (mediana de 13.066 tokens en el turno 7, porque el resumidor recibe el turno 1 entero con su adjunto). El segundo lado, y el más grave, son los **fallos duros**: 21 conversaciones mueren entre los turnos 13 y 20 porque el proyecto acumulado ya no cabe en el contrato de salida (más de 15 fases), y 5 mueren en el turno 1. Ningún test unitario los veía: aparecen con conversaciones largas y reales.

**Degradación silenciosa y caso límite para RAG.** Lo más peligroso no es el 502, que se ve, sino lo que pasa sin error. Los adjuntos se recortan a 15.000 caracteres sin avisar al usuario: a partir de 20 KB, el recall del final del documento cae del 100 % al 0 % mientras el del principio sigue al 100 %, y 20, 50 y 100 KB dan exactamente los mismos 15.030 caracteres, así que el coste deja de crecer porque se tira información. Con los hechos contradictorios pasa algo parecido: los 30.000 € siguen en la memoria a largo plazo en el 100 % de los turnos ≥ 10 (el turno de los 30.000 € es un ancla desde el turno 9 y, desde el 14, también el de los 80.000 €: las dos cifras conviven literales en el contexto) y React sigue en el 34 % tras el pivote. La metadata sí se limpia (0 %), pero el modelo recibe a la vez el hecho viejo y el nuevo. El caso que justifica saltar a RAG está en los datos: **documentos de más de unos 15 KB** (todo lo que pasa del límite se pierde en silencio, y subir el límite dispara latencia y coste por la curva 1) y **proyectos que crecen más allá de unos 13 turnos**, donde el contexto entero viaja en cada llamada y la latencia supera los 20 s. Con RAG, cada turno recuperaría solo los fragmentos relevantes del documento y del historial, en vez de un contexto que crece o se recorta a ciegas.

## Limitaciones

- Una sola ejecución de 45 conversaciones con un modelo pequeño: los porcentajes por celda (escenario × adjunto) se basan en pocas conversaciones.
- La latencia se midió con 8 conversaciones en paralelo; en serie sería menor (en la ráfaga, de 5,8 a 9,3 s de P50).
- La búsqueda literal de hechos obsoletos no distingue "se mantiene React" de "se cambió de React a Flutter": el 34 % del pivote es una cota superior.
