# Sanity check de los embeddings

Modelo `text-embedding-3-small` (1536 dimensiones), similitud coseno calculada a mano en `scripts/compare.py`. Ejecutado el 21 de noviembre de 2026.

| Pareja | Texto 1 | Texto 2 | Similitud | Expectativa |
|---|---|---|---|---|
| A, cercanos | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Authorization service using JSON Web Tokens for a banking application | **0.5957** | > 0.6 |
| B, no relacionados | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Database migration from MySQL to PostgreSQL with zero downtime | **0.1920** | < 0.4 |
| C, genéricos | Backend services | API development | **0.5407** | sin expectativa fija |

## Comentario

- **Discrimina, pero menos de lo que parece a primera vista.** La pareja B queda muy por debajo del umbral (0.19) y A está tres veces por encima. Sin embargo, A se queda justo por debajo del 0.6 orientativo, y C (0.54), dos frases de dos palabras sin contenido concreto, está casi a la altura de A.
- **A baja por el vocabulario, no por el significado.** Las dos frases dicen lo mismo con palabras distintas: "JWT" frente a "JSON Web Tokens", "fintech mobile app" frente a "banking application". Con una prueba extra, cambiando solo eso en el texto 1 ("OAuth 2.0 authentication backend with JSON Web Tokens for a banking mobile application"), la similitud sube a **0.8063**. El modelo no resuelve bien siglas ni sinónimos de dominio. Para el estimador importa: los presupuestos y las transcripciones están llenos de siglas (JWT, PSD2, ERP, SSO).
- **C sale alta porque los dos textos son vagos.** Los textos genéricos caen en una zona común del espacio vectorial y se parecen entre sí aunque no digan lo mismo. Un chunk muy corto o poco específico se parecería a casi cualquier consulta, lo que apoya añadir el contexto del presupuesto padre en cada chunk.
- **Los umbrales absolutos dependen del modelo.** Con `text-embedding-3-small`, las similitudes caen en un rango comprimido: lo no relacionado ronda 0.1–0.2, no 0. Un umbral fijo como 0.6 hay que calibrarlo con datos propios. Lo útil es el orden relativo (A > C > B), que sí es el esperado.

## Pipeline completo

`POST /embeddings/ingest` con los 15 presupuestos de `data/budgets_sample.json`:

- 52 chunks, uno por componente, de 1536 dimensiones.
- 4.126 tokens: entre 66 y 106 por chunk, con una mediana de 79. Ninguno se acerca al límite del modelo (8.191), así que el chunking estructural no necesita partir componentes.
- Coste estimado de 0,00008 USD; 5,3 s en una sola petición por lotes.
