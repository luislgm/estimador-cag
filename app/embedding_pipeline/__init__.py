"""Pipeline mínimo de embeddings: presupuestos JSON -> chunks estructurales -> vectores.

    POST /embeddings/ingest   presupuestos -> chunks con su embedding y estadísticas

- schemas.py: contratos (presupuesto, chunk, petición y respuesta).
- chunker.py: un componente del presupuesto = un chunk, con el contexto del padre.
- embedder.py: text-embedding-3-small de OpenAI, por lotes y con reintentos.
- router.py: el endpoint.

Los vectores se devuelven por HTTP y no se guardan: la persistencia (pgvector) y la búsqueda
llegan en la sesión 8. scripts/compare.py comprueba que los embeddings discriminan.
"""
