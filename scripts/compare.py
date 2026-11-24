"""Similitud coseno entre dos textos con los embeddings del pipeline.

    uv run python scripts/compare.py --text-a "..." --text-b "..."
    docker compose exec api python scripts/compare.py --text-a "..." --text-b "..."

Reutiliza OpenAIEmbedder (necesita OPENAI_API_KEY: fuera del contenedor la lee del .env) y
calcula la similitud a mano, sin numpy.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ejecutado como `python scripts/compare.py`, la raíz del proyecto no está en sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog  # noqa: E402

from app.embedding_pipeline.embedder import OpenAIEmbedder, cosine_similarity  # noqa: E402


def main(argv: list[str] | None = None, embedder: OpenAIEmbedder | None = None) -> int:
    parser = argparse.ArgumentParser(description="Similitud coseno entre dos textos")
    parser.add_argument("--text-a", required=True)
    parser.add_argument("--text-b", required=True)
    args = parser.parse_args(argv)
    # Solo el resultado en la salida: los logs informativos del embedder sobran aquí
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

    embedder = embedder or OpenAIEmbedder()
    similarity = cosine_similarity(embedder.embed_one(args.text_a), embedder.embed_one(args.text_b))
    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {similarity:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
