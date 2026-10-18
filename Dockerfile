# --- Etapa 1: instalar dependencias con uv ---
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Primero solo los archivos de dependencias, para aprovechar la caché de capas
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# --- Etapa 2: imagen final, sin uv ni herramientas de build ---
FROM python:3.11-slim

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY app ./app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    LOG_LEVEL=INFO

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
