# ── Stage 1: Frontend ─────────────────────────────────────────────
FROM node:20-slim AS frontend

WORKDIR /app/semanrag_webui
COPY semanrag_webui/package.json semanrag_webui/package-lock.json ./
RUN npm ci
COPY semanrag_webui/ ./
RUN npm run build

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.12-slim

LABEL org.opencontainers.image.title="SemanRAG" \
      org.opencontainers.image.description="Semantic, Graph-Augmented RAG with Production Guardrails" \
      org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY semanrag/ semanrag/

RUN pip install --no-cache-dir ".[api]" openai \
    && apt-get purge -y build-essential gcc && apt-get autoremove -y

COPY --from=frontend /app/semanrag_webui/dist /app/static

ENV SEMANRAG_STATIC_DIR=/app/static

RUN groupadd -r semanrag && useradd -r -g semanrag semanrag \
    && mkdir -p /app/data && chown -R semanrag:semanrag /app

USER semanrag

EXPOSE 9621

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:9621/api/health || exit 1

CMD ["semanrag-server"]
