# ── Stage 1: Builder ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc libpq-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install ".[all]"

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.12-slim

LABEL org.opencontainers.image.title="SemanRAG" \
      org.opencontainers.image.description="Semantic, Graph-Augmented RAG with Production Guardrails" \
      org.opencontainers.image.source="https://github.com/semanrag/semanrag" \
      org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r semanrag && useradd -r -g semanrag semanrag

COPY --from=builder /install /usr/local
COPY --from=builder /app /app

WORKDIR /app

RUN mkdir -p /app/data && chown -R semanrag:semanrag /app

USER semanrag

EXPOSE 9621

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:9621/health || exit 1

CMD ["semanrag-server"]
