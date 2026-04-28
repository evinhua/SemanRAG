"""SemanRAG observability — OpenTelemetry and Prometheus instrumentation."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

from semanrag.api.config import ObservabilityConfig

# ── Prometheus metrics (always importable; no-op if prometheus_client missing) ──

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False

if _HAS_PROM:
    request_count = Counter(
        "semanrag_request_total", "Total HTTP requests", ["method", "path", "status"]
    )
    request_latency_histogram = Histogram(
        "semanrag_request_latency_seconds", "Request latency", ["method", "path"]
    )
    ingestion_throughput = Counter(
        "semanrag_ingestion_docs_total", "Documents ingested"
    )
    query_latency = Histogram(
        "semanrag_query_latency_seconds", "Query latency by mode", ["mode"]
    )
    cache_hit_rate = Counter(
        "semanrag_cache_hits_total", "LLM cache hits", ["hit"]
    )
    verifier_pass_rate = Counter(
        "semanrag_verifier_results_total", "Verifier pass/fail", ["result"]
    )
else:
    # Provide no-op stubs so the rest of the code can reference them safely
    class _Noop:
        def labels(self, *a, **kw):
            return self
        def inc(self, *a, **kw): ...
        def observe(self, *a, **kw): ...

    request_count = _Noop()  # type: ignore[assignment]
    request_latency_histogram = _Noop()  # type: ignore[assignment]
    ingestion_throughput = _Noop()  # type: ignore[assignment]
    query_latency = _Noop()  # type: ignore[assignment]
    cache_hit_rate = _Noop()  # type: ignore[assignment]
    verifier_pass_rate = _Noop()  # type: ignore[assignment]


# ── OpenTelemetry setup ──────────────────────────────────────────────

def setup_otel(app: FastAPI, config: ObservabilityConfig) -> None:
    """Configure OpenTelemetry tracing middleware on *app*."""
    if not config.otel_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        import logging
        logging.getLogger("semanrag.api.telemetry").warning(
            "OpenTelemetry packages not installed — skipping OTEL setup"
        )
        return

    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=config.otel_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)


# ── Prometheus setup ─────────────────────────────────────────────────

def setup_prometheus(app: FastAPI) -> None:
    """Add ``/metrics`` endpoint and request-level instrumentation to *app*."""
    if not _HAS_PROM:
        return

    from fastapi import Request, Response
    from starlette.middleware.base import BaseHTTPMiddleware

    class _PrometheusMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            start = time.perf_counter()
            response = await call_next(request)
            elapsed = time.perf_counter() - start
            path = request.url.path
            request_count.labels(method=request.method, path=path, status=response.status_code).inc()
            request_latency_histogram.labels(method=request.method, path=path).observe(elapsed)
            return response

    app.add_middleware(_PrometheusMiddleware)

    @app.get("/metrics", include_in_schema=False)
    async def _metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
