"""SemanRAG FastAPI server — main application entry point."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from semanrag.api.config import (
    AuthConfig,
    LLMConfig,
    ObservabilityConfig,
    ServerConfig,
    StorageConfig,
    get_config,
)
from semanrag.utils import EmbeddingFunc, logger

# ── Optional imports (graceful degradation) ──────────────────────────

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    _HAS_SLOWAPI = True
except ImportError:
    _HAS_SLOWAPI = False


# ═════════════════════════════════════════════════════════════════════
# LLM / Embedding / Reranker factories
# ═════════════════════════════════════════════════════════════════════


def _build_llm_model_func(cfg: LLMConfig) -> Any:
    """Return an async LLM completion function based on *cfg.provider*."""
    provider = cfg.provider.lower()
    model = cfg.model
    api_key = cfg.api_key or None
    base_url = cfg.base_url or None

    if provider in ("openai", "azure_openai"):
        from semanrag.llm.openai_impl import openai_complete_if_cache

        return partial(openai_complete_if_cache, model=model, api_key=api_key, base_url=base_url)

    if provider == "ollama":
        from semanrag.llm.ollama_impl import ollama_model_complete

        return partial(ollama_model_complete, model=model, host=base_url)

    if provider in ("gemini", "google-genai", "google"):
        from semanrag.llm.gemini_impl import gemini_complete_if_cache

        return partial(gemini_complete_if_cache, model=model, api_key=api_key)

    if provider == "bedrock":
        from semanrag.llm.bedrock_impl import bedrock_complete_if_cache

        return partial(bedrock_complete_if_cache, model=model)

    if provider in ("anthropic", "claude"):
        from semanrag.llm.anthropic_impl import anthropic_complete_if_cache

        return partial(anthropic_complete_if_cache, model=model, api_key=api_key)

    if provider == "hf":
        from semanrag.llm.hf_impl import hf_model_complete

        return partial(hf_model_complete, model=model)

    if provider == "zhipu":
        from semanrag.llm.zhipu_impl import zhipu_complete_if_cache

        return partial(zhipu_complete_if_cache, model=model)

    if provider == "vllm":
        from semanrag.llm.vllm_impl import vllm_complete

        return partial(vllm_complete, model=model, base_url=base_url)

    if provider == "llama_index":
        from semanrag.llm.llama_index_impl import llama_index_complete_if_cache

        return partial(llama_index_complete_if_cache, model=model)

    raise ValueError(f"Unknown LLM provider: {provider!r}")


def _build_embedding_func(cfg: LLMConfig) -> EmbeddingFunc:
    """Return an :class:`EmbeddingFunc` based on *cfg.provider*."""
    provider = cfg.provider.lower()
    model = cfg.embedding_model
    api_key = cfg.api_key or None
    base_url = cfg.base_url or None

    if provider in ("openai", "azure_openai"):
        from semanrag.llm.openai_impl import openai_embed

        return EmbeddingFunc(
            embedding_dim=cfg.embedding_dim,
            max_token_size=cfg.max_token_size,
            func=partial(openai_embed, model=model, api_key=api_key, base_url=base_url),
        )

    if provider == "ollama":
        from semanrag.llm.ollama_impl import ollama_embed

        return EmbeddingFunc(
            embedding_dim=cfg.embedding_dim,
            max_token_size=cfg.max_token_size,
            func=partial(ollama_embed, model=model, host=base_url),
        )

    if provider in ("gemini", "google-genai", "google"):
        from semanrag.llm.gemini_impl import gemini_embed

        return EmbeddingFunc(
            embedding_dim=cfg.embedding_dim,
            max_token_size=cfg.max_token_size,
            func=partial(gemini_embed, model=model, api_key=api_key),
        )

    if provider == "bedrock":
        from semanrag.llm.bedrock_impl import bedrock_embed

        return EmbeddingFunc(
            embedding_dim=cfg.embedding_dim,
            max_token_size=cfg.max_token_size,
            func=partial(bedrock_embed, model=model),
        )

    if provider in ("anthropic", "claude"):
        from semanrag.llm.anthropic_impl import anthropic_embed

        return EmbeddingFunc(
            embedding_dim=cfg.embedding_dim,
            max_token_size=cfg.max_token_size,
            func=partial(anthropic_embed, model=model, api_key=api_key),
        )

    if provider == "hf":
        from semanrag.llm.hf_impl import hf_embed

        return EmbeddingFunc(
            embedding_dim=cfg.embedding_dim,
            max_token_size=cfg.max_token_size,
            func=partial(hf_embed, model=model),
        )

    if provider == "zhipu":
        from semanrag.llm.zhipu_impl import zhipu_embedding

        return EmbeddingFunc(
            embedding_dim=cfg.embedding_dim,
            max_token_size=cfg.max_token_size,
            func=partial(zhipu_embedding, model=model),
        )

    if provider == "vllm":
        from semanrag.llm.vllm_impl import vllm_embed

        return EmbeddingFunc(
            embedding_dim=cfg.embedding_dim,
            max_token_size=cfg.max_token_size,
            func=partial(vllm_embed, model=model, base_url=base_url),
        )

    raise ValueError(f"Unknown embedding provider: {provider!r}")


def _build_rerank_func(cfg: LLMConfig) -> Any | None:
    """Return a reranker function if a rerank model is configured, else None."""
    rerank_provider = os.environ.get("SEMANRAG_RERANK_PROVIDER", "").lower()
    if not rerank_provider:
        return None

    rerank_model = os.environ.get("SEMANRAG_RERANK_MODEL", "")

    if rerank_provider == "cohere":
        from semanrag.rerank.cohere_rerank import cohere_rerank

        return partial(cohere_rerank, model=rerank_model) if rerank_model else cohere_rerank

    if rerank_provider == "jina":
        from semanrag.rerank.jina_rerank import jina_rerank

        return partial(jina_rerank, model=rerank_model) if rerank_model else jina_rerank

    if rerank_provider == "bge":
        from semanrag.rerank.bge_rerank import bge_rerank

        return partial(bge_rerank, model=rerank_model) if rerank_model else bge_rerank

    if rerank_provider == "local":
        from semanrag.rerank.local_cross_encoder import local_cross_encoder_rerank

        return local_cross_encoder_rerank

    logger.warning("Unknown rerank provider %r — reranking disabled", rerank_provider)
    return None


def _build_verifier_func() -> Any | None:
    """Return the grounded-check verifier if enabled."""
    if os.environ.get("SEMANRAG_VERIFIER_ENABLED", "true").lower() in ("1", "true", "yes"):
        from semanrag.verify.grounded_check import grounded_check

        return grounded_check
    return None


# ═════════════════════════════════════════════════════════════════════
# Lifespan
# ═════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle for the SemanRAG server."""
    from semanrag.semanrag import SemanRAG

    config = get_config()
    llm_cfg: LLMConfig = config["llm"]
    storage_cfg: StorageConfig = config["storage"]

    embedding_func = _build_embedding_func(llm_cfg)
    llm_model_func = _build_llm_model_func(llm_cfg)
    rerank_func = _build_rerank_func(llm_cfg)
    verifier_func = _build_verifier_func()

    rag = SemanRAG(
        working_dir=storage_cfg.working_dir,
        workspace=storage_cfg.workspace or None,
        embedding_func=embedding_func,
        llm_model_func=llm_model_func,
        llm_model_name=llm_cfg.model,
        rerank_func=rerank_func,
        verifier_func=verifier_func,
        safety_config={
            "pii_policy": config["safety"].pii_policy,
            "prompt_injection_action": config["safety"].prompt_injection_action,
        },
    )
    await rag.initialize_storages()
    app.state.rag = rag
    app.state.config = config
    logger.info("SemanRAG server started (provider=%s, model=%s)", llm_cfg.provider, llm_cfg.model)

    yield

    await rag.finalize_storages()
    logger.info("SemanRAG server shut down")


# ═════════════════════════════════════════════════════════════════════
# App factory
# ═════════════════════════════════════════════════════════════════════


def create_app() -> FastAPI:
    config = get_config()
    server_cfg: ServerConfig = config["server"]
    auth_cfg: AuthConfig = config["auth"]
    obs_cfg: ObservabilityConfig = config["observability"]

    app = FastAPI(
        title="SemanRAG API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Rate limiting ─────────────────────────────────────────────
    if _HAS_SLOWAPI:
        limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Auth middleware ────────────────────────────────────────────
    if auth_cfg.enabled:
        from semanrag.api.auth import AuthHandler

        auth_handler = AuthHandler(auth_cfg)
        app.state.auth = auth_handler

    # ── Telemetry ─────────────────────────────────────────────────
    from semanrag.api.telemetry import setup_otel, setup_prometheus

    setup_otel(app, obs_cfg)
    if obs_cfg.prometheus_enabled:
        setup_prometheus(app)

    # ── Routers ───────────────────────────────────────────────────
    _include_routers(app)

    # ── Static files (WebUI) ──────────────────────────────────────
    static_dir = server_cfg.static_dir or os.environ.get("SEMANRAG_STATIC_DIR", "")
    if static_dir and Path(static_dir).is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="webui")

    # ── Global exception handler ──────────────────────────────────
    @app.exception_handler(Exception)
    async def _global_exc(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


def _include_routers(app: FastAPI) -> None:
    """Discover and include all routers from semanrag.api.routers."""
    import importlib
    import pkgutil

    import semanrag.api.routers as routers_pkg

    for importer, modname, ispkg in pkgutil.iter_modules(routers_pkg.__path__):
        try:
            mod = importlib.import_module(f"semanrag.api.routers.{modname}")
            if hasattr(mod, "router"):
                app.include_router(mod.router)
                logger.debug("Included router: %s", modname)
        except Exception:
            logger.exception("Failed to load router %s", modname)


app = create_app()


# ═════════════════════════════════════════════════════════════════════
# Entry points
# ═════════════════════════════════════════════════════════════════════


def main() -> None:
    """Run with uvicorn (``semanrag-server`` console script)."""
    import uvicorn

    config = get_config()
    server_cfg: ServerConfig = config["server"]
    uvicorn.run(
        "semanrag.api.semanrag_server:app",
        host=server_cfg.host,
        port=server_cfg.port,
        workers=server_cfg.workers,
        log_level="info",
    )


def gunicorn_main() -> None:
    """Run with gunicorn + uvicorn workers (``semanrag-gunicorn`` console script)."""
    config = get_config()
    server_cfg: ServerConfig = config["server"]
    argv = [
        "gunicorn",
        "semanrag.api.semanrag_server:app",
        "-k", "uvicorn.workers.UvicornWorker",
        "-b", f"{server_cfg.host}:{server_cfg.port}",
        "-w", str(server_cfg.workers),
        "--access-logfile", "-",
    ]
    os.execvp("gunicorn", argv)


if __name__ == "__main__":
    main()
