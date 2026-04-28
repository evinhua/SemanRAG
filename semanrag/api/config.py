"""SemanRAG API configuration — loads from environment variables with SEMANRAG_ prefix."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str = "") -> str:
    return os.environ.get(f"SEMANRAG_{key}", os.environ.get(key, default))


def _env_bool(key: str, default: bool = False) -> bool:
    return _env(key, str(default)).lower() in ("1", "true", "yes")


def _env_int(key: str, default: int = 0) -> int:
    raw = _env(key, "")
    return int(raw) if raw else default


@dataclass
class ServerConfig:
    host: str = ""
    port: int = 9621
    workers: int = 1
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    static_dir: str = ""


@dataclass
class LLMConfig:
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    embedding_model: str = ""
    embedding_dim: int = 1536
    max_token_size: int = 8192


@dataclass
class StorageConfig:
    kv_type: str = "json"
    vector_type: str = "nano"
    graph_type: str = "networkx"
    lexical_type: str = "inmemory_bm25"
    doc_status_type: str = "json"
    working_dir: str = "./semanrag_data"
    workspace: str = ""


@dataclass
class SafetyConfig:
    pii_policy: str = "flag"
    prompt_injection_action: str = "flag"
    output_sanitization: bool = True


@dataclass
class AuthConfig:
    enabled: bool = False
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    api_key_header: str = "X-API-Key"
    oidc_issuer: str = ""
    oidc_client_id: str = ""


@dataclass
class ObservabilityConfig:
    otel_enabled: bool = False
    otel_endpoint: str = ""
    prometheus_enabled: bool = True
    langfuse_enabled: bool = False


def get_config() -> dict:
    """Load all configuration sections from environment variables with SEMANRAG_ prefix."""
    server = ServerConfig(
        host=_env("HOST", "0.0.0.0"),
        port=_env_int("PORT", 9621),
        workers=_env_int("WORKERS", 1),
        cors_origins=[o.strip() for o in _env("CORS_ORIGINS", "*").split(",")],
        static_dir=_env("STATIC_DIR", ""),
    )
    llm = LLMConfig(
        provider=_env("LLM_BINDING", _env("LLM_PROVIDER", "openai")),
        model=_env("LLM_MODEL", "gpt-4o"),
        api_key=_env("LLM_API_KEY", ""),
        base_url=_env("LLM_API_BASE", _env("LLM_BASE_URL", "")),
        embedding_model=_env("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dim=_env_int("EMBEDDING_DIMENSION", _env_int("EMBEDDING_DIM", 1536)),
        max_token_size=_env_int("MAX_TOKEN_SIZE", 8192),
    )
    storage = StorageConfig(
        kv_type=_env("KV_TYPE", "json"),
        vector_type=_env("VECTOR_TYPE", "nano"),
        graph_type=_env("GRAPH_TYPE", "networkx"),
        lexical_type=_env("LEXICAL_TYPE", "inmemory_bm25"),
        doc_status_type=_env("DOC_STATUS_TYPE", "json"),
        working_dir=_env("WORKING_DIR", "./semanrag_data"),
        workspace=_env("WORKSPACE", ""),
    )
    safety = SafetyConfig(
        pii_policy=_env("PII_POLICY", "flag"),
        prompt_injection_action=_env("PROMPT_INJECTION_ACTION", "flag"),
        output_sanitization=_env_bool("OUTPUT_SANITIZATION", True),
    )
    auth = AuthConfig(
        enabled=_env_bool("AUTH_ENABLED", False),
        secret_key=_env("AUTH_SECRET_KEY", "change-me-in-production"),
        algorithm=_env("AUTH_ALGORITHM", "HS256"),
        access_token_expire_minutes=_env_int("AUTH_TOKEN_EXPIRE_MINUTES", 30),
        api_key_header=_env("AUTH_API_KEY_HEADER", "X-API-Key"),
        oidc_issuer=_env("OIDC_ISSUER", ""),
        oidc_client_id=_env("OIDC_CLIENT_ID", ""),
    )
    observability = ObservabilityConfig(
        otel_enabled=_env_bool("OTEL_ENABLED", False),
        otel_endpoint=_env("OTEL_ENDPOINT", "http://localhost:4317"),
        prometheus_enabled=_env_bool("PROMETHEUS_ENABLED", True),
        langfuse_enabled=_env_bool("LANGFUSE_ENABLED", False),
    )
    return {
        "server": server,
        "llm": llm,
        "storage": storage,
        "safety": safety,
        "auth": auth,
        "observability": observability,
    }
