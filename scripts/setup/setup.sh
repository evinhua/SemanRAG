#!/usr/bin/env bash
set -euo pipefail

# ── SemanRAG Interactive Setup Wizard ─────────────────────────────────
# Generates .env and optional docker-compose.override.yml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
COMPOSE_OVERRIDE="$PROJECT_ROOT/docker-compose.override.yml"

declare -A CONFIG

# ── Helpers ───────────────────────────────────────────────────────────

prompt_choice() {
    local var_name="$1" prompt="$2" default="$3"
    shift 3
    local options=("$@")
    echo ""
    echo "$prompt"
    for i in "${!options[@]}"; do
        local marker=" "
        [[ "${options[$i]}" == "$default" ]] && marker="*"
        echo "  $marker $((i+1))) ${options[$i]}"
    done
    read -rp "Choice [${default}]: " choice
    choice="${choice:-$default}"
    # If numeric, map to option
    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#options[@]} )); then
        choice="${options[$((choice-1))]}"
    fi
    CONFIG["$var_name"]="$choice"
}

prompt_input() {
    local var_name="$1" prompt="$2" default="${3:-}"
    read -rp "$prompt [${default}]: " value
    CONFIG["$var_name"]="${value:-$default}"
}

prompt_secret() {
    local var_name="$1" prompt="$2"
    read -rsp "$prompt: " value
    echo ""
    CONFIG["$var_name"]="${value:-}"
}

# ── Stage: env-base ──────────────────────────────────────────────────

stage_env_base() {
    echo "═══════════════════════════════════════════════════════════"
    echo "  Stage 1/7: LLM & Embedding Configuration"
    echo "═══════════════════════════════════════════════════════════"

    prompt_choice LLM_BINDING "LLM provider:" "google-genai" \
        "openai" "google-genai" "anthropic" "ollama" "bedrock" "zhipu" "vllm"

    case "${CONFIG[LLM_BINDING]}" in
        openai)       prompt_input LLM_MODEL "Model name" "gpt-4o" ;;
        google-genai) prompt_input LLM_MODEL "Model name" "gemini-2.0-flash" ;;
        anthropic)    prompt_input LLM_MODEL "Model name" "claude-sonnet-4-20250514" ;;
        ollama)       prompt_input LLM_MODEL "Model name" "llama3.1" ;;
        bedrock)      prompt_input LLM_MODEL "Model name" "anthropic.claude-3-sonnet-20240229-v1:0" ;;
        *)            prompt_input LLM_MODEL "Model name" "default" ;;
    esac

    prompt_secret LLM_API_KEY "API key (leave empty for local models)"
    prompt_input LLM_API_BASE "API base URL (leave empty for default)" ""
    prompt_input LLM_MAX_TOKENS "Max tokens" "4096"

    prompt_choice EMBEDDING_BINDING "Embedding provider:" "${CONFIG[LLM_BINDING]}" \
        "openai" "google-genai" "ollama" "huggingface" "bedrock"

    case "${CONFIG[EMBEDDING_BINDING]}" in
        openai)       prompt_input EMBEDDING_MODEL "Embedding model" "text-embedding-3-small" ;;
        google-genai) prompt_input EMBEDDING_MODEL "Embedding model" "text-embedding-004" ;;
        ollama)       prompt_input EMBEDDING_MODEL "Embedding model" "nomic-embed-text" ;;
        *)            prompt_input EMBEDDING_MODEL "Embedding model" "BAAI/bge-base-en-v1.5" ;;
    esac

    prompt_input EMBEDDING_DIMENSION "Embedding dimension" "768"
}

# ── Stage: env-storage ───────────────────────────────────────────────

stage_env_storage() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Stage 2/7: Storage Backends"
    echo "═══════════════════════════════════════════════════════════"

    prompt_choice KG_STORAGE "Knowledge graph backend:" "networkx" \
        "networkx" "neo4j" "memgraph" "postgres-age"
    [[ "${CONFIG[KG_STORAGE]}" == "neo4j" ]] && {
        prompt_input NEO4J_URI "Neo4j URI" "bolt://localhost:7687"
        prompt_input NEO4J_USER "Neo4j user" "neo4j"
        prompt_secret NEO4J_PASSWORD "Neo4j password"
    }

    prompt_choice VECTOR_STORAGE "Vector storage backend:" "nano-vectordb" \
        "nano-vectordb" "milvus" "qdrant" "faiss" "opensearch"

    prompt_choice DOC_STORAGE "Document/KV storage:" "json" \
        "json" "mongodb" "redis" "postgres"

    prompt_choice BM25_STORAGE "BM25 lexical search:" "inmemory" \
        "inmemory" "tantivy" "opensearch"

    prompt_input WORKING_DIR "Working directory" "./data"
}

# ── Stage: env-server ────────────────────────────────────────────────

stage_env_server() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Stage 3/7: API Server"
    echo "═══════════════════════════════════════════════════════════"

    prompt_input API_PORT "API port" "9621"
    prompt_input API_WORKERS "Worker count" "4"

    prompt_choice AUTH_ENABLED "Enable authentication?" "true" "true" "false"
    if [[ "${CONFIG[AUTH_ENABLED]}" == "true" ]]; then
        # Generate a random JWT secret
        JWT_DEFAULT=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))" 2>/dev/null || echo "change-me-$(date +%s)")
        prompt_input JWT_SECRET "JWT secret" "$JWT_DEFAULT"
        prompt_input JWT_EXPIRE_MINUTES "JWT expiry (minutes)" "60"
    else
        CONFIG[JWT_SECRET]="disabled"
        CONFIG[JWT_EXPIRE_MINUTES]="60"
    fi

    prompt_input RATE_LIMIT "Rate limit (requests/minute, 0=off)" "60"
}

# ── Stage: env-safety ────────────────────────────────────────────────

stage_env_safety() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Stage 4/7: Safety & PII"
    echo "═══════════════════════════════════════════════════════════"

    prompt_choice SAFETY_ENABLED "Enable PII detection?" "false" "true" "false"
    if [[ "${CONFIG[SAFETY_ENABLED]}" == "true" ]]; then
        prompt_input SAFETY_ENTITIES "PII entity types" "PERSON,EMAIL_ADDRESS,PHONE_NUMBER"
        prompt_choice PII_ACTION "PII action:" "redact" "redact" "mask" "block"
    fi

    prompt_choice PROMPT_INJECTION_ACTION "Prompt injection action:" "warn" "ignore" "warn" "block"
}

# ── Stage: env-queue ─────────────────────────────────────────────────

stage_env_queue() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Stage 5/7: Job Queue"
    echo "═══════════════════════════════════════════════════════════"

    prompt_choice QUEUE_TYPE "Job queue backend:" "none" "none" "celery" "arq"
    if [[ "${CONFIG[QUEUE_TYPE]}" != "none" ]]; then
        prompt_input REDIS_URL "Redis URL (for queue)" "redis://localhost:6379/0"
    fi
}

# ── Stage: env-observability ─────────────────────────────────────────

stage_env_observability() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Stage 6/7: Observability"
    echo "═══════════════════════════════════════════════════════════"

    prompt_choice OTEL_ENABLED "Enable OpenTelemetry?" "false" "true" "false"
    [[ "${CONFIG[OTEL_ENABLED]}" == "true" ]] && \
        prompt_input OTEL_EXPORTER_OTLP_ENDPOINT "OTel endpoint" "http://localhost:4317"

    prompt_choice PROMETHEUS_ENABLED "Enable Prometheus metrics?" "false" "true" "false"

    prompt_choice LANGFUSE_ENABLED "Enable Langfuse tracing?" "false" "true" "false"
    if [[ "${CONFIG[LANGFUSE_ENABLED]}" == "true" ]]; then
        prompt_input LANGFUSE_HOST "Langfuse host" "https://cloud.langfuse.com"
        prompt_secret LANGFUSE_PUBLIC_KEY "Langfuse public key"
        prompt_secret LANGFUSE_SECRET_KEY "Langfuse secret key"
    fi

    prompt_input LOG_LEVEL "Log level" "INFO"
}

# ── Generate .env ────────────────────────────────────────────────────

generate_env() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Stage 7/7: Generating Configuration"
    echo "═══════════════════════════════════════════════════════════"

    cat > "$ENV_FILE" << ENVEOF
# ── Generated by SemanRAG setup wizard ────────────────────────────
# $(date -Iseconds)

# ── LLM ───────────────────────────────────────────────────────────
LLM_MODEL=${CONFIG[LLM_MODEL]}
LLM_BINDING=${CONFIG[LLM_BINDING]}
LLM_API_KEY=${CONFIG[LLM_API_KEY]:-}
LLM_API_BASE=${CONFIG[LLM_API_BASE]:-}
LLM_MAX_TOKENS=${CONFIG[LLM_MAX_TOKENS]}
LLM_TEMPERATURE=0.0

# ── Embedding ─────────────────────────────────────────────────────
EMBEDDING_MODEL=${CONFIG[EMBEDDING_MODEL]}
EMBEDDING_BINDING=${CONFIG[EMBEDDING_BINDING]}
EMBEDDING_DIMENSION=${CONFIG[EMBEDDING_DIMENSION]}
EMBEDDING_BATCH_SIZE=64

# ── Storage ───────────────────────────────────────────────────────
KG_STORAGE=${CONFIG[KG_STORAGE]}
VECTOR_STORAGE=${CONFIG[VECTOR_STORAGE]}
DOC_STORAGE=${CONFIG[DOC_STORAGE]}
WORKING_DIR=${CONFIG[WORKING_DIR]}
ENVEOF

    # Conditional sections
    [[ -n "${CONFIG[NEO4J_URI]:-}" ]] && cat >> "$ENV_FILE" << ENVEOF
NEO4J_URI=${CONFIG[NEO4J_URI]}
NEO4J_USER=${CONFIG[NEO4J_USER]}
NEO4J_PASSWORD=${CONFIG[NEO4J_PASSWORD]}
ENVEOF

    cat >> "$ENV_FILE" << ENVEOF

# ── API Server ────────────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=${CONFIG[API_PORT]}
API_WORKERS=${CONFIG[API_WORKERS]}
JWT_SECRET=${CONFIG[JWT_SECRET]}
JWT_EXPIRE_MINUTES=${CONFIG[JWT_EXPIRE_MINUTES]}

# ── Safety ────────────────────────────────────────────────────────
SAFETY_ENABLED=${CONFIG[SAFETY_ENABLED]}
PROMPT_INJECTION_ACTION=${CONFIG[PROMPT_INJECTION_ACTION]}
ENVEOF

    [[ "${CONFIG[SAFETY_ENABLED]}" == "true" ]] && \
        echo "SAFETY_ENTITIES=${CONFIG[SAFETY_ENTITIES]}" >> "$ENV_FILE"

    [[ "${CONFIG[QUEUE_TYPE]}" != "none" ]] && cat >> "$ENV_FILE" << ENVEOF

# ── Queue ─────────────────────────────────────────────────────────
QUEUE_TYPE=${CONFIG[QUEUE_TYPE]}
REDIS_URL=${CONFIG[REDIS_URL]:-redis://localhost:6379/0}
ENVEOF

    cat >> "$ENV_FILE" << ENVEOF

# ── Observability ─────────────────────────────────────────────────
LOG_LEVEL=${CONFIG[LOG_LEVEL]}
ENVEOF

    [[ "${CONFIG[OTEL_ENABLED]}" == "true" ]] && \
        echo "OTEL_EXPORTER_OTLP_ENDPOINT=${CONFIG[OTEL_EXPORTER_OTLP_ENDPOINT]}" >> "$ENV_FILE"

    [[ "${CONFIG[LANGFUSE_ENABLED]}" == "true" ]] && cat >> "$ENV_FILE" << ENVEOF
LANGFUSE_HOST=${CONFIG[LANGFUSE_HOST]}
LANGFUSE_PUBLIC_KEY=${CONFIG[LANGFUSE_PUBLIC_KEY]}
LANGFUSE_SECRET_KEY=${CONFIG[LANGFUSE_SECRET_KEY]}
ENVEOF

    echo "  ✓ Generated $ENV_FILE"
}

# ── Generate docker-compose override ─────────────────────────────────

generate_compose() {
    local need_compose=false
    local services=""

    if [[ "${CONFIG[KG_STORAGE]}" == "neo4j" ]]; then
        need_compose=true
        services+="
  neo4j:
    image: neo4j:5-community
    ports: [\"7474:7474\", \"7687:7687\"]
    environment:
      NEO4J_AUTH: \"${CONFIG[NEO4J_USER]:-neo4j}/${CONFIG[NEO4J_PASSWORD]:-password}\"
    volumes: [neo4j-data:/data]
"
    fi

    if [[ "${CONFIG[VECTOR_STORAGE]}" == "qdrant" ]]; then
        need_compose=true
        services+="
  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports: [\"6333:6333\"]
    volumes: [qdrant-data:/qdrant/storage]
"
    fi

    if [[ "${CONFIG[QUEUE_TYPE]}" != "none" ]] || [[ "${CONFIG[DOC_STORAGE]}" == "redis" ]]; then
        need_compose=true
        services+="
  redis:
    image: redis:7-alpine
    ports: [\"6379:6379\"]
"
    fi

    if [[ "$need_compose" == "true" ]]; then
        cat > "$COMPOSE_OVERRIDE" << COMPEOF
# Generated by SemanRAG setup wizard
services:${services}
volumes:
COMPEOF
        # Add volume declarations
        [[ "${CONFIG[KG_STORAGE]}" == "neo4j" ]] && echo "  neo4j-data:" >> "$COMPOSE_OVERRIDE"
        [[ "${CONFIG[VECTOR_STORAGE]}" == "qdrant" ]] && echo "  qdrant-data:" >> "$COMPOSE_OVERRIDE"
        echo "  ✓ Generated $COMPOSE_OVERRIDE"
    fi
}

# ── Validate ─────────────────────────────────────────────────────────

validate_env() {
    local errors=0
    if [[ -z "${CONFIG[LLM_MODEL]}" ]]; then
        echo "  ✗ LLM_MODEL is required"
        errors=$((errors + 1))
    fi
    if [[ -z "${CONFIG[LLM_BINDING]}" ]]; then
        echo "  ✗ LLM_BINDING is required"
        errors=$((errors + 1))
    fi
    if [[ "${CONFIG[LLM_BINDING]}" != "ollama" ]] && [[ -z "${CONFIG[LLM_API_KEY]:-}" ]]; then
        echo "  ⚠ LLM_API_KEY is empty (required for cloud providers)"
    fi
    if (( errors > 0 )); then
        echo "  ✗ Validation failed with $errors error(s)"
        return 1
    fi
    echo "  ✓ Configuration validated"
    return 0
}

# ── Main ─────────────────────────────────────────────────────────────

main() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║           SemanRAG Interactive Setup Wizard              ║"
    echo "╚═══════════════════════════════════════════════════════════╝"

    if [[ -f "$ENV_FILE" ]]; then
        read -rp "Existing .env found. Overwrite? [y/N]: " overwrite
        [[ "$overwrite" != [yY]* ]] && { echo "Aborted."; exit 0; }
    fi

    stage_env_base
    stage_env_storage
    stage_env_server
    stage_env_safety
    stage_env_queue
    stage_env_observability
    generate_env
    generate_compose
    validate_env

    echo ""
    echo "Setup complete! Next steps:"
    echo "  1. Review .env and adjust as needed"
    echo "  2. pip install -e '.[api]'"
    echo "  3. semanrag serve"
    echo ""
}

main "$@"
