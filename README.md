# SemanRAG

**Semantic, Graph-Augmented RAG with Production Guardrails**

SemanRAG is a pluggable Retrieval-Augmented Generation framework that combines graph-augmented retrieval, hybrid lexical + dense search, structured LLM extraction, and production-grade safety/observability in a single architecture.

---

## Quick Start

### Prerequisites

- Python 3.10+
- An LLM API key (OpenAI, Gemini, Anthropic, Ollama, etc.)

### Install

```bash
# Clone and install
git clone https://github.com/your-org/semanrag.git
cd semanrag
pip install -e ".[api]"

# Or install everything
pip install -e ".[api,offline-llm,offline-storage,safety,observability]"
```

### Configure

```bash
# Copy and edit the environment file
cp env.example .env
# Set at minimum: LLM_API_KEY, EMBEDDING_API_KEY (or use Ollama for local)
```

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gemini-2.0-flash` | LLM model name |
| `LLM_BINDING` | `google-genai` | Provider: `openai`, `google-genai`, `anthropic`, `ollama`, `bedrock` |
| `LLM_API_KEY` | — | API key for the LLM provider |
| `EMBEDDING_MODEL` | `text-embedding-004` | Embedding model name |
| `EMBEDDING_DIMENSION` | `768` | Embedding vector dimension |
| `WORKING_DIR` | `./data` | Directory for local storage files |
| `KG_STORAGE` | `networkx` | Graph backend: `networkx`, `neo4j`, `postgres`, `mongo` |
| `VECTOR_STORAGE` | `nano-vectordb` | Vector backend: `nano-vectordb`, `milvus`, `qdrant`, `postgres` |

### Run

```bash
# Start the API server
semanrag serve

# Or use uvicorn directly
uvicorn semanrag.api.semanrag_server:app --host 0.0.0.0 --port 9621
```

The server starts at `http://localhost:9621` with Swagger docs at `/docs`.

---

## Usage

### Python API

```python
import asyncio
from semanrag.semanrag import SemanRAG
from semanrag.base import QueryParam

async def main():
    rag = SemanRAG(
        working_dir="./my_data",
        llm_model_func=my_llm_func,      # your async LLM callable
        embedding_func=my_embedding_func,  # your async embedding callable
    )
    await rag.initialize_storages()

    # Ingest documents
    await rag.ainsert("Your document text here...")
    await rag.ainsert(["Doc 1 content", "Doc 2 content"])

    # Query with different modes
    result = await rag.aquery("What is the relationship between X and Y?")
    print(result.content)

    # Query with specific mode
    result = await rag.aquery(
        "Summarize the main themes",
        QueryParam(mode="community")
    )

    # Streaming
    result = await rag.aquery_stream("Explain concept Z")
    async for chunk in result.response_iterator:
        print(chunk, end="")

    await rag.finalize_storages()

asyncio.run(main())
```

### Using with OpenAI

```python
from semanrag.semanrag import SemanRAG
from semanrag.llm.openai_impl import gpt_4o_mini_complete, openai_embed
from semanrag.utils import EmbeddingFunc

rag = SemanRAG(
    working_dir="./data",
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=lambda texts: openai_embed(texts, model="text-embedding-3-small"),
    ),
)
```

### Using with Ollama (local, no API key)

```python
from semanrag.semanrag import SemanRAG
from semanrag.llm.ollama_impl import ollama_model_complete, ollama_embed
from semanrag.utils import EmbeddingFunc
from functools import partial

rag = SemanRAG(
    working_dir="./data",
    llm_model_func=partial(ollama_model_complete, model="llama3.1"),
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(texts, model="nomic-embed-text"),
    ),
)
```

### CLI

```bash
# Ingest a file or directory
semanrag ingest ./documents/report.pdf
semanrag ingest ./documents/ --recursive

# Query
semanrag query "What are the key findings?" --mode hybrid
semanrag query "Summarize themes" --mode community

# Export knowledge graph
semanrag graph export --format json --output kg.json
semanrag graph export --format csv --output kg.csv

# Run evaluation
semanrag eval run --domain finance

# Administration
semanrag admin cache purge --scope all
semanrag admin budget --user alice --max-tokens 100000

# Start server
semanrag serve --host 0.0.0.0 --port 9621
```

### REST API

Once the server is running:

```bash
# Insert text
curl -X POST http://localhost:9621/documents/text \
  -H "Content-Type: application/json" \
  -d '{"content": "Your document text...", "id": "doc-001"}'

# Upload a file
curl -X POST http://localhost:9621/documents/upload \
  -F "file=@report.pdf"

# Query
curl -X POST http://localhost:9621/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is X?", "mode": "hybrid", "top_k": 20}'

# Query with streaming (SSE)
curl -X POST http://localhost:9621/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain Y", "mode": "naive"}'

# Get knowledge graph visualization data
curl http://localhost:9621/graph

# Get entity neighborhood
curl http://localhost:9621/graph/neighborhood/EntityName?hops=2

# Find path between entities
curl "http://localhost:9621/graph/path?src=EntityA&tgt=EntityB"
```

### WebUI

```bash
cd semanrag_webui
npm install   # or: bun install
npm run dev   # starts at http://localhost:5173, proxies API to :9621
```

Features: graph explorer (sigma.js), multi-turn chat with all 6 retrieval modes, document manager with drag-and-drop upload, admin console with token usage dashboards.

---

## Retrieval Modes

| Mode | What it does | Best for |
|---|---|---|
| `local` | Vector search on entities → connected relationships → source chunks | Entity-specific detail questions |
| `global` | Vector search on relationships → connected entities → source chunks | Relationship-driven questions |
| `hybrid` | `local` + `global` fused | Mixed detail and relationship |
| `naive` | Direct vector search on document chunks | Simple lookup |
| `mix` | KG retrieval + naive chunk retrieval | Coverage over precision |
| `community` | Query-matched community summaries (Leiden-clustered) | Broad thematic questions |
| `bypass` | Direct LLM call, no retrieval | Baseline / debugging |

Every non-`bypass` mode runs BM25 in parallel with dense retrieval and fuses via RRF. `naive` / `mix` additionally rerank with a cross-encoder by default.

---

## Document Ingestion

SemanRAG supports multiple file formats and chunking strategies:

```python
# Ingest from files (auto-detects format)
await rag.ainsert(content, file_paths=["report.pdf"])

# With ACL policy
from semanrag.base import ACLPolicy
await rag.ainsert(
    content,
    acl_policy=ACLPolicy(owner="alice", visible_to_groups=["engineering"], public=False),
)
```

**Supported formats:** PDF (text + tables + figures), DOCX, PPTX, XLSX, Markdown, plain text.

**Chunking strategies** (set via `chunking_strategy` parameter):
- `token` — fixed token-size chunks with overlap (default)
- `semantic` — split at embedding-drift boundaries
- `structure` — respect document structure (headings, sections)

**Safety scans** run automatically during ingestion:
- PII detection (configurable: flag / mask / redact / reject)
- Prompt-injection canary detection

---

## Storage Backends

| Storage type | Supported backends | Default |
|---|---|---|
| KV | JSON files, PostgreSQL, Redis, MongoDB, OpenSearch | JSON files |
| Vector | NanoVectorDB, PostgreSQL (pgvector), Milvus, Qdrant, MongoDB Atlas, OpenSearch, Faiss | NanoVectorDB |
| Graph | NetworkX, Neo4j, PostgreSQL (AGE), Memgraph, OpenSearch, MongoDB | NetworkX |
| DocStatus | JSON files, PostgreSQL, MongoDB, OpenSearch | JSON files |
| Lexical (BM25) | `rank-bm25` (in-memory), `tantivy`, PostgreSQL `tsvector`, OpenSearch | `rank-bm25` |

Switch backends via environment variables or constructor parameters. All backends support workspace isolation and document-level ACLs.

---

## Deployment

### Docker

```bash
# Simple (API only)
docker compose up -d

# Full stack (API + Postgres + Neo4j + Redis + Milvus + monitoring)
docker compose -f docker-compose-full.yml up -d
```

### Kubernetes

```bash
# Helm
helm install semanrag k8s-deploy/helm/ -f my-values.yaml

# Raw manifests
kubectl apply -f k8s-deploy/manifests/
```

### AWS Lambda

Deploy `deploy/lambda_function.py` with API Gateway. KG data stored in S3.

### Interactive Setup

```bash
bash scripts/setup/setup.sh
```

Walks through LLM provider, storage backends, auth, safety, and observability configuration. Generates `.env` and Docker Compose files.

---

## Evaluation

```bash
# Run evaluation against golden sets
semanrag eval run --domain cs

# Or programmatically
python -m semanrag.evaluation.runner \
  --goldens semanrag/evaluation/goldens/finance.jsonl \
  --output eval_report.json

# Check for regressions against baseline
python -m semanrag.evaluation.regression_gate \
  --current eval_report.json \
  --baseline semanrag/evaluation/baselines/baseline.json \
  --threshold 0.02
```

Metrics: context precision, faithfulness, answer relevancy, context recall (via RAGAS), grounded-check pass rate.

CI automatically blocks PRs that regress any metric by more than 2%.

---

## Observability

- **Prometheus** metrics at `/metrics` — query latency, ingestion throughput, cache hit rates, verifier pass rate
- **OpenTelemetry** traces across the async pipeline (configure `OTEL_EXPORTER_OTLP_ENDPOINT`)
- **Grafana** dashboards shipped in `observability/grafana/`
- **Langfuse** integration for LLM-specific tracing (optional)

```bash
# Start monitoring stack
docker compose -f docker-compose-full.yml up prometheus grafana otel-collector -d
```

---

## Project Structure

```
SemanRAG/
├── semanrag/                  # Python package
│   ├── base.py                # Data model & storage abstractions
│   ├── operate.py             # Ingestion pipeline & query engine
│   ├── semanrag.py            # Main orchestrator class
│   ├── utils.py               # Utilities (hashing, caching, RRF, etc.)
│   ├── prompt.py              # LLM prompt templates
│   ├── utils_graph.py         # KG management operations
│   ├── performance.py         # Caching & performance utilities
│   ├── kg/                    # Storage backend implementations (15+ files)
│   ├── llm/                   # LLM provider implementations (10 providers)
│   ├── rerank/                # Reranker implementations (5 providers)
│   ├── verify/                # Grounded-check verifier
│   ├── safety/                # PII, prompt injection, ACL, output sanitizer
│   ├── api/                   # FastAPI server & routes (45+ endpoints)
│   ├── cli/                   # Unified CLI
│   ├── tools/                 # Maintenance & admin tools
│   └── evaluation/            # Eval runner, metrics, golden sets
├── semanrag_webui/            # React 19 + TypeScript frontend
├── tests/                     # 32 test files, 113+ tests
├── specs/                     # Design specifications
├── scripts/setup/             # Interactive setup wizard
├── observability/             # Grafana dashboards, Prometheus config
├── k8s-deploy/                # Helm chart & K8s manifests
├── deploy/                    # Lambda handler
├── docker-compose.yml         # Simple deployment
├── docker-compose-full.yml    # Full stack (13 services)
├── Dockerfile                 # Multi-stage production build
├── Makefile                   # Dev targets
└── pyproject.toml             # Package configuration
```

---

## Development

```bash
# Install with all dev dependencies
make dev
# or: pip install -e ".[api,offline-llm,offline-storage,safety,test,evaluation]"

# Run tests
make test                    # unit tests
make test-integration        # integration tests (requires DBs)

# Lint & typecheck
make lint                    # ruff
make typecheck               # mypy

# Frontend
make frontend-install        # bun install
make frontend-dev            # dev server
make frontend-build          # production build
```

---

## Configuration Reference

### SemanRAG Constructor

```python
SemanRAG(
    working_dir="./data",           # Storage directory
    workspace="default",            # Workspace isolation namespace
    chunk_token_size=1200,          # Tokens per chunk
    chunk_overlap_token_size=100,   # Overlap between chunks
    chunking_strategy="token",      # "token" | "semantic" | "structure"
    entity_extract_max_gleaning=1,  # Extra extraction passes
    confidence_threshold=0.3,       # Min confidence to keep entities
    enable_entity_resolution=True,  # Deduplicate similar entities
    enable_community_detection=True,# Leiden community clustering
    community_levels=3,             # Hierarchy depth
    embedding_batch_num=32,         # Batch size for embeddings
    llm_model_max_async=4,          # Max concurrent LLM calls
    safety_config={                 # Safety settings
        "pii_policy": "flag",       # "flag" | "mask" | "redact" | "reject"
        "prompt_injection_action": "flag",
    },
)
```

### QueryParam

```python
QueryParam(
    mode="hybrid",              # Retrieval mode
    top_k=20,                   # Entities/relations to retrieve
    chunk_top_k=5,              # Chunks per entity
    enable_rerank=True,         # Cross-encoder reranking
    enable_hybrid_lexical=True, # BM25 fusion
    rrf_k=60,                   # RRF constant
    snapshot_at=None,           # Temporal filter (datetime)
    verifier_enabled=True,      # Grounded-check verification
    stream=False,               # Streaming response
)
```

---

## License

Apache 2.0
