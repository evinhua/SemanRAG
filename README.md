# SemanRAG

**Semantic, Graph-Augmented RAG with Production Guardrails**

SemanRAG is a pluggable Retrieval-Augmented Generation framework that combines graph-augmented retrieval, hybrid lexical + dense search, structured LLM extraction, and production-grade safety/observability in a single architecture.

---

## Why SemanRAG

Traditional RAG systems rely solely on vector similarity search over flat document chunks. That breaks down when you need to:

- Answer questions requiring relationships between entities, global themes, or multi-hop reasoning
- Retrieve rare proper nouns and acronyms that dense embeddings compress poorly
- Distinguish local detail queries from broad thematic queries
- Handle facts that change over time
- Enforce per-document access control in multi-tenant deployments
- Detect when the generated answer drifts away from the retrieved context
- Ingest non-text content (tables, figures) without losing structure

Existing graph-augmented systems improve retrieval but leave gaps: brittle delimited-text extraction, no hybrid lexical retrieval, no community-level summaries, no answer verification, no temporal support, and minimal operational guardrails. SemanRAG closes those gaps.

---

## Core Capabilities

### Retrieval
- **Graph-augmented retrieval** over an automatically-constructed knowledge graph (entities + relationships extracted per chunk, deduplicated, and entity-resolved)
- **Hybrid retrieval** — BM25 lexical search fused with dense vector search via Reciprocal Rank Fusion (RRF)
- **Query transformation** — conversation-aware rewriting, multi-hop decomposition, HyDE for sparse queries
- **Six retrieval modes** — `local`, `global`, `hybrid`, `naive`, `mix`, and `community` (Leiden-clustered hierarchical summaries), plus a `bypass` mode for direct LLM calls
- **Cross-encoder reranker** on by default for `naive` / `mix` modes

### Extraction
- **Structured-output extraction** via provider JSON mode / tool calling, enforced by Pydantic schemas; delimited-text prompt as fallback
- **Confidence scores** per entity and relation, used during merging
- Optional **user-provided entity-type schemas** for domain-specific extraction
- **Entity resolution** combining embedding similarity, rapidfuzz edit distance, and LLM adjudication

### Generation
- **Grounded answer verification** — a post-generation verifier scores each claim against retrieved context and flags unsupported claims; optional verifier-triggered retry
- **Reference tracking** with source-chunk provenance

### Data Model
- **Temporal knowledge graph** — `valid_from` / `valid_to` on edges, `snapshot_at` query parameter
- **Multi-modal ingestion** — table extraction (Camelot / docling) to Markdown, figure captioning via vision models, section-aware parent linking
- **Chunking strategies** — token-size, semantic (embedding-drift), structure-aware (Markdown headings, PDF sections, DOCX styles)

### Safety & Compliance
- **Document-level ACLs** (`owner`, `visible_to_groups`, `visible_to_users`) applied at the storage query layer, not post-hoc
- **PII detection** (Presidio) at ingestion with configurable policies: flag, mask, redact, reject
- **Prompt-injection canary** on ingested text (regex + LLM classifier)
- **Output sanitization** for leaked system-prompt fragments

### Operations
- **OpenTelemetry** traces across the async pipeline; **Prometheus** metrics (`/metrics`)
- Optional **Celery / Arq** job queue for durable ingestion
- **`TokenBudget`** enforcement per user / workspace / day
- **Evaluation harness** with checked-in golden sets and a **regression gate in CI**
- **User feedback loop** — thumbs up/down, structured relevance/accuracy/faithfulness ratings
- **Explicit incremental rebuild semantics** for document deletion and re-ingestion

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          WebUI (React 19)                       │
│   Graph Explorer · Chat · Document Manager · Admin Console      │
└─────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                           │
│   Auth (JWT/OIDC) · ACLs · Rate Limit · WebSocket · OpenAPI     │
└─────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────┐
│                         Query Engine                            │
│ Transform → Retrieve (6 modes) → Rerank → Assemble → Generate   │
│                     → Grounded-check Verifier                   │
└─────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────┐
│                      Ingestion Pipeline                         │
│   Parse → Chunk → PII/Injection Scan → Extract → Resolve →      │
│              Upsert KG → Community Detection                    │
└─────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────┐
│                     Pluggable Storage Layer                     │
│   KV · Vector · Graph · Lexical (BM25) · DocStatus              │
└─────────────────────────────────────────────────────────────────┘
```

All LLM providers, embedding models, storage backends, and rerankers are injected at init time — no hard dependency on any single provider.

---

## Tech Stack

- **Python 3.10+** — async-first with `asyncio`, Pydantic v2
- **TypeScript / React 19** — frontend with strict typing
- **FastAPI + uvicorn / gunicorn** — API server
- **PostgreSQL (pgvector + AGE) / Neo4j / MongoDB / Milvus / Qdrant / OpenSearch / Redis / Faiss / NanoVectorDB** — storage backends
- **OpenAI / Azure / Gemini / Anthropic / Bedrock / Ollama / HuggingFace / vLLM / Zhipu / LlamaIndex** — LLM providers
- **Cohere / Jina / Aliyun / sentence-transformers / BGE** — rerankers
- **networkx + graspologic** — default in-memory graph + Leiden community detection
- **rank-bm25 / tantivy / tsvector / OpenSearch** — BM25 lexical index
- **Presidio** — PII detection
- **OpenTelemetry + Prometheus + Grafana** — observability
- **RAGAS + custom golden-set harness** — evaluation with CI regression gates

See [`specs/tech-stack.md`](specs/tech-stack.md) for the full list.

---

## Storage Backends

| Storage type | Supported backends | Default |
|---|---|---|
| KV | JSON files, PostgreSQL, Redis, MongoDB, OpenSearch | JSON files |
| Vector | NanoVectorDB, PostgreSQL (pgvector), Milvus, Qdrant, MongoDB Atlas, OpenSearch, Faiss | NanoVectorDB |
| Graph | NetworkX, Neo4j, PostgreSQL (AGE), Memgraph, OpenSearch, MongoDB | NetworkX |
| DocStatus | JSON files, PostgreSQL, MongoDB, OpenSearch | JSON files |
| Lexical (BM25) | `rank-bm25` (in-memory), `tantivy`, PostgreSQL `tsvector`, OpenSearch | `rank-bm25` |

15+ implementations across 5 storage types with workspace isolation and document-level ACLs.

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

## Project Status

SemanRAG is a specification-driven project. Implementation is organized into 11 phases — see [`specs/roadmap.md`](specs/roadmap.md):

1. Core data model & storage abstractions
2. Document ingestion pipeline
3. Query engine
4. LLM & embedding provider integration
5. Production storage backends
6. API server
7. Knowledge graph management
8. Evaluation & quality gates
9. Operational tooling & deployment
10. WebUI (React 19, graph-centric)
11. Performance & scale

---

## Deployment Targets

- **Docker / Docker Compose** — multi-service orchestration
- **Kubernetes** — Helm chart in `k8s-deploy/helm/`, raw manifests in `k8s-deploy/manifests/`
- **AWS Lambda** — `deploy/lambda_function.py` + API Gateway with S3 KG storage
- **Systemd** — `semanrag.service.example`
- **Interactive wizard** — `scripts/setup/setup.sh`

---

## Success Criteria

- Outperforms NaiveRAG, RQ-RAG, HyDE, and GraphRAG on comprehensiveness, diversity, empowerment, and **faithfulness** across Agriculture, CS, Legal, Finance, and Mixed domains
- **Regression-gated CI** — no merge drops context-precision or faithfulness by more than the configured threshold (default 2%)
- Sub-second query latency for cached KG lookups
- >95% of generated answers pass the grounded-check verifier on the golden eval set
- Zero undetected PII leakage on the ingestion benchmark
- <1% orphan-entity rate after document deletion

---

## Target Users

- Developers building RAG applications who need better retrieval quality than naive vector search
- Researchers studying knowledge-graph construction, community detection, and graph-augmented generation
- Enterprise teams deploying multi-tenant RAG with compliance, cost, and access-control requirements
- Regulated-industry teams (legal, finance, healthcare) that need auditability, PII handling, and time-aware knowledge

---

## Repository Layout

```
SemanRAG/
├── specs/
│   ├── mission.md       # Problem, solution, value proposition, success metrics
│   ├── tech-stack.md    # Full library and backend matrix
│   └── roadmap.md       # 11-phase build order
├── prompts.md           # Implementation prompts
└── README.md
```

---

## License

Apache 2.0
