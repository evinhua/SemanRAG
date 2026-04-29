# Roadmap

> **Status: All 11 phases fully implemented.** See `prompts.md` for the generation prompts used.

This roadmap reconstructs the logical build order for SemanRAG — the sequence of prompts and implementation phases needed to build the project from scratch.

## Phase 1: Core Data Model & Storage Abstractions

1. Define `TextChunkSchema`, `QueryParam`, `QueryResult`, `DocStatus`, `ACLPolicy`, `TemporalEdge` dataclasses in `base.py`
2. Define Pydantic extraction schemas: `ExtractedEntity`, `ExtractedRelation`, `ExtractionResult` with `confidence` fields
3. Define abstract storage interfaces: `BaseKVStorage`, `BaseVectorStorage`, `BaseGraphStorage`, `BaseLexicalStorage` (BM25), `DocStatusStorage`
4. Implement `StorageNameSpace` with workspace isolation plus ACL filter injection hook
5. Implement default local storage backends: `JsonKVStorage`, `NanoVectorDBStorage`, `NetworkXStorage`, `JsonDocStatusStorage`, `InMemoryBM25Storage`

## Phase 2: Document Ingestion Pipeline

1. Implement chunking strategies:
   - `chunking_by_token_size` (default)
   - `chunking_semantic` — embedding-drift-based splits
   - `chunking_structure_aware` — respect Markdown headings, PDF sections, DOCX styles
2. Multi-modal parsing:
   - Table extraction (Camelot/docling) → Markdown tables with preceding-section context
   - Figure captioning via vision models; caption linked to parent doc + page
   - OCR fallback for scanned PDFs
3. Safety pre-checks:
   - PII scan (Presidio) with configurable action (flag/mask/redact/reject)
   - Prompt-injection canary (regex + LLM classifier) flags suspicious ingested text
4. Structured-output entity extraction in `operate.py`:
   - Provider-native JSON mode / tool calling enforced by Pydantic schema
   - Delimited-text prompt retained as fallback for providers without structured output
   - Optional user-provided `entity_type_schema` for domain-specific extraction
   - `confidence` score per entity/relation
5. Entity resolution pass:
   - Candidate blocking via embedding similarity + rapidfuzz edit distance
   - LLM adjudicator for ambiguous matches
   - Merges with `merge_strategy` (concatenate / keep_first / join_unique / confidence_weighted)
6. Knowledge-graph upsert with temporal fields (`valid_from`, `valid_to`)
7. Document-status tracking and pipeline orchestration in `semanrag.py` with durable state
8. Community detection:
   - Leiden clustering via `graspologic`
   - Per-community summaries via LLM (stored for **community retrieval mode**)
9. Pluggable **job queue** backend (in-process default, Celery/Arq optional)

## Phase 3: Query Engine

1. Query-transformation layer:
   - Conversation-aware rewriting (pronoun/entity resolution)
   - Multi-hop decomposition into sub-queries
   - HyDE — generate hypothetical answer, embed it
2. Keyword extraction (high-level + low-level) using LLM prompt with structured output
3. Multi-mode retrieval:
   - **Local**: vector search on entities → expand to connected relationships → gather source chunks
   - **Global**: vector search on relationships → connected entities → source chunks
   - **Hybrid**: local + global fused
   - **Naive**: direct vector search on document chunks
   - **Mix**: KG retrieval + naive chunk retrieval
   - **Community**: query-matched community summaries for broad thematic questions
4. **Hybrid lexical + dense retrieval** — BM25 run in parallel with vector search; merged via Reciprocal Rank Fusion (RRF)
5. **Cross-encoder reranking on by default** for naive/mix modes
6. Context assembly with token budget management (`_apply_token_truncation`)
7. RAG response generation with system prompt, context injection, reference tracking
8. **Grounded answer verification**:
   - Post-generation verifier scores each claim against retrieved context
   - Unsupported claims flagged in response metadata
   - Optional verifier-triggered retry with expanded context
9. Query caching with content-hash keys; **per-user** cache scoping
10. ACL filter applied at storage query time (not post-hoc)
11. Temporal queries: `snapshot_at` parameter filters edges to those active at the given timestamp

## Phase 4: LLM & Embedding Provider Integration

1. Injection pattern: `llm_model_func`, `embedding_func`, `rerank_func`, `verifier_func` as constructor parameters
2. Structured-output capability detection — providers advertise JSON-mode / tool-calling support; fallback otherwise
3. Implement OpenAI provider with caching, JSON mode, tool calling
4. Add providers: Azure OpenAI, Gemini, Anthropic, Bedrock, Ollama, HuggingFace, LlamaIndex, Zhipu, vLLM
5. Vision-capable completion for figure captioning (OpenAI, Anthropic, Gemini, Bedrock/Claude)
6. `EmbeddingFunc` wrapper with `wrap_embedding_func_with_attrs` decorator
7. Reranker support (Cohere, Jina, Aliyun, local cross-encoder, BGE)

## Phase 5: Production Storage Backends

1. PostgreSQL all-in-one: KV + Vector (pgvector) + Graph (Apache AGE) + DocStatus + BM25 (`tsvector`) + RLS-based ACLs
2. Neo4j graph storage with fulltext index, workspace labels, temporal properties
3. MongoDB: KV + Vector (Atlas Search) + Graph + DocStatus
4. Milvus vector storage with configurable index types (HNSW, IVF, DISKANN, etc.)
5. Redis KV and DocStatus storage with persistence configuration
6. Qdrant vector storage with payload-based workspace and ACL partitioning
7. OpenSearch unified storage (KV + Vector + Graph + DocStatus + BM25) with PPL graph traversal
8. Memgraph and Faiss implementations
9. Shared locking system (`shared_storage.py`) for concurrent entity/relation updates
10. **Incremental rebuild semantics** for deletion:
    - Orphan-entity detection and cleanup
    - Re-summarization of descriptions when a source doc is removed
    - Document versioning so re-uploads don't orphan prior entities

## Phase 6: API Server

1. FastAPI server (`semanrag_server.py`) with lifespan management and storage initialization
2. Document routes: upload, scan, insert, delete, pipeline status, **ACL management**
3. Query routes: text query, streaming (SSE + WebSocket), structured data query with references, **answer comparison endpoint** (`/query/compare`)
4. Graph routes: CRUD for entities/relations, knowledge-graph visualization, **community endpoints**, **temporal snapshot** endpoints
5. Feedback routes: thumbs up/down, structured ratings, comments
6. Admin routes: token budget, cost reports, eval runs, PII-scan reports, audit log
7. Ollama-compatible chat API for Open WebUI
8. Authentication: JWT tokens, bcrypt password hashing, API key support, **OIDC/SAML SSO** (optional)
9. Authorization: user/group-based document ACLs enforced at storage layer
10. Rate limiting (`slowapi`) per user/workspace
11. OpenAPI schema + Swagger UI + ReDoc
12. WebUI (React 19 + TypeScript) is built separately — see Phase 10

## Phase 7: Knowledge Graph Management

> **Status: Implemented.** Full KG builder pipeline with entity extraction, resolution, community detection, and CRUD operations.

### KG Construction Pipeline

The automated pipeline runs on every `ainsert()` call:

```
Document → Parse → Chunk → Extract (LLM) → Resolve → Upsert → Communities → Index
```

- **Parsing**: PDF (text + tables + figures), DOCX, PPTX, XLSX, Markdown, plain text
- **Chunking**: Token-based (1200 tokens, 100 overlap), semantic (embedding-drift), structure-aware (headings)
- **Extraction**: LLM structured output → entities (name, type, description, confidence) + relations (source, target, keywords, weight)
- **Resolution**: Embedding similarity blocking → edit distance scoring → LLM adjudication → description merging
- **Upsert**: Nodes and edges with embeddings, source chunk provenance links, temporal edges
- **Communities**: Leiden hierarchical clustering (3 levels) with LLM-generated summaries
- **Indexing**: Entity/relation/chunk embeddings → vector store; all text → BM25

### KG Management Operations

1. Entity/relation CRUD (create, edit, delete) with edit history
2. Entity merging with configurable strategies (concatenate, keep_first, join_unique, confidence_weighted)
3. Document deletion with smart KG cleanup and incremental rebuilding
4. Custom KG insertion API for programmatic ingestion
5. Data export: CSV, Excel, Markdown, Text, **RDF/Turtle**, **GraphML**, **Cypher dump**
6. Scheduled maintenance jobs: orphan scan, staleness scan, entity-resolution sweep, community re-detection

### Inbox Upload (Docker Deployments)

For stable file transfer bypassing HTTP upload timeouts:
- `POST /documents/inbox/upload` — Stream file to volume (fast copy, no ingestion)
- `POST /documents/inbox/scan` — Enqueue inbox files for background ingestion
- `GET /documents/inbox` — List queued files
- Files removed from inbox after successful ingestion

## Phase 8: Evaluation & Quality Gates

1. Checked-in **golden eval sets** (50–200 Q&A per domain: Agriculture, CS, Legal, Finance, Mixed)
2. RAGAS-based metrics: context precision, faithfulness, answer relevancy, context recall
3. **Regression gate** in GitHub Actions: PR blocked if any metric drops >configured threshold (default 2%)
4. Prompt A/B framework — swap `prompt.py` variants, compare across golden sets
5. Grounded-check verifier accuracy reporting
6. Entity-resolution precision/recall benchmark
7. Langfuse observability integration for production trace sampling

## Phase 9: Operational Tooling & Deployment

1. Interactive setup wizard (`scripts/setup/`) for `.env` and Docker Compose generation
2. LLM cache migration and cleanup tools
3. **CLI** beyond wizard: `semanrag query`, `semanrag ingest`, `semanrag graph export`, `semanrag eval run`, `semanrag admin`
4. Docker and Docker Compose deployment with multi-service orchestration
5. Kubernetes: Helm chart + raw manifests + database installation scripts
6. AWS Lambda handler with S3 KG storage
7. **OpenTelemetry** traces; **Prometheus** `/metrics` endpoint; Grafana dashboards
8. `TokenTracker` usage accounting; `TokenBudget` enforcement
9. Feedback-loop storage schema and offline-analysis notebooks
10. Audit log for all mutating operations (who, what, when, from where)

## Phase 10: WebUI — Feature-Rich, Graph-Centric

React 19 + TypeScript application (sigma.js + graphology for graph rendering). Full feature spec in Prompt 11.

1. **Graph Explorer**: WebGL rendering (100k+ nodes), community overlay with drill-down, temporal slider, neighborhood isolation, path finding, property editing with diff viewer, entity merge dialog, export (PNG/SVG/JSON)
2. **Alternate Graph Views**: Cytoscape hierarchical, adjacency matrix, community dendrogram
3. **Document Manager**: drag-and-drop upload, bulk operations, live WebSocket status, document preview with PII/injection reports, version history
4. **Retrieval Testing & Chat**: multi-turn threaded chat, all six retrieval modes plus `bypass`, streaming, query-explain panel, grounded-check badges, inline references, feedback (thumbs + structured ratings), side-by-side answer comparison
5. **Query Builder**: no-code visual filter for entity-type/property/relationship constraints
6. **Admin Console**: user/group/ACL management, token usage + budget dashboards, cache management, pipeline/job-queue status, audit log, eval dashboard with regression alerts, observability deep-links
7. **Global UX**: command palette (⌘K), global search, keyboard shortcuts, responsive layout, WCAG 2.1 AA, onboarding tour
8. **Auth UI**: login, SSO redirect, password reset, MFA (TOTP), session management, API key issuance

## Phase 11: Performance & Scale

Performance requirements are surfaced in Prompt 18 and threaded into the relevant implementation prompts (4, 7, 9, 14). Items:

1. Async-first pipeline with configurable concurrency (`llm_model_max_async`, `embedding_func_max_async`) — Prompt 7 ctor, Prompt 4 `priority_limit_async_func_call`
2. Batch embedding with configurable batch sizes — Prompt 7 ctor
3. Pipeline document processing with enqueue/process pattern (optional Celery/Arq) — Prompt 14 `JobQueueAdapter`
4. Embedding cache with similarity threshold for repeated questions — Prompt 7 `embedding_cache_config`
5. Token budget system for query context assembly — Prompt 4 `TokenBudget`
6. Priority-based LLM request scheduling — Prompt 4 `priority_limit_async_func_call`
7. Graph query result caching with invalidation on mutation — Prompt 18
8. Incremental community detection on graph mutation (avoid full recomputation) — Prompt 7 method #14, Prompt 12 `aincremental_community_update`, Prompt 18
9. Read replicas / connection-pool tuning for PostgreSQL backend — Prompt 9 `postgres_impl.py`, Prompt 18
10. Lazy-loaded WebUI routes, code splitting, service worker caching for static assets — Prompt 11, Prompt 18
