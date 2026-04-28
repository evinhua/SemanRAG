# Mission

## Project Name

SemanRAG — Semantic, Graph-Augmented RAG with Production Guardrails

## Problem Statement

Traditional RAG systems rely solely on vector similarity search over flat document chunks. This limits their ability to:

- Answer questions that require understanding relationships between entities, global themes, or multi-hop reasoning across documents
- Retrieve rare proper nouns and acronyms that dense embeddings compress poorly
- Distinguish between local detail queries and broad thematic queries
- Handle temporal knowledge (facts that change over time)
- Enforce per-document access control in multi-tenant deployments
- Detect when the generated answer drifts away from the retrieved context
- Ingest non-text content (tables, figures) without losing structure

Existing graph-augmented systems improve retrieval but leave significant gaps: brittle delimited-text extraction, no hybrid lexical retrieval, no community-level summaries, no answer verification, no temporal support, and minimal operational guardrails (PII, prompt-injection defenses, cost budgets, ACLs, regression-gated evaluation).

## Solution

SemanRAG combines graph-augmented retrieval, hybrid lexical + dense search, structured LLM extraction, and production-grade safety/observability into a single pluggable architecture.

**Retrieval**
- Graph-augmented retrieval over an automatically-constructed knowledge graph (entities + relationships extracted per chunk, deduplicated, and entity-resolved)
- **Hybrid retrieval** — BM25 lexical search fused with dense vector search via reciprocal rank fusion (RRF)
- **Query transformation layer** — conversation-aware rewriting, multi-hop decomposition, HyDE for sparse queries
- **Multi-mode retrieval** — six modes (local, global, hybrid, naive, mix, **community** with Leiden-clustered hierarchical summaries) plus a `bypass` mode for direct LLM calls
- Cross-encoder reranker on by default for naive/mix modes

**Extraction**
- **Structured-output extraction** via provider JSON mode / tool calling, enforced by Pydantic schemas; delimited-text as fallback
- Per-entity and per-relation **confidence scores** used during merging
- Optional user-provided **entity-type schemas** for domain-specific extraction
- **Entity resolution** combining embedding similarity, edit distance (rapidfuzz), and LLM adjudication

**Generation**
- **Grounded answer verification** — post-generation check that scores claim support against retrieved context; flags unsupported claims; optional verifier-triggered retry
- Reference tracking with source chunk provenance

**Data model**
- **Temporal knowledge graph** — `valid_from` / `valid_to` on edges, `snapshot_at` query parameter
- **Multi-modal ingestion** — table extraction (Camelot/docling) to Markdown, figure captioning via vision models, section-aware parent linking
- **Chunking strategies** — token-size, semantic (embedding drift), and structure-aware (Markdown headings, PDF sections, DOCX styles)

**Safety & compliance**
- **Document-level ACLs** — `owner`, `visible_to_groups`, `visible_to_users`; applied at the storage query layer (not post-hoc)
- **PII detection** (Presidio) at ingestion with configurable policies (flag, mask, redact, reject)
- **Prompt-injection canary** on ingested text (regex + LLM classifier)
- Output sanitization for leaked system-prompt fragments

**Operations**
- **OpenTelemetry** traces across the async pipeline; **Prometheus** metrics (retrieval latency, extraction throughput, cache hit rate, queue depth, verification pass rate)
- Optional **Celery/Arq** job queue for durable ingestion
- **`TokenBudget`** enforcement per user / workspace / day
- **Evaluation harness** with checked-in golden sets and **regression gate in CI**
- **User feedback loop** — thumbs up/down, structured relevance/accuracy/faithfulness ratings captured for offline analysis and prompt tuning
- **Explicit incremental rebuild semantics** for document deletion and re-ingestion

## Core Value Proposition

- **Graph + hybrid retrieval** — structural context from the KG, lexical precision from BM25, semantic recall from dense vectors, fused via RRF
- **Multi-mode, multi-modal querying** — match retrieval strategy to question type; ingest text, tables, and figures
- **Pluggable architecture** — any LLM provider, any embedding model, any storage backend, any reranker, injected at init time
- **Production-grade guardrails** — ACLs, PII scanning, prompt-injection defenses, cost budgets, observability, regression-gated evals
- **Verifiable answers** — grounded-check on every response, confidence-weighted extraction, temporal snapshots
- **Multi-tenant ready** — 15+ storage backend implementations across 5 storage types, workspace isolation, document-level access control

## Project Status

SemanRAG is fully implemented across all 11 phases. The system is deployed and operational with:

- Complete Python backend with 78 source files
- React 19 frontend with D3 force-directed graph visualization
- Docker deployment with full-stack orchestration (13 services)
- Azure OpenAI integration (GPT-5.4, text-embedding-3-large)
- 110+ passing unit tests with CI regression gates
- Knowledge graph with 300+ entities extracted from ingested documents

- Developers building RAG applications who need better retrieval quality than naive vector search
- Researchers studying knowledge-graph construction, community detection, and graph-augmented generation
- Enterprise teams deploying multi-tenant RAG with compliance, cost, and access-control requirements
- Regulated-industry teams (legal, finance, healthcare) that need auditability, PII handling, and time-aware knowledge

## Success Metrics

- Outperforms NaiveRAG, RQ-RAG, HyDE, and GraphRAG on comprehensiveness, diversity, empowerment, and **faithfulness** benchmarks across Agriculture, CS, Legal, Finance, and Mixed domains
- **Regression-gated CI**: no merge drops context-precision or faithfulness by more than a configured threshold (default 2%)
- Supports 15+ storage backend implementations across 5 storage types without changes to the core pipeline
- Sub-second query latency for cached knowledge-graph lookups
- >95% of generated answers pass the grounded-check verifier on the golden eval set
- Zero undetected PII leakage on the ingestion benchmark
- <1% orphan-entity rate after document deletion

## License

Apache 2.0
