# Prompts

Sequence of prompts to generate the SemanRAG project from scratch using Claude Code. Each prompt builds on the previous output. Reference `specs/mission.md`, `specs/roadmap.md`, and `specs/tech-stack.md` for context.

## Phase → Prompt Mapping

| Roadmap Phase | Prompts |
|---|---|
| Phase 1: Core Data Model & Storage Abstractions | Prompt 2, Prompt 3 |
| Phase 2: Document Ingestion Pipeline | Prompt 6 |
| Phase 3: Query Engine | Prompt 6 (query section) |
| Phase 4: LLM & Embedding Provider Integration | Prompt 8 |
| Phase 5: Production Storage Backends | Prompt 9 |
| Phase 6: API Server | Prompt 10 |
| Phase 7: Knowledge Graph Management | Prompt 12 |
| Phase 8: Evaluation & Quality Gates | Prompt 16 |
| Phase 9: Operational Tooling & Deployment | Prompt 13 |
| Phase 10: WebUI (frontend) | Prompt 11 |
| Phase 11: Performance & Scale | Prompt 18 (plus threads into Prompts 4, 7, 9, 14) |
| Cross-cutting: Project scaffolding | Prompt 1 |
| Cross-cutting: Prompt templates | Prompt 5 |
| Cross-cutting: Utilities | Prompt 4 |
| Cross-cutting: Concurrency & locking | Prompt 14 |
| Cross-cutting: Safety & PII | Prompt 15 |
| Cross-cutting: Test suite | Prompt 17 |

---

## Prompt 1: Project Scaffolding

```
Create a Python project called "semanrag" with the following structure:

- pyproject.toml with setuptools build, Python 3.10+, Apache 2.0 license
- semanrag/ package with __init__.py and _version.py (version "0.1.0")
- Core dependencies: aiohttp, httpx, pydantic>=2, tiktoken, networkx, graspologic, nano-vectordb, rank-bm25, numpy, scipy, rapidfuzz, json_repair, tenacity, python-dotenv, pandas, xlsxwriter, pypinyin, pipmaster, configparser, google-genai, google-api-core, packaging
- Optional dependency groups:
  - api: FastAPI, uvicorn, gunicorn, PyJWT, python-jose, bcrypt, python-multipart, aiofiles, slowapi, pypdf, pdfplumber, python-docx, python-pptx, openpyxl
  - multimodal: camelot-py, docling, pytesseract, rapidocr-onnxruntime
  - offline-storage: redis, neo4j, pymilvus, pymongo, motor, asyncpg, pgvector, qdrant-client, opensearch-py, tantivy
  - offline-llm: openai, anthropic, ollama, zhipuai, aioboto3, google-genai, llama-index, sentence-transformers, FlagEmbedding
  - safety: presidio-analyzer, presidio-anonymizer
  - queue: celery, arq
  - observability: opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp, prometheus-client, langfuse
  - test: pytest, pytest-asyncio, pytest-cov, ruff, mypy, pre-commit
  - evaluation: ragas, datasets
  - export: rdflib
- Entry points: semanrag (unified CLI), semanrag-server, semanrag-gunicorn, semanrag-hash-password, semanrag-download-cache, semanrag-clean-llmqc, semanrag-eval
- tests/ with conftest.py supporting markers: offline, integration, requires_db, requires_api, requires_vision, regression
- Makefile with targets: dev, env-base, env-storage, env-server, env-validate, env-security-check, eval, lint, typecheck
- .gitignore, env.example, config.ini.example, Dockerfile, Dockerfile.lite, docker-compose.yml, docker-compose-full.yml
- observability/ directory with Grafana dashboard JSON and Prometheus config
```

---

## Prompt 2: Storage Abstractions & Data Model

```
In semanrag/base.py, create the core data model and storage abstractions:

1. TextChunkSchema: TypedDict with tokens, content, full_doc_id, chunk_order_index, section_path, page_number, modality (text/table/figure_caption)
2. TemporalEdge: dataclass with valid_from, valid_to, source, confidence — used for time-aware graph edges
3. ACLPolicy: dataclass with owner, visible_to_groups, visible_to_users, public flag
4. StorageNameSpace: base class with global_config, namespace, workspace, ACL filter hook
5. BaseKVStorage(StorageNameSpace): abstract async — get_by_id, get_by_ids, filter_keys, upsert, delete, drop, initialize, finalize, index_done_callback
6. BaseVectorStorage(StorageNameSpace): abstract async — upsert, query, delete, delete_entity, delete_entity_relation, get_by_id, get_by_ids, get_vectors_by_ids, drop, initialize, finalize. Must validate embedding_func. Must accept optional ACL filter.
7. BaseGraphStorage(StorageNameSpace): abstract async — has_node, has_edge, node_degree, edge_degree, get_node, get_edge, get_node_edges, upsert_node, upsert_edge, delete_node, remove_nodes, remove_edges, get_all_labels, get_knowledge_graph, search_labels, get_popular_labels, get_subgraph_at(snapshot_at), detect_communities, get_community_summary, drop, initialize, finalize
8. BaseLexicalStorage(StorageNameSpace): abstract async — upsert, search_bm25, delete, drop, initialize, finalize
9. DocStatusStorage(StorageNameSpace): abstract async — get_status_counts, get_docs_by_status, get_docs_paginated, get_all_status_counts, get_doc_by_file_path. DocStatus dataclass fields: id, content, content_summary, content_length, chunks_count, status, created_at, updated_at, file_path, chunks_list, error_message, pii_findings, prompt_injection_flags, acl_policy, version
10. QueryParam dataclass: mode — one of six retrieval modes (local/global/hybrid/naive/mix/community) or `bypass` for direct LLM calls — plus only_need_context, only_need_prompt, response_type, stream, top_k, chunk_top_k, max_entity_tokens, max_relation_tokens, max_total_tokens, conversation_history, model_func, user_prompt, enable_rerank (default True for naive/mix), enable_hybrid_lexical (default True), rrf_k, snapshot_at, user_id, user_groups, verifier_enabled (default True)
11. QueryResult dataclass: content, raw_data, response_iterator, is_streaming, references, grounded_check (list of per-claim support scores), communities_used, latency_ms, tokens_used
12. Pydantic extraction schemas: ExtractedEntity (name, type, description, confidence), ExtractedRelation (source, target, keywords, description, confidence, valid_from, valid_to), ExtractionResult (entities, relations)
```

---

## Prompt 3: Default Local Storage Implementations

```
Implement the default local storage backends:

1. semanrag/kg/json_kv_impl.py — JsonKVStorage: file-based JSON with workspace subdirectories, lazy loading, atomic writes, legacy cache migration
2. semanrag/kg/nano_vector_db_impl.py — NanoVectorDBStorage: wraps nano-vectordb, cosine similarity threshold filtering, workspace subdirectories, ACL metadata filtering
3. semanrag/kg/networkx_impl.py — NetworkXStorage: NetworkX graph with JSON serialization, workspace subdirectories, temporal edge properties, snapshot extraction, BFS subgraph, community detection via graspologic Leiden
4. semanrag/kg/json_doc_status_impl.py — JsonDocStatusStorage: JSON file per workspace, status filtering, pagination, ACL-aware listing
5. semanrag/kg/inmemory_bm25_impl.py — InMemoryBM25Storage: rank-bm25 backed, per-workspace index, incremental updates, JSON snapshot for restart recovery. Default BM25 backend for local dev and small workspaces; tantivy (Prompt 9 #10) is the persistent single-node option for larger corpora.

All implementations must:
- Inherit from the appropriate base class in base.py
- Support workspace-based data isolation
- Implement initialize/finalize lifecycle
- Apply ACL filters when provided
- Handle concurrent access safely
```

---

## Prompt 4: Utility Functions

```
Create semanrag/utils.py with these core utilities:

1. compute_mdhash_id(content, prefix=""): MD5 hash with optional prefix
2. EmbeddingFunc wrapper + wrap_embedding_func_with_attrs decorator
3. TokenizerInterface protocol; TiktokenTokenizer default
4. truncate_list_by_token_size(items, key, max_token_size, tokenizer)
5. pack_user_ass_to_openai_messages(user_msg, assistant_msg)
6. use_llm_func_with_cache(prompt, llm_func, system_prompt, llm_response_cache, cache_type, chunk_id, user_id): call LLM with transparent caching, user-scoped cache keys, returns (result, timestamp)
7. handle_cache / save_to_cache / CacheData with content-hash keys
8. priority_limit_async_func_call: async semaphore with priority queue
9. TokenTracker: context manager tracking token usage
10. TokenBudget: enforcement — per-user, per-workspace, per-day limits; raises BudgetExceededError
11. setup_logger: configure semanrag logger with SafeStreamHandler, JSON formatter, trace_id propagation
12. write_json / load_json with SanitizingJSONEncoder
13. safe_vdb_operation_with_exception: retry wrapper
14. reciprocal_rank_fusion(result_lists, k=60): merge ranked lists by RRF
15. detect_pii(text, policy): returns list of PIIFinding
16. detect_prompt_injection(text): regex + optional LLM classifier, returns risk score
17. sanitize_output(text, forbidden_patterns): strip leaked system-prompt fragments
18. otel_tracer helpers: span decorator for async functions
```

---

## Prompt 5: Prompt Templates

```
Create semanrag/prompt.py containing all LLM prompt templates as a PROMPTS dict:

1. DEFAULT_TUPLE_DELIMITER = "<|#|>"
2. DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"

3. entity_extraction_system_prompt (fallback, for providers without structured output):
   Role as Knowledge Graph Specialist. Instructions for entity and relation extraction with delimiter protocol, confidence scoring, {language}, {entity_types} (optional user-provided schema), {examples}.

4. entity_extraction_user_prompt: task text with {input_text}, {completion_delimiter}

5. entity_continue_extraction_user_prompt: correction/addition pass

6. entity_extraction_examples: 3 few-shot examples covering narrative, finance, sports domains, each with confidence values

7. entity_extraction_structured_instructions: natural-language instructions that accompany the Pydantic schema when using JSON mode / tool calling. Emphasize: entity_name title-case, confidence 0.0-1.0, entity_type from schema or "Other".

8. summarize_entity_descriptions: synthesis prompt with {description_type}, {description_name}, {description_list}, {summary_length}, {language}

9. entity_resolution_adjudicator: given two candidate entities with descriptions and source contexts, decide SAME / DIFFERENT / UNCERTAIN with reasoning

10. community_report: given community members (entities + relations), produce a title, summary, key findings (ranked), rating. Used for community-mode retrieval.

11. rag_response: expert AI assistant. Synthesize from Knowledge Graph Data + Document Chunks + Community Summaries. Track reference_ids. References section. Variables: {response_type}, {user_prompt}, {context_data}

12. naive_rag_response: Document Chunks only

13. kg_query_context: template assembling entities JSON, relations JSON, document chunks JSON, community summaries JSON, reference list

14. naive_query_context: document chunks JSON + reference list

15. keywords_extraction: extract high_level_keywords and low_level_keywords as JSON

16. keywords_extraction_examples: 3 examples

17. query_rewrite: given conversation history and new query, resolve pronouns, expand abbreviations, output rewritten standalone query

18. query_decomposition: split multi-hop query into 2-5 sub-queries; return JSON array, or empty if already atomic

19. hyde_generation: produce a plausible hypothetical answer paragraph to be embedded

20. grounded_check: given (claim, retrieved_context), return score 0-1 for "is this claim supported" and quoted supporting span

21. prompt_injection_classifier: given user-supplied ingested text, classify as safe / suspicious / malicious with reasoning

22. figure_caption: given image + surrounding document context, produce factual caption

23. fail_response: "Sorry, I'm not able to provide an answer to that question.[no-context]"
```

---

## Prompt 6: Document Ingestion Pipeline

```
Create semanrag/operate.py with the document processing and query pipeline.

INGESTION:

1. Chunking strategies:
   - chunking_by_token_size(content, overlap_token_size, max_token_size, tokenizer, split_by_character, split_by_character_only)
   - chunking_semantic(content, embedding_func, drift_threshold, min_size, max_size)
   - chunking_structure_aware(content, modality, section_headers): respect Markdown h1-h6, PDF sections, DOCX styles. Attach section_path to each chunk.

2. Multi-modal parsing:
   - parse_pdf(path, extract_tables=True, extract_figures=True): pypdf/pdfplumber for text, camelot for tables, vision-model captions for figures
   - parse_docx / parse_pptx / parse_xlsx
   - All return list of (TextChunkSchema, modality) with consistent chunk ordering

3. Safety pre-checks:
   - pii_scan(chunks, policy): Presidio-based; policy = flag|mask|redact|reject
   - prompt_injection_scan(chunks): flags and optionally rejects

4. extract_entities(chunks, global_config, pipeline_status, llm_response_cache):
   For each chunk:
   - If provider supports structured output: call with ExtractionResult Pydantic schema
   - Else: format entity_extraction_system_prompt + entity_extraction_user_prompt with delimiters
   - Use use_llm_func_with_cache (cache_type="extract")
   - Run gleaning pass if entity_extract_max_gleaning > 0 (token-limit guarded)
   - Validate and normalize; record confidence
   - Return merged (nodes, edges)

5. _process_extraction_result: parse LLM output (structured or delimited) into nodes dict and edges list. Validate confidence in [0,1]. Normalize entity names via _truncate_entity_identifier.

6. resolve_entities(candidate_entities, knowledge_graph, entities_vdb):
   - Block candidates by embedding similarity + rapidfuzz edit distance
   - For ambiguous pairs, call entity_resolution_adjudicator LLM
   - Return groups of entities to merge

7. _merge_nodes_then_upsert: merge descriptions, entity_types, source IDs, file paths, confidences. Call _handle_entity_relation_summary when multiple descriptions. Upsert to graph + vector + lexical BM25.

8. _merge_edges_then_upsert: similar for relationships. Preserve temporal fields.

9. _handle_entity_relation_summary / _summarize_descriptions: summarize_entity_descriptions with JSONL list, priority=8, cache_type="summary"

10. build_communities(knowledge_graph):
    - Run Leiden clustering via graspologic
    - For each community: gather members, format community_report prompt, cache summary
    - Store hierarchy (level-0, level-1, ...) for drill-down

QUERY:

11. rewrite_query(query, conversation_history): apply query_rewrite prompt

12. maybe_decompose(query): apply query_decomposition; return list of sub-queries

13. maybe_hyde(query): if mode in {naive, mix} and query is short/sparse, generate HyDE passage for embedding

14. get_keywords_from_query: keywords_extraction with structured output; cache by content hash

15. kg_query: full knowledge-graph flow:
    - rewrite + decompose
    - for each sub-query: extract keywords, _build_query_context (vector + BM25 fusion, graph traversal, chunk gathering, community matching, reranking)
    - aggregate contexts
    - format rag_response
    - stream or return
    - run grounded_check on output
    - return QueryResult with references, grounded_check, communities_used

16. naive_query: hybrid vector + BM25 (RRF) search on chunks_vdb + chunks_bm25, optional HyDE, rerank, naive_rag_response, grounded_check

17. community_query: match query keywords against community summaries; assemble top-N community reports as context; generate answer

18. _build_query_context: orchestrate across modes; apply RRF for hybrid lexical+dense; rerank; assemble with kg_query_context or naive_query_context; apply _apply_token_truncation

19. apply_acl_filter(results, user_id, user_groups): filter retrieved chunks/entities by ACL

20. apply_temporal_filter(results, snapshot_at): keep only edges valid at timestamp
```

---

## Prompt 7: SemanRAG Orchestrator

```
Create semanrag/semanrag.py with the main SemanRAG class (dataclass):

Constructor parameters:
- working_dir, workspace, kv_storage, vector_storage, graph_storage, lexical_storage, doc_status_storage
- chunk_token_size (1200), chunk_overlap_token_size (100), chunking_strategy ("token"|"semantic"|"structure"), tokenizer, tiktoken_model_name
- entity_extract_max_gleaning (1), entity_type_schema (optional), confidence_threshold (0.3)
- enable_entity_resolution (True), resolution_similarity_threshold (0.88)
- enable_community_detection (True), community_levels (3)
- embedding_func, embedding_batch_num (32), embedding_func_max_async (16)
- llm_model_func, llm_model_name, llm_model_max_async (4), llm_model_kwargs
- rerank_func (optional; default cross-encoder for naive/mix)
- verifier_func (optional; defaults to llm_model_func)
- vision_model_func (optional; for figure captions)
- vector_db_storage_cls_kwargs, enable_llm_cache (True), enable_llm_cache_for_entity_extract (True)
- addon_params (language, entity_types), embedding_cache_config
- safety_config: pii_policy, prompt_injection_action, output_sanitization_patterns
- token_budget (optional TokenBudget instance)
- job_queue ("inprocess"|"celery"|"arq")
- tracer (OpenTelemetry tracer, optional)

Core methods:
1. initialize_storages(): instantiate 5 storage types, run migrations, initialize community cache
2. finalize_storages(): clean shutdown
3. insert(content, ids, file_paths, acl_policy): sync wrapper
4. ainsert(): full pipeline — parse (multi-modal) → chunk → safety scan → extract → resolve → upsert → BM25 index → community detect (incremental) → status track
5. query / aquery: route to kg_query / naive_query / community_query by param.mode; apply rewrite, decompose, HyDE, ACL, temporal filters; run grounded_check
6. aquery_data(): structured QueryResult
7. aquery_stream(): async generator with per-token streaming and trailing metadata
8. delete_by_doc_id / delete_by_entity / delete_by_relation: incremental rebuild with orphan cleanup and re-summarization
9. create_entity / edit_entity / create_relation / edit_relation: CRUD with edit history
10. merge_entities(source_entities, target_entity, merge_strategy, target_entity_data)
11. export_data(path, file_format, include_vector_data): CSV, Excel, MD, TXT, RDF/Turtle, GraphML, Cypher
12. get_knowledge_graph(snapshot_at=None, community_level=None): visualization payload
13. get_community_summary(community_id)
14. apipeline_enqueue_documents / apipeline_process_enqueue_documents: queue-backed when configured
15. clear_cache(scope="all"|"query"|"keywords"|"extract"|"summary")
16. run_maintenance(tasks=["orphan_scan", "staleness_scan", "resolution_sweep", "community_redetect"])

Storage class resolution: _get_storage_class() maps string names to implementation classes from semanrag/kg/
```

---

## Prompt 8: LLM Provider Implementations

```
Create semanrag/llm/ package with provider implementations. Each provider must support:
- Async completion: async def func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, response_schema=None, **kwargs) -> str
- Streaming via stream=True returning AsyncIterator
- Structured output when response_schema provided (JSON mode / tool calling / grammar-guided)
- Vision input when messages include image parts (where supported)
- Caching integration via use_llm_func_with_cache
- Capability flags: supports_json_mode, supports_tools, supports_vision

Implementations:
1. openai.py: openai_complete_if_cache, gpt_4o_complete, gpt_4o_mini_complete, openai_embed, azure_openai_complete_if_cache, azure_openai_embed. AsyncOpenAI with configurable base_url and api_key.
2. ollama.py: ollama_model_complete, ollama_embed. Support num_ctx and format=json.
3. gemini.py: gemini_complete_if_cache, gemini_model_complete, gemini_embed. Uses response_schema.
4. bedrock.py: bedrock_complete_if_cache, bedrock_complete, bedrock_embed. aioboto3 Bedrock runtime.
5. anthropic.py: anthropic_complete_if_cache, anthropic_complete, anthropic_embed. Tool-use for structured output. Vision-capable.
6. hf.py: hf_model_complete, hf_embed. transformers AutoModel/AutoTokenizer. Optional Outlines grammar-guided output.
7. llama_index_impl.py: llama_index_complete_if_cache, llama_index_embed. Pydantic program for structured output.
8. zhipu.py: zhipu_complete_if_cache, zhipu_embedding.
9. vllm.py: OpenAI-compatible. Optional guided_json.

Rerankers (semanrag/rerank/):
10. cohere.py, jina.py, ali.py, local_cross_encoder.py (sentence-transformers CrossEncoder), bge_rerank.py. Each with chunk_documents_for_rerank and aggregate_chunk_scores.

Verifier utilities (semanrag/verify/):
11. grounded_check.py: per-claim support scoring given (answer, contexts); optional retry orchestration.
```

---

## Prompt 9: Production Storage Backends

```
Implement production storage backends in semanrag/kg/:

1. postgres_impl.py: PostgreSQLDB connection manager (asyncpg pool, retry, SSL). Implement:
   - PGKVStorage: JSONB with workspace column, batch upsert, flattened LLM cache keys
   - PGVectorStorage: pgvector extension, configurable dimensions, halfvec, HNSW indexing, ACL metadata filter
   - PGGraphStorage: Apache AGE, Cypher queries, BFS subgraph, temporal edge properties
   - PGDocStatusStorage: full document lifecycle, PII findings, ACL storage
   - PGLexicalStorage: tsvector + GIN index for BM25-like lexical search
   - Row-level security policies for ACL enforcement
   - Migration system (alembic-style)

2. neo4j_impl.py: Neo4JStorage — async driver, workspace labels, fulltext index (Chinese-capable), Cypher graph ops, temporal relationships

3. mongo_impl.py: MongoKVStorage, MongoVectorDBStorage (Atlas Search), MongoGraphStorage, MongoDocStatusStorage. Uses `motor` as the async client (not blocking `pymongo`). Workspace via collection prefix. Use `motor` for async MongoDB operations wrapping `pymongo`.

4. milvus_impl.py: MilvusVectorDBStorage with MilvusIndexConfig (AUTOINDEX, HNSW, HNSW_SQ, IVF_FLAT, DISKANN, SCANN). Version validation. Three config paths: env, kwargs, config.ini. Scalar filtering for ACL.

5. redis_impl.py: RedisKVStorage, RedisDocStatusStorage with connection pooling, persistence configuration

6. qdrant_impl.py: QdrantVectorDBStorage — payload-based workspace + ACL partitioning, batch upsert with size estimation

7. opensearch_impl.py: all 5 storage types (KV, Vector, Graph, DocStatus, BM25 via standard analyzer). PPL graphlookup for server-side BFS with client-side fallback.

8. memgraph_impl.py: MemgraphStorage (Neo4j Bolt-compatible)

9. faiss_impl.py: FaissVectorDBStorage (faiss-cpu/faiss-gpu). Separate ACL metadata sidecar for filtering.

10. tantivy_lexical_impl.py: persistent BM25 via tantivy for single-node deployments

Also create:
11. shared_storage.py: KeyedUnifiedLock for concurrent entity/relation updates, NamespaceLock, pipeline status management, MutableBoolean update flags
```

---

## Prompt 10: FastAPI Server & Routes

```
Create semanrag/api/ package:

1. semanrag_server.py: FastAPI application with:
   - Lifespan manager: initialize SemanRAG instance, configure LLM/embedding/rerank/verifier/vision from .env
   - LLM factory functions for OpenAI, Azure, Gemini, Bedrock, Anthropic, Ollama, HuggingFace, Zhipu, vLLM with caching
   - Embedding function factory with batching
   - Reranker and vision-model injection
   - Static file serving for WebUI
   - CORS configuration
   - Auth middleware (JWT + API key + optional OIDC/SAML)
   - Rate limiting (slowapi) per user/workspace
   - OpenTelemetry middleware; /metrics Prometheus endpoint
   - Global exception handler with structured error responses

2. routers/document_routes.py: DocumentManager class with:
   - POST /documents/text, /documents/texts — insert text with acl_policy
   - POST /documents/upload — file upload (DOCX/PPTX/XLSX/PDF/images) with multi-modal extraction
   - POST /documents/scan — scan input directory
   - GET /documents — paginated, status-filtered, ACL-filtered list
   - GET /documents/{doc_id} — detail with extracted chunks, figures, PII findings
   - DELETE /documents/{doc_id} — delete with KG cleanup
   - GET /documents/pipeline-status — live status (also via WebSocket)
   - POST /documents/pipeline-cancel — cancel running pipeline
   - PUT /documents/{doc_id}/acl — update ACL
   - POST /documents/{doc_id}/reingest — re-ingest with version bump

3. routers/query_routes.py:
   - POST /query — non-streaming
   - POST /query/stream — SSE streaming
   - WS /query/ws — WebSocket streaming with live metadata
   - POST /query/data — structured with references + grounded_check
   - POST /query/explain — returns retrieval breakdown without generation
   - POST /query/compare — run the same query under two QueryParam variants (e.g., different modes or prompt versions) and return both QueryResults side-by-side with a diff summary

4. routers/graph_routes.py:
   - GET /graph — KG visualization data (supports snapshot_at, community_level)
   - GET /graph/labels, /graph/labels/popular
   - POST /graph/entities, /graph/relations — create
   - PUT /graph/entities/{name}, /graph/relations — update
   - DELETE /graph/entities/{name}, /graph/relations
   - POST /graph/entities/merge — merge with preview endpoint
   - GET /graph/communities — list with summaries; GET /graph/communities/{id}
   - GET /graph/path — shortest path between two nodes
   - GET /graph/neighborhood/{name}?hops=n

5. routers/feedback_routes.py:
   - POST /feedback — thumbs, rating, comment, linked query_id
   - GET /feedback — admin list

6. routers/admin_routes.py:
   - GET /admin/users, /admin/groups — management
   - POST /admin/budget — configure TokenBudget
   - GET /admin/cost-report — per-user/workspace breakdown
   - GET /admin/audit-log
   - POST /admin/eval/run — trigger eval run
   - GET /admin/eval/history
   - GET /admin/pii-report
   - POST /admin/cache/purge

7. routers/ollama_api.py: Ollama-compatible chat API for Open WebUI

8. config.py: Configuration from .env with DefaultRAGStorageConfig, SafetyConfig, ObservabilityConfig
9. auth.py: AuthHandler with JWT, bcrypt, OIDC/SAML adapters
10. passwords.py: hashing utilities
11. acl.py: ACL enforcement helpers, group resolution
12. telemetry.py: OTel setup, Prometheus metrics registration
```

---

## Prompt 11: React WebUI — Graph-Centric & Feature-Rich

```
Create semanrag_webui/ as a React 19 + TypeScript application.

Build tools: Bun + Vite, Tailwind CSS, shadcn/ui + Radix UI
State: Zustand + Immer stores (state.ts, graph.ts, settings.ts, chat.ts, eval.ts, admin.ts)
Data fetching: @tanstack/react-query
Forms: react-hook-form + zod
Routing: React Router v6 with auth guards
i18n: i18next (EN/ZH/JA/ES)
Accessibility: WCAG 2.1 AA; axe-core in dev

Pages/Features:

1. GraphExplorer — sigma.js + graphology:
   - WebGL rendering; scales to 100k+ nodes
   - Layouts: force-directed (fa2), hierarchical, circular, cluster-preserving
   - Community overlay — color per community, summary side panel, drill-down through Leiden levels
   - TemporalSlider — scrub timeline; edges fade per valid_from/valid_to; ingestion-history overlay (vis-timeline)
   - Minimap, infinite zoom, fisheye lens
   - NeighborhoodIsolation — n-hop subgraph extraction; PathFinding between two nodes
   - PropertyEditDialog with EditablePropertyRow, validation, edit history, undo/redo
   - DiffViewer (react-diff-viewer-continued) for entity/relation changes
   - MergeDialog with side-by-side preview and conflict resolution
   - GraphLabels autocomplete search, SavedSearches, shareable permalinks
   - Legend with color-scheme editor; PNG/SVG/JSON export
   - Full-screen and split-pane modes (graph + chat)

2. AlternateGraphViews:
   - CytoscapeHierarchicalView — entity-type taxonomy
   - AdjacencyMatrixView — dense-neighborhood inspection
   - CommunityDendrogram — Leiden hierarchy tree

3. DocumentManager:
   - FileUploader with drag-and-drop (react-dropzone), per-file progress, MIME validation, size quotas
   - BulkOperations (select, delete, reassign ACL, retry failed)
   - DocumentTable — server-side pagination, column sort/filter, saved filter presets
   - StatusIndicators with live WebSocket updates; pipeline-stage breakdown
   - DocumentPreview panel: text, tables, figures, linked entities, PII report, prompt-injection flags
   - VersionHistory with reingest

4. RetrievalTesting & Chat:
   - Threaded chat (multi-turn history) with markdown + code + tables + math (react-markdown + remark-gfm + rehype-highlight + rehype-katex)
   - Mode selector (local/global/hybrid/naive/mix/community) with tooltips
   - QuerySettingsPanel: top_k, chunk_top_k, reranker toggle, token-budget cap, snapshot_at, ACL scope
   - Streaming with token-by-token rendering
   - QueryExplainPanel — chunks, entities, relations, communities used; inline highlights in context
   - GroundedCheckBadge per claim (supported/partial/unsupported); hover for source
   - Inline references with preview tooltip, click-to-jump to chunk or graph node
   - Feedback — thumbs + structured rating (relevance, accuracy, faithfulness) + comment
   - RegenerateWithMode, CompareTwoAnswersSideBySide
   - ExportChat to Markdown / JSON

5. QueryBuilder (no-code):
   - Visual filter for entity-type, property, relationship constraints
   - Save/load named queries

6. AdminConsole:
   - Users & Groups with ACL role templates
   - TokenUsageDashboard — recharts; per user/workspace/model; cost breakdown
   - TokenBudgetConfig
   - CacheManagement — inspect, purge by type
   - PipelineRuns & JobQueue status, failed-task retry
   - AuditLogViewer with filters, export
   - EvalDashboard — latest vs baseline, per-domain drilldown, regression alerts
   - ObservabilityLinks — Grafana, Langfuse deep-links
   - PIIReportViewer

7. Settings:
   - Theme (light/dark/system), density, color-blind palettes
   - Language (EN/ZH/JA/ES)
   - Keybinding customization
   - Feature flags (experimental modes)

8. Global UX:
   - CommandPalette (cmdk, ⌘K) — jump-to, run query, open doc, admin
   - GlobalSearch with scoped results (docs/entities/relations/chats)
   - Keyboard shortcuts, discoverable via `?`
   - Responsive layout to tablet; mobile-aware chat
   - Toast queue; persistent IncidentBanner for degraded backends
   - OnboardingTour and in-app HelpDrawer

9. Auth:
   - LoginPage with SSO redirect
   - PasswordReset, MFA (TOTP)
   - SessionManagement, revoke sessions
   - ApiKeyIssuance

API client: semanrag.ts with typed request/response interfaces; ApiError normalization; auto token refresh
WebSocket client: ws.ts with reconnect/backoff
```

---

## Prompt 12: Knowledge Graph Management Operations

```
Create semanrag/utils_graph.py with graph-manipulation operations. All operations must:
- Maintain consistency between graph, vector, and BM25 storage
- Record an edit-history entry (who, when, before, after)
- Respect ACL policy on the owning document(s)

1. aedit_entity(entity_name, data, knowledge_graph, entities_vdb, lexical_storage): update attributes, support renaming with relationship migration. Validate non-empty descriptions.

2. aedit_relation(src, tgt, data, knowledge_graph, relationships_vdb): update attributes. Validate non-empty descriptions.

3. acreate_entity(entity_name, data, knowledge_graph, entities_vdb, lexical_storage): generate embedding + BM25 index entry.

4. acreate_relation(src, tgt, data, knowledge_graph, relationships_vdb, entities_vdb): validate both entities exist; optional temporal fields.

5. adelete_by_entity(entity_name, knowledge_graph, entities_vdb, relationships_vdb, lexical_storage): remove entity and connected relationships. Orphan-scan affected neighbors.

6. adelete_by_relation(src, tgt, knowledge_graph, relationships_vdb)

7. amerge_entities(source_entities, target_entity, knowledge_graph, entities_vdb, relationships_vdb, merge_strategy, target_entity_data): redirect all relationships, prevent self-loops, merge attributes per strategy (concatenate/keep_first/join_unique/confidence_weighted).

8. get_entity_info / get_relation_info / get_entity_edit_history / get_relation_edit_history

9. afind_path(src, tgt, knowledge_graph, max_hops, snapshot_at): shortest path with optional temporal filter

10. aneighborhood(entity_name, knowledge_graph, hops, snapshot_at): subgraph extraction

11. arun_entity_resolution(knowledge_graph, entities_vdb, threshold): sweep for near-duplicates, surface merge suggestions

12. abuild_communities / aincremental_community_update
```

---

## Prompt 13: Operational Tooling & Deployment

```
Create the operational infrastructure:

1. scripts/setup/setup.sh (3000+ lines): interactive setup wizard with staged flows:
   - env-base: LLM provider, embedding model, reranker, verifier, vision model
   - env-storage: DB backends (KV, vector, graph, BM25, doc-status)
   - env-server: port, auth (incl. OIDC/SAML), SSL, rate-limits, ACL mode
   - env-safety: PII policy, prompt-injection action, output sanitization
   - env-queue: in-process / Celery / Arq
   - env-observability: OTel endpoint, Prometheus scrape, Langfuse
   - Docker Compose generation from templates in scripts/setup/templates/
   - Validation, security audit, backup functions
   Supporting libraries in scripts/setup/lib/: prompts.sh, validation.sh, file_ops.sh, presets.sh

2. semanrag/cli/ unified CLI (entry point `semanrag`):
   - semanrag query "..." [--mode] [--snapshot-at] [--user] 
   - semanrag ingest <path> [--acl] [--workspace]
   - semanrag graph export [--format rdf|graphml|cypher|csv|excel|md]
   - semanrag eval run [--domain] [--baseline]
   - semanrag admin [budget|users|cache|audit]
   - semanrag serve (dev shortcut)

3. semanrag/tools/:
   - migrate_llm_cache.py: migrate cache between storage backends
   - clean_llm_query_cache.py: selective cleanup
   - hash_password.py: CLI for bcrypt hashes
   - download_cache.py: pre-download tiktoken cache
   - check_initialization.py: verify SemanRAG setup
   - scan_orphans.py, scan_stale.py, resolve_entities.py: maintenance commands

4. Deployment files:
   - Dockerfile, Dockerfile.lite: multi-stage builds
   - docker-compose.yml, docker-compose-full.yml: service orchestration (API, DBs, OTel collector, Prometheus, Grafana)
   - k8s-deploy/helm/ Helm chart
   - k8s-deploy/manifests/: raw manifests
   - deploy/lambda_function.py: AWS Lambda with S3 KG storage

5. semanrag/evaluation/:
   - eval_rag_quality.py: RAGAS metrics (context precision, faithfulness, answer relevancy, context recall)
   - golden_sets/: checked-in Q&A per domain (Agriculture, CS, Legal, Finance, Mixed)
   - regression_gate.py: compare run vs. baseline, exit non-zero on regression > threshold
   - ab_prompt.py: prompt A/B harness
   - langfuse_integration.py: trace sampling + attribute attachment

6. observability/:
   - grafana/: dashboard JSON (ingestion, query latency, token usage, verifier pass rate, job queue)
   - prometheus.yml: scrape config
   - otel-collector-config.yaml

7. .github/workflows/:
   - ci.yml: lint, typecheck, unit tests
   - eval-regression.yml: golden-set regression gate (blocks PRs)
   - docker.yml: multi-arch image builds

8. Makefile: dev setup, env wizard targets, test runners, frontend build, eval, observability stack up/down
```

---

## Prompt 14: Concurrency, Locking & Shared State

```
Create semanrag/kg/shared_storage.py for concurrent access management:

1. UnifiedLock: async-compatible lock with threading and asyncio support. Methods: __aenter__/__aexit__, __enter__/__exit__, locked property.

2. KeyedUnifiedLock: lock manager keyed by entity/relation names. Supports:
   - Atomic acquisition of multiple locks (sorted to prevent deadlocks)
   - Rollback on partial failure
   - Expiry-based cleanup of stale locks
   - Debug counters for lock acquisition

3. NamespaceLock: per-namespace data-initialization tracking with try_initialize_namespace and get_namespace_data

4. MutableBoolean: update flags for tracking storage state changes across namespaces

5. Pipeline status management: initialize_pipeline_status, get_pipeline_status_lock; WebSocket-pub for UI

6. Workspace management: get_default_workspace / set_default_workspace for cross-storage coordination

7. JobQueueAdapter: pluggable interface for in-process / Celery / Arq backends; at-least-once delivery, retry policy, dead-letter handling

All locking must be safe for use with asyncio event loops and support the keyed locking pattern used by _locked_process_entity_name and _locked_process_edges in operate.py.
```

---

## Prompt 15: Safety, PII, and Prompt-Injection Defenses

```
Create semanrag/safety/:

1. pii.py:
   - PIIPolicy (flag | mask | redact | reject) per PII category (EMAIL, PHONE, SSN, CREDIT_CARD, IP, PERSON, LOCATION, custom regex)
   - scan(text, policy) using Presidio analyzer
   - apply_policy(text, findings, policy): mask / redact / return flags
   - PIIFinding dataclass stored with DocStatus

2. prompt_injection.py:
   - pattern-based detector (regex library shipped in safety/patterns.yml)
   - LLM-classifier detector using prompt_injection_classifier prompt
   - combined score with configurable action (flag / reject)

3. output_sanitizer.py:
   - strip leaked system-prompt fragments from LLM output
   - configurable forbidden-pattern list
   - apply to all query responses

4. acl.py:
   - ACLPolicy validation
   - Group resolution via AuthHandler
   - authorize(user_id, user_groups, acl_policy) -> bool
   - storage-layer filter injection helpers
```

---

## Prompt 16: Evaluation & Regression Gate

```
Create semanrag/evaluation/:

1. goldens/: one JSONL per domain (agriculture.jsonl, cs.jsonl, legal.jsonl, finance.jsonl, mixed.jsonl). Each record: id, query, expected_answer, ground_truth_contexts, tags

2. runner.py:
   - run_eval(goldens, rag_instance, metrics) -> EvalReport
   - Supports per-mode evaluation
   - Emits JSON + Markdown summary
   - Uploads to artifact store / GitHub Actions artifacts

3. metrics.py:
   - ragas_context_precision, ragas_faithfulness, ragas_answer_relevancy, ragas_context_recall
   - grounded_check_pass_rate (internal metric)
   - entity_resolution_precision, entity_resolution_recall
   - verifier_agreement_with_judges

4. regression_gate.py:
   - Compare EvalReport vs. baseline.json
   - Exit non-zero if any configured metric drops > threshold (default 2%)
   - GitHub Actions step example in .github/workflows/eval-regression.yml

5. ab_prompt.py:
   - Harness to run same golden set with two prompt variants; emits paired comparison report

6. baselines/: versioned baseline.json per domain; CI updates via approved workflow
```

---

## Prompt 17: Test Suite

```
Create the test suite in tests/:

Core tests:
- test_chunking.py: token / semantic / structure-aware splitting
- test_extract_entities.py: structured-output path, fallback delimiter path, gleaning, token limits
- test_entity_resolution.py: embedding + edit-distance blocking, LLM adjudicator
- test_community_detection.py: Leiden clustering correctness, hierarchy
- test_workspace_isolation.py: multi-workspace concurrency
- test_doc_status_chunk_preservation.py: pipeline failure recovery and retry
- test_batch_embeddings.py: batching and mode-specific keyword handling
- test_rrf_fusion.py: reciprocal rank fusion correctness
- test_description_api_validation.py: entity/relation CRUD validation
- test_write_json_optimization.py: unicode sanitization and JSON encoding
- test_temporal_queries.py: snapshot_at filtering
- test_grounded_check.py: per-claim scoring, retry
- test_query_transformation.py: rewrite, decomposition, HyDE
- test_token_budget.py: enforcement at user/workspace/day granularity

Safety tests:
- test_pii.py: each policy mode; custom regex
- test_prompt_injection.py: regex + LLM classifier
- test_output_sanitizer.py
- test_acl.py: storage-layer filter injection, cross-tenant isolation

Storage tests:
- test_opensearch_storage.py: mock-based for all 5 storage types
- test_postgres_migration.py / test_postgres_upsert.py / test_postgres_rls.py
- test_qdrant_migration.py
- test_milvus_index_config.py
- test_neo4j_fulltext_index.py / test_neo4j_temporal.py
- test_memgraph_storage.py
- test_faiss_meta_inconsistency.py
- test_bm25_impl.py: in-memory + tantivy + opensearch + pg_tsvector

API tests:
- test_auth.py: JWT, bcrypt, guest tokens, OIDC stub
- test_token_auto_renewal.py
- test_aquery_data_endpoint.py: structured response validation
- test_streaming.py: SSE + WebSocket
- test_feedback.py, test_admin.py

Infrastructure tests:
- test_interactive_setup_outputs.py: wizard env/compose generation
- test_runtime_target_validation.py: host vs container detection
- test_otel_instrumentation.py, test_prometheus_metrics.py

Evaluation tests:
- test_regression_gate.py: threshold logic
- test_ab_prompt.py

Frontend tests:
- semanrag_webui/__tests__/ Vitest unit tests
- semanrag_webui/e2e/ Playwright scenarios: upload → ingest → query → explain → feedback; graph exploration; community drill-down; temporal slider; admin flows

Configuration: tests/conftest.py with --run-integration, --keep-artifacts, --stress-test, --test-workers, --run-regression
```

---

## Prompt 18: Performance & Scale

```
Implement performance and scale features across the existing modules. This prompt collects requirements that cut across earlier prompts so they can be verified as a unit.

Async & concurrency (touches Prompt 4, Prompt 7):
1. Confirm semanrag/utils.py priority_limit_async_func_call supports per-priority queues, backpressure, and a max-concurrency cap. Add metrics (queue depth, wait time) exposed via Prometheus.
2. SemanRAG ctor must honor llm_model_max_async, embedding_func_max_async; document defaults and recommended values per provider.
3. Batch embedding: embedding_batch_num with adaptive batching when provider rate-limits.

Caching (touches Prompt 4, Prompt 7):
4. Embedding cache with configurable similarity threshold for query-embedding reuse.
5. Graph query result cache keyed by (query_signature, workspace, snapshot_at). Invalidate on any upsert_node / upsert_edge / delete_* in the affected neighborhood; use dirty-set propagation rather than full flush.
6. Token budget enforcement (TokenBudget) applied at query time and during long-running ingestion.

Incremental community detection (touches Prompt 7, Prompt 12):
7. aincremental_community_update must only re-run Leiden on the affected subgraph, not the whole graph. Use a change-set tracked during upsert operations.
8. Community summary cache invalidated per-community on member change, not globally.

Database tuning (touches Prompt 9):
9. postgres_impl.py: connection-pool sizing guidance, read-replica routing for read-heavy graph queries, statement_timeout per query type.
10. Milvus / Qdrant: batch-size tuning guidance; memory pressure monitoring.
11. Neo4j / Memgraph: Cypher query-plan capture for slow queries.

Job queue (touches Prompt 14):
12. JobQueueAdapter must support priority queues, retry with exponential backoff, dead-letter queue, and at-least-once semantics. Document idempotency contract for consumers.

WebUI performance (touches Prompt 11):
13. Route-level code splitting and lazy imports for GraphExplorer, AdminConsole.
14. Service worker with stale-while-revalidate for static assets.
15. Graph virtualization — sigma.js + graphology viewport culling; only render visible nodes/edges above a configurable threshold.
16. Query result streaming with chunked rendering; avoid full re-render on each token.

Observability requirements (touches Prompt 4, Prompt 10):
17. Prometheus metrics: ingestion throughput, query latency histograms, cache hit rates (per cache type), verifier pass rate, queue depth, token usage.
18. OpenTelemetry spans around: parse, chunk, extract, resolve, upsert, community_detect, rewrite, decompose, HyDE, retrieve (per mode), rerank, generate, verify.

Benchmarks:
19. tests/bench/ micro-benchmarks for chunking, RRF fusion, entity resolution, community detection on 10k / 100k / 1M edge graphs.
20. End-to-end ingestion throughput and query latency regression tests (nightly CI, not per-PR).
```
