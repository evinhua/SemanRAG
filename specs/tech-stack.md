# Tech Stack

## Language & Runtime

- **Python 3.10+** — async-first with `asyncio`, type hints, dataclasses, Pydantic v2
- **TypeScript** — React 19 frontend with strict typing

## Build & Package Management

- **setuptools** — Python build backend (`pyproject.toml`)
- **uv** — recommended Python package manager (fast, reliable)
- **Bun** — JavaScript runtime and package manager for the WebUI
- **Vite** — frontend build tool and dev server
- **Makefile** — canonical entry point for dev setup and wizard targets

## Core Framework

| Component | Library | Purpose |
|---|---|---|
| Async HTTP | `aiohttp`, `httpx` | Async HTTP client for LLM/embedding API calls |
| Data validation | `pydantic` v2 | Request/response models, configuration, **structured-output extraction schemas** |
| Tokenization | `tiktoken` | Token counting for chunking and budget management |
| Graph library | `networkx`, `graspologic` | Default in-memory graph storage; community detection (Leiden) |
| Vector DB | `nano-vectordb` | Default lightweight local vector storage |
| Lexical search | `rank-bm25`, `tantivy` (optional) | BM25 retrieval for hybrid search |
| JSON repair | `json_repair` | Robust fallback parsing of LLM JSON outputs |
| Retry logic | `tenacity` | Exponential backoff for API calls |
| Environment | `python-dotenv` | `.env` file loading |
| Numerical | `numpy`, `scipy` | Embedding vector operations, RRF fusion |
| Fuzzy matching | `rapidfuzz` | Entity resolution edit-distance scoring |
| Data export | `pandas`, `xlsxwriter` | CSV/Excel export of knowledge-graph data |
| Graph export | `rdflib` (optional) | RDF/Turtle serialization |

## Ingestion & Multi-modal Parsing

| Component | Library | Purpose |
|---|---|---|
| PDF text | `pypdf`, `pdfplumber` | Text extraction |
| Tables | `camelot-py`, `docling` (optional) | Table extraction to Markdown |
| DOCX / PPTX / XLSX | `python-docx`, `python-pptx`, `openpyxl` | Office doc extraction |
| Vision captions | `openai`, `anthropic`, `google-genai` | Figure/image captioning |
| Semantic chunking | `langchain-text-splitters` (optional) or custom | Embedding-drift-based splits |
| OCR (optional) | `pytesseract`, `rapidocr-onnxruntime` | Scanned-doc fallback |

## Query Transformation

| Component | Purpose |
|---|---|
| Query rewriter | Conversation-aware pronoun/entity resolution |
| Query decomposer | Split multi-hop queries into sub-queries |
| HyDE generator | Produce hypothetical answer for embedding |
| Keyword extractor | High-level + low-level keyword JSON extraction |

## API Server

| Component | Library | Purpose |
|---|---|---|
| Web framework | `FastAPI` | REST API with OpenAPI docs |
| ASGI server | `uvicorn` | Development server |
| WSGI server | `gunicorn` | Production multi-worker server |
| Auth | `PyJWT`, `python-jose`, `bcrypt` | JWT tokens, password hashing |
| File handling | `python-multipart`, `aiofiles` | Document upload |
| Rate limiting | `slowapi` | Per-user/workspace request limits |
| WebSocket | FastAPI WebSocket | Live pipeline status + chat streaming |

## LLM Providers

All providers are injected via function parameters — no hard dependency on any single provider. All providers supporting it use **structured output / tool calling** for entity extraction.

| Provider | Library | Structured output | Vision |
|---|---|---|---|
| OpenAI / compatible | `openai` | JSON mode, tools | ✓ |
| Azure OpenAI | `openai` | JSON mode, tools | ✓ |
| Google Gemini | `google-genai` | response_schema | ✓ |
| Anthropic | `anthropic` | Tool use | ✓ |
| AWS Bedrock | `aioboto3` | Via provider (Claude, Titan) | ✓ (Claude) |
| Ollama | `ollama` | Format=json | — |
| HuggingFace | `transformers` | Outlines / grammar-guided | — |
| LlamaIndex | `llama-index` | Pydantic program | — |
| Zhipu AI | `zhipuai` | — | — |
| vLLM | OpenAI-compatible | Outlines / guided | — |

## Rerankers

| Provider | Library |
|---|---|
| Cohere Rerank | `cohere` |
| Jina Rerank | `jina` HTTP API |
| Aliyun Rerank | `aliyun-python-sdk-core` |
| Local cross-encoder | `sentence-transformers` |
| BGE reranker | `FlagEmbedding` (optional) |

## Storage Backends

### KV Storage (documents, chunks, LLM cache)

| Backend | Library | Default |
|---|---|---|
| JSON files | built-in | ✓ |
| PostgreSQL | `asyncpg` | |
| Redis | `redis` | |
| MongoDB | `pymongo`, `motor` | |
| OpenSearch | `opensearch-py` | |

### Vector Storage

| Backend | Library | Default |
|---|---|---|
| NanoVectorDB | `nano-vectordb` | ✓ |
| PostgreSQL (pgvector) | `asyncpg`, `pgvector` | |
| Milvus | `pymilvus` | |
| Qdrant | `qdrant-client` | |
| MongoDB Atlas | `pymongo` | |
| OpenSearch | `opensearch-py` | |
| Faiss | `faiss-cpu`/`faiss-gpu` | |

### Graph Storage

| Backend | Library | Default |
|---|---|---|
| NetworkX | `networkx` | ✓ |
| Neo4j | `neo4j` | |
| PostgreSQL (AGE) | `asyncpg` | |
| Memgraph | `neo4j` (Bolt protocol) | |
| OpenSearch | `opensearch-py` | |
| MongoDB | `pymongo` | |

### Document Status Storage

| Backend | Library | Default |
|---|---|---|
| JSON files | built-in | ✓ |
| PostgreSQL | `asyncpg` | |
| MongoDB | `pymongo` | |
| OpenSearch | `opensearch-py` | |

### Lexical Index (BM25)

| Backend | Library |
|---|---|
| In-memory | `rank-bm25` |
| Persistent | `tantivy` (Rust, optional) |
| PostgreSQL `tsvector` | `asyncpg` |
| OpenSearch | `opensearch-py` |

## Job Queue (optional, for large-scale ingestion)

| Library | Notes |
|---|---|
| `celery` | Battle-tested, Redis/RabbitMQ broker |
| `arq` | Async-native, Redis broker |
| In-process | Default — `apipeline_*` coroutines |

## Safety & Compliance

| Component | Library | Purpose |
|---|---|---|
| PII detection | `presidio-analyzer`, `presidio-anonymizer` | Scan + redact/mask at ingestion |
| Prompt-injection detection | Custom regex + LLM classifier | Ingested-text canaries |
| Output sanitization | Custom | Strip leaked system-prompt fragments |
| ACL enforcement | Storage-layer filters | Document-level `visible_to_*` |

## Frontend

| Component | Library | Purpose |
|---|---|---|
| UI framework | React 19 | Functional components with hooks, Suspense, Server Components where applicable |
| Styling | Tailwind CSS | Utility-first CSS |
| Component library | shadcn/ui, Radix UI | Accessible primitives |
| Graph visualization | `sigma.js` + `graphology` | High-performance canvas/WebGL graph rendering (replaces vis-network) |
| Alternate graph view | `cytoscape.js` | Hierarchical and compound layouts |
| Charts | `recharts`, `visx` | Metrics, community-distribution, token-usage plots |
| Timeline | `vis-timeline` | Temporal graph slider, document ingestion history |
| Markdown | `react-markdown`, `remark-gfm`, `rehype-highlight`, `rehype-katex` | Rich answer rendering with code, tables, math |
| Diff viewer | `react-diff-viewer-continued` | Entity/relation edit history |
| Command palette | `cmdk` | Global `⌘K` actions |
| Drag-and-drop | `react-dropzone` | File upload |
| State management | Zustand + Immer | Typed stores (`state.ts`, `graph.ts`, `settings.ts`, `chat.ts`, `eval.ts`) |
| Data fetching | `@tanstack/react-query` | Caching, retries, optimistic updates |
| HTTP client | Custom fetch wrapper | API communication (`semanrag.ts`) |
| WebSocket | native `WebSocket` | Live pipeline status, streaming chat |
| Routing | React Router v6 | SPA navigation with auth guards |
| i18n | `i18next`, `react-i18next` | English, Simplified Chinese, Japanese, Spanish |
| Forms | `react-hook-form` + `zod` | Typed form validation |
| Accessibility | `axe-core` (dev) | WCAG 2.1 AA compliance |
| Keyboard UX | `react-hotkeys-hook` | Shortcuts |
| Build | Vite + Bun | Fast builds and HMR |

## Testing & Quality

| Tool | Purpose |
|---|---|
| `pytest` + `pytest-asyncio` | Test runner with async support |
| `ruff` | Python linter |
| `mypy` | Type checking |
| `pre-commit` | Git hook management |
| `Vitest` | Frontend unit tests |
| `Playwright` | Frontend end-to-end tests |
| RAGAS | RAG evaluation — context precision, faithfulness, answer relevancy |
| Custom golden-set harness | Regression gate in CI |

## Observability

| Tool | Purpose |
|---|---|
| **OpenTelemetry** | Distributed tracing across async pipeline (OTLP exporter) |
| **Prometheus** | Metrics scraping endpoint (`/metrics`) |
| **Grafana** | Dashboard templates shipped in `observability/grafana/` |
| Langfuse | LLM-specific tracing and analytics (optional) |
| `TokenTracker` | Built-in token-usage accounting |
| `TokenBudget` | Per-user / per-workspace / per-day enforcement |
| Python `logging` | Structured JSON logging via `semanrag.utils.logger` with `trace_id` propagation |

## Deployment

| Method | Tools |
|---|---|
| Docker | `Dockerfile`, `docker-compose.yml` |
| Kubernetes | Helm chart in `k8s-deploy/helm/`, raw manifests in `k8s-deploy/manifests/` |
| AWS Lambda | `deploy/lambda_function.py` + API Gateway |
| Systemd | `semanrag.service.example` |
| Interactive setup | `scripts/setup/setup.sh` wizard |
| CI | GitHub Actions workflows with regression-gated eval |
