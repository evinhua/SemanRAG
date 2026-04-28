"""SemanRAG – Main orchestrator class."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from semanrag.base import (
    ACLPolicy,
    BaseGraphStorage,
    BaseKVStorage,
    BaseLexicalStorage,
    BaseVectorStorage,
    DocStatus,
    DocStatusStorage,
    QueryParam,
    QueryResult,
    TextChunkSchema,
)
from semanrag.operate import (
    _merge_edges_then_upsert,
    _merge_nodes_then_upsert,
    build_communities,
    chunking_by_token_size,
    chunking_semantic,
    chunking_structure_aware,
    community_query,
    extract_entities,
    kg_query,
    naive_query,
    parse_docx,
    parse_pdf,
    parse_pptx,
    parse_xlsx,
    pii_scan,
    prompt_injection_scan,
    resolve_entities,
    rewrite_query,
)
from semanrag.utils import (
    EmbeddingFunc,
    TiktokenTokenizer,
    TokenBudget,
    compute_mdhash_id,
    load_json,
    logger,
    write_json,
)


@dataclass
class SemanRAG:
    """Central orchestrator for the SemanRAG pipeline."""

    # --- workspace ---
    working_dir: str = "./semanrag_data"
    workspace: str | None = None

    # --- storage instances (created in initialize if None) ---
    kv_storage: BaseKVStorage | None = None
    vector_storage: BaseVectorStorage | None = None
    graph_storage: BaseGraphStorage | None = None
    lexical_storage: BaseLexicalStorage | None = None
    doc_status_storage: DocStatusStorage | None = None

    # --- chunking config ---
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    chunking_strategy: str = "token"
    tokenizer: Any = None
    tiktoken_model_name: str = "gpt-4o"

    # --- extraction config ---
    entity_extract_max_gleaning: int = 1
    entity_type_schema: dict | None = None
    confidence_threshold: float = 0.3
    enable_entity_resolution: bool = True
    resolution_similarity_threshold: float = 0.88

    # --- community config ---
    enable_community_detection: bool = True
    community_levels: int = 3

    # --- model functions ---
    embedding_func: EmbeddingFunc | None = None
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16
    llm_model_func: Any = None
    llm_model_name: str = "gpt-4o"
    llm_model_max_async: int = 4
    llm_model_kwargs: dict = field(default_factory=dict)
    rerank_func: Any = None
    verifier_func: Any = None
    vision_model_func: Any = None

    # --- storage config ---
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    enable_llm_cache: bool = True
    enable_llm_cache_for_entity_extract: bool = True

    # --- addon params ---
    addon_params: dict = field(
        default_factory=lambda: {"language": "English", "entity_types": []}
    )
    embedding_cache_config: dict | None = None

    # --- safety ---
    safety_config: dict = field(
        default_factory=lambda: {
            "pii_policy": "flag",
            "prompt_injection_action": "flag",
        }
    )

    # --- budget ---
    token_budget: TokenBudget | None = None

    # --- job queue ---
    job_queue: str = "inprocess"

    # --- observability ---
    tracer: Any = None

    # --- internal state (init=False) ---
    _global_config: dict = field(default_factory=dict, init=False, repr=False)
    _initialized: bool = field(default=False, init=False)
    _llm_response_cache: BaseKVStorage | None = field(
        default=None, init=False, repr=False
    )
    _entities_vdb: BaseVectorStorage | None = field(
        default=None, init=False, repr=False
    )
    _relationships_vdb: BaseVectorStorage | None = field(
        default=None, init=False, repr=False
    )
    _chunks_vdb: BaseVectorStorage | None = field(
        default=None, init=False, repr=False
    )
    _chunks_bm25: BaseLexicalStorage | None = field(
        default=None, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # 1. __post_init__
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.tokenizer is None:
            self.tokenizer = TiktokenTokenizer(self.tiktoken_model_name)
        self._global_config = self._build_global_config()

    # ------------------------------------------------------------------
    # 2. _build_global_config
    # ------------------------------------------------------------------
    def _build_global_config(self) -> dict:
        return {
            "working_dir": self.working_dir,
            "llm_model_func": self.llm_model_func,
            "embedding_func": self.embedding_func,
            "rerank_func": self.rerank_func,
            "verifier_func": self.verifier_func,
            "vision_model_func": self.vision_model_func,
            "entity_extract_max_gleaning": self.entity_extract_max_gleaning,
            "entity_type_schema": self.entity_type_schema,
            "confidence_threshold": self.confidence_threshold,
            "enable_entity_resolution": self.enable_entity_resolution,
            "resolution_similarity_threshold": self.resolution_similarity_threshold,
            "chunk_token_size": self.chunk_token_size,
            "chunk_overlap_token_size": self.chunk_overlap_token_size,
            "tiktoken_model_name": self.tiktoken_model_name,
            "addon_params": self.addon_params,
            "enable_llm_cache": self.enable_llm_cache,
            "safety_config": self.safety_config,
            "community_levels": self.community_levels,
            "use_structured_output": True,
            "llm_model_max_async": self.llm_model_max_async,
            "embedding_func_max_async": self.embedding_func_max_async,
        }

    # ------------------------------------------------------------------
    # 3. initialize_storages
    # ------------------------------------------------------------------
    async def initialize_storages(self) -> None:
        from semanrag.kg.inmemory_bm25_impl import InMemoryBM25Storage
        from semanrag.kg.json_doc_status_impl import JsonDocStatusStorage
        from semanrag.kg.json_kv_impl import JsonKVStorage
        from semanrag.kg.nano_vector_db_impl import NanoVectorDBStorage
        from semanrag.kg.networkx_impl import NetworkXStorage

        os.makedirs(self.working_dir, exist_ok=True)
        gc = self._global_config

        if self.kv_storage is None:
            self.kv_storage = JsonKVStorage(gc, "doc_chunks", self.workspace)
        if self.vector_storage is None:
            self.vector_storage = NanoVectorDBStorage(
                gc, "chunks", self.workspace, embedding_func=self.embedding_func
            )
        if self.graph_storage is None:
            self.graph_storage = NetworkXStorage(gc, "knowledge_graph", self.workspace)
        if self.lexical_storage is None:
            self.lexical_storage = InMemoryBM25Storage(gc, "chunks_bm25", self.workspace)
        if self.doc_status_storage is None:
            self.doc_status_storage = JsonDocStatusStorage(gc, "doc_status", self.workspace)

        # Internal storages
        self._llm_response_cache = JsonKVStorage(gc, "llm_response_cache", self.workspace)
        self._entities_vdb = NanoVectorDBStorage(
            gc, "entities", self.workspace, embedding_func=self.embedding_func
        )
        self._relationships_vdb = NanoVectorDBStorage(
            gc, "relationships", self.workspace, embedding_func=self.embedding_func
        )
        self._chunks_vdb = NanoVectorDBStorage(
            gc, "chunks_vdb", self.workspace, embedding_func=self.embedding_func
        )
        self._chunks_bm25 = InMemoryBM25Storage(gc, "entities_bm25", self.workspace)

        # Initialize all
        for store in (
            self.kv_storage,
            self.vector_storage,
            self.graph_storage,
            self.lexical_storage,
            self.doc_status_storage,
            self._llm_response_cache,
            self._entities_vdb,
            self._relationships_vdb,
            self._chunks_vdb,
            self._chunks_bm25,
        ):
            await store.initialize()

        self._initialized = True
        logger.info("SemanRAG storages initialized (working_dir=%s)", self.working_dir)

    # ------------------------------------------------------------------
    # 4. finalize_storages
    # ------------------------------------------------------------------
    async def finalize_storages(self) -> None:
        for store in (
            self.kv_storage,
            self.vector_storage,
            self.graph_storage,
            self.lexical_storage,
            self.doc_status_storage,
            self._llm_response_cache,
            self._entities_vdb,
            self._relationships_vdb,
            self._chunks_vdb,
            self._chunks_bm25,
        ):
            if store is not None:
                await store.finalize()
        self._initialized = False
        logger.info("SemanRAG storages finalized")

    # ------------------------------------------------------------------
    # 5. _get_storage_class
    # ------------------------------------------------------------------
    def _get_storage_class(self, name: str):
        from semanrag.kg.inmemory_bm25_impl import InMemoryBM25Storage
        from semanrag.kg.json_doc_status_impl import JsonDocStatusStorage
        from semanrag.kg.json_kv_impl import JsonKVStorage
        from semanrag.kg.nano_vector_db_impl import NanoVectorDBStorage
        from semanrag.kg.networkx_impl import NetworkXStorage

        mapping = {
            "json_kv": JsonKVStorage,
            "nano_vector_db": NanoVectorDBStorage,
            "networkx": NetworkXStorage,
            "json_doc_status": JsonDocStatusStorage,
            "inmemory_bm25": InMemoryBM25Storage,
        }
        cls = mapping.get(name)
        if cls is None:
            raise ValueError(f"Unknown storage class: {name!r}. Available: {sorted(mapping)}")
        return cls

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "SemanRAG storages not initialized. Call await initialize_storages() first."
            )

    # ------------------------------------------------------------------
    # 6. insert (sync wrapper)
    # ------------------------------------------------------------------
    def insert(
        self,
        content: str | list[str],
        ids: list[str] | None = None,
        file_paths: list[str] | None = None,
        acl_policy: ACLPolicy | None = None,
    ) -> None:
        asyncio.run(self.ainsert(content, ids=ids, file_paths=file_paths, acl_policy=acl_policy))

    # ------------------------------------------------------------------
    # 7. ainsert
    # ------------------------------------------------------------------
    async def ainsert(
        self,
        content: str | list[str],
        ids: list[str] | None = None,
        file_paths: list[str] | None = None,
        acl_policy: ACLPolicy | None = None,
    ) -> None:
        self._ensure_initialized()

        if isinstance(content, str):
            content = [content]
        if ids is None:
            ids = [compute_mdhash_id(c, prefix="doc-") for c in content]
        if file_paths is None:
            file_paths = [""] * len(content)

        for idx, (doc_text, doc_id, fpath) in enumerate(zip(content, ids, file_paths)):
            doc_status = DocStatus(
                id=doc_id,
                content=doc_text[:500],
                content_length=len(doc_text),
                status="processing",
                file_path=fpath,
                acl_policy=acl_policy,
            )
            await self.doc_status_storage.upsert(doc_id, doc_status)

            try:
                # --- a. Chunk ---
                if fpath and fpath.lower().endswith(".pdf"):
                    chunks = await parse_pdf(fpath, global_config=self._global_config)
                elif fpath and fpath.lower().endswith(".docx"):
                    chunks = await parse_docx(fpath)
                elif fpath and fpath.lower().endswith(".pptx"):
                    chunks = await parse_pptx(fpath)
                elif fpath and fpath.lower().endswith(".xlsx"):
                    chunks = await parse_xlsx(fpath)
                elif self.chunking_strategy == "semantic":
                    chunks = chunking_semantic(
                        doc_text,
                        embedding_func=self.embedding_func,
                    )
                elif self.chunking_strategy == "structure":
                    chunks = chunking_structure_aware(doc_text)
                else:
                    chunks = chunking_by_token_size(
                        doc_text,
                        overlap_token_size=self.chunk_overlap_token_size,
                        max_token_size=self.chunk_token_size,
                        tokenizer=self.tokenizer,
                    )

                # Stamp doc id on each chunk
                for c in chunks:
                    c["full_doc_id"] = doc_id

                # --- b. Safety scans ---
                pii_findings: list[dict] = []
                pi_flags: list[dict] = []
                pii_policy = self.safety_config.get("pii_policy", "flag")
                if pii_policy != "off":
                    chunks, pii_findings = pii_scan(chunks, policy=pii_policy)
                pi_action = self.safety_config.get("prompt_injection_action", "flag")
                if pi_action != "off":
                    chunks, pi_flags = prompt_injection_scan(chunks, action=pi_action)

                doc_status.pii_findings = pii_findings
                doc_status.prompt_injection_flags = pi_flags

                if not chunks:
                    doc_status.status = "completed"
                    doc_status.chunks_count = 0
                    doc_status.updated_at = datetime.now(UTC).isoformat()
                    await self.doc_status_storage.upsert(doc_id, doc_status)
                    continue

                # --- c. Extract entities ---
                cache = self._llm_response_cache if self.enable_llm_cache_for_entity_extract else None
                nodes, edges = await extract_entities(
                    chunks, self._global_config, llm_response_cache=cache
                )

                # --- d. Resolve entities ---
                if self.enable_entity_resolution and nodes:
                    merge_groups = await resolve_entities(
                        nodes, self.graph_storage, self._entities_vdb, self._global_config
                    )
                    for group in merge_groups:
                        canonical = group[0]
                        for alias in group[1:]:
                            if alias in nodes:
                                alias_data = nodes.pop(alias)
                                if canonical in nodes:
                                    nodes[canonical]["description"] += " | " + alias_data.get("description", "")
                                    nodes[canonical]["source_id"] += "," + alias_data.get("source_id", "")
                                    nodes[canonical]["confidence"] = max(
                                        nodes[canonical]["confidence"],
                                        alias_data.get("confidence", 0.0),
                                    )
                                else:
                                    nodes[canonical] = alias_data
                            for edge in edges:
                                if edge["src_id"] == alias:
                                    edge["src_id"] = canonical
                                if edge["tgt_id"] == alias:
                                    edge["tgt_id"] = canonical

                # --- e. Merge & upsert nodes and edges ---
                await _merge_nodes_then_upsert(
                    nodes,
                    self.graph_storage,
                    self._entities_vdb,
                    self._chunks_bm25,
                    self._global_config,
                    llm_response_cache=self._llm_response_cache,
                )
                await _merge_edges_then_upsert(
                    edges,
                    self.graph_storage,
                    self._relationships_vdb,
                    self._global_config,
                    llm_response_cache=self._llm_response_cache,
                )

                # --- f. Upsert chunks to VDB and BM25 ---
                chunk_ids: list[str] = []
                chunks_vdb_data: dict[str, dict] = {}
                chunks_bm25_data: dict[str, dict] = {}
                for c in chunks:
                    cid = compute_mdhash_id(c["content"])
                    chunk_ids.append(cid)
                    chunks_vdb_data[cid] = {
                        "content": c["content"],
                        "full_doc_id": doc_id,
                        "chunk_order_index": c["chunk_order_index"],
                    }
                    chunks_bm25_data[cid] = {"content": c["content"], "full_doc_id": doc_id}

                await self._chunks_vdb.upsert(chunks_vdb_data)
                await self._chunks_bm25.upsert(chunks_bm25_data)

                # Also store chunks in KV
                kv_data: dict[str, dict] = {}
                for c in chunks:
                    cid = compute_mdhash_id(c["content"])
                    kv_data[cid] = dict(c)
                await self.kv_storage.upsert(kv_data)

                # --- g. Build communities ---
                if self.enable_community_detection:
                    await build_communities(
                        self.graph_storage,
                        self._global_config,
                        llm_response_cache=self._llm_response_cache,
                    )

                # --- h. Update doc status ---
                doc_status.status = "completed"
                doc_status.chunks_count = len(chunks)
                doc_status.chunks_list = chunk_ids
                doc_status.updated_at = datetime.now(UTC).isoformat()
                await self.doc_status_storage.upsert(doc_id, doc_status)

            except Exception as exc:
                logger.error("ainsert failed for doc %s: %s", doc_id, exc)
                doc_status.status = "failed"
                doc_status.error_message = str(exc)
                doc_status.updated_at = datetime.now(UTC).isoformat()
                await self.doc_status_storage.upsert(doc_id, doc_status)

    # ------------------------------------------------------------------
    # 8. query (sync wrapper)
    # ------------------------------------------------------------------
    def query(self, query: str, param: QueryParam | None = None) -> QueryResult:
        return asyncio.run(self.aquery(query, param=param))

    # ------------------------------------------------------------------
    # 9. aquery
    # ------------------------------------------------------------------
    async def aquery(self, query: str, param: QueryParam | None = None) -> QueryResult:
        self._ensure_initialized()
        if param is None:
            param = QueryParam()
        if param.model_func is None:
            param.model_func = self.llm_model_func

        t0 = time.monotonic()

        if param.mode == "bypass":
            if param.stream:
                async def _stream():
                    async for chunk in self.llm_model_func(query, stream=True):
                        yield chunk
                return QueryResult(
                    response_iterator=_stream(),
                    is_streaming=True,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            answer = await self.llm_model_func(query)
            return QueryResult(
                content=answer,
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        if param.mode == "naive":
            return await naive_query(
                query,
                param,
                self._global_config,
                self._chunks_vdb,
                self._chunks_bm25,
                llm_response_cache=self._llm_response_cache,
            )

        if param.mode == "community":
            return await community_query(
                query,
                param,
                self._global_config,
                self.graph_storage,
                llm_response_cache=self._llm_response_cache,
            )

        # local / global / hybrid / mix → kg_query
        return await kg_query(
            query,
            param,
            self._global_config,
            self.graph_storage,
            self._entities_vdb,
            self._relationships_vdb,
            self._chunks_vdb,
            self._chunks_bm25,
            llm_response_cache=self._llm_response_cache,
        )

    # ------------------------------------------------------------------
    # 10. aquery_data (non-streaming)
    # ------------------------------------------------------------------
    async def aquery_data(self, query: str, param: QueryParam | None = None) -> QueryResult:
        if param is None:
            param = QueryParam()
        param.stream = False
        return await self.aquery(query, param=param)

    # ------------------------------------------------------------------
    # 11. aquery_stream
    # ------------------------------------------------------------------
    async def aquery_stream(self, query: str, param: QueryParam | None = None) -> QueryResult:
        if param is None:
            param = QueryParam()
        param.stream = True
        return await self.aquery(query, param=param)

    # ------------------------------------------------------------------
    # 12. delete_by_doc_id
    # ------------------------------------------------------------------
    async def delete_by_doc_id(self, doc_id: str) -> None:
        self._ensure_initialized()
        doc = await self.doc_status_storage.get(doc_id)
        if doc is None:
            logger.warning("delete_by_doc_id: doc %s not found", doc_id)
            return

        chunk_ids = doc.chunks_list or []

        # Collect entities sourced only from this doc's chunks
        orphan_entities: list[str] = []
        all_labels = await self.graph_storage.get_all_labels()
        for label in all_labels:
            node = await self.graph_storage.get_node(label)
            if node is None:
                continue
            sources = set(node.get("source_id", "").split(","))
            chunk_set = set(chunk_ids)
            remaining = sources - chunk_set
            if not remaining or remaining == {""}:
                orphan_entities.append(label)
            elif sources & chunk_set:
                # Partial overlap: update source_id
                node["source_id"] = ",".join(s for s in remaining if s)
                await self.graph_storage.upsert_node(label, node)

        # Delete orphan entities and their edges
        for ent in orphan_entities:
            edges = await self.graph_storage.get_node_edges(ent)
            await self.graph_storage.remove_edges(edges)
            await self.graph_storage.delete_node(ent)
            await self._entities_vdb.delete_entity(ent)
            await self._relationships_vdb.delete_entity_relation(ent)

        # Delete chunks
        if chunk_ids:
            await self._chunks_vdb.delete(chunk_ids)
            await self._chunks_bm25.delete(chunk_ids)
            await self.kv_storage.delete(chunk_ids)

        await self.doc_status_storage.delete(doc_id)
        logger.info("Deleted doc %s (%d chunks, %d orphan entities)", doc_id, len(chunk_ids), len(orphan_entities))

    # ------------------------------------------------------------------
    # 13. delete_by_entity
    # ------------------------------------------------------------------
    async def delete_by_entity(self, entity_name: str) -> None:
        self._ensure_initialized()
        edges = await self.graph_storage.get_node_edges(entity_name)
        await self.graph_storage.remove_edges(edges)
        await self.graph_storage.delete_node(entity_name)
        await self._entities_vdb.delete_entity(entity_name)
        await self._relationships_vdb.delete_entity_relation(entity_name)
        try:
            eid = compute_mdhash_id(entity_name)
            await self._chunks_bm25.delete([eid])
        except Exception:
            pass
        logger.info("Deleted entity %s and %d connected edges", entity_name, len(edges))

    # ------------------------------------------------------------------
    # 14. delete_by_relation
    # ------------------------------------------------------------------
    async def delete_by_relation(self, src: str, tgt: str) -> None:
        self._ensure_initialized()
        await self.graph_storage.remove_edges([(src, tgt)])
        edge_id = compute_mdhash_id(f"{src}-{tgt}")
        await self._relationships_vdb.delete([edge_id])
        logger.info("Deleted relation %s -> %s", src, tgt)

    # ------------------------------------------------------------------
    # 15. create_entity
    # ------------------------------------------------------------------
    async def create_entity(self, entity_name: str, entity_data: dict) -> None:
        self._ensure_initialized()
        await self.graph_storage.upsert_node(entity_name, entity_data)
        desc = entity_data.get("description", "")
        embeddings = await self.embedding_func([f"{entity_name}: {desc}"])
        embed_vec = embeddings[0] if len(embeddings) > 0 else None
        eid = compute_mdhash_id(entity_name)
        await self._entities_vdb.upsert({
            eid: {
                "entity_name": entity_name,
                "content": f"{entity_name}: {desc}",
                "embedding": embed_vec,
            }
        })
        await self._chunks_bm25.upsert({
            eid: {
                "entity_name": entity_name,
                "content": f"{entity_name} ({entity_data.get('type', '')}): {desc}",
            }
        })
        logger.info("Created entity %s", entity_name)

    # ------------------------------------------------------------------
    # 16. edit_entity
    # ------------------------------------------------------------------
    async def edit_entity(self, entity_name: str, entity_data: dict) -> None:
        self._ensure_initialized()
        existing = await self.graph_storage.get_node(entity_name)
        if existing is None:
            raise ValueError(f"Entity {entity_name!r} not found")
        merged = {**existing, **entity_data}
        await self.graph_storage.upsert_node(entity_name, merged)
        desc = merged.get("description", "")
        embeddings = await self.embedding_func([f"{entity_name}: {desc}"])
        embed_vec = embeddings[0] if len(embeddings) > 0 else None
        eid = compute_mdhash_id(entity_name)
        await self._entities_vdb.upsert({
            eid: {
                "entity_name": entity_name,
                "content": f"{entity_name}: {desc}",
                "embedding": embed_vec,
            }
        })
        logger.info("Edited entity %s", entity_name)

    # ------------------------------------------------------------------
    # 17. create_relation
    # ------------------------------------------------------------------
    async def create_relation(self, src: str, tgt: str, relation_data: dict) -> None:
        self._ensure_initialized()
        await self.graph_storage.upsert_edge(src, tgt, relation_data)
        desc = relation_data.get("description", "")
        content = f"{src} -> {tgt}: {desc}"
        embeddings = await self.embedding_func([content])
        embed_vec = embeddings[0] if len(embeddings) > 0 else None
        edge_id = compute_mdhash_id(f"{src}-{tgt}")
        await self._relationships_vdb.upsert({
            edge_id: {
                "src_id": src,
                "tgt_id": tgt,
                "content": content,
                "embedding": embed_vec,
            }
        })
        logger.info("Created relation %s -> %s", src, tgt)

    # ------------------------------------------------------------------
    # 18. edit_relation
    # ------------------------------------------------------------------
    async def edit_relation(self, src: str, tgt: str, relation_data: dict) -> None:
        self._ensure_initialized()
        existing = await self.graph_storage.get_edge(src, tgt)
        if existing is None:
            raise ValueError(f"Relation {src!r} -> {tgt!r} not found")
        merged = {**existing, **relation_data}
        await self.graph_storage.upsert_edge(src, tgt, merged)
        desc = merged.get("description", "")
        content = f"{src} -> {tgt}: {desc}"
        embeddings = await self.embedding_func([content])
        embed_vec = embeddings[0] if len(embeddings) > 0 else None
        edge_id = compute_mdhash_id(f"{src}-{tgt}")
        await self._relationships_vdb.upsert({
            edge_id: {
                "src_id": src,
                "tgt_id": tgt,
                "content": content,
                "embedding": embed_vec,
            }
        })
        logger.info("Edited relation %s -> %s", src, tgt)

    # ------------------------------------------------------------------
    # 19. merge_entities
    # ------------------------------------------------------------------
    async def merge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: str = "concatenate",
        target_entity_data: dict | None = None,
    ) -> None:
        self._ensure_initialized()
        merged_desc_parts: list[str] = []
        merged_sources: set[str] = set()
        max_confidence = 0.0
        merged_type = ""

        for ent_name in source_entities:
            node = await self.graph_storage.get_node(ent_name)
            if node is None:
                continue
            merged_desc_parts.append(node.get("description", ""))
            for s in node.get("source_id", "").split(","):
                if s.strip():
                    merged_sources.add(s.strip())
            max_confidence = max(max_confidence, node.get("confidence", 0.0))
            if not merged_type:
                merged_type = node.get("type", "")

        if merge_strategy == "concatenate":
            merged_desc = " | ".join(d for d in merged_desc_parts if d)
        else:
            merged_desc = merged_desc_parts[0] if merged_desc_parts else ""

        node_data = {
            "type": merged_type,
            "description": merged_desc,
            "source_id": ",".join(merged_sources),
            "confidence": max_confidence,
        }
        if target_entity_data:
            node_data.update(target_entity_data)

        # Collect edges from all source entities
        all_edges: list[tuple[str, str, dict]] = []
        for ent_name in source_entities:
            edges = await self.graph_storage.get_node_edges(ent_name)
            for src, tgt in edges:
                edge_data = await self.graph_storage.get_edge(src, tgt)
                if edge_data:
                    new_src = target_entity if src == ent_name else src
                    new_tgt = target_entity if tgt == ent_name else tgt
                    all_edges.append((new_src, new_tgt, edge_data))

        # Delete source entities
        for ent_name in source_entities:
            if ent_name != target_entity:
                await self.delete_by_entity(ent_name)

        # Create/update target entity
        await self.graph_storage.upsert_node(target_entity, node_data)
        desc = node_data.get("description", "")
        embeddings = await self.embedding_func([f"{target_entity}: {desc}"])
        embed_vec = embeddings[0] if len(embeddings) > 0 else None
        eid = compute_mdhash_id(target_entity)
        await self._entities_vdb.upsert({
            eid: {
                "entity_name": target_entity,
                "content": f"{target_entity}: {desc}",
                "embedding": embed_vec,
            }
        })

        # Re-create edges
        for src, tgt, edata in all_edges:
            await self.graph_storage.upsert_edge(src, tgt, edata)

        logger.info("Merged %d entities into %s", len(source_entities), target_entity)

    # ------------------------------------------------------------------
    # 20. export_data
    # ------------------------------------------------------------------
    async def export_data(
        self,
        path: str,
        file_format: str = "csv",
        include_vector_data: bool = False,
    ) -> None:
        self._ensure_initialized()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        all_labels = await self.graph_storage.get_all_labels()
        nodes: list[dict] = []
        edges: list[dict] = []

        for label in all_labels:
            node = await self.graph_storage.get_node(label)
            if node:
                entry = {"id": label, **node}
                if include_vector_data:
                    vec = await self._entities_vdb.get_by_id(compute_mdhash_id(label))
                    if vec:
                        entry["vector"] = vec.get("embedding")
                nodes.append(entry)
            node_edges = await self.graph_storage.get_node_edges(label)
            for src, tgt in node_edges:
                edge = await self.graph_storage.get_edge(src, tgt)
                if edge:
                    edges.append({"src": src, "tgt": tgt, **edge})

        # Deduplicate edges
        seen_edges: set[str] = set()
        unique_edges: list[dict] = []
        for e in edges:
            key = f"{e['src']}|{e['tgt']}"
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(e)

        if file_format == "json":
            write_json({"nodes": nodes, "edges": unique_edges}, path)
        elif file_format == "csv":
            import csv

            nodes_path = path.replace(".csv", "_nodes.csv")
            edges_path = path.replace(".csv", "_edges.csv")
            if nodes:
                with open(nodes_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(nodes[0].keys()))
                    writer.writeheader()
                    writer.writerows(nodes)
            if unique_edges:
                with open(edges_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(unique_edges[0].keys()))
                    writer.writeheader()
                    writer.writerows(unique_edges)
        else:
            write_json({"nodes": nodes, "edges": unique_edges}, path)

        logger.info("Exported %d nodes and %d edges to %s", len(nodes), len(unique_edges), path)

    # ------------------------------------------------------------------
    # 21. get_knowledge_graph
    # ------------------------------------------------------------------
    async def get_knowledge_graph(
        self,
        snapshot_at=None,
        community_level: int | None = None,
    ) -> dict:
        self._ensure_initialized()

        if snapshot_at is not None:
            return await self.graph_storage.get_subgraph_at(snapshot_at)

        all_labels = await self.graph_storage.get_all_labels()
        nodes: list[dict] = []
        edges: list[dict] = []

        for label in all_labels:
            node = await self.graph_storage.get_node(label)
            if node is None:
                continue
            if community_level is not None:
                node_community = node.get("community", "")
                # Filter by community level if metadata available
            nodes.append({"id": label, **node})
            node_edges = await self.graph_storage.get_node_edges(label)
            for src, tgt in node_edges:
                edge = await self.graph_storage.get_edge(src, tgt)
                if edge:
                    edges.append({"src": src, "tgt": tgt, **edge})

        # Deduplicate edges
        seen: set[str] = set()
        unique_edges: list[dict] = []
        for e in edges:
            key = f"{e['src']}|{e['tgt']}"
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)

        return {"nodes": nodes, "edges": unique_edges}

    # ------------------------------------------------------------------
    # 22. get_community_summary
    # ------------------------------------------------------------------
    async def get_community_summary(self, community_id: str) -> str | None:
        self._ensure_initialized()
        return await self.graph_storage.get_community_summary(community_id)

    # ------------------------------------------------------------------
    # 23. apipeline_enqueue_documents
    # ------------------------------------------------------------------
    async def apipeline_enqueue_documents(self, documents: list[dict]) -> None:
        self._ensure_initialized()
        for doc in documents:
            doc_id = doc.get("id") or compute_mdhash_id(doc.get("content", ""), prefix="doc-")
            doc_status = DocStatus(
                id=doc_id,
                content=doc.get("content", "")[:500],
                content_length=len(doc.get("content", "")),
                status="pending",
                file_path=doc.get("file_path", ""),
            )
            acl = doc.get("acl_policy")
            if isinstance(acl, dict):
                doc_status.acl_policy = ACLPolicy(**acl)
            elif isinstance(acl, ACLPolicy):
                doc_status.acl_policy = acl
            await self.doc_status_storage.upsert(doc_id, doc_status)
        logger.info("Enqueued %d documents for processing", len(documents))

    # ------------------------------------------------------------------
    # 24. apipeline_process_enqueue_documents
    # ------------------------------------------------------------------
    async def apipeline_process_enqueue_documents(self) -> None:
        self._ensure_initialized()
        pending = await self.doc_status_storage.get_docs_by_status("pending")
        if not pending:
            logger.info("No pending documents to process")
            return

        contents: list[str] = []
        ids: list[str] = []
        file_paths: list[str] = []
        acl_policies: list[ACLPolicy | None] = []

        for doc in pending:
            # Retrieve full content from KV or use stored summary
            full_doc = await self.kv_storage.get_by_id(doc.id)
            content = full_doc.get("content", doc.content) if full_doc else doc.content
            contents.append(content)
            ids.append(doc.id)
            file_paths.append(doc.file_path)
            acl_policies.append(doc.acl_policy)

        for content, doc_id, fpath, acl in zip(contents, ids, file_paths, acl_policies):
            await self.ainsert(content, ids=[doc_id], file_paths=[fpath], acl_policy=acl)

        logger.info("Processed %d enqueued documents", len(pending))

    # ------------------------------------------------------------------
    # 25. clear_cache
    # ------------------------------------------------------------------
    async def clear_cache(self, scope: str = "all") -> None:
        self._ensure_initialized()
        if scope in ("all", "llm"):
            if self._llm_response_cache is not None:
                await self._llm_response_cache.drop()
                await self._llm_response_cache.initialize()
                logger.info("Cleared LLM response cache")
        if scope in ("all", "vectors"):
            for vdb in (self._entities_vdb, self._relationships_vdb, self._chunks_vdb):
                if vdb is not None:
                    await vdb.drop()
                    await vdb.initialize()
            logger.info("Cleared vector caches")
        if scope in ("all", "lexical"):
            if self._chunks_bm25 is not None:
                await self._chunks_bm25.drop()
                await self._chunks_bm25.initialize()
            logger.info("Cleared lexical cache")

    # ------------------------------------------------------------------
    # 26. run_maintenance
    # ------------------------------------------------------------------
    async def run_maintenance(self, tasks: list[str] | None = None) -> None:
        self._ensure_initialized()
        if tasks is None:
            tasks = ["rebuild_communities", "cleanup_orphans", "persist"]

        if "rebuild_communities" in tasks and self.enable_community_detection:
            await build_communities(
                self.graph_storage,
                self._global_config,
                llm_response_cache=self._llm_response_cache,
            )
            logger.info("Maintenance: rebuilt communities")

        if "cleanup_orphans" in tasks:
            all_labels = await self.graph_storage.get_all_labels()
            orphans = []
            for label in all_labels:
                degree = await self.graph_storage.node_degree(label)
                node = await self.graph_storage.get_node(label)
                if degree == 0 and node:
                    source = node.get("source_id", "")
                    if not source or source == "":
                        orphans.append(label)
            if orphans:
                await self.graph_storage.remove_nodes(orphans)
                for ent in orphans:
                    await self._entities_vdb.delete_entity(ent)
                logger.info("Maintenance: removed %d orphan nodes", len(orphans))

        if "persist" in tasks:
            for store in (
                self.kv_storage,
                self.vector_storage,
                self.graph_storage,
                self.lexical_storage,
                self.doc_status_storage,
                self._llm_response_cache,
                self._entities_vdb,
                self._relationships_vdb,
                self._chunks_vdb,
                self._chunks_bm25,
            ):
                if store is not None:
                    await store.finalize()
                    await store.initialize()
            logger.info("Maintenance: persisted all storages")
