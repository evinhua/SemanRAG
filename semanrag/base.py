from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import (
    Any,
    Literal,
    TypedDict,
)

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. TextChunkSchema
# ---------------------------------------------------------------------------
class TextChunkSchema(TypedDict):
    tokens: int
    content: str
    full_doc_id: str
    chunk_order_index: int
    section_path: str | None
    page_number: int | None
    modality: Literal["text", "table", "figure_caption"]


# ---------------------------------------------------------------------------
# 2. TemporalEdge
# ---------------------------------------------------------------------------
@dataclass
class TemporalEdge:
    source: str
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# 3. ACLPolicy
# ---------------------------------------------------------------------------
@dataclass
class ACLPolicy:
    owner: str = ""
    visible_to_groups: list[str] = field(default_factory=list)
    visible_to_users: list[str] = field(default_factory=list)
    public: bool = True

    def can_access(self, user_id: str, user_groups: list[str]) -> bool:
        if self.public:
            return True
        if user_id == self.owner:
            return True
        if user_id in self.visible_to_users:
            return True
        if set(self.visible_to_groups) & set(user_groups):
            return True
        return False


# ---------------------------------------------------------------------------
# 4. StorageNameSpace
# ---------------------------------------------------------------------------
class StorageNameSpace:
    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
    ) -> None:
        self._global_config = global_config
        self._namespace = namespace
        self._workspace = workspace

    @property
    def full_namespace(self) -> str:
        return f"{self._workspace}/{self._namespace}" if self._workspace else self._namespace

    def inject_acl_filter(self, user_id: str, user_groups: list[str]) -> dict:
        return {"user_id": user_id, "user_groups": user_groups}


# ---------------------------------------------------------------------------
# 5. BaseKVStorage
# ---------------------------------------------------------------------------
class BaseKVStorage(StorageNameSpace, ABC):
    @abstractmethod
    async def get_by_id(self, id: str) -> dict | None: ...

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict | None]: ...

    @abstractmethod
    async def filter_keys(self, data: set[str]) -> set[str]: ...

    @abstractmethod
    async def upsert(self, data: dict[str, dict]) -> None: ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None: ...

    @abstractmethod
    async def drop(self) -> None: ...

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def finalize(self) -> None: ...

    @abstractmethod
    async def index_done_callback(self) -> None: ...


# ---------------------------------------------------------------------------
# 6. BaseVectorStorage
# ---------------------------------------------------------------------------
class BaseVectorStorage(StorageNameSpace, ABC):
    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
        embedding_func: Any = None,
    ) -> None:
        super().__init__(global_config, namespace, workspace)
        if embedding_func is None:
            raise ValueError("embedding_func is required")
        if not hasattr(embedding_func, "embedding_dim"):
            raise ValueError("embedding_func must have an 'embedding_dim' attribute")
        if not hasattr(embedding_func, "max_token_size"):
            raise ValueError("embedding_func must have a 'max_token_size' attribute")
        self.embedding_func = embedding_func

    @abstractmethod
    async def upsert(self, data: dict[str, dict]) -> None: ...

    @abstractmethod
    async def query(
        self, query: str, top_k: int, acl_filter: dict | None = None
    ) -> list[dict]: ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None: ...

    @abstractmethod
    async def delete_entity(self, entity_name: str) -> None: ...

    @abstractmethod
    async def delete_entity_relation(self, entity_name: str) -> None: ...

    @abstractmethod
    async def get_by_id(self, id: str) -> dict | None: ...

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> list[dict | None]: ...

    @abstractmethod
    async def get_vectors_by_ids(self, ids: list[str]) -> list[dict | None]: ...

    @abstractmethod
    async def drop(self) -> None: ...

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def finalize(self) -> None: ...


# ---------------------------------------------------------------------------
# 7. BaseGraphStorage
# ---------------------------------------------------------------------------
class BaseGraphStorage(StorageNameSpace, ABC):
    @abstractmethod
    async def has_node(self, node_id: str) -> bool: ...

    @abstractmethod
    async def has_edge(self, src: str, tgt: str) -> bool: ...

    @abstractmethod
    async def node_degree(self, node_id: str) -> int: ...

    @abstractmethod
    async def edge_degree(self, src: str, tgt: str) -> int: ...

    @abstractmethod
    async def get_node(self, node_id: str) -> dict | None: ...

    @abstractmethod
    async def get_edge(self, src: str, tgt: str) -> dict | None: ...

    @abstractmethod
    async def get_node_edges(self, node_id: str) -> list[tuple[str, str]]: ...

    @abstractmethod
    async def upsert_node(self, node_id: str, node_data: dict) -> None: ...

    @abstractmethod
    async def upsert_edge(self, src: str, tgt: str, edge_data: dict) -> None: ...

    @abstractmethod
    async def delete_node(self, node_id: str) -> None: ...

    @abstractmethod
    async def remove_nodes(self, nodes: list[str]) -> None: ...

    @abstractmethod
    async def remove_edges(self, edges: list[tuple[str, str]]) -> None: ...

    @abstractmethod
    async def get_all_labels(self) -> list[str]: ...

    @abstractmethod
    async def get_knowledge_graph(
        self, node_label: str | None, max_depth: int
    ) -> dict: ...

    @abstractmethod
    async def search_labels(self, query: str) -> list[str]: ...

    @abstractmethod
    async def get_popular_labels(self, top_n: int) -> list[tuple[str, int]]: ...

    @abstractmethod
    async def get_subgraph_at(self, snapshot_at: datetime) -> dict: ...

    @abstractmethod
    async def detect_communities(self, algorithm: str, levels: int) -> dict: ...

    @abstractmethod
    async def get_community_summary(self, community_id: str) -> str | None: ...

    @abstractmethod
    async def drop(self) -> None: ...

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def finalize(self) -> None: ...


# ---------------------------------------------------------------------------
# 8. BaseLexicalStorage
# ---------------------------------------------------------------------------
class BaseLexicalStorage(StorageNameSpace, ABC):
    @abstractmethod
    async def upsert(self, data: dict[str, dict]) -> None: ...

    @abstractmethod
    async def search_bm25(self, query: str, top_k: int) -> list[dict]: ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None: ...

    @abstractmethod
    async def drop(self) -> None: ...

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def finalize(self) -> None: ...


# ---------------------------------------------------------------------------
# 9. DocStatus
# ---------------------------------------------------------------------------
@dataclass
class DocStatus:
    id: str
    content: str = ""
    content_summary: str = ""
    content_length: int = 0
    chunks_count: int = 0
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = ""
    file_path: str = ""
    chunks_list: list[str] = field(default_factory=list)
    error_message: str = ""
    pii_findings: list[dict] = field(default_factory=list)
    prompt_injection_flags: list[dict] = field(default_factory=list)
    acl_policy: ACLPolicy | None = None
    version: int = 1


# ---------------------------------------------------------------------------
# 10. DocStatusStorage
# ---------------------------------------------------------------------------
class DocStatusStorage(StorageNameSpace, ABC):
    @abstractmethod
    async def get(self, doc_id: str) -> DocStatus | None: ...

    @abstractmethod
    async def upsert(self, doc_id: str, status: DocStatus) -> None: ...

    @abstractmethod
    async def delete(self, doc_id: str) -> None: ...

    @abstractmethod
    async def get_status_counts(self) -> dict[str, int]: ...

    @abstractmethod
    async def get_docs_by_status(self, status: str) -> list[DocStatus]: ...

    @abstractmethod
    async def get_docs_paginated(
        self,
        offset: int,
        limit: int,
        status: str | None = None,
        acl_filter: dict | None = None,
    ) -> tuple[list[DocStatus], int]: ...

    @abstractmethod
    async def get_all_status_counts(self) -> dict[str, int]: ...

    @abstractmethod
    async def get_doc_by_file_path(self, file_path: str) -> DocStatus | None: ...

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def finalize(self) -> None: ...


# ---------------------------------------------------------------------------
# 11. QueryParam
# ---------------------------------------------------------------------------
ALLOWED_QUERY_MODES = {"local", "global", "hybrid", "naive", "mix", "community", "bypass"}


@dataclass
class QueryParam:
    mode: str = "local"
    only_need_context: bool = False
    only_need_prompt: bool = False
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    top_k: int = 20
    chunk_top_k: int = 5
    max_entity_tokens: int = 4000
    max_relation_tokens: int = 4000
    max_total_tokens: int = 12000
    conversation_history: list[dict] = field(default_factory=list)
    model_func: Callable | None = None
    user_prompt: str = ""
    enable_rerank: bool = True
    enable_hybrid_lexical: bool = True
    rrf_k: int = 60
    snapshot_at: datetime | None = None
    user_id: str | None = None
    user_groups: list[str] = field(default_factory=list)
    verifier_enabled: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ALLOWED_QUERY_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of {sorted(ALLOWED_QUERY_MODES)}"
            )


# ---------------------------------------------------------------------------
# 12. QueryResult
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    content: str = ""
    raw_data: dict = field(default_factory=dict)
    response_iterator: AsyncIterator | None = None
    is_streaming: bool = False
    references: list[dict] = field(default_factory=list)
    grounded_check: list[dict] = field(default_factory=list)
    communities_used: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens_used: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 13. Pydantic extraction schemas
# ---------------------------------------------------------------------------
class ExtractedEntity(BaseModel):
    name: str
    type: str = "UNKNOWN"
    description: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ExtractedRelation(BaseModel):
    source: str
    target: str
    keywords: str = ""
    description: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    valid_from: str | None = None
    valid_to: str | None = None


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
