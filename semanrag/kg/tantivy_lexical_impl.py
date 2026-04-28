"""Tantivy-backed persistent BM25 lexical storage."""

from __future__ import annotations

import logging
import os
from typing import Any

from semanrag.base import BaseLexicalStorage

logger = logging.getLogger(__name__)

try:
    import tantivy

    TANTIVY_AVAILABLE = True
except ImportError:
    tantivy = None  # type: ignore[assignment]
    TANTIVY_AVAILABLE = False
    logger.warning("tantivy is not installed. TantivyLexicalStorage will not work.")


class TantivyLexicalStorage(BaseLexicalStorage):
    """Persistent BM25 lexical search backed by tantivy Python bindings."""

    def __init__(self, global_config: dict, namespace: str, workspace: str | None = None) -> None:
        if not TANTIVY_AVAILABLE:
            raise ImportError("tantivy is required but not installed. pip install tantivy")
        super().__init__(global_config, namespace, workspace)
        working_dir = global_config.get("working_dir", "./data")
        if workspace:
            working_dir = os.path.join(working_dir, workspace)
        safe_ns = self.full_namespace.replace("/", "_")
        self._index_dir = os.path.join(working_dir, f"{safe_ns}_tantivy")
        self._index: Any = None
        self._schema: Any = None

    async def initialize(self) -> None:
        os.makedirs(self._index_dir, exist_ok=True)
        builder = tantivy.SchemaBuilder()
        builder.add_text_field("id", stored=True)
        builder.add_text_field("content", stored=True, tokenizer_name="default")
        builder.add_text_field("workspace", stored=True)
        self._schema = builder.build()
        self._index = tantivy.Index(self._schema, path=self._index_dir)

    async def finalize(self) -> None:
        # Index is persisted on commit; nothing extra needed
        pass

    async def upsert(self, data: dict[str, dict]) -> None:
        if not data:
            return
        writer = self._index.writer()
        for doc_id, doc in data.items():
            # Delete existing doc with same id first (tantivy doesn't have native upsert)
            writer.delete_documents("id", doc_id)
            writer.add_document(tantivy.Document(
                id=doc_id,
                content=doc.get("content", ""),
                workspace=self._workspace or "",
            ))
        writer.commit()
        self._index.reload()

    async def search_bm25(self, query: str, top_k: int) -> list[dict]:
        if not query.strip():
            return []
        searcher = self._index.searcher()
        query_parser = tantivy.QueryParser.for_index(self._index, ["content"])
        try:
            parsed = query_parser.parse_query(query)
        except Exception:
            # If query parsing fails (special chars etc.), escape and retry
            escaped = query.replace("\\", "\\\\").replace('"', '\\"')
            parsed = query_parser.parse_query(f'"{escaped}"')
        results = []
        search_result = searcher.search(parsed, top_k)
        for score, doc_address in search_result.hits:
            doc = searcher.doc(doc_address)
            results.append({
                "id": doc["id"][0],
                "content": doc["content"][0],
                "workspace": doc["workspace"][0],
                "score": score,
            })
        return results

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        writer = self._index.writer()
        for doc_id in ids:
            writer.delete_documents("id", doc_id)
        writer.commit()
        self._index.reload()

    async def drop(self) -> None:
        # Delete all documents by recreating the index
        if os.path.isdir(self._index_dir):
            import shutil

            shutil.rmtree(self._index_dir)
        # Re-initialize with empty index
        await self.initialize()
