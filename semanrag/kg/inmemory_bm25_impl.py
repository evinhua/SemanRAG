from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from typing import Any

from semanrag.base import BaseLexicalStorage

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None  # type: ignore[assignment,misc]
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 is not installed. BM25 search will return empty results.")

_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return _PUNCTUATION_RE.sub("", text.lower()).split()


class InMemoryBM25Storage(BaseLexicalStorage):
    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
    ) -> None:
        super().__init__(global_config, namespace, workspace)
        self._documents: dict[str, dict] = {}
        self._corpus: list[list[str]] = []
        self._doc_ids: list[str] = []
        self._bm25: Any = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """Rebuild BM25 index from current _documents."""
        self._doc_ids = list(self._documents.keys())
        self._corpus = [
            _tokenize(self._documents[did].get("content", ""))
            for did in self._doc_ids
        ]
        if self._corpus and BM25_AVAILABLE:
            self._bm25 = BM25Okapi(self._corpus)
        else:
            self._bm25 = None

    @property
    def _snapshot_path(self) -> str:
        working_dir = self._global_config.get("working_dir", ".")
        safe_ns = self.full_namespace.replace("/", "_")
        return os.path.join(working_dir, f"{safe_ns}_bm25.json")

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    async def upsert(self, data: dict[str, dict]) -> None:
        for doc_id, doc in data.items():
            self._documents[doc_id] = doc
        self._rebuild_index()

    async def search_bm25(self, query: str, top_k: int) -> list[dict]:
        if not self._doc_ids or self._bm25 is None:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        scored = sorted(
            zip(self._doc_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results: list[dict] = []
        for doc_id, score in scored:
            doc = self._documents[doc_id]
            entry = {**doc, "id": doc_id, "score": float(score)}
            results.append(entry)
        return results

    async def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self._documents.pop(doc_id, None)
        self._rebuild_index()

    async def drop(self) -> None:
        self._documents.clear()
        self._corpus.clear()
        self._doc_ids.clear()
        self._bm25 = None

    async def initialize(self) -> None:
        path = self._snapshot_path
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self._documents = json.load(f)
                self._rebuild_index()
                logger.info("Loaded BM25 snapshot from %s (%d docs)", path, len(self._documents))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load BM25 snapshot from %s: %s", path, exc)
                self._documents = {}
                self._rebuild_index()

    async def finalize(self) -> None:
        path = self._snapshot_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Atomic write: write to temp file then rename
        dir_name = os.path.dirname(path) or "."
        try:
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._documents, f, ensure_ascii=False)
                os.replace(tmp_path, path)
                logger.info("Saved BM25 snapshot to %s (%d docs)", path, len(self._documents))
            except BaseException:
                os.unlink(tmp_path)
                raise
        except OSError as exc:
            logger.error("Failed to save BM25 snapshot to %s: %s", path, exc)
