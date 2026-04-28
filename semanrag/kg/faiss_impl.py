"""FAISS-backed vector storage with JSON sidecar for metadata and ACL filtering."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from semanrag.base import BaseVectorStorage

logger = logging.getLogger(__name__)

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore[assignment]
    FAISS_AVAILABLE = False
    logger.warning("faiss is not installed. FaissVectorDBStorage will not work.")


class FaissVectorDBStorage(BaseVectorStorage):
    """Vector storage backed by FAISS with a JSON sidecar for metadata/ACL."""

    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
        embedding_func: Any = None,
    ) -> None:
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required but not installed. pip install faiss-cpu")
        super().__init__(global_config, namespace, workspace, embedding_func)
        working_dir = global_config.get("working_dir", "./data")
        if workspace:
            working_dir = os.path.join(working_dir, workspace)
        safe_ns = self.full_namespace.replace("/", "_")
        self._index_path = os.path.join(working_dir, f"{safe_ns}_faiss.index")
        self._sidecar_path = os.path.join(working_dir, f"{safe_ns}_faiss_meta.json")
        self._use_ivf = global_config.get("faiss_use_ivf", False)
        self._nlist = global_config.get("faiss_nlist", 100)
        self._nprobe = global_config.get("faiss_nprobe", 10)
        self._index: Any = None
        # Sidecar: maps internal integer position -> {id, metadata}
        # _id_to_pos: external id -> internal position
        self._meta: dict[int, dict] = {}
        self._id_to_pos: dict[str, int] = {}
        self._next_pos: int = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self) -> None:
        Path(self._index_path).parent.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, self._index_path)
        sidecar = {"meta": {str(k): v for k, v in self._meta.items()}, "id_to_pos": self._id_to_pos, "next_pos": self._next_pos}
        with open(self._sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, default=str)

    def _load(self) -> None:
        if os.path.exists(self._index_path):
            self._index = faiss.read_index(self._index_path)
        if os.path.exists(self._sidecar_path):
            try:
                with open(self._sidecar_path, encoding="utf-8") as f:
                    sidecar = json.load(f)
                self._meta = {int(k): v for k, v in sidecar.get("meta", {}).items()}
                self._id_to_pos = sidecar.get("id_to_pos", {})
                self._next_pos = sidecar.get("next_pos", 0)
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt FAISS sidecar %s – starting fresh", self._sidecar_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        self._load()
        if self._index is None:
            dim = self.embedding_func.embedding_dim
            if self._use_ivf:
                quantizer = faiss.IndexFlatIP(dim)
                self._index = faiss.IndexIVFFlat(quantizer, dim, self._nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                self._index = faiss.IndexFlatIP(dim)

    async def finalize(self) -> None:
        self._save()

    async def drop(self) -> None:
        self._index = None
        self._meta.clear()
        self._id_to_pos.clear()
        self._next_pos = 0
        for p in (self._index_path, self._sidecar_path):
            if os.path.exists(p):
                os.remove(p)

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------
    async def upsert(self, data: dict[str, dict]) -> None:
        if not data:
            return
        vectors = []
        positions = []
        for id_, fields in data.items():
            vec = fields.get("__vector__")
            if vec is None:
                continue
            vec_np = np.array(vec, dtype=np.float32).reshape(1, -1)
            # Normalize for inner-product (cosine) similarity
            norm = np.linalg.norm(vec_np)
            if norm > 0:
                vec_np = vec_np / norm

            meta = {k: v for k, v in fields.items() if k != "__vector__"}
            meta["__id__"] = id_

            if id_ in self._id_to_pos:
                # Update: we cannot remove from FAISS easily, so we mark old position as deleted
                old_pos = self._id_to_pos[id_]
                self._meta.pop(old_pos, None)

            pos = self._next_pos
            self._next_pos += 1
            self._id_to_pos[id_] = pos
            self._meta[pos] = meta
            vectors.append(vec_np)
            positions.append(pos)

        if vectors:
            mat = np.vstack(vectors)
            if self._use_ivf and not self._index.is_trained:
                self._index.train(mat)
            self._index.add(mat)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    async def query(self, query: str, top_k: int, acl_filter: dict | None = None) -> list[dict]:
        if self._index is None or self._index.ntotal == 0:
            return []
        embedding = await self.embedding_func([query])
        vec = np.array(embedding[0], dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if self._use_ivf:
            self._index.nprobe = self._nprobe

        # Search more than top_k to allow for deleted entries and ACL filtering
        search_k = min(top_k * 4, self._index.ntotal)
        scores, indices = self._index.search(vec, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._meta.get(int(idx))
            if meta is None:
                continue  # Deleted entry
            entry = {**meta, "__score__": float(score)}
            results.append(entry)

        # ACL post-filter
        if acl_filter:
            user_id = acl_filter.get("user_id")
            user_groups = acl_filter.get("user_groups", [])
            filtered = []
            for r in results:
                if r.get("acl_public", True) or user_id and (r.get("acl_owner") == user_id or user_id in r.get("acl_visible_to_users", [])) or user_groups and set(user_groups) & set(r.get("acl_visible_to_groups", [])):
                    filtered.append(r)
            results = filtered

        return results[:top_k]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    async def delete(self, ids: list[str]) -> None:
        for id_ in ids:
            pos = self._id_to_pos.pop(id_, None)
            if pos is not None:
                self._meta.pop(pos, None)

    async def delete_entity(self, entity_name: str) -> None:
        to_remove = [id_ for id_, pos in self._id_to_pos.items() if self._meta.get(pos, {}).get("entity_name") == entity_name]
        await self.delete(to_remove)

    async def delete_entity_relation(self, entity_name: str) -> None:
        to_remove = []
        for id_, pos in self._id_to_pos.items():
            m = self._meta.get(pos, {})
            if m.get("src_id") == entity_name or m.get("tgt_id") == entity_name:
                to_remove.append(id_)
        await self.delete(to_remove)

    # ------------------------------------------------------------------
    # Get
    # ------------------------------------------------------------------
    async def get_by_id(self, id: str) -> dict | None:
        pos = self._id_to_pos.get(id)
        if pos is None:
            return None
        return self._meta.get(pos)

    async def get_by_ids(self, ids: list[str]) -> list[dict | None]:
        return [await self.get_by_id(i) for i in ids]

    async def get_vectors_by_ids(self, ids: list[str]) -> list[dict | None]:
        if not ids or self._index is None:
            return [None] * len(ids)
        results: list[dict | None] = []
        for id_ in ids:
            pos = self._id_to_pos.get(id_)
            if pos is None or pos >= self._index.ntotal:
                results.append(None)
                continue
            meta = self._meta.get(pos)
            if meta is None:
                results.append(None)
                continue
            vec = self._index.reconstruct(pos)
            entry = {**meta, "__vector__": vec.tolist()}
            results.append(entry)
        return results
