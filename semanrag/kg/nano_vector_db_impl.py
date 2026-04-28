from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from semanrag.base import BaseVectorStorage

logger = logging.getLogger(__name__)

try:
    from nano_vectordb import NanoVectorDB
except ImportError:
    NanoVectorDB = None


class NanoVectorDBStorage(BaseVectorStorage):
    """Vector storage backed by nano-vectordb with workspace isolation and ACL filtering."""

    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
        embedding_func: Any = None,
    ) -> None:
        super().__init__(global_config, namespace, workspace, embedding_func)
        if NanoVectorDB is None:
            raise ImportError(
                "nano-vectordb is required but not installed. "
                "Install it with: pip install nano-vectordb"
            )
        self._db: NanoVectorDB | None = None
        working_dir = self._global_config.get("working_dir", "./data")
        if self._workspace:
            working_dir = os.path.join(working_dir, self._workspace)
        self._storage_path = os.path.join(
            working_dir,
            f"{self.full_namespace}_vdb_{self.embedding_func.embedding_dim}.json",
        )

    async def initialize(self) -> None:
        os.makedirs(os.path.dirname(self._storage_path), exist_ok=True)
        self._db = NanoVectorDB(
            embedding_dim=self.embedding_func.embedding_dim,
            storage_file=self._storage_path,
        )

    async def finalize(self) -> None:
        if self._db is not None:
            self._db.save()

    async def upsert(self, data: dict[str, dict]) -> None:
        records = []
        for id_, fields in data.items():
            record: dict[str, Any] = {"__id__": id_}
            for k, v in fields.items():
                if k == "__vector__":
                    record["__vector__"] = np.array(v, dtype=np.float32)
                else:
                    record[k] = v
            records.append(record)
        if records:
            self._db.upsert(records)

    async def query(
        self, query: str, top_k: int, acl_filter: dict | None = None
    ) -> list[dict]:
        embedding = await self.embedding_func([query])
        vector = np.array(embedding[0], dtype=np.float32)
        threshold = self._global_config.get(
            "query_better_than_threshold", 0.2
        )
        results = self._db.query(
            vector, top_k=top_k, better_than_threshold=threshold
        )
        if acl_filter:
            user_id = acl_filter.get("user_id")
            user_groups = acl_filter.get("user_groups", [])
            filtered = []
            for r in results:
                if r.get("acl_public", True):
                    filtered.append(r)
                    continue
                if user_id and (
                    r.get("acl_owner") == user_id
                    or user_id in r.get("acl_visible_to_users", [])
                ):
                    filtered.append(r)
                    continue
                if user_groups and set(user_groups) & set(
                    r.get("acl_visible_to_groups", [])
                ):
                    filtered.append(r)
            results = filtered
        return results

    async def delete(self, ids: list[str]) -> None:
        self._db.delete(ids)

    async def delete_entity(self, entity_name: str) -> None:
        storage = self._db._NanoVectorDB__storage
        ids_to_delete = [
            d["__id__"]
            for d in storage["data"]
            if d.get("entity_name") == entity_name
        ]
        if ids_to_delete:
            self._db.delete(ids_to_delete)

    async def delete_entity_relation(self, entity_name: str) -> None:
        storage = self._db._NanoVectorDB__storage
        ids_to_delete = [
            d["__id__"]
            for d in storage["data"]
            if d.get("src_id") == entity_name or d.get("tgt_id") == entity_name
        ]
        if ids_to_delete:
            self._db.delete(ids_to_delete)

    async def get_by_id(self, id: str) -> dict | None:
        results = self._db.get([id])
        return results[0] if results else None

    async def get_by_ids(self, ids: list[str]) -> list[dict | None]:
        if not ids:
            return []
        results = self._db.get(ids)
        found = {r["__id__"]: r for r in results}
        return [found.get(id_) for id_ in ids]

    async def get_vectors_by_ids(self, ids: list[str]) -> list[dict | None]:
        if not ids:
            return []
        storage = self._db._NanoVectorDB__storage
        id_to_idx = {
            d["__id__"]: i for i, d in enumerate(storage["data"])
        }
        output: list[dict | None] = []
        for id_ in ids:
            idx = id_to_idx.get(id_)
            if idx is None:
                output.append(None)
            else:
                entry = dict(storage["data"][idx])
                entry["__vector__"] = storage["matrix"][idx].tolist()
                output.append(entry)
        return output

    async def drop(self) -> None:
        if os.path.exists(self._storage_path):
            os.remove(self._storage_path)
        self._db = None
