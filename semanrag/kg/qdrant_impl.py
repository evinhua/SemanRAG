from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from semanrag.base import BaseVectorStorage

logger = logging.getLogger(__name__)

try:
    from qdrant_client import AsyncQdrantClient, models

    QDRANT_AVAILABLE = True
except ImportError as _import_err:
    QDRANT_AVAILABLE = False
    _qdrant_error = _import_err

# Rough per-point overhead for batch size estimation (bytes)
_POINT_OVERHEAD = 128
_MAX_BATCH_BYTES = 32 * 1024 * 1024  # 32 MiB


class QdrantVectorDBStorage(BaseVectorStorage):
    """Qdrant-backed vector storage with workspace isolation and ACL payload filtering."""

    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
        embedding_func: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(global_config, namespace, workspace, embedding_func)
        if not QDRANT_AVAILABLE:
            raise ImportError(
                f"qdrant-client is required but not installed: {_qdrant_error}. "
                "Install with: pip install qdrant-client"
            )

        self._url = (
            os.environ.get("QDRANT_URL")
            or kwargs.get("url")
            or global_config.get("qdrant_url", "http://localhost:6333")
        )
        self._api_key = (
            os.environ.get("QDRANT_API_KEY")
            or kwargs.get("api_key")
            or global_config.get("qdrant_api_key")
            or None
        )
        self._prefer_grpc = (
            os.environ.get("QDRANT_PREFER_GRPC", "").lower() in ("1", "true")
        ) or kwargs.get("prefer_grpc", global_config.get("qdrant_prefer_grpc", False))

        ws = self._workspace or "default"
        self._collection_name = f"{self._namespace}_{ws}"
        self._dim = self.embedding_func.embedding_dim
        self._client: AsyncQdrantClient | None = None

    async def initialize(self) -> None:
        self._client = AsyncQdrantClient(
            url=self._url,
            api_key=self._api_key,
            prefer_grpc=self._prefer_grpc,
        )
        collections = await self._client.get_collections()
        existing = {c.name for c in collections.collections}
        if self._collection_name not in existing:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s' (dim=%d)", self._collection_name, self._dim)
        else:
            logger.info("Qdrant collection '%s' already exists", self._collection_name)

    async def finalize(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    def _estimate_batch_size(self, sample_vector_len: int) -> int:
        """Estimate max points per batch to stay under _MAX_BATCH_BYTES."""
        per_point = sample_vector_len * 4 + _POINT_OVERHEAD  # float32
        return max(1, _MAX_BATCH_BYTES // per_point)

    async def upsert(self, data: dict[str, dict]) -> None:
        if not data:
            return

        points: list[models.PointStruct] = []
        ws = self._workspace or "default"
        for id_, fields in data.items():
            vector = fields.get("__vector__")
            if vector is None:
                continue
            payload = {k: v for k, v in fields.items() if k != "__vector__"}
            payload["workspace"] = ws
            # Deterministic UUID from string id for Qdrant point id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, id_))
            payload["_original_id"] = id_
            points.append(
                models.PointStruct(id=point_id, vector=list(vector), payload=payload)
            )

        if not points:
            return

        batch_size = self._estimate_batch_size(len(points[0].vector))
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self._client.upsert(
                collection_name=self._collection_name,
                points=batch,
            )

    def _build_filter(self, acl_filter: dict | None = None) -> models.Filter:
        """Build Qdrant filter for workspace + optional ACL."""
        ws = self._workspace or "default"
        must = [
            models.FieldCondition(
                key="workspace",
                match=models.MatchValue(value=ws),
            )
        ]

        if acl_filter:
            user_id = acl_filter.get("user_id", "")
            user_groups = acl_filter.get("user_groups", [])
            acl_should: list[models.Condition] = [
                models.FieldCondition(key="acl_public", match=models.MatchValue(value=True)),
            ]
            if user_id:
                acl_should.append(
                    models.FieldCondition(key="acl_owner", match=models.MatchValue(value=user_id))
                )
                acl_should.append(
                    models.FieldCondition(
                        key="acl_visible_to_users",
                        match=models.MatchAny(any=[user_id]),
                    )
                )
            if user_groups:
                acl_should.append(
                    models.FieldCondition(
                        key="acl_visible_to_groups",
                        match=models.MatchAny(any=user_groups),
                    )
                )
            must.append(models.Filter(should=acl_should))

        return models.Filter(must=must)

    async def query(
        self, query: str, top_k: int, acl_filter: dict | None = None
    ) -> list[dict]:
        embedding = await self.embedding_func([query])
        vector = list(embedding[0])

        results = await self._client.search(
            collection_name=self._collection_name,
            query_vector=vector,
            limit=top_k,
            query_filter=self._build_filter(acl_filter),
            with_payload=True,
        )

        output = []
        for hit in results:
            entry = dict(hit.payload or {})
            entry["__id__"] = entry.pop("_original_id", str(hit.id))
            entry["__score__"] = hit.score
            output.append(entry)
        return output

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, i)) for i in ids]
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=point_ids),
        )

    async def delete_entity(self, entity_name: str) -> None:
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="entity_name",
                            match=models.MatchValue(value=entity_name),
                        )
                    ]
                )
            ),
        )

    async def delete_entity_relation(self, entity_name: str) -> None:
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="src_id", match=models.MatchValue(value=entity_name)
                        ),
                        models.FieldCondition(
                            key="tgt_id", match=models.MatchValue(value=entity_name)
                        ),
                    ]
                )
            ),
        )

    async def get_by_id(self, id: str) -> dict | None:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, id))
        results = await self._client.retrieve(
            collection_name=self._collection_name,
            ids=[point_id],
            with_payload=True,
        )
        if not results:
            return None
        entry = dict(results[0].payload or {})
        entry["__id__"] = entry.pop("_original_id", id)
        return entry

    async def get_by_ids(self, ids: list[str]) -> list[dict | None]:
        if not ids:
            return []
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, i)) for i in ids]
        results = await self._client.retrieve(
            collection_name=self._collection_name,
            ids=point_ids,
            with_payload=True,
        )
        found: dict[str, dict] = {}
        for r in results:
            entry = dict(r.payload or {})
            orig_id = entry.pop("_original_id", str(r.id))
            entry["__id__"] = orig_id
            found[orig_id] = entry
        return [found.get(i) for i in ids]

    async def get_vectors_by_ids(self, ids: list[str]) -> list[dict | None]:
        if not ids:
            return []
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, i)) for i in ids]
        results = await self._client.retrieve(
            collection_name=self._collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=True,
        )
        found: dict[str, dict] = {}
        for r in results:
            entry = dict(r.payload or {})
            orig_id = entry.pop("_original_id", str(r.id))
            entry["__id__"] = orig_id
            entry["__vector__"] = r.vector if r.vector else []
            found[orig_id] = entry
        return [found.get(i) for i in ids]

    async def drop(self) -> None:
        if self._client:
            collections = await self._client.get_collections()
            if self._collection_name in {c.name for c in collections.collections}:
                await self._client.delete_collection(self._collection_name)
