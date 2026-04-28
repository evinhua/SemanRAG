from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from semanrag.base import BaseVectorStorage

logger = logging.getLogger(__name__)

PYMILVUS_MIN_VERSION = "2.4.0"

try:
    import pymilvus
    from pymilvus import (
        CollectionSchema,
        DataType,
        FieldSchema,
        MilvusClient,
    )

    # Version validation
    _ver = tuple(int(x) for x in pymilvus.__version__.split(".")[:3])
    _min = tuple(int(x) for x in PYMILVUS_MIN_VERSION.split(".")[:3])
    if _ver < _min:
        raise ImportError(
            f"pymilvus >= {PYMILVUS_MIN_VERSION} required, found {pymilvus.__version__}"
        )
    PYMILVUS_AVAILABLE = True
except ImportError as _import_err:
    PYMILVUS_AVAILABLE = False
    _pymilvus_error = _import_err


@dataclass
class MilvusIndexConfig:
    index_type: str = "AUTOINDEX"  # AUTOINDEX|HNSW|HNSW_SQ|IVF_FLAT|DISKANN|SCANN
    metric_type: str = "COSINE"  # COSINE|L2|IP
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_index = {"AUTOINDEX", "HNSW", "HNSW_SQ", "IVF_FLAT", "DISKANN", "SCANN"}
        valid_metric = {"COSINE", "L2", "IP"}
        if self.index_type not in valid_index:
            raise ValueError(f"index_type must be one of {valid_index}")
        if self.metric_type not in valid_metric:
            raise ValueError(f"metric_type must be one of {valid_metric}")


class MilvusVectorDBStorage(BaseVectorStorage):
    """Milvus-backed vector storage with workspace isolation and ACL filtering."""

    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
        embedding_func: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(global_config, namespace, workspace, embedding_func)
        if not PYMILVUS_AVAILABLE:
            raise ImportError(
                f"pymilvus is required but not installed or incompatible: {_pymilvus_error}. "
                f"Install with: pip install pymilvus>={PYMILVUS_MIN_VERSION}"
            )

        # Config resolution: env vars > kwargs > global_config
        self._uri = (
            os.environ.get("MILVUS_URI")
            or kwargs.get("uri")
            or global_config.get("milvus_uri", "http://localhost:19530")
        )
        self._token = (
            os.environ.get("MILVUS_TOKEN")
            or kwargs.get("token")
            or global_config.get("milvus_token", "")
        )

        idx_cfg = kwargs.get("index_config") or global_config.get("milvus_index_config", {})
        if isinstance(idx_cfg, MilvusIndexConfig):
            self._index_config = idx_cfg
        elif isinstance(idx_cfg, dict):
            self._index_config = MilvusIndexConfig(**idx_cfg)
        else:
            self._index_config = MilvusIndexConfig()

        self._dim = self.embedding_func.embedding_dim
        ws = self._workspace or "default"
        self._collection_name = f"{self._namespace}_{ws}_{self._dim}"
        self._client: MilvusClient | None = None
        self._batch_size = int(kwargs.get("batch_size", global_config.get("milvus_batch_size", 1000)))

    def _build_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self._dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        return CollectionSchema(fields=fields, enable_dynamic_field=False)

    async def initialize(self) -> None:
        connect_params: dict[str, Any] = {"uri": self._uri}
        if self._token:
            connect_params["token"] = self._token
        self._client = MilvusClient(**connect_params)

        if not self._client.has_collection(self._collection_name):
            schema = self._build_schema()
            self._client.create_collection(
                collection_name=self._collection_name,
                schema=schema,
            )
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type=self._index_config.index_type,
                metric_type=self._index_config.metric_type,
                params=self._index_config.params,
            )
            self._client.create_index(
                collection_name=self._collection_name,
                index_params=index_params,
            )
        self._client.load_collection(self._collection_name)
        logger.info("Milvus collection '%s' ready", self._collection_name)

    async def finalize(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    async def upsert(self, data: dict[str, dict]) -> None:
        if not data:
            return
        rows = []
        for id_, fields in data.items():
            vector = fields.get("__vector__")
            if vector is None:
                continue
            meta = {k: v for k, v in fields.items() if k != "__vector__"}
            meta["workspace"] = self._workspace or "default"
            rows.append({"id": id_, "vector": list(vector), "metadata": meta})

        # Batch insert
        for i in range(0, len(rows), self._batch_size):
            batch = rows[i : i + self._batch_size]
            self._client.upsert(collection_name=self._collection_name, data=batch)
        self._client.flush(self._collection_name)

    async def query(
        self, query: str, top_k: int, acl_filter: dict | None = None
    ) -> list[dict]:
        embedding = await self.embedding_func([query])
        vector = list(embedding[0])

        search_params = {"metric_type": self._index_config.metric_type}

        # Build scalar filter for workspace + ACL
        ws = self._workspace or "default"
        filter_expr = f'metadata["workspace"] == "{ws}"'

        if acl_filter:
            user_id = acl_filter.get("user_id", "")
            user_groups = acl_filter.get("user_groups", [])
            acl_parts = ['metadata["acl_public"] == true']
            if user_id:
                acl_parts.append(f'metadata["acl_owner"] == "{user_id}"')
                acl_parts.append(f'array_contains(metadata["acl_visible_to_users"], "{user_id}")')
            for g in user_groups:
                acl_parts.append(f'array_contains(metadata["acl_visible_to_groups"], "{g}")')
            filter_expr += " and (" + " or ".join(acl_parts) + ")"

        results = self._client.search(
            collection_name=self._collection_name,
            data=[vector],
            limit=top_k,
            output_fields=["id", "metadata"],
            search_params=search_params,
            filter=filter_expr,
        )

        output = []
        for hit in results[0]:
            entry = dict(hit["entity"].get("metadata", {}))
            entry["__id__"] = hit["entity"].get("id", hit.get("id"))
            entry["__distance__"] = hit.get("distance", 0.0)
            output.append(entry)
        return output

    async def delete(self, ids: list[str]) -> None:
        if ids:
            self._client.delete(
                collection_name=self._collection_name,
                filter=f'id in {json.dumps(ids)}',
            )

    async def delete_entity(self, entity_name: str) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            filter=f'metadata["entity_name"] == "{entity_name}"',
        )

    async def delete_entity_relation(self, entity_name: str) -> None:
        self._client.delete(
            collection_name=self._collection_name,
            filter=f'metadata["src_id"] == "{entity_name}" or metadata["tgt_id"] == "{entity_name}"',
        )

    async def get_by_id(self, id: str) -> dict | None:
        results = self._client.get(
            collection_name=self._collection_name,
            ids=[id],
            output_fields=["id", "metadata"],
        )
        if not results:
            return None
        entry = dict(results[0].get("metadata", {}))
        entry["__id__"] = results[0]["id"]
        return entry

    async def get_by_ids(self, ids: list[str]) -> list[dict | None]:
        if not ids:
            return []
        results = self._client.get(
            collection_name=self._collection_name,
            ids=ids,
            output_fields=["id", "metadata"],
        )
        found = {}
        for r in results:
            entry = dict(r.get("metadata", {}))
            entry["__id__"] = r["id"]
            found[r["id"]] = entry
        return [found.get(i) for i in ids]

    async def get_vectors_by_ids(self, ids: list[str]) -> list[dict | None]:
        if not ids:
            return []
        results = self._client.get(
            collection_name=self._collection_name,
            ids=ids,
            output_fields=["id", "vector", "metadata"],
        )
        found = {}
        for r in results:
            entry = dict(r.get("metadata", {}))
            entry["__id__"] = r["id"]
            entry["__vector__"] = r.get("vector", [])
            found[r["id"]] = entry
        return [found.get(i) for i in ids]

    async def drop(self) -> None:
        if self._client and self._client.has_collection(self._collection_name):
            self._client.drop_collection(self._collection_name)
