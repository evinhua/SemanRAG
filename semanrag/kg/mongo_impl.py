"""MongoDB-based storage implementations using motor (async driver)."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any

from semanrag.base import (
    ACLPolicy,
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocStatus,
    DocStatusStorage,
)

logger = logging.getLogger(__name__)

try:
    from motor.motor_asyncio import AsyncIOMotorClient
except ImportError:
    AsyncIOMotorClient = None  # type: ignore[assignment,misc]


def _get_motor_client(global_config: dict) -> Any:
    if AsyncIOMotorClient is None:
        raise ImportError(
            "motor is required but not installed. "
            "Install it with: pip install motor"
        )
    mongo_cfg = global_config.get("mongo", {})
    uri = mongo_cfg.get("uri", "mongodb://localhost:27017")
    return AsyncIOMotorClient(uri)


def _get_db(global_config: dict) -> Any:
    client = _get_motor_client(global_config)
    db_name = global_config.get("mongo", {}).get("database", "semanrag")
    return client[db_name]


# =========================================================================
# MongoKVStorage
# =========================================================================
class MongoKVStorage(BaseKVStorage):
    """Key-value storage backed by MongoDB with one collection per namespace."""

    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
    ) -> None:
        super().__init__(global_config, namespace, workspace)
        self._db = _get_db(global_config)
        self._ws = workspace or "default"
        self._col_name = f"kv_{namespace}"
        self._col: Any = None

    async def initialize(self) -> None:
        self._col = self._db[self._col_name]
        await self._col.create_index("workspace")

    async def finalize(self) -> None:
        pass

    async def get_by_id(self, id: str) -> dict | None:
        doc = await self._col.find_one({"_id": id, "workspace": self._ws})
        if doc:
            doc.pop("_id", None)
            doc.pop("workspace", None)
        return doc

    async def get_by_ids(self, ids: list[str]) -> list[dict | None]:
        cursor = self._col.find({"_id": {"$in": ids}, "workspace": self._ws})
        found: dict[str, dict] = {}
        async for doc in cursor:
            did = doc.pop("_id")
            doc.pop("workspace", None)
            found[did] = doc
        return [found.get(i) for i in ids]

    async def filter_keys(self, data: set[str]) -> set[str]:
        cursor = self._col.find(
            {"_id": {"$in": list(data)}, "workspace": self._ws},
            projection={"_id": 1},
        )
        existing = {doc["_id"] async for doc in cursor}
        return data & existing

    async def upsert(self, data: dict[str, dict]) -> None:
        from pymongo import UpdateOne

        ops = [
            UpdateOne(
                {"_id": k, "workspace": self._ws},
                {"$set": {**v, "workspace": self._ws}},
                upsert=True,
            )
            for k, v in data.items()
        ]
        if ops:
            await self._col.bulk_write(ops, ordered=False)

    async def delete(self, ids: list[str]) -> None:
        await self._col.delete_many({"_id": {"$in": ids}, "workspace": self._ws})

    async def drop(self) -> None:
        await self._col.delete_many({"workspace": self._ws})

    async def index_done_callback(self) -> None:
        pass


# =========================================================================
# MongoVectorDBStorage
# =========================================================================
class MongoVectorDBStorage(BaseVectorStorage):
    """Vector storage using MongoDB Atlas Search $vectorSearch."""

    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
        embedding_func: Any = None,
    ) -> None:
        super().__init__(global_config, namespace, workspace, embedding_func)
        self._db = _get_db(global_config)
        self._ws = workspace or "default"
        self._col_name = f"vec_{namespace}"
        self._col: Any = None
        self._index_name = f"vec_idx_{namespace}"

    async def initialize(self) -> None:
        self._col = self._db[self._col_name]
        await self._col.create_index("workspace")
        await self._col.create_index("entity_name")

    async def finalize(self) -> None:
        pass

    async def upsert(self, data: dict[str, dict]) -> None:
        from pymongo import UpdateOne

        ops = []
        for id_, fields in data.items():
            doc = {"workspace": self._ws}
            for k, v in fields.items():
                if k == "__vector__":
                    doc["embedding"] = list(v) if not isinstance(v, list) else v
                else:
                    doc[k] = v
            ops.append(
                UpdateOne({"_id": id_, "workspace": self._ws}, {"$set": doc}, upsert=True)
            )
        if ops:
            await self._col.bulk_write(ops, ordered=False)

    async def query(
        self, query: str, top_k: int, acl_filter: dict | None = None
    ) -> list[dict]:
        embedding = await self.embedding_func([query])
        vector = list(embedding[0])

        pipeline: list[dict] = [
            {
                "$vectorSearch": {
                    "index": self._index_name,
                    "path": "embedding",
                    "queryVector": vector,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                    "filter": {"workspace": self._ws},
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        ]

        if acl_filter:
            uid = acl_filter.get("user_id", "")
            groups = acl_filter.get("user_groups", [])
            pipeline.append(
                {
                    "$match": {
                        "$or": [
                            {"acl_public": True},
                            {"acl_owner": uid},
                            {"acl_visible_to_users": uid},
                            {"acl_visible_to_groups": {"$in": groups}},
                        ]
                    }
                }
            )

        results = []
        async for doc in self._col.aggregate(pipeline):
            doc_id = doc.pop("_id", None)
            doc.pop("embedding", None)
            doc.pop("workspace", None)
            if doc_id is not None:
                doc["__id__"] = doc_id
            results.append(doc)
        return results

    async def delete(self, ids: list[str]) -> None:
        await self._col.delete_many({"_id": {"$in": ids}, "workspace": self._ws})

    async def delete_entity(self, entity_name: str) -> None:
        await self._col.delete_many({"entity_name": entity_name, "workspace": self._ws})

    async def delete_entity_relation(self, entity_name: str) -> None:
        await self._col.delete_many(
            {
                "$or": [{"src_id": entity_name}, {"tgt_id": entity_name}],
                "workspace": self._ws,
            }
        )

    async def get_by_id(self, id: str) -> dict | None:
        doc = await self._col.find_one({"_id": id, "workspace": self._ws})
        if doc:
            doc["__id__"] = doc.pop("_id")
            doc.pop("embedding", None)
            doc.pop("workspace", None)
        return doc

    async def get_by_ids(self, ids: list[str]) -> list[dict | None]:
        cursor = self._col.find({"_id": {"$in": ids}, "workspace": self._ws})
        found: dict[str, dict] = {}
        async for doc in cursor:
            did = doc.pop("_id")
            doc.pop("embedding", None)
            doc.pop("workspace", None)
            doc["__id__"] = did
            found[did] = doc
        return [found.get(i) for i in ids]

    async def get_vectors_by_ids(self, ids: list[str]) -> list[dict | None]:
        cursor = self._col.find({"_id": {"$in": ids}, "workspace": self._ws})
        found: dict[str, dict] = {}
        async for doc in cursor:
            did = doc.pop("_id")
            doc.pop("workspace", None)
            doc["__id__"] = did
            if "embedding" in doc:
                doc["__vector__"] = doc.pop("embedding")
            found[did] = doc
        return [found.get(i) for i in ids]

    async def drop(self) -> None:
        await self._col.delete_many({"workspace": self._ws})


# =========================================================================
# MongoGraphStorage
# =========================================================================
class MongoGraphStorage(BaseGraphStorage):
    """Graph storage using MongoDB nodes/edges collections with $graphLookup."""

    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
    ) -> None:
        super().__init__(global_config, namespace, workspace)
        self._db = _get_db(global_config)
        self._ws = workspace or "default"
        self._nodes: Any = None
        self._edges: Any = None
        self._community_summaries: dict[str, str] = {}

    async def initialize(self) -> None:
        self._nodes = self._db[f"graph_nodes_{self._namespace}"]
        self._edges = self._db[f"graph_edges_{self._namespace}"]
        await self._nodes.create_index([("name", 1), ("workspace", 1)], unique=True)
        await self._edges.create_index([("src", 1), ("tgt", 1), ("workspace", 1)])
        await self._edges.create_index("src")
        await self._edges.create_index("tgt")

    async def finalize(self) -> None:
        pass

    async def has_node(self, node_id: str) -> bool:
        return await self._nodes.count_documents(
            {"name": node_id, "workspace": self._ws}, limit=1
        ) > 0

    async def has_edge(self, src: str, tgt: str) -> bool:
        return await self._edges.count_documents(
            {"src": src, "tgt": tgt, "workspace": self._ws}, limit=1
        ) > 0

    async def node_degree(self, node_id: str) -> int:
        return await self._edges.count_documents(
            {"$or": [{"src": node_id}, {"tgt": node_id}], "workspace": self._ws}
        )

    async def edge_degree(self, src: str, tgt: str) -> int:
        d1 = await self.node_degree(src)
        d2 = await self.node_degree(tgt)
        return d1 + d2

    async def get_node(self, node_id: str) -> dict | None:
        doc = await self._nodes.find_one({"name": node_id, "workspace": self._ws})
        if doc:
            doc.pop("_id", None)
            doc.pop("workspace", None)
        return doc

    async def get_edge(self, src: str, tgt: str) -> dict | None:
        doc = await self._edges.find_one({"src": src, "tgt": tgt, "workspace": self._ws})
        if doc:
            doc.pop("_id", None)
            doc.pop("workspace", None)
        return doc

    async def get_node_edges(self, node_id: str) -> list[tuple[str, str]]:
        cursor = self._edges.find(
            {"$or": [{"src": node_id}, {"tgt": node_id}], "workspace": self._ws},
            projection={"src": 1, "tgt": 1},
        )
        return [(doc["src"], doc["tgt"]) async for doc in cursor]

    async def upsert_node(self, node_id: str, node_data: dict) -> None:
        await self._nodes.update_one(
            {"name": node_id, "workspace": self._ws},
            {"$set": {**node_data, "name": node_id, "workspace": self._ws}},
            upsert=True,
        )

    async def upsert_edge(self, src: str, tgt: str, edge_data: dict) -> None:
        props = {}
        for k, v in edge_data.items():
            props[k] = v.isoformat() if isinstance(v, datetime) else v
        await self._edges.update_one(
            {"src": src, "tgt": tgt, "workspace": self._ws},
            {"$set": {**props, "src": src, "tgt": tgt, "workspace": self._ws}},
            upsert=True,
        )

    async def delete_node(self, node_id: str) -> None:
        await self._nodes.delete_one({"name": node_id, "workspace": self._ws})
        await self._edges.delete_many(
            {"$or": [{"src": node_id}, {"tgt": node_id}], "workspace": self._ws}
        )

    async def remove_nodes(self, nodes: list[str]) -> None:
        await self._nodes.delete_many({"name": {"$in": nodes}, "workspace": self._ws})
        await self._edges.delete_many(
            {"$or": [{"src": {"$in": nodes}}, {"tgt": {"$in": nodes}}], "workspace": self._ws}
        )

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        if not edges:
            return
        conditions = [{"src": s, "tgt": t} for s, t in edges]
        await self._edges.delete_many({"$or": conditions, "workspace": self._ws})

    async def get_all_labels(self) -> list[str]:
        cursor = self._nodes.find(
            {"workspace": self._ws}, projection={"name": 1}
        ).sort("name", 1)
        return [doc["name"] async for doc in cursor]

    async def get_knowledge_graph(self, node_label: str | None, max_depth: int) -> dict:
        if node_label is None:
            return {"nodes": [], "edges": []}

        pipeline = [
            {"$match": {"name": node_label, "workspace": self._ws}},
            {
                "$graphLookup": {
                    "from": self._edges.name,
                    "startWith": "$name",
                    "connectFromField": "tgt",
                    "connectToField": "src",
                    "as": "reachable",
                    "maxDepth": max_depth - 1,
                    "restrictSearchWithMatch": {"workspace": self._ws},
                }
            },
        ]
        result = await self._nodes.aggregate(pipeline).to_list(1)
        if not result:
            return {"nodes": [], "edges": []}

        reachable_names = {node_label}
        for edge_doc in result[0].get("reachable", []):
            reachable_names.add(edge_doc.get("src", ""))
            reachable_names.add(edge_doc.get("tgt", ""))
        reachable_names.discard("")

        node_cursor = self._nodes.find(
            {"name": {"$in": list(reachable_names)}, "workspace": self._ws}
        )
        nodes = []
        async for doc in node_cursor:
            doc.pop("_id", None)
            doc.pop("workspace", None)
            nodes.append(doc)

        edge_cursor = self._edges.find(
            {
                "src": {"$in": list(reachable_names)},
                "tgt": {"$in": list(reachable_names)},
                "workspace": self._ws,
            }
        )
        edges = []
        async for doc in edge_cursor:
            doc.pop("_id", None)
            doc.pop("workspace", None)
            edges.append(doc)

        return {"nodes": nodes, "edges": edges}

    async def search_labels(self, query: str) -> list[str]:
        q_lower = query.lower()
        cursor = self._nodes.find(
            {
                "workspace": self._ws,
                "name": {"$regex": q_lower, "$options": "i"},
            },
            projection={"name": 1},
        )
        return [doc["name"] async for doc in cursor]

    async def get_popular_labels(self, top_n: int) -> list[tuple[str, int]]:
        pipeline = [
            {"$match": {"workspace": self._ws}},
            {
                "$group": {
                    "_id": "$src",
                    "deg": {"$sum": 1},
                }
            },
            {"$sort": {"deg": -1}},
            {"$limit": top_n},
        ]
        rows = await self._edges.aggregate(pipeline).to_list(top_n)
        return [(r["_id"], r["deg"]) for r in rows]

    async def get_subgraph_at(self, snapshot_at: datetime) -> dict:
        ts = snapshot_at.isoformat() if isinstance(snapshot_at, datetime) else str(snapshot_at)
        edge_cursor = self._edges.find(
            {
                "workspace": self._ws,
                "$and": [
                    {"$or": [{"valid_from": {"$exists": False}}, {"valid_from": {"$lte": ts}}]},
                    {"$or": [{"valid_to": {"$exists": False}}, {"valid_to": {"$gte": ts}}]},
                ],
            }
        )
        node_names: set[str] = set()
        edges = []
        async for doc in edge_cursor:
            doc.pop("_id", None)
            doc.pop("workspace", None)
            node_names.add(doc["src"])
            node_names.add(doc["tgt"])
            edges.append(doc)

        node_cursor = self._nodes.find(
            {"name": {"$in": list(node_names)}, "workspace": self._ws}
        )
        nodes = []
        async for doc in node_cursor:
            doc.pop("_id", None)
            doc.pop("workspace", None)
            nodes.append(doc)

        return {"nodes": nodes, "edges": edges}

    async def detect_communities(self, algorithm: str = "leiden", levels: int = 3) -> dict:
        try:
            from graspologic.partition import hierarchical_leiden
        except ImportError:
            logger.warning("graspologic not installed – skipping community detection")
            return {}

        import networkx as nx

        G = nx.Graph()
        async for doc in self._nodes.find({"workspace": self._ws}, {"name": 1}):
            G.add_node(doc["name"])
        async for doc in self._edges.find({"workspace": self._ws}, {"src": 1, "tgt": 1}):
            G.add_edge(doc["src"], doc["tgt"])

        if G.number_of_nodes() == 0:
            return {}

        results = hierarchical_leiden(G, max_cluster_size=len(G.nodes()) + 1)
        communities: dict[str, list[str]] = {}
        for entry in results:
            cid = str(entry.cluster)
            communities.setdefault(cid, []).append(entry.node)

        from pymongo import UpdateOne

        ops = [
            UpdateOne(
                {"name": node, "workspace": self._ws},
                {"$set": {"community": cid}},
            )
            for cid, members in communities.items()
            for node in members
        ]
        if ops:
            await self._nodes.bulk_write(ops, ordered=False)

        return communities

    async def get_community_summary(self, community_id: str) -> str | None:
        return self._community_summaries.get(community_id)

    async def drop(self) -> None:
        await self._nodes.delete_many({"workspace": self._ws})
        await self._edges.delete_many({"workspace": self._ws})
        self._community_summaries.clear()


# =========================================================================
# MongoDocStatusStorage
# =========================================================================
class MongoDocStatusStorage(DocStatusStorage):
    """Document status storage backed by MongoDB."""

    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
    ) -> None:
        super().__init__(global_config, namespace, workspace)
        self._db = _get_db(global_config)
        self._ws = workspace or "default"
        self._col: Any = None

    async def initialize(self) -> None:
        self._col = self._db[f"doc_status_{self._namespace}"]
        await self._col.create_index("workspace")
        await self._col.create_index([("workspace", 1), ("status", 1)])
        await self._col.create_index([("workspace", 1), ("file_path", 1)])

    async def finalize(self) -> None:
        pass

    @staticmethod
    def _to_doc(doc_id: str, status: DocStatus, workspace: str) -> dict:
        d = asdict(status)
        if status.acl_policy is not None:
            d["acl_policy"] = asdict(status.acl_policy)
        d["_id"] = doc_id
        d["workspace"] = workspace
        return d

    @staticmethod
    def _from_doc(doc: dict) -> DocStatus:
        doc.pop("_id", None)
        doc.pop("workspace", None)
        acl = doc.get("acl_policy")
        if isinstance(acl, dict):
            doc["acl_policy"] = ACLPolicy(**acl)
        else:
            doc["acl_policy"] = None
        return DocStatus(**doc)

    async def get(self, doc_id: str) -> DocStatus | None:
        doc = await self._col.find_one({"_id": doc_id, "workspace": self._ws})
        return self._from_doc(doc) if doc else None

    async def upsert(self, doc_id: str, status: DocStatus) -> None:
        doc = self._to_doc(doc_id, status, self._ws)
        await self._col.replace_one(
            {"_id": doc_id, "workspace": self._ws}, doc, upsert=True
        )

    async def delete(self, doc_id: str) -> None:
        await self._col.delete_one({"_id": doc_id, "workspace": self._ws})

    async def get_status_counts(self) -> dict[str, int]:
        pipeline = [
            {"$match": {"workspace": self._ws}},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]
        rows = await self._col.aggregate(pipeline).to_list(100)
        return {r["_id"]: r["count"] for r in rows}

    async def get_all_status_counts(self) -> dict[str, int]:
        return await self.get_status_counts()

    async def get_docs_by_status(self, status: str) -> list[DocStatus]:
        cursor = self._col.find({"workspace": self._ws, "status": status})
        return [self._from_doc(doc) async for doc in cursor]

    async def get_docs_paginated(
        self,
        offset: int,
        limit: int,
        status: str | None = None,
        acl_filter: dict | None = None,
    ) -> tuple[list[DocStatus], int]:
        query: dict[str, Any] = {"workspace": self._ws}
        if status is not None:
            query["status"] = status
        if acl_filter:
            uid = acl_filter.get("user_id", "")
            groups = acl_filter.get("user_groups", [])
            query["$or"] = [
                {"acl_policy": None},
                {"acl_policy.public": True},
                {"acl_policy.owner": uid},
                {"acl_policy.visible_to_users": uid},
                {"acl_policy.visible_to_groups": {"$in": groups}},
            ]
        total = await self._col.count_documents(query)
        cursor = self._col.find(query).skip(offset).limit(limit)
        docs = [self._from_doc(doc) async for doc in cursor]
        return docs, total

    async def get_doc_by_file_path(self, file_path: str) -> DocStatus | None:
        doc = await self._col.find_one({"workspace": self._ws, "file_path": file_path})
        return self._from_doc(doc) if doc else None
