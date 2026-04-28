"""OpenSearch-backed storage implementations for KV, Vector, Graph, DocStatus, and Lexical."""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict
from datetime import datetime
from typing import Any

from semanrag.base import (
    ACLPolicy,
    BaseGraphStorage,
    BaseKVStorage,
    BaseLexicalStorage,
    BaseVectorStorage,
    DocStatus,
    DocStatusStorage,
)

logger = logging.getLogger(__name__)

try:
    from opensearchpy import AsyncOpenSearch, NotFoundError, RequestError

    OPENSEARCH_AVAILABLE = True
except ImportError:
    AsyncOpenSearch = None  # type: ignore[assignment,misc]
    NotFoundError = Exception  # type: ignore[assignment,misc]
    RequestError = Exception  # type: ignore[assignment,misc]
    OPENSEARCH_AVAILABLE = False
    logger.warning("opensearch-py is not installed. OpenSearch storage backends will not work.")


def _get_client(global_config: dict) -> Any:
    """Extract or create an AsyncOpenSearch client from global_config."""
    client = global_config.get("opensearch_client")
    if client is not None:
        return client
    hosts = global_config.get("opensearch_hosts", [{"host": "localhost", "port": 9200}])
    auth = global_config.get("opensearch_auth")
    use_ssl = global_config.get("opensearch_use_ssl", False)
    verify_certs = global_config.get("opensearch_verify_certs", False)
    kwargs: dict[str, Any] = {"hosts": hosts, "use_ssl": use_ssl, "verify_certs": verify_certs}
    if auth:
        kwargs["http_auth"] = auth
    return AsyncOpenSearch(**kwargs)


def _safe_index_name(name: str) -> str:
    return name.lower().replace("/", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# OSKVStorage
# ---------------------------------------------------------------------------
class OSKVStorage(BaseKVStorage):
    """Key-value storage backed by an OpenSearch index (one doc per key)."""

    def __init__(self, global_config: dict, namespace: str, workspace: str | None = None) -> None:
        if not OPENSEARCH_AVAILABLE:
            raise ImportError("opensearch-py is required but not installed.")
        super().__init__(global_config, namespace, workspace)
        self._client = _get_client(global_config)
        self._index = _safe_index_name(f"semanrag_kv_{self.full_namespace}")

    async def initialize(self) -> None:
        if not await self._client.indices.exists(index=self._index):
            await self._client.indices.create(index=self._index, body={"settings": {"number_of_shards": 1}})

    async def finalize(self) -> None:
        await self._client.indices.refresh(index=self._index)

    async def get_by_id(self, id: str) -> dict | None:
        try:
            resp = await self._client.get(index=self._index, id=id)
            return resp["_source"]
        except NotFoundError:
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict | None]:
        if not ids:
            return []
        body = {"docs": [{"_index": self._index, "_id": i} for i in ids]}
        resp = await self._client.mget(body=body)
        return [doc["_source"] if doc.get("found") else None for doc in resp["docs"]]

    async def filter_keys(self, data: set[str]) -> set[str]:
        if not data:
            return set()
        body = {"query": {"ids": {"values": list(data)}}, "_source": False, "size": len(data)}
        resp = await self._client.search(index=self._index, body=body)
        return {hit["_id"] for hit in resp["hits"]["hits"]}

    async def upsert(self, data: dict[str, dict]) -> None:
        if not data:
            return
        actions = []
        for k, v in data.items():
            actions.append(json.dumps({"index": {"_index": self._index, "_id": k}}))
            actions.append(json.dumps(v, default=str))
        body = "\n".join(actions) + "\n"
        await self._client.bulk(body=body)

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        actions = [json.dumps({"delete": {"_index": self._index, "_id": i}}) for i in ids]
        await self._client.bulk(body="\n".join(actions) + "\n")

    async def drop(self) -> None:
        if await self._client.indices.exists(index=self._index):
            await self._client.indices.delete(index=self._index)

    async def index_done_callback(self) -> None:
        await self._client.indices.refresh(index=self._index)


# ---------------------------------------------------------------------------
# OSVectorStorage
# ---------------------------------------------------------------------------
class OSVectorStorage(BaseVectorStorage):
    """Vector storage backed by OpenSearch k-NN plugin."""

    def __init__(self, global_config: dict, namespace: str, workspace: str | None = None, embedding_func: Any = None) -> None:
        if not OPENSEARCH_AVAILABLE:
            raise ImportError("opensearch-py is required but not installed.")
        super().__init__(global_config, namespace, workspace, embedding_func)
        self._client = _get_client(global_config)
        self._index = _safe_index_name(f"semanrag_vec_{self.full_namespace}")
        self._engine = global_config.get("opensearch_knn_engine", "nmslib")

    async def initialize(self) -> None:
        if await self._client.indices.exists(index=self._index):
            return
        dim = self.embedding_func.embedding_dim
        body = {
            "settings": {"index": {"knn": True, "number_of_shards": 1}},
            "mappings": {
                "properties": {
                    "vector": {"type": "knn_vector", "dimension": dim, "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": self._engine}},
                    "entity_name": {"type": "keyword"},
                    "src_id": {"type": "keyword"},
                    "tgt_id": {"type": "keyword"},
                    "workspace": {"type": "keyword"},
                }
            },
        }
        await self._client.indices.create(index=self._index, body=body)

    async def finalize(self) -> None:
        await self._client.indices.refresh(index=self._index)

    async def upsert(self, data: dict[str, dict]) -> None:
        if not data:
            return
        actions = []
        for id_, fields in data.items():
            doc = {k: v for k, v in fields.items() if k != "__vector__"}
            if "__vector__" in fields:
                doc["vector"] = fields["__vector__"]
            if self._workspace:
                doc["workspace"] = self._workspace
            actions.append(json.dumps({"index": {"_index": self._index, "_id": id_}}))
            actions.append(json.dumps(doc, default=str))
        await self._client.bulk(body="\n".join(actions) + "\n")

    async def query(self, query: str, top_k: int, acl_filter: dict | None = None) -> list[dict]:
        embedding = await self.embedding_func([query])
        vector = embedding[0]
        body: dict[str, Any] = {"size": top_k, "query": {"knn": {"vector": {"vector": vector, "k": top_k}}}}
        resp = await self._client.search(index=self._index, body=body)
        results = []
        for hit in resp["hits"]["hits"]:
            doc = hit["_source"]
            doc["__id__"] = hit["_id"]
            doc["__score__"] = hit["_score"]
            results.append(doc)
        if acl_filter:
            user_id = acl_filter.get("user_id")
            user_groups = acl_filter.get("user_groups", [])
            filtered = []
            for r in results:
                if r.get("acl_public", True) or user_id and (r.get("acl_owner") == user_id or user_id in r.get("acl_visible_to_users", [])) or user_groups and set(user_groups) & set(r.get("acl_visible_to_groups", [])):
                    filtered.append(r)
            results = filtered
        return results

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        actions = [json.dumps({"delete": {"_index": self._index, "_id": i}}) for i in ids]
        await self._client.bulk(body="\n".join(actions) + "\n")

    async def delete_entity(self, entity_name: str) -> None:
        await self._client.delete_by_query(index=self._index, body={"query": {"term": {"entity_name": entity_name}}})

    async def delete_entity_relation(self, entity_name: str) -> None:
        await self._client.delete_by_query(
            index=self._index,
            body={"query": {"bool": {"should": [{"term": {"src_id": entity_name}}, {"term": {"tgt_id": entity_name}}]}}},
        )

    async def get_by_id(self, id: str) -> dict | None:
        try:
            resp = await self._client.get(index=self._index, id=id)
            doc = resp["_source"]
            doc["__id__"] = resp["_id"]
            return doc
        except NotFoundError:
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict | None]:
        if not ids:
            return []
        body = {"docs": [{"_index": self._index, "_id": i} for i in ids]}
        resp = await self._client.mget(body=body)
        out: list[dict | None] = []
        for doc in resp["docs"]:
            if doc.get("found"):
                d = doc["_source"]
                d["__id__"] = doc["_id"]
                out.append(d)
            else:
                out.append(None)
        return out

    async def get_vectors_by_ids(self, ids: list[str]) -> list[dict | None]:
        return await self.get_by_ids(ids)

    async def drop(self) -> None:
        if await self._client.indices.exists(index=self._index):
            await self._client.indices.delete(index=self._index)


# ---------------------------------------------------------------------------
# OSGraphStorage
# ---------------------------------------------------------------------------
class OSGraphStorage(BaseGraphStorage):
    """Graph storage using two OpenSearch indices (nodes + edges) with client-side BFS."""

    def __init__(self, global_config: dict, namespace: str, workspace: str | None = None) -> None:
        if not OPENSEARCH_AVAILABLE:
            raise ImportError("opensearch-py is required but not installed.")
        super().__init__(global_config, namespace, workspace)
        self._client = _get_client(global_config)
        prefix = _safe_index_name(f"semanrag_graph_{self.full_namespace}")
        self._nodes_index = f"{prefix}_nodes"
        self._edges_index = f"{prefix}_edges"

    async def initialize(self) -> None:
        if not await self._client.indices.exists(index=self._nodes_index):
            await self._client.indices.create(
                index=self._nodes_index,
                body={"settings": {"number_of_shards": 1}, "mappings": {"properties": {"node_id": {"type": "keyword"}, "data": {"type": "object", "enabled": False}}}},
            )
        if not await self._client.indices.exists(index=self._edges_index):
            await self._client.indices.create(
                index=self._edges_index,
                body={
                    "settings": {"number_of_shards": 1},
                    "mappings": {"properties": {"src": {"type": "keyword"}, "tgt": {"type": "keyword"}, "data": {"type": "object", "enabled": False}}},
                },
            )

    async def finalize(self) -> None:
        for idx in (self._nodes_index, self._edges_index):
            if await self._client.indices.exists(index=idx):
                await self._client.indices.refresh(index=idx)

    async def drop(self) -> None:
        for idx in (self._nodes_index, self._edges_index):
            if await self._client.indices.exists(index=idx):
                await self._client.indices.delete(index=idx)

    # -- node ops --
    async def has_node(self, node_id: str) -> bool:
        try:
            await self._client.get(index=self._nodes_index, id=node_id)
            return True
        except NotFoundError:
            return False

    async def get_node(self, node_id: str) -> dict | None:
        try:
            resp = await self._client.get(index=self._nodes_index, id=node_id)
            return resp["_source"].get("data", {})
        except NotFoundError:
            return None

    async def node_degree(self, node_id: str) -> int:
        body = {"query": {"bool": {"should": [{"term": {"src": node_id}}, {"term": {"tgt": node_id}}]}}, "track_total_hits": True}
        resp = await self._client.search(index=self._edges_index, body=body, size=0)
        return resp["hits"]["total"]["value"]

    async def upsert_node(self, node_id: str, node_data: dict) -> None:
        await self._client.index(index=self._nodes_index, id=node_id, body={"node_id": node_id, "data": node_data})

    async def delete_node(self, node_id: str) -> None:
        try:
            await self._client.delete(index=self._nodes_index, id=node_id)
        except NotFoundError:
            pass
        await self._client.delete_by_query(
            index=self._edges_index,
            body={"query": {"bool": {"should": [{"term": {"src": node_id}}, {"term": {"tgt": node_id}}]}}},
        )

    async def remove_nodes(self, nodes: list[str]) -> None:
        for n in nodes:
            await self.delete_node(n)

    # -- edge ops --
    async def has_edge(self, src: str, tgt: str) -> bool:
        edge_id = f"{src}||{tgt}"
        try:
            await self._client.get(index=self._edges_index, id=edge_id)
            return True
        except NotFoundError:
            return False

    async def get_edge(self, src: str, tgt: str) -> dict | None:
        edge_id = f"{src}||{tgt}"
        try:
            resp = await self._client.get(index=self._edges_index, id=edge_id)
            return resp["_source"].get("data", {})
        except NotFoundError:
            return None

    async def edge_degree(self, src: str, tgt: str) -> int:
        return await self.node_degree(src) + await self.node_degree(tgt)

    async def get_node_edges(self, node_id: str) -> list[tuple[str, str]]:
        body = {"query": {"bool": {"should": [{"term": {"src": node_id}}, {"term": {"tgt": node_id}}]}}, "size": 10000}
        resp = await self._client.search(index=self._edges_index, body=body)
        return [(hit["_source"]["src"], hit["_source"]["tgt"]) for hit in resp["hits"]["hits"]]

    async def upsert_edge(self, src: str, tgt: str, edge_data: dict) -> None:
        for key in ("valid_from", "valid_to"):
            val = edge_data.get(key)
            if isinstance(val, datetime):
                edge_data[key] = val.isoformat()
        edge_id = f"{src}||{tgt}"
        await self._client.index(index=self._edges_index, id=edge_id, body={"src": src, "tgt": tgt, "data": edge_data})

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        if not edges:
            return
        actions = [json.dumps({"delete": {"_index": self._edges_index, "_id": f"{s}||{t}"}}) for s, t in edges]
        await self._client.bulk(body="\n".join(actions) + "\n")

    # -- graph queries (client-side BFS) --
    async def get_all_labels(self) -> list[str]:
        body = {"query": {"match_all": {}}, "_source": ["node_id"], "size": 10000}
        resp = await self._client.search(index=self._nodes_index, body=body)
        return sorted(hit["_source"]["node_id"] for hit in resp["hits"]["hits"])

    async def get_knowledge_graph(self, node_label: str | None, max_depth: int) -> dict:
        if node_label is None or not await self.has_node(node_label):
            return {"nodes": [], "edges": []}
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(node_label, 0)])
        visited.add(node_label)
        collected_edges: list[tuple[str, str]] = []
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            edges = await self.get_node_edges(current)
            for s, t in edges:
                collected_edges.append((s, t))
                neighbor = t if s == current else s
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        nodes = []
        for nid in visited:
            data = await self.get_node(nid) or {}
            nodes.append({"id": nid, **data})
        edge_set = set()
        edges_out = []
        for s, t in collected_edges:
            key = (s, t)
            if key not in edge_set:
                edge_set.add(key)
                data = await self.get_edge(s, t) or {}
                edges_out.append({"src": s, "tgt": t, **data})
        return {"nodes": nodes, "edges": edges_out}

    async def search_labels(self, query: str) -> list[str]:
        body = {"query": {"wildcard": {"node_id": {"value": f"*{query.lower()}*"}}}, "size": 100, "_source": ["node_id"]}
        resp = await self._client.search(index=self._nodes_index, body=body)
        return sorted(hit["_source"]["node_id"] for hit in resp["hits"]["hits"])

    async def get_popular_labels(self, top_n: int) -> list[tuple[str, int]]:
        labels = await self.get_all_labels()
        results = []
        for label in labels:
            deg = await self.node_degree(label)
            results.append((label, deg))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    async def get_subgraph_at(self, snapshot_at: datetime) -> dict:
        ts = snapshot_at.isoformat() if isinstance(snapshot_at, datetime) else str(snapshot_at)
        body = {"query": {"match_all": {}}, "size": 10000}
        resp = await self._client.search(index=self._edges_index, body=body)
        kept_edges = []
        node_ids: set[str] = set()
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            data = src.get("data", {})
            vf = data.get("valid_from")
            vt = data.get("valid_to")
            if vf is not None and vf > ts:
                continue
            if vt is not None and vt < ts:
                continue
            kept_edges.append({"src": src["src"], "tgt": src["tgt"], **data})
            node_ids.update((src["src"], src["tgt"]))
        nodes = []
        for nid in node_ids:
            nd = await self.get_node(nid) or {}
            nodes.append({"id": nid, **nd})
        return {"nodes": nodes, "edges": kept_edges}

    async def detect_communities(self, algorithm: str = "leiden", levels: int = 3) -> dict:
        try:
            import networkx as nx
            from graspologic.partition import hierarchical_leiden
        except ImportError:
            logger.warning("networkx/graspologic not installed – skipping community detection")
            return {}
        labels = await self.get_all_labels()
        if not labels:
            return {}
        G = nx.Graph()
        G.add_nodes_from(labels)
        body = {"query": {"match_all": {}}, "size": 10000}
        resp = await self._client.search(index=self._edges_index, body=body)
        for hit in resp["hits"]["hits"]:
            G.add_edge(hit["_source"]["src"], hit["_source"]["tgt"])
        results = hierarchical_leiden(G, max_cluster_size=len(G.nodes()) + 1)
        communities: dict[str, list[str]] = {}
        for entry in results:
            cid = str(entry.cluster)
            communities.setdefault(cid, []).append(entry.node)
        return communities

    async def get_community_summary(self, community_id: str) -> str | None:
        try:
            resp = await self._client.get(index=self._nodes_index, id=f"__community_summary__{community_id}")
            return resp["_source"].get("data", {}).get("summary")
        except NotFoundError:
            return None


# ---------------------------------------------------------------------------
# OSDocStatusStorage
# ---------------------------------------------------------------------------
class OSDocStatusStorage(DocStatusStorage):
    """Document status storage backed by OpenSearch."""

    def __init__(self, global_config: dict, namespace: str, workspace: str | None = None) -> None:
        if not OPENSEARCH_AVAILABLE:
            raise ImportError("opensearch-py is required but not installed.")
        super().__init__(global_config, namespace, workspace)
        self._client = _get_client(global_config)
        self._index = _safe_index_name(f"semanrag_docstatus_{self.full_namespace}")

    async def initialize(self) -> None:
        if not await self._client.indices.exists(index=self._index):
            await self._client.indices.create(
                index=self._index,
                body={
                    "settings": {"number_of_shards": 1},
                    "mappings": {"properties": {"status": {"type": "keyword"}, "file_path": {"type": "keyword"}, "created_at": {"type": "keyword"}, "updated_at": {"type": "keyword"}}},
                },
            )

    async def finalize(self) -> None:
        await self._client.indices.refresh(index=self._index)

    @staticmethod
    def _to_dict(status: DocStatus) -> dict:
        d = asdict(status)
        if status.acl_policy is not None:
            d["acl_policy"] = asdict(status.acl_policy)
        return d

    @staticmethod
    def _from_dict(d: dict) -> DocStatus:
        acl = d.get("acl_policy")
        if isinstance(acl, dict):
            d = {**d, "acl_policy": ACLPolicy(**acl)}
        else:
            d = {**d, "acl_policy": None}
        return DocStatus(**d)

    async def get(self, doc_id: str) -> DocStatus | None:
        try:
            resp = await self._client.get(index=self._index, id=doc_id)
            return self._from_dict(resp["_source"])
        except NotFoundError:
            return None

    async def upsert(self, doc_id: str, status: DocStatus) -> None:
        await self._client.index(index=self._index, id=doc_id, body=self._to_dict(status), refresh="true")

    async def delete(self, doc_id: str) -> None:
        try:
            await self._client.delete(index=self._index, id=doc_id, refresh="true")
        except NotFoundError:
            pass

    async def get_status_counts(self) -> dict[str, int]:
        body = {"size": 0, "aggs": {"statuses": {"terms": {"field": "status", "size": 100}}}}
        resp = await self._client.search(index=self._index, body=body)
        return {b["key"]: b["doc_count"] for b in resp["aggregations"]["statuses"]["buckets"]}

    async def get_all_status_counts(self) -> dict[str, int]:
        return await self.get_status_counts()

    async def get_docs_by_status(self, status: str) -> list[DocStatus]:
        body = {"query": {"term": {"status": status}}, "size": 10000}
        resp = await self._client.search(index=self._index, body=body)
        return [self._from_dict(hit["_source"]) for hit in resp["hits"]["hits"]]

    async def get_docs_paginated(self, offset: int, limit: int, status: str | None = None, acl_filter: dict | None = None) -> tuple[list[DocStatus], int]:
        query: dict[str, Any] = {"match_all": {}}
        if status is not None:
            query = {"term": {"status": status}}
        body: dict[str, Any] = {"query": query, "from": offset, "size": limit, "track_total_hits": True}
        resp = await self._client.search(index=self._index, body=body)
        total = resp["hits"]["total"]["value"]
        docs = [self._from_dict(hit["_source"]) for hit in resp["hits"]["hits"]]
        if acl_filter:
            uid = acl_filter.get("user_id", "")
            groups = acl_filter.get("user_groups", [])
            docs = [d for d in docs if d.acl_policy is None or d.acl_policy.can_access(uid, groups)]
        return docs, total

    async def get_doc_by_file_path(self, file_path: str) -> DocStatus | None:
        body = {"query": {"term": {"file_path": file_path}}, "size": 1}
        resp = await self._client.search(index=self._index, body=body)
        hits = resp["hits"]["hits"]
        return self._from_dict(hits[0]["_source"]) if hits else None


# ---------------------------------------------------------------------------
# OSLexicalStorage
# ---------------------------------------------------------------------------
class OSLexicalStorage(BaseLexicalStorage):
    """BM25-like lexical search backed by OpenSearch standard analyzer."""

    def __init__(self, global_config: dict, namespace: str, workspace: str | None = None) -> None:
        if not OPENSEARCH_AVAILABLE:
            raise ImportError("opensearch-py is required but not installed.")
        super().__init__(global_config, namespace, workspace)
        self._client = _get_client(global_config)
        self._index = _safe_index_name(f"semanrag_lex_{self.full_namespace}")

    async def initialize(self) -> None:
        if not await self._client.indices.exists(index=self._index):
            await self._client.indices.create(
                index=self._index,
                body={
                    "settings": {"number_of_shards": 1, "analysis": {"analyzer": {"default": {"type": "standard"}}}},
                    "mappings": {"properties": {"content": {"type": "text", "analyzer": "standard"}, "workspace": {"type": "keyword"}}},
                },
            )

    async def finalize(self) -> None:
        await self._client.indices.refresh(index=self._index)

    async def upsert(self, data: dict[str, dict]) -> None:
        if not data:
            return
        actions = []
        for doc_id, doc in data.items():
            body = {**doc}
            if self._workspace:
                body["workspace"] = self._workspace
            actions.append(json.dumps({"index": {"_index": self._index, "_id": doc_id}}))
            actions.append(json.dumps(body, default=str))
        await self._client.bulk(body="\n".join(actions) + "\n")
        await self._client.indices.refresh(index=self._index)

    async def search_bm25(self, query: str, top_k: int) -> list[dict]:
        body: dict[str, Any] = {"query": {"match": {"content": query}}, "size": top_k}
        resp = await self._client.search(index=self._index, body=body)
        results = []
        for hit in resp["hits"]["hits"]:
            doc = hit["_source"]
            doc["id"] = hit["_id"]
            doc["score"] = hit["_score"]
            results.append(doc)
        return results

    async def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        actions = [json.dumps({"delete": {"_index": self._index, "_id": i}}) for i in ids]
        await self._client.bulk(body="\n".join(actions) + "\n")

    async def drop(self) -> None:
        if await self._client.indices.exists(index=self._index):
            await self._client.indices.delete(index=self._index)
