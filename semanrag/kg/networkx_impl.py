"""NetworkX-based graph storage implementation."""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from datetime import datetime
from pathlib import Path

import networkx as nx

from semanrag.base import BaseGraphStorage

logger = logging.getLogger(__name__)


class NetworkXStorage(BaseGraphStorage):
    """Graph storage backed by an in-memory networkx.Graph with JSON persistence."""

    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
    ) -> None:
        super().__init__(global_config, namespace, workspace)
        self._graph = nx.Graph()
        self._community_summaries: dict[str, str] = {}

        working_dir = global_config.get("working_dir", "./data")
        base = Path(working_dir)
        if workspace:
            base = base / workspace
        self._persist_path = base / f"{self.full_namespace.replace('/', '_')}_graph.json"

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _save(self) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "graph": nx.node_link_data(self._graph),
            "community_summaries": self._community_summaries,
        }
        tmp = self._persist_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
        os.replace(tmp, self._persist_path)

    def _load(self) -> None:
        if not self._persist_path.exists():
            self._graph = nx.Graph()
            self._community_summaries = {}
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
            self._graph = nx.node_link_graph(raw["graph"])
            self._community_summaries = raw.get("community_summaries", {})
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Corrupt graph file %s – starting fresh", self._persist_path)
            self._graph = nx.Graph()
            self._community_summaries = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        self._load()

    async def finalize(self) -> None:
        self._save()

    async def drop(self) -> None:
        self._graph.clear()
        self._community_summaries.clear()
        if self._persist_path.exists():
            self._persist_path.unlink()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------
    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def get_node(self, node_id: str) -> dict | None:
        if not self._graph.has_node(node_id):
            return None
        return dict(self._graph.nodes[node_id])

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def upsert_node(self, node_id: str, node_data: dict) -> None:
        if self._graph.has_node(node_id):
            self._graph.nodes[node_id].update(node_data)
        else:
            self._graph.add_node(node_id, **node_data)

    async def delete_node(self, node_id: str) -> None:
        if self._graph.has_node(node_id):
            self._graph.remove_node(node_id)

    async def remove_nodes(self, nodes: list[str]) -> None:
        for n in nodes:
            if self._graph.has_node(n):
                self._graph.remove_node(n)

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------
    async def has_edge(self, src: str, tgt: str) -> bool:
        return self._graph.has_edge(src, tgt)

    async def get_edge(self, src: str, tgt: str) -> dict | None:
        if not self._graph.has_edge(src, tgt):
            return None
        return dict(self._graph.edges[src, tgt])

    async def edge_degree(self, src: str, tgt: str) -> int:
        d = 0
        if self._graph.has_node(src):
            d += self._graph.degree(src)
        if self._graph.has_node(tgt):
            d += self._graph.degree(tgt)
        return d

    async def get_node_edges(self, node_id: str) -> list[tuple[str, str]]:
        if not self._graph.has_node(node_id):
            return []
        return list(self._graph.edges(node_id))

    async def upsert_edge(self, src: str, tgt: str, edge_data: dict) -> None:
        # Convert temporal fields to ISO strings
        for key in ("valid_from", "valid_to"):
            val = edge_data.get(key)
            if isinstance(val, datetime):
                edge_data[key] = val.isoformat()

        if self._graph.has_edge(src, tgt):
            self._graph.edges[src, tgt].update(edge_data)
        else:
            self._graph.add_edge(src, tgt, **edge_data)

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        for src, tgt in edges:
            if self._graph.has_edge(src, tgt):
                self._graph.remove_edge(src, tgt)

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------
    async def get_all_labels(self) -> list[str]:
        return sorted(self._graph.nodes())

    async def get_knowledge_graph(self, node_label: str | None, max_depth: int) -> dict:
        if node_label is None or not self._graph.has_node(node_label):
            return {"nodes": [], "edges": []}

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(node_label, 0)])
        visited.add(node_label)

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor in self._graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        sub = self._graph.subgraph(visited)
        nodes = [{"id": n, **sub.nodes[n]} for n in sub.nodes()]
        edges = [{"src": u, "tgt": v, **sub.edges[u, v]} for u, v in sub.edges()]
        return {"nodes": nodes, "edges": edges}

    async def search_labels(self, query: str) -> list[str]:
        q = query.lower()
        return sorted(n for n in self._graph.nodes() if q in str(n).lower())

    async def get_popular_labels(self, top_n: int) -> list[tuple[str, int]]:
        return sorted(self._graph.degree(), key=lambda x: x[1], reverse=True)[:top_n]

    async def get_subgraph_at(self, snapshot_at: datetime) -> dict:
        ts = snapshot_at.isoformat() if isinstance(snapshot_at, datetime) else str(snapshot_at)
        kept_edges = []
        for u, v, data in self._graph.edges(data=True):
            vf = data.get("valid_from")
            vt = data.get("valid_to")
            if vf is not None and vf > ts:
                continue
            if vt is not None and vt < ts:
                continue
            kept_edges.append((u, v, data))

        node_ids: set[str] = set()
        for u, v, _ in kept_edges:
            node_ids.update((u, v))

        nodes = [{"id": n, **self._graph.nodes[n]} for n in node_ids]
        edges = [{"src": u, "tgt": v, **d} for u, v, d in kept_edges]
        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # Community detection
    # ------------------------------------------------------------------
    async def detect_communities(self, algorithm: str = "leiden", levels: int = 3) -> dict:
        try:
            from graspologic.partition import hierarchical_leiden
        except ImportError:
            logger.warning("graspologic not installed – skipping community detection")
            return {}

        if self._graph.number_of_nodes() == 0:
            return {}

        results = hierarchical_leiden(self._graph, max_cluster_size=len(self._graph.nodes()) + 1)

        communities: dict[str, list[str]] = {}
        for node_id in self._graph.nodes():
            self._graph.nodes[node_id].pop("community", None)

        for entry in results:
            cid = str(entry.cluster)
            node = entry.node
            communities.setdefault(cid, []).append(node)
            if self._graph.has_node(node):
                self._graph.nodes[node]["community"] = cid

        return communities

    async def get_community_summary(self, community_id: str) -> str | None:
        return self._community_summaries.get(community_id)
