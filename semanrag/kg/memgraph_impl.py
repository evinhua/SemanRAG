"""Memgraph-backed graph storage using the Neo4j Bolt driver."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from semanrag.base import BaseGraphStorage

logger = logging.getLogger(__name__)

try:
    from neo4j import AsyncGraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    AsyncGraphDatabase = None  # type: ignore[assignment,misc]
    NEO4J_AVAILABLE = False
    logger.warning("neo4j driver is not installed. MemgraphStorage will not work.")


class MemgraphStorage(BaseGraphStorage):
    """Graph storage backed by Memgraph via the Neo4j Bolt protocol.

    Workspace isolation is achieved by labelling every node with a
    workspace-specific label (e.g. ``ws_<workspace>``).
    """

    def __init__(self, global_config: dict, namespace: str, workspace: str | None = None) -> None:
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j driver is required but not installed. pip install neo4j")
        super().__init__(global_config, namespace, workspace)
        host = global_config.get("memgraph_host", global_config.get("neo4j_host", "localhost"))
        port = global_config.get("memgraph_port", global_config.get("neo4j_port", 7687))
        user = global_config.get("memgraph_user", global_config.get("neo4j_user", ""))
        password = global_config.get("memgraph_password", global_config.get("neo4j_password", ""))
        uri = f"bolt://{host}:{port}"
        auth = (user, password) if user else None
        self._driver = AsyncGraphDatabase.driver(uri, auth=auth)
        # Workspace label for isolation
        safe_ws = (workspace or "default").replace("-", "_").replace(" ", "_")
        self._ws_label = f"ws_{safe_ws}"
        self._ns_label = f"ns_{namespace.replace('-', '_').replace(' ', '_')}"

    @property
    def _label(self) -> str:
        return f"{self._ws_label}:{self._ns_label}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _run(self, query: str, **params: Any) -> list[dict]:
        async with self._driver.session() as session:
            result = await session.run(query, **params)
            return [record.data() async for record in result]

    async def _run_single(self, query: str, **params: Any) -> dict | None:
        rows = await self._run(query, **params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        # Memgraph uses CREATE INDEX ON :Label(property) syntax
        try:
            await self._run(f"CREATE INDEX ON :{self._ns_label}(node_id)")
        except Exception:
            pass  # Index may already exist

    async def finalize(self) -> None:
        await self._driver.close()

    async def drop(self) -> None:
        await self._run(f"MATCH (n:{self._label}) DETACH DELETE n")

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------
    async def has_node(self, node_id: str) -> bool:
        row = await self._run_single(
            f"MATCH (n:{self._label} {{node_id: $nid}}) RETURN n LIMIT 1", nid=node_id
        )
        return row is not None

    async def get_node(self, node_id: str) -> dict | None:
        row = await self._run_single(
            f"MATCH (n:{self._label} {{node_id: $nid}}) RETURN properties(n) AS props", nid=node_id
        )
        if row is None:
            return None
        props = dict(row["props"])
        props.pop("node_id", None)
        return props

    async def node_degree(self, node_id: str) -> int:
        row = await self._run_single(
            f"MATCH (n:{self._label} {{node_id: $nid}})-[r]-() RETURN count(r) AS deg", nid=node_id
        )
        return row["deg"] if row else 0

    async def upsert_node(self, node_id: str, node_data: dict) -> None:
        # MERGE on node_id, then SET properties
        set_clause = ", ".join(f"n.{k} = ${k}" for k in node_data)
        params: dict[str, Any] = {"nid": node_id, **node_data}
        q = f"MERGE (n:{self._label} {{node_id: $nid}})"
        if set_clause:
            q += f" SET {set_clause}"
        await self._run(q, **params)

    async def delete_node(self, node_id: str) -> None:
        await self._run(f"MATCH (n:{self._label} {{node_id: $nid}}) DETACH DELETE n", nid=node_id)

    async def remove_nodes(self, nodes: list[str]) -> None:
        if not nodes:
            return
        await self._run(
            f"MATCH (n:{self._label}) WHERE n.node_id IN $ids DETACH DELETE n", ids=nodes
        )

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------
    async def has_edge(self, src: str, tgt: str) -> bool:
        row = await self._run_single(
            f"MATCH (a:{self._label} {{node_id: $src}})-[r:RELATES_TO]->(b:{self._label} {{node_id: $tgt}}) RETURN r LIMIT 1",
            src=src, tgt=tgt,
        )
        return row is not None

    async def get_edge(self, src: str, tgt: str) -> dict | None:
        row = await self._run_single(
            f"MATCH (a:{self._label} {{node_id: $src}})-[r:RELATES_TO]->(b:{self._label} {{node_id: $tgt}}) RETURN properties(r) AS props",
            src=src, tgt=tgt,
        )
        return dict(row["props"]) if row else None

    async def edge_degree(self, src: str, tgt: str) -> int:
        return await self.node_degree(src) + await self.node_degree(tgt)

    async def get_node_edges(self, node_id: str) -> list[tuple[str, str]]:
        rows = await self._run(
            f"MATCH (a:{self._label} {{node_id: $nid}})-[r:RELATES_TO]-(b:{self._label}) RETURN a.node_id AS src, b.node_id AS tgt",
            nid=node_id,
        )
        return [(r["src"], r["tgt"]) for r in rows]

    async def upsert_edge(self, src: str, tgt: str, edge_data: dict) -> None:
        for key in ("valid_from", "valid_to"):
            val = edge_data.get(key)
            if isinstance(val, datetime):
                edge_data[key] = val.isoformat()
        set_clause = ", ".join(f"r.{k} = ${k}" for k in edge_data)
        params: dict[str, Any] = {"src": src, "tgt": tgt, **edge_data}
        q = (
            f"MERGE (a:{self._label} {{node_id: $src}}) "
            f"MERGE (b:{self._label} {{node_id: $tgt}}) "
            f"MERGE (a)-[r:RELATES_TO]->(b)"
        )
        if set_clause:
            q += f" SET {set_clause}"
        await self._run(q, **params)

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        for s, t in edges:
            await self._run(
                f"MATCH (a:{self._label} {{node_id: $src}})-[r:RELATES_TO]->(b:{self._label} {{node_id: $tgt}}) DELETE r",
                src=s, tgt=t,
            )

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------
    async def get_all_labels(self) -> list[str]:
        rows = await self._run(f"MATCH (n:{self._label}) RETURN n.node_id AS nid ORDER BY nid")
        return [r["nid"] for r in rows]

    async def get_knowledge_graph(self, node_label: str | None, max_depth: int) -> dict:
        if node_label is None or not await self.has_node(node_label):
            return {"nodes": [], "edges": []}
        # Memgraph supports variable-length paths
        rows = await self._run(
            f"MATCH path = (start:{self._label} {{node_id: $nid}})-[*1..{max_depth}]-(end:{self._label}) "
            f"UNWIND nodes(path) AS n "
            f"WITH COLLECT(DISTINCT n) AS all_nodes "
            f"UNWIND all_nodes AS nd "
            f"RETURN nd.node_id AS nid, properties(nd) AS props",
            nid=node_label,
        )
        node_ids = set()
        nodes = []
        for r in rows:
            nid = r["nid"]
            if nid not in node_ids:
                node_ids.add(nid)
                props = dict(r["props"])
                props.pop("node_id", None)
                nodes.append({"id": nid, **props})
        # Fetch edges between collected nodes
        edge_rows = await self._run(
            f"MATCH (a:{self._label})-[r:RELATES_TO]->(b:{self._label}) "
            f"WHERE a.node_id IN $ids AND b.node_id IN $ids "
            f"RETURN a.node_id AS src, b.node_id AS tgt, properties(r) AS props",
            ids=list(node_ids),
        )
        edges = [{"src": r["src"], "tgt": r["tgt"], **dict(r["props"])} for r in edge_rows]
        return {"nodes": nodes, "edges": edges}

    async def search_labels(self, query: str) -> list[str]:
        # Memgraph supports CONTAINS for string matching
        rows = await self._run(
            f"MATCH (n:{self._label}) WHERE toLower(n.node_id) CONTAINS toLower($q) RETURN n.node_id AS nid ORDER BY nid",
            q=query,
        )
        return [r["nid"] for r in rows]

    async def get_popular_labels(self, top_n: int) -> list[tuple[str, int]]:
        rows = await self._run(
            f"MATCH (n:{self._label})-[r]-() "
            f"RETURN n.node_id AS nid, count(r) AS deg "
            f"ORDER BY deg DESC LIMIT $top_n",
            top_n=top_n,
        )
        return [(r["nid"], r["deg"]) for r in rows]

    async def get_subgraph_at(self, snapshot_at: datetime) -> dict:
        ts = snapshot_at.isoformat() if isinstance(snapshot_at, datetime) else str(snapshot_at)
        rows = await self._run(
            f"MATCH (a:{self._label})-[r:RELATES_TO]->(b:{self._label}) "
            f"WHERE (r.valid_from IS NULL OR r.valid_from <= $ts) "
            f"AND (r.valid_to IS NULL OR r.valid_to >= $ts) "
            f"RETURN a.node_id AS src, b.node_id AS tgt, properties(r) AS props, "
            f"properties(a) AS a_props, properties(b) AS b_props",
            ts=ts,
        )
        node_map: dict[str, dict] = {}
        edges = []
        for r in rows:
            for nid, p_key in ((r["src"], "a_props"), (r["tgt"], "b_props")):
                if nid not in node_map:
                    props = dict(r[p_key])
                    props.pop("node_id", None)
                    node_map[nid] = {"id": nid, **props}
            edges.append({"src": r["src"], "tgt": r["tgt"], **dict(r["props"])})
        return {"nodes": list(node_map.values()), "edges": edges}

    async def detect_communities(self, algorithm: str = "leiden", levels: int = 3) -> dict:
        try:
            import networkx as nx
            from graspologic.partition import hierarchical_leiden
        except ImportError:
            logger.warning("networkx/graspologic not installed – skipping community detection")
            return {}
        # Build a networkx graph from Memgraph data
        node_rows = await self._run(f"MATCH (n:{self._label}) RETURN n.node_id AS nid")
        if not node_rows:
            return {}
        G = nx.Graph()
        for r in node_rows:
            G.add_node(r["nid"])
        edge_rows = await self._run(
            f"MATCH (a:{self._label})-[r:RELATES_TO]->(b:{self._label}) RETURN a.node_id AS src, b.node_id AS tgt"
        )
        for r in edge_rows:
            G.add_edge(r["src"], r["tgt"])
        results = hierarchical_leiden(G, max_cluster_size=len(G.nodes()) + 1)
        communities: dict[str, list[str]] = {}
        for entry in results:
            cid = str(entry.cluster)
            communities.setdefault(cid, []).append(entry.node)
        return communities

    async def get_community_summary(self, community_id: str) -> str | None:
        row = await self._run_single(
            f"MATCH (n:{self._label} {{node_id: $cid}}) RETURN n.summary AS summary",
            cid=f"__community_summary__{community_id}",
        )
        return row["summary"] if row else None
