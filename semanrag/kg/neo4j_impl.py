"""Neo4j-based graph storage implementation using the async driver."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from semanrag.base import BaseGraphStorage

logger = logging.getLogger(__name__)

try:
    from neo4j import AsyncGraphDatabase
except ImportError:
    AsyncGraphDatabase = None  # type: ignore[assignment,misc]


class Neo4JStorage(BaseGraphStorage):
    """Graph storage backed by Neo4j with workspace isolation via node labels."""

    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
    ) -> None:
        super().__init__(global_config, namespace, workspace)
        if AsyncGraphDatabase is None:
            raise ImportError(
                "neo4j is required but not installed. "
                "Install it with: pip install neo4j"
            )
        neo4j_cfg = global_config.get("neo4j", {})
        self._uri = neo4j_cfg.get("uri", "bolt://localhost:7687")
        self._user = neo4j_cfg.get("user", "neo4j")
        self._password = neo4j_cfg.get("password", "neo4j")
        self._database = neo4j_cfg.get("database", "neo4j")
        self._driver: Any = None
        # Sanitise workspace for use as a Neo4j label (alphanumeric + _)
        self._ws_label = (workspace or "default").replace("-", "_").replace(" ", "_")
        self._community_summaries: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _label(self) -> str:
        return f"Entity:{self._ws_label}"

    async def _run(self, query: str, params: dict | None = None) -> list[dict]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, params or {})
            return [record.data() async for record in result]

    async def _run_single(self, query: str, params: dict | None = None) -> dict | None:
        rows = await self._run(query, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        # Ensure uniqueness constraint on name within workspace
        await self._run(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{self._ws_label}) "
            f"REQUIRE n.name IS UNIQUE"
        )
        # Fulltext index for search_labels (supports CJK via standard-no-stop-words)
        idx_name = f"ft_{self._ws_label}"
        try:
            await self._run(
                f"CREATE FULLTEXT INDEX {idx_name} IF NOT EXISTS "
                f"FOR (n:{self._ws_label}) ON EACH [n.name] "
                f"OPTIONS {{indexConfig: {{`fulltext.analyzer`: 'standard-no-stop-words'}}}}"
            )
        except Exception:
            logger.debug("Fulltext index %s may already exist or analyzer unavailable", idx_name)

    async def finalize(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def drop(self) -> None:
        await self._run(
            f"MATCH (n:{self._ws_label}) DETACH DELETE n"
        )
        self._community_summaries.clear()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------
    async def has_node(self, node_id: str) -> bool:
        row = await self._run_single(
            f"MATCH (n:{self._ws_label} {{name: $name}}) RETURN count(n) AS c",
            {"name": node_id},
        )
        return bool(row and row["c"] > 0)

    async def get_node(self, node_id: str) -> dict | None:
        row = await self._run_single(
            f"MATCH (n:{self._ws_label} {{name: $name}}) RETURN properties(n) AS props",
            {"name": node_id},
        )
        return row["props"] if row else None

    async def node_degree(self, node_id: str) -> int:
        row = await self._run_single(
            f"MATCH (n:{self._ws_label} {{name: $name}})-[r]-() RETURN count(r) AS d",
            {"name": node_id},
        )
        return row["d"] if row else 0

    async def upsert_node(self, node_id: str, node_data: dict) -> None:
        props = {k: v for k, v in node_data.items() if k != "name"}
        await self._run(
            f"MERGE (n:{self._ws_label} {{name: $name}}) SET n += $props",
            {"name": node_id, "props": props},
        )

    async def delete_node(self, node_id: str) -> None:
        await self._run(
            f"MATCH (n:{self._ws_label} {{name: $name}}) DETACH DELETE n",
            {"name": node_id},
        )

    async def remove_nodes(self, nodes: list[str]) -> None:
        await self._run(
            f"MATCH (n:{self._ws_label}) WHERE n.name IN $names DETACH DELETE n",
            {"names": nodes},
        )

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------
    async def has_edge(self, src: str, tgt: str) -> bool:
        row = await self._run_single(
            f"MATCH (a:{self._ws_label} {{name: $src}})-[r]-(b:{self._ws_label} {{name: $tgt}}) "
            f"RETURN count(r) AS c",
            {"src": src, "tgt": tgt},
        )
        return bool(row and row["c"] > 0)

    async def get_edge(self, src: str, tgt: str) -> dict | None:
        row = await self._run_single(
            f"MATCH (a:{self._ws_label} {{name: $src}})-[r]->(b:{self._ws_label} {{name: $tgt}}) "
            f"RETURN properties(r) AS props",
            {"src": src, "tgt": tgt},
        )
        return row["props"] if row else None

    async def edge_degree(self, src: str, tgt: str) -> int:
        row = await self._run_single(
            f"OPTIONAL MATCH (a:{self._ws_label} {{name: $src}})-[r1]-() "
            f"WITH count(r1) AS d1 "
            f"OPTIONAL MATCH (b:{self._ws_label} {{name: $tgt}})-[r2]-() "
            f"RETURN d1 + count(r2) AS d",
            {"src": src, "tgt": tgt},
        )
        return row["d"] if row else 0

    async def get_node_edges(self, node_id: str) -> list[tuple[str, str]]:
        rows = await self._run(
            f"MATCH (a:{self._ws_label} {{name: $name}})-[r]-(b:{self._ws_label}) "
            f"RETURN a.name AS src, b.name AS tgt",
            {"name": node_id},
        )
        return [(r["src"], r["tgt"]) for r in rows]

    async def upsert_edge(self, src: str, tgt: str, edge_data: dict) -> None:
        props = {}
        for k, v in edge_data.items():
            if isinstance(v, datetime):
                props[k] = v.isoformat()
            else:
                props[k] = v
        await self._run(
            f"MATCH (a:{self._ws_label} {{name: $src}}), (b:{self._ws_label} {{name: $tgt}}) "
            f"MERGE (a)-[r:RELATED]->(b) SET r += $props",
            {"src": src, "tgt": tgt, "props": props},
        )

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        for src, tgt in edges:
            await self._run(
                f"MATCH (a:{self._ws_label} {{name: $src}})-[r]->(b:{self._ws_label} {{name: $tgt}}) "
                f"DELETE r",
                {"src": src, "tgt": tgt},
            )

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------
    async def get_all_labels(self) -> list[str]:
        rows = await self._run(
            f"MATCH (n:{self._ws_label}) RETURN n.name AS name ORDER BY name"
        )
        return [r["name"] for r in rows]

    async def get_knowledge_graph(self, node_label: str | None, max_depth: int) -> dict:
        if node_label is None:
            return {"nodes": [], "edges": []}
        rows = await self._run(
            f"MATCH (start:{self._ws_label} {{name: $name}}) "
            f"CALL apoc.path.subgraphAll(start, {{maxLevel: $depth}}) "
            f"YIELD nodes, relationships "
            f"RETURN nodes, relationships",
            {"name": node_label, "depth": max_depth},
        )
        if not rows:
            # Fallback: variable-length path pattern (no APOC needed)
            rows = await self._run(
                f"MATCH path = (start:{self._ws_label} {{name: $name}})"
                f"-[*1..{max_depth}]-(end:{self._ws_label}) "
                f"UNWIND nodes(path) AS n "
                f"WITH collect(DISTINCT n) AS ns, collect(DISTINCT relationships(path)) AS rss "
                f"UNWIND rss AS rs UNWIND rs AS r "
                f"RETURN ns, collect(DISTINCT r) AS rels",
                {"name": node_label},
            )
            if not rows:
                return {"nodes": [], "edges": []}
            ns = rows[0].get("ns", [])
            rels = rows[0].get("rels", [])
            nodes = [dict(n) for n in ns] if ns else []
            edges = []
            for r in rels or []:
                try:
                    edges.append({
                        "src": r.start_node["name"],
                        "tgt": r.end_node["name"],
                        **dict(r),
                    })
                except Exception:
                    edges.append(dict(r) if hasattr(r, "__iter__") else {})
            return {"nodes": nodes, "edges": edges}

        ns = rows[0].get("nodes", [])
        rels = rows[0].get("relationships", [])
        nodes = [dict(n) for n in ns]
        edges = []
        for r in rels:
            try:
                edges.append({
                    "src": r.start_node["name"],
                    "tgt": r.end_node["name"],
                    **dict(r),
                })
            except Exception:
                edges.append(dict(r) if hasattr(r, "__iter__") else {})
        return {"nodes": nodes, "edges": edges}

    async def search_labels(self, query: str) -> list[str]:
        idx_name = f"ft_{self._ws_label}"
        # Escape Lucene special chars for safety
        safe_q = query.replace("\\", "\\\\").replace('"', '\\"')
        try:
            rows = await self._run(
                f'CALL db.index.fulltext.queryNodes("{idx_name}", $q) '
                f"YIELD node RETURN node.name AS name",
                {"q": f"*{safe_q}*"},
            )
            return [r["name"] for r in rows if r["name"]]
        except Exception:
            # Fallback: CONTAINS
            rows = await self._run(
                f"MATCH (n:{self._ws_label}) WHERE toLower(n.name) CONTAINS toLower($q) "
                f"RETURN n.name AS name",
                {"q": query},
            )
            return [r["name"] for r in rows if r["name"]]

    async def get_popular_labels(self, top_n: int) -> list[tuple[str, int]]:
        rows = await self._run(
            f"MATCH (n:{self._ws_label})-[r]-() "
            f"RETURN n.name AS name, count(r) AS deg ORDER BY deg DESC LIMIT $n",
            {"n": top_n},
        )
        return [(r["name"], r["deg"]) for r in rows]

    async def get_subgraph_at(self, snapshot_at: datetime) -> dict:
        ts = snapshot_at.isoformat() if isinstance(snapshot_at, datetime) else str(snapshot_at)
        rows = await self._run(
            f"MATCH (a:{self._ws_label})-[r]->(b:{self._ws_label}) "
            f"WHERE (r.valid_from IS NULL OR r.valid_from <= $ts) "
            f"  AND (r.valid_to IS NULL OR r.valid_to >= $ts) "
            f"RETURN a.name AS src, properties(a) AS src_props, "
            f"       b.name AS tgt, properties(b) AS tgt_props, "
            f"       properties(r) AS rel_props",
            {"ts": ts},
        )
        node_map: dict[str, dict] = {}
        edges = []
        for r in rows:
            node_map.setdefault(r["src"], r["src_props"])
            node_map.setdefault(r["tgt"], r["tgt_props"])
            edges.append({"src": r["src"], "tgt": r["tgt"], **r["rel_props"]})
        nodes = [{"id": k, **v} for k, v in node_map.items()]
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

        # Fetch all nodes and edges into memory for graspologic
        import networkx as nx

        node_rows = await self._run(
            f"MATCH (n:{self._ws_label}) RETURN n.name AS name"
        )
        if not node_rows:
            return {}

        edge_rows = await self._run(
            f"MATCH (a:{self._ws_label})-[r]->(b:{self._ws_label}) "
            f"RETURN a.name AS src, b.name AS tgt"
        )

        G = nx.Graph()
        for r in node_rows:
            G.add_node(r["name"])
        for r in edge_rows:
            G.add_edge(r["src"], r["tgt"])

        if G.number_of_nodes() == 0:
            return {}

        results = hierarchical_leiden(G, max_cluster_size=len(G.nodes()) + 1)
        communities: dict[str, list[str]] = {}
        for entry in results:
            cid = str(entry.cluster)
            communities.setdefault(cid, []).append(entry.node)

        # Write community IDs back to Neo4j
        for cid, members in communities.items():
            await self._run(
                f"MATCH (n:{self._ws_label}) WHERE n.name IN $names "
                f"SET n.community = $cid",
                {"names": members, "cid": cid},
            )

        return communities

    async def get_community_summary(self, community_id: str) -> str | None:
        return self._community_summaries.get(community_id)
