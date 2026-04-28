"""PostgreSQL storage implementations for SemanRAG."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any

import asyncpg

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


def _get_pg_db(global_config: dict) -> PostgreSQLDB:
    """Get or create a PostgreSQLDB instance from global_config."""
    if "pg_db" in global_config:
        return global_config["pg_db"]
    return PostgreSQLDB(global_config)


class PostgreSQLDB:
    """Async connection pool manager for PostgreSQL using asyncpg."""

    def __init__(self, config: dict) -> None:
        self._host = config.get("pg_host", "localhost")
        self._port = config.get("pg_port", 5432)
        self._database = config.get("pg_database", "semanrag")
        self._user = config.get("pg_user", "postgres")
        self._password = config.get("pg_password", "")
        self._ssl = config.get("pg_ssl")
        self._min_size = config.get("pg_min_size", 2)
        self._max_size = config.get("pg_max_size", 10)
        self._statement_timeout = config.get("pg_statement_timeout", 30000)
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        for attempt in range(3):
            try:
                self._pool = await asyncpg.create_pool(
                    host=self._host,
                    port=self._port,
                    database=self._database,
                    user=self._user,
                    password=self._password,
                    ssl=self._ssl,
                    min_size=self._min_size,
                    max_size=self._max_size,
                    command_timeout=self._statement_timeout / 1000,
                )
                logger.info("PostgreSQL pool created (attempt %d)", attempt + 1)
                return
            except (asyncpg.PostgresError, OSError) as exc:
                logger.warning("Connection attempt %d failed: %s", attempt + 1, exc)
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def execute(self, query: str, *args: Any) -> str:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"SET statement_timeout = {self._statement_timeout}"
            )
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"SET statement_timeout = {self._statement_timeout}"
            )
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"SET statement_timeout = {self._statement_timeout}"
            )
            return await conn.fetchrow(query, *args)

    async def __aenter__(self) -> PostgreSQLDB:
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


class PGKVStorage(BaseKVStorage):
    """PostgreSQL key-value storage with JSONB data column."""

    def __init__(
        self, global_config: dict, namespace: str, workspace: str | None = None
    ) -> None:
        super().__init__(global_config, namespace, workspace)
        self._db = _get_pg_db(global_config)
        self._table = f"{namespace}_kv"

    async def initialize(self) -> None:
        await self._db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                id TEXT PRIMARY KEY,
                data JSONB NOT NULL DEFAULT '{{}}',
                workspace TEXT,
                updated_at TIMESTAMPTZ DEFAULT now()
            )
        """)
        await self._db.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table}_ws
            ON {self._table} (workspace)
        """)

    async def finalize(self) -> None:
        pass

    async def get_by_id(self, id: str) -> dict | None:
        row = await self._db.fetchrow(
            f"SELECT data FROM {self._table} WHERE id = $1"
            + (" AND workspace = $2" if self._workspace else ""),
            id, *([self._workspace] if self._workspace else []),
        )
        return json.loads(row["data"]) if row else None

    async def get_by_ids(self, ids: list[str]) -> list[dict | None]:
        if not ids:
            return []
        rows = await self._db.fetch(
            f"SELECT id, data FROM {self._table} WHERE id = ANY($1)"
            + (" AND workspace = $2" if self._workspace else ""),
            ids, *([self._workspace] if self._workspace else []),
        )
        found = {r["id"]: json.loads(r["data"]) for r in rows}
        return [found.get(i) for i in ids]

    async def filter_keys(self, data: set[str]) -> set[str]:
        if not data:
            return set()
        rows = await self._db.fetch(
            f"SELECT id FROM {self._table} WHERE id = ANY($1)",
            list(data),
        )
        return {r["id"] for r in rows}

    async def upsert(self, data: dict[str, dict]) -> None:
        if not data:
            return
        async with self._db._pool.acquire() as conn:
            await conn.executemany(
                f"""INSERT INTO {self._table} (id, data, workspace, updated_at)
                    VALUES ($1, $2::jsonb, $3, now())
                    ON CONFLICT (id) DO UPDATE
                    SET data = EXCLUDED.data, updated_at = now()""",
                [(k, json.dumps(v), self._workspace) for k, v in data.items()],
            )

    async def delete(self, ids: list[str]) -> None:
        if ids:
            await self._db.execute(
                f"DELETE FROM {self._table} WHERE id = ANY($1)", ids
            )

    async def drop(self) -> None:
        await self._db.execute(f"DROP TABLE IF EXISTS {self._table}")

    async def index_done_callback(self) -> None:
        pass


class PGVectorStorage(BaseVectorStorage):
    """PostgreSQL vector storage using pgvector extension with HNSW indexing."""

    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
        embedding_func: Any = None,
    ) -> None:
        super().__init__(global_config, namespace, workspace, embedding_func)
        self._db = _get_pg_db(global_config)
        self._table = f"{namespace}_vec"
        self._dim = self.embedding_func.embedding_dim
        self._use_halfvec = global_config.get("pg_use_halfvec", False)
        self._hnsw_m = global_config.get("pg_hnsw_m", 16)
        self._hnsw_ef = global_config.get("pg_hnsw_ef_construction", 64)

    @property
    def _vec_type(self) -> str:
        return f"halfvec({self._dim})" if self._use_halfvec else f"vector({self._dim})"

    @property
    def _dist_op(self) -> str:
        return "<=>"  # cosine distance

    async def initialize(self) -> None:
        await self._db.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await self._db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                id TEXT PRIMARY KEY,
                vector {self._vec_type},
                metadata JSONB NOT NULL DEFAULT '{{}}',
                workspace TEXT
            )
        """)
        idx_name = f"idx_{self._table}_hnsw"
        ops_class = "halfvec_cosine_ops" if self._use_halfvec else "vector_cosine_ops"
        await self._db.execute(f"""
            CREATE INDEX IF NOT EXISTS {idx_name}
            ON {self._table} USING hnsw (vector {ops_class})
            WITH (m = {self._hnsw_m}, ef_construction = {self._hnsw_ef})
        """)

    async def finalize(self) -> None:
        pass

    async def upsert(self, data: dict[str, dict]) -> None:
        if not data:
            return
        rows = []
        for id_, fields in data.items():
            vec = fields.get("__vector__")
            meta = {k: v for k, v in fields.items() if k != "__vector__"}
            vec_str = "[" + ",".join(str(float(x)) for x in vec) + "]" if vec else None
            rows.append((id_, vec_str, json.dumps(meta), self._workspace))
        async with self._db._pool.acquire() as conn:
            await conn.executemany(
                f"""INSERT INTO {self._table} (id, vector, metadata, workspace)
                    VALUES ($1, $2::{self._vec_type.split('(')[0]}, $3::jsonb, $4)
                    ON CONFLICT (id) DO UPDATE
                    SET vector = EXCLUDED.vector,
                        metadata = EXCLUDED.metadata""",
                rows,
            )

    async def query(
        self, query: str, top_k: int, acl_filter: dict | None = None
    ) -> list[dict]:
        embedding = await self.embedding_func([query])
        vec_str = "[" + ",".join(str(float(x)) for x in embedding[0]) + "]"

        where_clauses = []
        params: list[Any] = [vec_str, top_k]
        idx = 3

        if self._workspace:
            where_clauses.append(f"workspace = ${idx}")
            params.append(self._workspace)
            idx += 1

        if acl_filter:
            user_id = acl_filter.get("user_id", "")
            user_groups = acl_filter.get("user_groups", [])
            where_clauses.append(f"""(
                (metadata->>'acl_public')::boolean IS NOT FALSE
                OR metadata->>'acl_owner' = ${idx}
                OR metadata->'acl_visible_to_users' ? ${idx}
                OR metadata->'acl_visible_to_groups' ?| ${idx + 1}
            )""")
            params.extend([user_id, user_groups])
            idx += 2

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        rows = await self._db.fetch(
            f"""SELECT id, metadata, vector {self._dist_op} $1::{self._vec_type.split('(')[0]} AS distance
                FROM {self._table}
                {where_sql}
                ORDER BY distance
                LIMIT $2""",
            *params,
        )
        return [
            {**json.loads(r["metadata"]), "id": r["id"], "distance": float(r["distance"])}
            for r in rows
        ]

    async def delete(self, ids: list[str]) -> None:
        if ids:
            await self._db.execute(
                f"DELETE FROM {self._table} WHERE id = ANY($1)", ids
            )

    async def delete_entity(self, entity_name: str) -> None:
        await self._db.execute(
            f"DELETE FROM {self._table} WHERE metadata->>'entity_name' = $1",
            entity_name,
        )

    async def delete_entity_relation(self, entity_name: str) -> None:
        await self._db.execute(
            f"""DELETE FROM {self._table}
                WHERE metadata->>'src_id' = $1 OR metadata->>'tgt_id' = $1""",
            entity_name,
        )

    async def get_by_id(self, id: str) -> dict | None:
        row = await self._db.fetchrow(
            f"SELECT id, metadata FROM {self._table} WHERE id = $1", id
        )
        return {**json.loads(row["metadata"]), "id": row["id"]} if row else None

    async def get_by_ids(self, ids: list[str]) -> list[dict | None]:
        if not ids:
            return []
        rows = await self._db.fetch(
            f"SELECT id, metadata FROM {self._table} WHERE id = ANY($1)", ids
        )
        found = {r["id"]: {**json.loads(r["metadata"]), "id": r["id"]} for r in rows}
        return [found.get(i) for i in ids]

    async def get_vectors_by_ids(self, ids: list[str]) -> list[dict | None]:
        if not ids:
            return []
        rows = await self._db.fetch(
            f"SELECT id, metadata, vector::text AS vec_text FROM {self._table} WHERE id = ANY($1)",
            ids,
        )
        found = {}
        for r in rows:
            vec_list = [float(x) for x in r["vec_text"].strip("[]").split(",")]
            found[r["id"]] = {
                **json.loads(r["metadata"]),
                "id": r["id"],
                "__vector__": vec_list,
            }
        return [found.get(i) for i in ids]

    async def drop(self) -> None:
        await self._db.execute(f"DROP TABLE IF EXISTS {self._table}")


class PGGraphStorage(BaseGraphStorage):
    """PostgreSQL graph storage using Apache AGE with adjacency-list fallback."""

    def __init__(
        self, global_config: dict, namespace: str, workspace: str | None = None
    ) -> None:
        super().__init__(global_config, namespace, workspace)
        self._db = _get_pg_db(global_config)
        self._graph_name = f"{namespace}_graph"
        self._nodes_table = f"{namespace}_nodes"
        self._edges_table = f"{namespace}_edges"
        self._community_table = f"{namespace}_communities"
        self._use_age = global_config.get("pg_use_age", True)
        self._age_available = False

    async def _try_init_age(self) -> bool:
        try:
            await self._db.execute("CREATE EXTENSION IF NOT EXISTS age")
            await self._db.execute("LOAD 'age'")
            await self._db.execute(
                "SET search_path = ag_catalog, \"$user\", public"
            )
            exists = await self._db.fetchrow(
                "SELECT * FROM ag_catalog.ag_graph WHERE name = $1",
                self._graph_name,
            )
            if not exists:
                await self._db.execute(
                    f"SELECT create_graph('{self._graph_name}')"
                )
            return True
        except (asyncpg.PostgresError, Exception) as exc:
            logger.warning("AGE not available, falling back to tables: %s", exc)
            return False

    async def _init_fallback_tables(self) -> None:
        await self._db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._nodes_table} (
                id TEXT PRIMARY KEY,
                properties JSONB NOT NULL DEFAULT '{{}}'
            )
        """)
        await self._db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._edges_table} (
                src TEXT NOT NULL,
                tgt TEXT NOT NULL,
                properties JSONB NOT NULL DEFAULT '{{}}',
                PRIMARY KEY (src, tgt)
            )
        """)
        await self._db.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._edges_table}_src
            ON {self._edges_table} (src)
        """)
        await self._db.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._edges_table}_tgt
            ON {self._edges_table} (tgt)
        """)
        await self._db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._community_table} (
                community_id TEXT PRIMARY KEY,
                summary TEXT DEFAULT ''
            )
        """)

    async def initialize(self) -> None:
        if self._use_age:
            self._age_available = await self._try_init_age()
        if not self._age_available:
            await self._init_fallback_tables()

    async def finalize(self) -> None:
        pass

    async def drop(self) -> None:
        if self._age_available:
            try:
                await self._db.execute(
                    f"SELECT drop_graph('{self._graph_name}', true)"
                )
            except asyncpg.PostgresError:
                pass
        await self._db.execute(f"DROP TABLE IF EXISTS {self._nodes_table}")
        await self._db.execute(f"DROP TABLE IF EXISTS {self._edges_table}")
        await self._db.execute(f"DROP TABLE IF EXISTS {self._community_table}")

    # -- AGE cypher helper --
    async def _cypher(self, cypher_query: str, params: str = "NULL") -> list[asyncpg.Record]:
        sql = f"""SELECT * FROM ag_catalog.cypher(
            '{self._graph_name}', $$ {cypher_query} $$, {params}
        ) AS (result agtype)"""
        return await self._db.fetch(sql)

    # -- Node operations --
    async def has_node(self, node_id: str) -> bool:
        if self._age_available:
            rows = await self._cypher(
                f"MATCH (n {{id: '{_esc(node_id)}'}}) RETURN n"
            )
            return len(rows) > 0
        row = await self._db.fetchrow(
            f"SELECT 1 FROM {self._nodes_table} WHERE id = $1", node_id
        )
        return row is not None

    async def has_edge(self, src: str, tgt: str) -> bool:
        if self._age_available:
            rows = await self._cypher(
                f"MATCH ({{id: '{_esc(src)}'}})-[r]->({{id: '{_esc(tgt)}'}}) RETURN r"
            )
            return len(rows) > 0
        row = await self._db.fetchrow(
            f"SELECT 1 FROM {self._edges_table} WHERE src = $1 AND tgt = $2",
            src, tgt,
        )
        return row is not None

    async def node_degree(self, node_id: str) -> int:
        if self._age_available:
            rows = await self._cypher(
                f"MATCH ({{id: '{_esc(node_id)}'}})-[r]-() RETURN count(r)"
            )
            return _age_int(rows[0]["result"]) if rows else 0
        row = await self._db.fetchrow(
            f"""SELECT count(*) AS cnt FROM {self._edges_table}
                WHERE src = $1 OR tgt = $1""",
            node_id,
        )
        return int(row["cnt"]) if row else 0

    async def edge_degree(self, src: str, tgt: str) -> int:
        return await self.node_degree(src) + await self.node_degree(tgt)

    async def get_node(self, node_id: str) -> dict | None:
        if self._age_available:
            rows = await self._cypher(
                f"MATCH (n {{id: '{_esc(node_id)}'}}) RETURN properties(n)"
            )
            return _age_json(rows[0]["result"]) if rows else None
        row = await self._db.fetchrow(
            f"SELECT properties FROM {self._nodes_table} WHERE id = $1", node_id
        )
        return json.loads(row["properties"]) if row else None

    async def get_edge(self, src: str, tgt: str) -> dict | None:
        if self._age_available:
            rows = await self._cypher(
                f"MATCH ({{id: '{_esc(src)}'}})-[r]->({{id: '{_esc(tgt)}'}}) RETURN properties(r)"
            )
            return _age_json(rows[0]["result"]) if rows else None
        row = await self._db.fetchrow(
            f"SELECT properties FROM {self._edges_table} WHERE src = $1 AND tgt = $2",
            src, tgt,
        )
        return json.loads(row["properties"]) if row else None

    async def get_node_edges(self, node_id: str) -> list[tuple[str, str]]:
        if self._age_available:
            rows = await self._cypher(
                f"MATCH (a {{id: '{_esc(node_id)}'}})-[r]-(b) RETURN a.id, b.id"
            )
            return [(_age_str(r["result"]), _age_str(r["result"])) for r in rows]
        rows = await self._db.fetch(
            f"""SELECT src, tgt FROM {self._edges_table}
                WHERE src = $1 OR tgt = $1""",
            node_id,
        )
        return [(r["src"], r["tgt"]) for r in rows]

    async def upsert_node(self, node_id: str, node_data: dict) -> None:
        if self._age_available:
            props = _props_str({**node_data, "id": node_id})
            await self._cypher(
                f"MERGE (n {{id: '{_esc(node_id)}'}}) SET n += {props} RETURN n"
            )
            return
        await self._db.execute(
            f"""INSERT INTO {self._nodes_table} (id, properties)
                VALUES ($1, $2::jsonb)
                ON CONFLICT (id) DO UPDATE
                SET properties = {self._nodes_table}.properties || $2::jsonb""",
            node_id, json.dumps(node_data),
        )

    async def upsert_edge(self, src: str, tgt: str, edge_data: dict) -> None:
        for key in ("valid_from", "valid_to"):
            val = edge_data.get(key)
            if isinstance(val, datetime):
                edge_data[key] = val.isoformat()

        if self._age_available:
            props = _props_str(edge_data)
            await self._cypher(
                f"""MERGE (a {{id: '{_esc(src)}'}})
                    MERGE (b {{id: '{_esc(tgt)}'}})
                    MERGE (a)-[r:RELATES]->(b)
                    SET r += {props}
                    RETURN r"""
            )
            return
        await self._db.execute(
            f"""INSERT INTO {self._edges_table} (src, tgt, properties)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (src, tgt) DO UPDATE
                SET properties = {self._edges_table}.properties || $3::jsonb""",
            src, tgt, json.dumps(edge_data),
        )

    async def delete_node(self, node_id: str) -> None:
        if self._age_available:
            await self._cypher(
                f"MATCH (n {{id: '{_esc(node_id)}'}}) DETACH DELETE n"
            )
            return
        await self._db.execute(
            f"DELETE FROM {self._edges_table} WHERE src = $1 OR tgt = $1", node_id
        )
        await self._db.execute(
            f"DELETE FROM {self._nodes_table} WHERE id = $1", node_id
        )

    async def remove_nodes(self, nodes: list[str]) -> None:
        for n in nodes:
            await self.delete_node(n)

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        for src, tgt in edges:
            if self._age_available:
                await self._cypher(
                    f"MATCH ({{id: '{_esc(src)}'}})-[r]->({{id: '{_esc(tgt)}'}}) DELETE r"
                )
            else:
                await self._db.execute(
                    f"DELETE FROM {self._edges_table} WHERE src = $1 AND tgt = $2",
                    src, tgt,
                )

    async def get_all_labels(self) -> list[str]:
        if self._age_available:
            rows = await self._cypher("MATCH (n) RETURN n.id")
            return sorted(_age_str(r["result"]) for r in rows)
        rows = await self._db.fetch(
            f"SELECT id FROM {self._nodes_table} ORDER BY id"
        )
        return [r["id"] for r in rows]

    async def get_knowledge_graph(self, node_label: str | None, max_depth: int) -> dict:
        if node_label is None:
            return {"nodes": [], "edges": []}
        if not await self.has_node(node_label):
            return {"nodes": [], "edges": []}

        if self._age_available:
            rows = await self._cypher(
                f"""MATCH path = (start {{id: '{_esc(node_label)}'}})-[*1..{max_depth}]-(end)
                    UNWIND nodes(path) AS n
                    UNWIND relationships(path) AS r
                    RETURN DISTINCT properties(n), properties(r),
                           startNode(r).id, endNode(r).id"""
            )
            nodes_map: dict[str, dict] = {}
            edges_list: list[dict] = []
            for r in rows:
                data = _age_json(r["result"])
                if data and "id" in data:
                    nodes_map[data["id"]] = data
            return {"nodes": list(nodes_map.values()), "edges": edges_list}

        # Fallback: BFS via SQL
        visited: set[str] = {node_label}
        frontier = [node_label]
        all_edges: list[dict] = []
        for _ in range(max_depth):
            if not frontier:
                break
            rows = await self._db.fetch(
                f"""SELECT src, tgt, properties FROM {self._edges_table}
                    WHERE src = ANY($1) OR tgt = ANY($1)""",
                frontier,
            )
            next_frontier = []
            for r in rows:
                all_edges.append({
                    "src": r["src"], "tgt": r["tgt"],
                    **json.loads(r["properties"]),
                })
                for nid in (r["src"], r["tgt"]):
                    if nid not in visited:
                        visited.add(nid)
                        next_frontier.append(nid)
            frontier = next_frontier

        node_rows = await self._db.fetch(
            f"SELECT id, properties FROM {self._nodes_table} WHERE id = ANY($1)",
            list(visited),
        )
        nodes = [{"id": r["id"], **json.loads(r["properties"])} for r in node_rows]
        return {"nodes": nodes, "edges": all_edges}

    async def search_labels(self, query: str) -> list[str]:
        if self._age_available:
            rows = await self._cypher(
                f"MATCH (n) WHERE n.id CONTAINS '{_esc(query)}' RETURN n.id"
            )
            return sorted(_age_str(r["result"]) for r in rows)
        rows = await self._db.fetch(
            f"SELECT id FROM {self._nodes_table} WHERE id ILIKE $1 ORDER BY id",
            f"%{query}%",
        )
        return [r["id"] for r in rows]

    async def get_popular_labels(self, top_n: int) -> list[tuple[str, int]]:
        rows = await self._db.fetch(
            f"""SELECT node_id, cnt FROM (
                    SELECT src AS node_id, count(*) AS cnt FROM {self._edges_table} GROUP BY src
                    UNION ALL
                    SELECT tgt, count(*) FROM {self._edges_table} GROUP BY tgt
                ) sub
                GROUP BY node_id
                ORDER BY sum(cnt) DESC
                LIMIT $1""",
            top_n,
        )
        return [(r["node_id"], int(r["cnt"])) for r in rows] if rows else []

    async def get_subgraph_at(self, snapshot_at: datetime) -> dict:
        ts = snapshot_at.isoformat() if isinstance(snapshot_at, datetime) else str(snapshot_at)
        rows = await self._db.fetch(
            f"""SELECT src, tgt, properties FROM {self._edges_table}
                WHERE (properties->>'valid_from' IS NULL OR properties->>'valid_from' <= $1)
                  AND (properties->>'valid_to' IS NULL OR properties->>'valid_to' >= $1)""",
            ts,
        )
        node_ids: set[str] = set()
        edges = []
        for r in rows:
            node_ids.update((r["src"], r["tgt"]))
            edges.append({"src": r["src"], "tgt": r["tgt"], **json.loads(r["properties"])})

        node_rows = await self._db.fetch(
            f"SELECT id, properties FROM {self._nodes_table} WHERE id = ANY($1)",
            list(node_ids),
        ) if node_ids else []
        nodes = [{"id": r["id"], **json.loads(r["properties"])} for r in node_rows]
        return {"nodes": nodes, "edges": edges}

    async def detect_communities(self, algorithm: str = "leiden", levels: int = 3) -> dict:
        try:
            from graspologic.partition import hierarchical_leiden
        except ImportError:
            logger.warning("graspologic not installed – skipping community detection")
            return {}

        import networkx as nx

        G = nx.Graph()
        rows = await self._db.fetch(
            f"SELECT src, tgt FROM {self._edges_table}"
        )
        for r in rows:
            G.add_edge(r["src"], r["tgt"])

        if G.number_of_nodes() == 0:
            return {}

        results = hierarchical_leiden(G, max_cluster_size=len(G.nodes()) + 1)
        communities: dict[str, list[str]] = {}
        for entry in results:
            cid = str(entry.cluster)
            communities.setdefault(cid, []).append(entry.node)
            # Store community assignment in node properties
            if not self._age_available:
                await self._db.execute(
                    f"""UPDATE {self._nodes_table}
                        SET properties = properties || jsonb_build_object('community', $2)
                        WHERE id = $1""",
                    entry.node, cid,
                )
        return communities

    async def get_community_summary(self, community_id: str) -> str | None:
        row = await self._db.fetchrow(
            f"SELECT summary FROM {self._community_table} WHERE community_id = $1",
            community_id,
        )
        return row["summary"] if row else None


def _esc(s: str) -> str:
    """Escape single quotes for Cypher string literals."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _props_str(d: dict) -> str:
    """Convert a dict to a Cypher properties literal."""
    parts = []
    for k, v in d.items():
        if isinstance(v, str):
            parts.append(f"{k}: '{_esc(v)}'")
        elif isinstance(v, bool):
            parts.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}: {v}")
        else:
            parts.append(f"{k}: '{_esc(json.dumps(v))}'")
    return "{" + ", ".join(parts) + "}"


def _age_json(val: Any) -> dict | None:
    """Parse an agtype result into a Python dict."""
    if val is None:
        return None
    s = str(val)
    try:
        return json.loads(s.rstrip("::vertex").rstrip("::edge").rstrip("::agtype"))
    except (json.JSONDecodeError, ValueError):
        return None


def _age_str(val: Any) -> str:
    """Extract a string from an agtype result."""
    s = str(val).strip('"').rstrip("::agtype")
    return s


def _age_int(val: Any) -> int:
    """Extract an int from an agtype result."""
    s = str(val).rstrip("::agtype")
    try:
        return int(s)
    except ValueError:
        return 0


class PGDocStatusStorage(DocStatusStorage):
    """PostgreSQL document status storage with ACL-aware pagination."""

    def __init__(
        self, global_config: dict, namespace: str, workspace: str | None = None
    ) -> None:
        super().__init__(global_config, namespace, workspace)
        self._db = _get_pg_db(global_config)
        self._table = f"{namespace}_doc_status"

    async def initialize(self) -> None:
        await self._db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                id TEXT PRIMARY KEY,
                data JSONB NOT NULL DEFAULT '{{}}',
                workspace TEXT,
                status TEXT DEFAULT 'pending',
                file_path TEXT DEFAULT '',
                created_at TIMESTAMPTZ DEFAULT now(),
                updated_at TIMESTAMPTZ DEFAULT now()
            )
        """)
        await self._db.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table}_status
            ON {self._table} (status)
        """)
        await self._db.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table}_fp
            ON {self._table} (file_path)
        """)
        await self._db.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table}_ws
            ON {self._table} (workspace)
        """)

    async def finalize(self) -> None:
        pass

    @staticmethod
    def _row_to_status(row: asyncpg.Record) -> DocStatus:
        d = json.loads(row["data"])
        acl = d.pop("acl_policy", None)
        if isinstance(acl, dict):
            d["acl_policy"] = ACLPolicy(**acl)
        else:
            d["acl_policy"] = None
        return DocStatus(**d)

    @staticmethod
    def _status_to_dict(status: DocStatus) -> dict:
        d = asdict(status)
        if status.acl_policy is not None:
            d["acl_policy"] = asdict(status.acl_policy)
        else:
            d.pop("acl_policy", None)
        return d

    async def get(self, doc_id: str) -> DocStatus | None:
        row = await self._db.fetchrow(
            f"SELECT data FROM {self._table} WHERE id = $1", doc_id
        )
        return self._row_to_status(row) if row else None

    async def upsert(self, doc_id: str, status: DocStatus) -> None:
        data = json.dumps(self._status_to_dict(status))
        await self._db.execute(
            f"""INSERT INTO {self._table} (id, data, workspace, status, file_path, created_at, updated_at)
                VALUES ($1, $2::jsonb, $3, $4, $5, now(), now())
                ON CONFLICT (id) DO UPDATE
                SET data = EXCLUDED.data, status = EXCLUDED.status,
                    file_path = EXCLUDED.file_path, updated_at = now()""",
            doc_id, data, self._workspace, status.status, status.file_path,
        )

    async def delete(self, doc_id: str) -> None:
        await self._db.execute(
            f"DELETE FROM {self._table} WHERE id = $1", doc_id
        )

    async def get_status_counts(self) -> dict[str, int]:
        where = " WHERE workspace = $1" if self._workspace else ""
        params = [self._workspace] if self._workspace else []
        rows = await self._db.fetch(
            f"SELECT status, count(*) AS cnt FROM {self._table}{where} GROUP BY status",
            *params,
        )
        return {r["status"]: int(r["cnt"]) for r in rows}

    async def get_all_status_counts(self) -> dict[str, int]:
        rows = await self._db.fetch(
            f"SELECT status, count(*) AS cnt FROM {self._table} GROUP BY status"
        )
        return {r["status"]: int(r["cnt"]) for r in rows}

    async def get_docs_by_status(self, status: str) -> list[DocStatus]:
        rows = await self._db.fetch(
            f"SELECT data FROM {self._table} WHERE status = $1"
            + (" AND workspace = $2" if self._workspace else ""),
            status, *([self._workspace] if self._workspace else []),
        )
        return [self._row_to_status(r) for r in rows]

    async def get_docs_paginated(
        self,
        offset: int,
        limit: int,
        status: str | None = None,
        acl_filter: dict | None = None,
    ) -> tuple[list[DocStatus], int]:
        where_clauses: list[str] = []
        params: list[Any] = []
        idx = 1

        if self._workspace:
            where_clauses.append(f"workspace = ${idx}")
            params.append(self._workspace)
            idx += 1

        if status is not None:
            where_clauses.append(f"status = ${idx}")
            params.append(status)
            idx += 1

        if acl_filter:
            user_id = acl_filter.get("user_id", "")
            user_groups = acl_filter.get("user_groups", [])
            where_clauses.append(f"""(
                (data->'acl_policy'->>'public')::boolean IS NOT FALSE
                OR data->'acl_policy'->>'owner' = ${idx}
                OR data->'acl_policy'->'visible_to_users' ? ${idx}
                OR data->'acl_policy'->'visible_to_groups' ?| ${idx + 1}
            )""")
            params.extend([user_id, user_groups])
            idx += 2

        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        count_row = await self._db.fetchrow(
            f"SELECT count(*) AS cnt FROM {self._table}{where_sql}", *params
        )
        total = int(count_row["cnt"])

        params.extend([limit, offset])
        rows = await self._db.fetch(
            f"""SELECT data FROM {self._table}{where_sql}
                ORDER BY updated_at DESC
                LIMIT ${idx} OFFSET ${idx + 1}""",
            *params,
        )
        return [self._row_to_status(r) for r in rows], total

    async def get_doc_by_file_path(self, file_path: str) -> DocStatus | None:
        row = await self._db.fetchrow(
            f"SELECT data FROM {self._table} WHERE file_path = $1", file_path
        )
        return self._row_to_status(row) if row else None


class PGLexicalStorage(BaseLexicalStorage):
    """PostgreSQL full-text search storage using tsvector and GIN index."""

    def __init__(
        self, global_config: dict, namespace: str, workspace: str | None = None
    ) -> None:
        super().__init__(global_config, namespace, workspace)
        self._db = _get_pg_db(global_config)
        self._table = f"{namespace}_lexical"

    async def initialize(self) -> None:
        await self._db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL DEFAULT '',
                tsv tsvector,
                workspace TEXT
            )
        """)
        await self._db.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self._table}_tsv
            ON {self._table} USING gin(tsv)
        """)

    async def finalize(self) -> None:
        pass

    async def upsert(self, data: dict[str, dict]) -> None:
        if not data:
            return
        rows = []
        for id_, fields in data.items():
            content = fields.get("content", "")
            rows.append((id_, content, self._workspace))
        async with self._db._pool.acquire() as conn:
            await conn.executemany(
                f"""INSERT INTO {self._table} (id, content, tsv, workspace)
                    VALUES ($1, $2, to_tsvector('english', $2), $3)
                    ON CONFLICT (id) DO UPDATE
                    SET content = EXCLUDED.content,
                        tsv = to_tsvector('english', EXCLUDED.content)""",
                rows,
            )

    async def search_bm25(self, query: str, top_k: int) -> list[dict]:
        if not query.strip():
            return []
        where_clauses = ["tsv @@ plainto_tsquery('english', $1)"]
        params: list[Any] = [query, top_k]
        idx = 3

        if self._workspace:
            where_clauses.append(f"workspace = ${idx}")
            params.append(self._workspace)
            idx += 1

        where_sql = " AND ".join(where_clauses)
        rows = await self._db.fetch(
            f"""SELECT id, content,
                       ts_rank(tsv, plainto_tsquery('english', $1)) AS score
                FROM {self._table}
                WHERE {where_sql}
                ORDER BY score DESC
                LIMIT $2""",
            *params,
        )
        return [
            {"id": r["id"], "content": r["content"], "score": float(r["score"])}
            for r in rows
        ]

    async def delete(self, ids: list[str]) -> None:
        if ids:
            await self._db.execute(
                f"DELETE FROM {self._table} WHERE id = ANY($1)", ids
            )

    async def drop(self) -> None:
        await self._db.execute(f"DROP TABLE IF EXISTS {self._table}")
