"""Knowledge-graph CRUD, merge, traversal, and community operations."""

from __future__ import annotations

import copy
from collections import deque
from datetime import UTC, datetime
from typing import Any

from semanrag.base import BaseGraphStorage, BaseLexicalStorage, BaseVectorStorage
from semanrag.operate import _merge_nodes_then_upsert, build_communities
from semanrag.utils import EmbeddingFunc, compute_mdhash_id, logger

# ── module-level edit history ────────────────────────────────────────────
_edit_history: dict[str, list[dict]] = {}


def _record_history(
    key: str, action: str, user_id: str, before: Any, after: Any
) -> None:
    _edit_history.setdefault(key, []).append(
        {
            "action": action,
            "user_id": user_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "before": before,
            "after": after,
        }
    )


# ── internal helpers ─────────────────────────────────────────────────────

def _entity_key(name: str) -> str:
    return f"entity::{name}"


def _edge_key(src: str, tgt: str) -> str:
    return f"edge::{src}::{tgt}"


async def _embed_and_upsert_entity(
    name: str,
    node_data: dict,
    entities_vdb: BaseVectorStorage,
    lexical_storage: BaseLexicalStorage | None,
    embedding_func: EmbeddingFunc,
) -> None:
    content = f"{name}: {node_data.get('description', '')}"
    embeddings = await embedding_func([content])
    embed_vec = embeddings[0] if len(embeddings) > 0 else None
    await entities_vdb.upsert(
        {
            compute_mdhash_id(name): {
                "entity_name": name,
                "content": content,
                "embedding": embed_vec,
            }
        }
    )
    if lexical_storage is not None:
        await lexical_storage.upsert(
            {
                compute_mdhash_id(name): {
                    "entity_name": name,
                    "content": f"{name} ({node_data.get('type', '')}): {node_data.get('description', '')}",
                }
            }
        )


async def _embed_and_upsert_edge(
    src: str,
    tgt: str,
    edge_data: dict,
    relationships_vdb: BaseVectorStorage,
    embedding_func: EmbeddingFunc,
) -> None:
    content = f"{src} -> {tgt}: {edge_data.get('description', '')}"
    embeddings = await embedding_func([content])
    embed_vec = embeddings[0] if len(embeddings) > 0 else None
    edge_id = compute_mdhash_id(f"{src}-{tgt}")
    await relationships_vdb.upsert(
        {
            edge_id: {
                "src_id": src,
                "tgt_id": tgt,
                "content": content,
                "embedding": embed_vec,
            }
        }
    )


# ═════════════════════════════════════════════════════════════════════════
# 1. aedit_entity
# ═════════════════════════════════════════════════════════════════════════

async def aedit_entity(
    entity_name: str,
    data: dict,
    knowledge_graph: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    lexical_storage: BaseLexicalStorage,
    global_config: dict,
    user_id: str = "",
) -> dict:
    existing = await knowledge_graph.get_node(entity_name)
    if existing is None:
        raise ValueError(f"Entity '{entity_name}' not found")

    if "description" in data and not data["description"].strip():
        raise ValueError("Description must not be empty")

    before = copy.deepcopy(existing)
    embedding_func = global_config["embedding_func"]
    new_name = data.pop("new_name", None)

    # Apply attribute updates to existing node data
    updated = {**existing, **{k: v for k, v in data.items() if v is not None}}

    if new_name and new_name != entity_name:
        # Migrate relationships to new name
        edges = await knowledge_graph.get_node_edges(entity_name)
        for src, tgt in edges:
            edge_data = await knowledge_graph.get_edge(src, tgt)
            if edge_data is None:
                continue
            old_edge_id = compute_mdhash_id(f"{src}-{tgt}")
            new_src = new_name if src == entity_name else src
            new_tgt = new_name if tgt == entity_name else tgt
            await knowledge_graph.upsert_edge(new_src, new_tgt, edge_data)
            await _embed_and_upsert_edge(new_src, new_tgt, edge_data, entities_vdb, embedding_func)
            # Remove old edge artifacts
            await knowledge_graph.remove_edges([(src, tgt)])
            try:
                await entities_vdb.delete([old_edge_id])
            except Exception:
                pass

        # Remove old entity from all stores
        old_id = compute_mdhash_id(entity_name)
        await knowledge_graph.delete_node(entity_name)
        try:
            await entities_vdb.delete([old_id])
        except Exception:
            pass
        try:
            await lexical_storage.delete([old_id])
        except Exception:
            pass

        # Create under new name
        await knowledge_graph.upsert_node(new_name, updated)
        await _embed_and_upsert_entity(new_name, updated, entities_vdb, lexical_storage, embedding_func)
        _record_history(_entity_key(new_name), "rename", user_id, before, {**updated, "old_name": entity_name})
        return {**updated, "entity_name": new_name}

    await knowledge_graph.upsert_node(entity_name, updated)
    await _embed_and_upsert_entity(entity_name, updated, entities_vdb, lexical_storage, embedding_func)
    _record_history(_entity_key(entity_name), "edit", user_id, before, updated)
    return {**updated, "entity_name": entity_name}


# ═════════════════════════════════════════════════════════════════════════
# 2. aedit_relation
# ═════════════════════════════════════════════════════════════════════════

async def aedit_relation(
    src: str,
    tgt: str,
    data: dict,
    knowledge_graph: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
    user_id: str = "",
) -> dict:
    existing = await knowledge_graph.get_edge(src, tgt)
    if existing is None:
        raise ValueError(f"Relation '{src}' -> '{tgt}' not found")

    if "description" in data and not data["description"].strip():
        raise ValueError("Description must not be empty")

    before = copy.deepcopy(existing)
    updated = {**existing, **{k: v for k, v in data.items() if v is not None}}

    await knowledge_graph.upsert_edge(src, tgt, updated)
    embedding_func = global_config["embedding_func"]
    await _embed_and_upsert_edge(src, tgt, updated, relationships_vdb, embedding_func)

    _record_history(_edge_key(src, tgt), "edit", user_id, before, updated)
    return {**updated, "src": src, "tgt": tgt}


# ═════════════════════════════════════════════════════════════════════════
# 3. acreate_entity
# ═════════════════════════════════════════════════════════════════════════

async def acreate_entity(
    entity_name: str,
    data: dict,
    knowledge_graph: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    lexical_storage: BaseLexicalStorage,
    global_config: dict,
    user_id: str = "",
) -> dict:
    if await knowledge_graph.has_node(entity_name):
        raise ValueError(f"Entity '{entity_name}' already exists")

    node_data = {
        "type": data.get("type", "UNKNOWN"),
        "description": data.get("description", ""),
        "source_id": data.get("source_id", ""),
        "confidence": data.get("confidence", 0.5),
    }

    await knowledge_graph.upsert_node(entity_name, node_data)
    embedding_func = global_config["embedding_func"]
    await _embed_and_upsert_entity(entity_name, node_data, entities_vdb, lexical_storage, embedding_func)

    _record_history(_entity_key(entity_name), "create", user_id, None, node_data)
    return {**node_data, "entity_name": entity_name}


# ═════════════════════════════════════════════════════════════════════════
# 4. acreate_relation
# ═════════════════════════════════════════════════════════════════════════

async def acreate_relation(
    src: str,
    tgt: str,
    data: dict,
    knowledge_graph: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    entities_vdb: BaseVectorStorage,
    global_config: dict,
    user_id: str = "",
) -> dict:
    if not await knowledge_graph.has_node(src):
        raise ValueError(f"Source entity '{src}' does not exist")
    if not await knowledge_graph.has_node(tgt):
        raise ValueError(f"Target entity '{tgt}' does not exist")

    edge_data = {
        "keywords": data.get("keywords", ""),
        "description": data.get("description", ""),
        "source_id": data.get("source_id", ""),
        "confidence": data.get("confidence", 0.5),
        "valid_from": data.get("valid_from"),
        "valid_to": data.get("valid_to"),
    }

    await knowledge_graph.upsert_edge(src, tgt, edge_data)
    embedding_func = global_config["embedding_func"]
    await _embed_and_upsert_edge(src, tgt, edge_data, relationships_vdb, embedding_func)

    _record_history(_edge_key(src, tgt), "create", user_id, None, edge_data)
    return {**edge_data, "src": src, "tgt": tgt}


# ═════════════════════════════════════════════════════════════════════════
# 5. adelete_by_entity
# ═════════════════════════════════════════════════════════════════════════

async def adelete_by_entity(
    entity_name: str,
    knowledge_graph: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    lexical_storage: BaseLexicalStorage,
    global_config: dict,
    user_id: str = "",
) -> dict:
    existing = await knowledge_graph.get_node(entity_name)
    if existing is None:
        raise ValueError(f"Entity '{entity_name}' not found")

    edges = await knowledge_graph.get_node_edges(entity_name)
    deleted_edges: list[dict] = []
    neighbors: set[str] = set()

    for src, tgt in edges:
        edge_data = await knowledge_graph.get_edge(src, tgt)
        deleted_edges.append({"src": src, "tgt": tgt, **(edge_data or {})})
        neighbor = tgt if src == entity_name else src
        neighbors.add(neighbor)
        # Remove edge from VDB
        edge_id = compute_mdhash_id(f"{src}-{tgt}")
        try:
            await relationships_vdb.delete([edge_id])
        except Exception:
            pass

    # Remove edges from graph
    if edges:
        await knowledge_graph.remove_edges(edges)

    # Remove entity from graph, VDB, BM25
    entity_id = compute_mdhash_id(entity_name)
    await knowledge_graph.delete_node(entity_name)
    try:
        await entities_vdb.delete([entity_id])
    except Exception:
        pass
    try:
        await lexical_storage.delete([entity_id])
    except Exception:
        pass

    # Orphan scan: flag neighbors that now have degree 0
    orphans: list[str] = []
    for nb in neighbors:
        if await knowledge_graph.has_node(nb):
            deg = await knowledge_graph.node_degree(nb)
            if deg == 0:
                orphans.append(nb)
                logger.warning(f"Orphan detected after deleting '{entity_name}': '{nb}' has degree 0")

    summary = {
        "entity": entity_name,
        "entity_data": existing,
        "deleted_edges": deleted_edges,
        "orphaned_neighbors": orphans,
    }
    _record_history(_entity_key(entity_name), "delete", user_id, existing, None)
    for e in deleted_edges:
        _record_history(_edge_key(e["src"], e["tgt"]), "delete", user_id, e, None)
    return summary


# ═════════════════════════════════════════════════════════════════════════
# 6. adelete_by_relation
# ═════════════════════════════════════════════════════════════════════════

async def adelete_by_relation(
    src: str,
    tgt: str,
    knowledge_graph: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
    user_id: str = "",
) -> dict:
    existing = await knowledge_graph.get_edge(src, tgt)
    if existing is None:
        raise ValueError(f"Relation '{src}' -> '{tgt}' not found")

    await knowledge_graph.remove_edges([(src, tgt)])
    edge_id = compute_mdhash_id(f"{src}-{tgt}")
    try:
        await relationships_vdb.delete([edge_id])
    except Exception:
        pass

    _record_history(_edge_key(src, tgt), "delete", user_id, existing, None)
    return {**existing, "src": src, "tgt": tgt}


# ═════════════════════════════════════════════════════════════════════════
# 7. amerge_entities
# ═════════════════════════════════════════════════════════════════════════

def _merge_descriptions(existing: str, incoming: str, strategy: str) -> str:
    if strategy == "keep_first":
        return existing if existing else incoming
    if strategy == "join_unique":
        seen: set[str] = set()
        result: list[str] = []
        for s in (existing + "\n" + incoming).split("\n"):
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                result.append(s)
        return "\n".join(result)
    # default: concatenate
    parts = [p for p in [existing, incoming] if p.strip()]
    return "\n".join(parts)


async def amerge_entities(
    source_entities: list[str],
    target_entity: str,
    knowledge_graph: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    lexical_storage: BaseLexicalStorage,
    global_config: dict,
    merge_strategy: str = "concatenate",
    target_entity_data: dict | None = None,
    user_id: str = "",
) -> dict:
    embedding_func = global_config["embedding_func"]

    # Get or create target node
    target_node = await knowledge_graph.get_node(target_entity)
    if target_node is None:
        target_node = target_entity_data or {
            "type": "UNKNOWN",
            "description": "",
            "source_id": "",
            "confidence": 0.5,
        }
        await knowledge_graph.upsert_node(target_entity, target_node)

    before_target = copy.deepcopy(target_node)
    merged_sources: set[str] = set(target_node.get("source_id", "").split(","))
    merged_desc = target_node.get("description", "")
    merged_confidence = target_node.get("confidence", 0.0)
    merged_type = target_node.get("type", "UNKNOWN")
    sources_merged: list[str] = []

    for src_entity in source_entities:
        if src_entity == target_entity:
            continue
        src_node = await knowledge_graph.get_node(src_entity)
        if src_node is None:
            logger.warning(f"Merge source '{src_entity}' not found, skipping")
            continue

        sources_merged.append(src_entity)
        src_desc = src_node.get("description", "")
        merged_desc = _merge_descriptions(merged_desc, src_desc, merge_strategy)

        if merge_strategy == "confidence_weighted":
            if src_node.get("confidence", 0.0) > merged_confidence:
                merged_confidence = src_node["confidence"]
                merged_type = src_node.get("type", merged_type)
        else:
            merged_confidence = max(merged_confidence, src_node.get("confidence", 0.0))

        merged_sources |= set(src_node.get("source_id", "").split(","))

        # Redirect relationships
        edges = await knowledge_graph.get_node_edges(src_entity)
        for e_src, e_tgt in edges:
            edge_data = await knowledge_graph.get_edge(e_src, e_tgt)
            if edge_data is None:
                continue
            new_src = target_entity if e_src == src_entity else e_src
            new_tgt = target_entity if e_tgt == src_entity else e_tgt
            # Prevent self-loops
            if new_src == new_tgt:
                continue
            await knowledge_graph.upsert_edge(new_src, new_tgt, edge_data)
            await _embed_and_upsert_edge(new_src, new_tgt, edge_data, relationships_vdb, embedding_func)

        # Remove old edges and source entity from all stores
        if edges:
            await knowledge_graph.remove_edges(edges)
            for e_src, e_tgt in edges:
                old_eid = compute_mdhash_id(f"{e_src}-{e_tgt}")
                try:
                    await relationships_vdb.delete([old_eid])
                except Exception:
                    pass

        src_id = compute_mdhash_id(src_entity)
        await knowledge_graph.delete_node(src_entity)
        try:
            await entities_vdb.delete([src_id])
        except Exception:
            pass
        try:
            await lexical_storage.delete([src_id])
        except Exception:
            pass

    # Update target node
    merged_sources.discard("")
    updated_target = {
        "type": merged_type,
        "description": merged_desc,
        "source_id": ",".join(merged_sources),
        "confidence": merged_confidence,
    }
    await knowledge_graph.upsert_node(target_entity, updated_target)
    await _embed_and_upsert_entity(target_entity, updated_target, entities_vdb, lexical_storage, embedding_func)

    _record_history(
        _entity_key(target_entity),
        "merge",
        user_id,
        {"target_before": before_target, "sources": sources_merged},
        updated_target,
    )
    return {**updated_target, "entity_name": target_entity, "merged_from": sources_merged}


# ═════════════════════════════════════════════════════════════════════════
# 8–11. Info & history getters
# ═════════════════════════════════════════════════════════════════════════

async def get_entity_info(
    entity_name: str, knowledge_graph: BaseGraphStorage
) -> dict | None:
    node = await knowledge_graph.get_node(entity_name)
    if node is None:
        return None
    return {**node, "entity_name": entity_name}


async def get_relation_info(
    src: str, tgt: str, knowledge_graph: BaseGraphStorage
) -> dict | None:
    edge = await knowledge_graph.get_edge(src, tgt)
    if edge is None:
        return None
    return {**edge, "src": src, "tgt": tgt}


async def get_entity_edit_history(
    entity_name: str, knowledge_graph: BaseGraphStorage
) -> list[dict]:
    return list(_edit_history.get(_entity_key(entity_name), []))


async def get_relation_edit_history(
    src: str, tgt: str, knowledge_graph: BaseGraphStorage
) -> list[dict]:
    return list(_edit_history.get(_edge_key(src, tgt), []))


# ═════════════════════════════════════════════════════════════════════════
# 12. afind_path
# ═════════════════════════════════════════════════════════════════════════

async def afind_path(
    src: str,
    tgt: str,
    knowledge_graph: BaseGraphStorage,
    max_hops: int = 5,
    snapshot_at=None,
) -> list[str]:
    if not await knowledge_graph.has_node(src) or not await knowledge_graph.has_node(tgt):
        return []

    visited: set[str] = {src}
    queue: deque[tuple[str, list[str]]] = deque([(src, [src])])

    while queue:
        current, path = queue.popleft()
        if current == tgt:
            return path
        if len(path) - 1 >= max_hops:
            continue
        edges = await knowledge_graph.get_node_edges(current)
        for e_src, e_tgt in edges:
            # Optional temporal filter
            if snapshot_at is not None:
                edge_data = await knowledge_graph.get_edge(e_src, e_tgt)
                if edge_data:
                    vf = edge_data.get("valid_from")
                    vt = edge_data.get("valid_to")
                    if vf and snapshot_at < vf:
                        continue
                    if vt and snapshot_at > vt:
                        continue
            neighbor = e_tgt if e_src == current else e_src
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []


# ═════════════════════════════════════════════════════════════════════════
# 13. aneighborhood
# ═════════════════════════════════════════════════════════════════════════

async def aneighborhood(
    entity_name: str,
    knowledge_graph: BaseGraphStorage,
    hops: int = 2,
    snapshot_at=None,
) -> dict:
    if not await knowledge_graph.has_node(entity_name):
        return {"nodes": [], "edges": []}

    visited: set[str] = {entity_name}
    queue: deque[tuple[str, int]] = deque([(entity_name, 0)])
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()

    while queue:
        current, depth = queue.popleft()
        node_data = await knowledge_graph.get_node(current)
        nodes.append({"id": current, **(node_data or {})})

        if depth >= hops:
            continue

        node_edges = await knowledge_graph.get_node_edges(current)
        for e_src, e_tgt in node_edges:
            if snapshot_at is not None:
                edge_data = await knowledge_graph.get_edge(e_src, e_tgt)
                if edge_data:
                    vf = edge_data.get("valid_from")
                    vt = edge_data.get("valid_to")
                    if vf and snapshot_at < vf:
                        continue
                    if vt and snapshot_at > vt:
                        continue

            edge_key = (e_src, e_tgt)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edge_data = await knowledge_graph.get_edge(e_src, e_tgt)
                edges.append({"src": e_src, "tgt": e_tgt, **(edge_data or {})})

            neighbor = e_tgt if e_src == current else e_src
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return {"nodes": nodes, "edges": edges}


# ═════════════════════════════════════════════════════════════════════════
# 14. arun_entity_resolution
# ═════════════════════════════════════════════════════════════════════════

async def arun_entity_resolution(
    knowledge_graph: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    global_config: dict,
    threshold: float = 0.88,
) -> list[dict]:
    try:
        import numpy as _np
    except ImportError:
        logger.error("numpy is required for entity resolution")
        return []

    labels = await knowledge_graph.get_all_labels()
    if len(labels) < 2:
        return []

    # Fetch embeddings for all entities
    ids = [compute_mdhash_id(label) for label in labels]
    vectors = await entities_vdb.get_vectors_by_ids(ids)

    # Build label->vector mapping (skip missing)
    label_vecs: list[tuple[str, Any]] = []
    for label, vec_data in zip(labels, vectors):
        if vec_data is not None and vec_data.get("embedding") is not None:
            label_vecs.append((label, _np.array(vec_data["embedding"], dtype=_np.float32)))

    suggestions: list[dict] = []
    for i in range(len(label_vecs)):
        name_a, vec_a = label_vecs[i]
        norm_a = _np.linalg.norm(vec_a)
        if norm_a == 0:
            continue
        for j in range(i + 1, len(label_vecs)):
            name_b, vec_b = label_vecs[j]
            norm_b = _np.linalg.norm(vec_b)
            if norm_b == 0:
                continue
            sim = float(_np.dot(vec_a, vec_b) / (norm_a * norm_b))
            if sim >= threshold:
                suggestions.append(
                    {
                        "entity_a": name_a,
                        "entity_b": name_b,
                        "similarity": round(sim, 4),
                        "suggestion": f"Consider merging '{name_a}' and '{name_b}' (similarity={sim:.4f})",
                    }
                )

    suggestions.sort(key=lambda x: x["similarity"], reverse=True)
    return suggestions


# ═════════════════════════════════════════════════════════════════════════
# 15. abuild_communities
# ═════════════════════════════════════════════════════════════════════════

async def abuild_communities(
    knowledge_graph: BaseGraphStorage,
    global_config: dict,
    llm_response_cache=None,
) -> dict:
    return await build_communities(knowledge_graph, global_config, llm_response_cache)


# ═════════════════════════════════════════════════════════════════════════
# 16. aincremental_community_update
# ═════════════════════════════════════════════════════════════════════════

async def aincremental_community_update(
    changed_entities: list[str],
    knowledge_graph: BaseGraphStorage,
    global_config: dict,
    llm_response_cache=None,
) -> dict:
    # Collect the affected subgraph: changed entities + their immediate neighbors
    affected: set[str] = set(changed_entities)
    for entity in changed_entities:
        if not await knowledge_graph.has_node(entity):
            continue
        edges = await knowledge_graph.get_node_edges(entity)
        for src, tgt in edges:
            affected.add(src)
            affected.add(tgt)

    # Get current community assignments for affected nodes
    old_assignments: dict[str, str] = {}
    affected_communities: set[str] = set()
    for node_name in affected:
        node = await knowledge_graph.get_node(node_name)
        if node and "community" in node:
            cid = str(node["community"])
            old_assignments[node_name] = cid
            affected_communities.add(cid)

    # Re-run full community detection (the graph storage handles the algorithm)
    full_result = await build_communities(knowledge_graph, global_config, llm_response_cache)

    # Filter to only the communities that were affected
    communities = full_result.get("communities", {})
    updated: dict[str, Any] = {}
    for cid, cdata in communities.items():
        members = cdata.get("members", [])
        if affected_communities & {cid} or affected & set(members):
            updated[cid] = cdata

    return {
        "affected_entities": list(affected),
        "updated_communities": updated,
        "total_communities": len(communities),
    }
