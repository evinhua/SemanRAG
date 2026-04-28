"""Knowledge-graph CRUD and visualization routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/graph", tags=["graph"])


# ── Pydantic models ──────────────────────────────────────────────────

class EntityCreateRequest(BaseModel):
    name: str
    type: str = "UNKNOWN"
    description: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_id: str = ""


class EntityUpdateRequest(BaseModel):
    type: Optional[str] = None
    description: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class RelationCreateRequest(BaseModel):
    src: str
    tgt: str
    keywords: str = ""
    description: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None


class MergeRequest(BaseModel):
    source_entities: list[str]
    target_entity: str
    merge_strategy: str = "concatenate"
    target_entity_data: Optional[dict] = None


class GraphResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]


class CommunityResponse(BaseModel):
    community_id: str
    summary: Optional[str] = None
    members: list[str] = Field(default_factory=list)


# ── Routes ───────────────────────────────────────────────────────────

@router.get("", response_model=GraphResponse)
async def get_graph(
    request: Request,
    snapshot_at: Optional[str] = Query(None),
    community_level: Optional[int] = Query(None),
):
    rag = request.app.state.rag
    snap = datetime.fromisoformat(snapshot_at) if snapshot_at else None
    data = await rag.get_knowledge_graph(snapshot_at=snap, community_level=community_level)
    return GraphResponse(nodes=data.get("nodes", []), edges=data.get("edges", []))


@router.get("/labels")
async def get_labels(request: Request):
    rag = request.app.state.rag
    labels = await rag.graph_storage.get_all_labels()
    return {"labels": labels}


@router.get("/labels/popular")
async def get_popular_labels(request: Request, top_n: int = Query(20, ge=1)):
    rag = request.app.state.rag
    popular = await rag.graph_storage.get_popular_labels(top_n)
    return {"labels": [{"name": name, "degree": deg} for name, deg in popular]}


@router.post("/entities", status_code=201)
async def create_entity(body: EntityCreateRequest, request: Request):
    rag = request.app.state.rag
    data = body.model_dump(exclude={"name"})
    await rag.create_entity(body.name, data)
    return {"name": body.name, "status": "created"}


@router.put("/entities/{name}")
async def update_entity(name: str, body: EntityUpdateRequest, request: Request):
    rag = request.app.state.rag
    data = body.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(400, "No fields to update")
    await rag.edit_entity(name, data)
    return {"name": name, "status": "updated"}


@router.delete("/entities/{name}")
async def delete_entity(name: str, request: Request):
    rag = request.app.state.rag
    await rag.delete_by_entity(name)
    return {"name": name, "status": "deleted"}


@router.post("/relations", status_code=201)
async def create_relation(body: RelationCreateRequest, request: Request):
    rag = request.app.state.rag
    data = body.model_dump(exclude={"src", "tgt"})
    await rag.create_relation(body.src, body.tgt, data)
    return {"src": body.src, "tgt": body.tgt, "status": "created"}


@router.put("/relations")
async def update_relation(body: RelationCreateRequest, request: Request):
    rag = request.app.state.rag
    data = body.model_dump(exclude={"src", "tgt"})
    await rag.edit_relation(body.src, body.tgt, data)
    return {"src": body.src, "tgt": body.tgt, "status": "updated"}


@router.delete("/relations")
async def delete_relation(
    request: Request,
    src: str = Query(...),
    tgt: str = Query(...),
):
    rag = request.app.state.rag
    await rag.delete_by_relation(src, tgt)
    return {"src": src, "tgt": tgt, "status": "deleted"}


@router.post("/entities/merge", status_code=200)
async def merge_entities(body: MergeRequest, request: Request):
    rag = request.app.state.rag
    await rag.merge_entities(
        source_entities=body.source_entities,
        target_entity=body.target_entity,
        merge_strategy=body.merge_strategy,
        target_entity_data=body.target_entity_data,
    )
    return {"target_entity": body.target_entity, "merged": body.source_entities}


@router.get("/communities")
async def list_communities(request: Request):
    rag = request.app.state.rag
    labels = await rag.graph_storage.get_all_labels()
    communities: dict[str, list[str]] = {}
    for label in labels:
        node = await rag.graph_storage.get_node(label)
        if node and "community" in node:
            cid = str(node["community"])
            communities.setdefault(cid, []).append(label)

    results = []
    for cid, members in communities.items():
        summary = await rag.get_community_summary(cid)
        results.append(CommunityResponse(community_id=cid, summary=summary, members=members).model_dump())
    return {"communities": results}


@router.get("/communities/{community_id}", response_model=CommunityResponse)
async def get_community(community_id: str, request: Request):
    rag = request.app.state.rag
    summary = await rag.get_community_summary(community_id)
    labels = await rag.graph_storage.get_all_labels()
    members = []
    for label in labels:
        node = await rag.graph_storage.get_node(label)
        if node and str(node.get("community", "")) == community_id:
            members.append(label)
    if not members and summary is None:
        raise HTTPException(404, f"Community {community_id} not found")
    return CommunityResponse(community_id=community_id, summary=summary, members=members)


@router.get("/path")
async def shortest_path(
    request: Request,
    src: str = Query(...),
    tgt: str = Query(...),
):
    rag = request.app.state.rag
    # BFS shortest path
    graph = rag.graph_storage
    visited: set[str] = set()
    queue: list[tuple[str, list[str]]] = [(src, [src])]
    visited.add(src)

    while queue:
        current, path = queue.pop(0)
        if current == tgt:
            return {"path": path, "length": len(path) - 1}
        edges = await graph.get_node_edges(current)
        for s, t in edges:
            neighbor = t if s == current else s
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    raise HTTPException(404, f"No path found between {src} and {tgt}")


@router.get("/neighborhood/{name}", response_model=GraphResponse)
async def get_neighborhood(
    name: str,
    request: Request,
    hops: int = Query(1, ge=1, le=5),
):
    rag = request.app.state.rag
    data = await rag.graph_storage.get_knowledge_graph(name, max_depth=hops)
    return GraphResponse(nodes=data.get("nodes", []), edges=data.get("edges", []))
