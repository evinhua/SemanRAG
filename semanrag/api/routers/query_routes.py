"""Query routes – non-streaming, SSE, WebSocket, data, explain, compare."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from semanrag.base import QueryParam, QueryResult

router = APIRouter(prefix="/query", tags=["query"])


# ── Pydantic models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    mode: str = "local"
    top_k: int = 20
    stream: bool = False
    conversation_history: list[dict] = Field(default_factory=list)
    snapshot_at: str | None = None
    user_id: str | None = None
    user_groups: list[str] = Field(default_factory=list)
    response_type: str = "Multiple Paragraphs"
    enable_rerank: bool = True
    verifier_enabled: bool = True


class QueryResponse(BaseModel):
    answer: str
    references: list[dict] = Field(default_factory=list)
    communities_used: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    tokens_used: dict = Field(default_factory=dict)


class QueryDataResponse(BaseModel):
    answer: str
    references: list[dict] = Field(default_factory=list)
    grounded_check: list[dict] = Field(default_factory=list)
    latency_ms: float = 0.0


class QueryExplainResponse(BaseModel):
    query: str
    mode: str
    retrieved_entities: list[dict] = Field(default_factory=list)
    retrieved_relations: list[dict] = Field(default_factory=list)
    retrieved_chunks: list[dict] = Field(default_factory=list)
    communities_used: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class QueryParamVariant(BaseModel):
    mode: str = "local"
    top_k: int = 20
    enable_rerank: bool = True
    response_type: str = "Multiple Paragraphs"


class QueryCompareRequest(BaseModel):
    query: str
    variant_a: QueryParamVariant
    variant_b: QueryParamVariant
    user_id: str | None = None
    user_groups: list[str] = Field(default_factory=list)


class QueryCompareResponse(BaseModel):
    query: str
    result_a: QueryResponse
    result_b: QueryResponse


# ── Helpers ──────────────────────────────────────────────────────────

def _build_param(body: QueryRequest) -> QueryParam:
    snapshot = None
    if body.snapshot_at:
        snapshot = datetime.fromisoformat(body.snapshot_at)
    return QueryParam(
        mode=body.mode,
        top_k=body.top_k,
        stream=body.stream,
        conversation_history=body.conversation_history,
        snapshot_at=snapshot,
        user_id=body.user_id,
        user_groups=body.user_groups,
        response_type=body.response_type,
        enable_rerank=body.enable_rerank,
        verifier_enabled=body.verifier_enabled,
    )


def _result_to_response(result: QueryResult) -> QueryResponse:
    return QueryResponse(
        answer=result.content,
        references=result.references,
        communities_used=result.communities_used,
        latency_ms=result.latency_ms,
        tokens_used=result.tokens_used,
    )


# ── Routes ───────────────────────────────────────────────────────────

@router.post("", response_model=QueryResponse)
async def query_non_streaming(body: QueryRequest, request: Request):
    rag = request.app.state.rag
    param = _build_param(body)
    param.stream = False
    result = await rag.aquery(body.query, param=param)
    return _result_to_response(result)


@router.post("/stream")
async def query_stream(body: QueryRequest, request: Request):
    rag = request.app.state.rag
    param = _build_param(body)
    param.stream = True
    result = await rag.aquery(body.query, param=param)

    async def event_generator():
        if result.is_streaming and result.response_iterator:
            async for chunk in result.response_iterator:
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True, 'latency_ms': result.latency_ms})}\n\n"
        else:
            yield f"data: {json.dumps({'chunk': result.content, 'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.websocket("/ws")
async def query_websocket(websocket: WebSocket):
    await websocket.accept()
    rag = websocket.app.state.rag
    try:
        while True:
            data = await websocket.receive_json()
            body = QueryRequest(**data)
            param = _build_param(body)
            param.stream = True
            t0 = time.monotonic()
            result = await rag.aquery(body.query, param=param)

            if result.is_streaming and result.response_iterator:
                async for chunk in result.response_iterator:
                    await websocket.send_json({"type": "chunk", "content": chunk})
            else:
                await websocket.send_json({"type": "chunk", "content": result.content})

            await websocket.send_json({
                "type": "done",
                "references": result.references,
                "communities_used": result.communities_used,
                "latency_ms": (time.monotonic() - t0) * 1000,
                "tokens_used": result.tokens_used,
            })
    except WebSocketDisconnect:
        pass


@router.post("/data", response_model=QueryDataResponse)
async def query_data(body: QueryRequest, request: Request):
    rag = request.app.state.rag
    param = _build_param(body)
    param.stream = False
    result = await rag.aquery_data(body.query, param=param)
    return QueryDataResponse(
        answer=result.content,
        references=result.references,
        grounded_check=result.grounded_check,
        latency_ms=result.latency_ms,
    )


@router.post("/explain", response_model=QueryExplainResponse)
async def query_explain(body: QueryRequest, request: Request):
    rag = request.app.state.rag
    param = _build_param(body)
    param.only_need_context = True
    param.stream = False
    t0 = time.monotonic()
    result = await rag.aquery(body.query, param=param)
    raw = result.raw_data or {}
    return QueryExplainResponse(
        query=body.query,
        mode=body.mode,
        retrieved_entities=raw.get("entities", []),
        retrieved_relations=raw.get("relations", []),
        retrieved_chunks=raw.get("chunks", []),
        communities_used=result.communities_used,
        latency_ms=(time.monotonic() - t0) * 1000,
    )


@router.post("/compare", response_model=QueryCompareResponse)
async def query_compare(body: QueryCompareRequest, request: Request):
    rag = request.app.state.rag

    async def _run(variant: QueryParamVariant) -> QueryResult:
        param = QueryParam(
            mode=variant.mode,
            top_k=variant.top_k,
            enable_rerank=variant.enable_rerank,
            response_type=variant.response_type,
            stream=False,
            user_id=body.user_id,
            user_groups=body.user_groups,
        )
        return await rag.aquery(body.query, param=param)

    result_a, result_b = await asyncio.gather(
        _run(body.variant_a), _run(body.variant_b)
    )
    return QueryCompareResponse(
        query=body.query,
        result_a=_result_to_response(result_a),
        result_b=_result_to_response(result_b),
    )
