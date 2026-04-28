"""Ollama-compatible API routes for Open WebUI integration."""

from __future__ import annotations

import json
import time
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from semanrag.base import QueryParam

router = APIRouter(prefix="/api", tags=["ollama"])


# ── Pydantic models (Ollama wire format) ─────────────────────────────

class OllamaMessage(BaseModel):
    role: str
    content: str


class OllamaChatRequest(BaseModel):
    model: str = "semanrag"
    messages: list[OllamaMessage] = Field(default_factory=list)
    stream: bool = True
    options: Optional[dict] = None


class OllamaGenerateRequest(BaseModel):
    model: str = "semanrag"
    prompt: str = ""
    stream: bool = True
    options: Optional[dict] = None


# ── Helpers ──────────────────────────────────────────────────────────

def _build_param_from_ollama(options: dict | None, stream: bool, history: list[dict]) -> QueryParam:
    opts = options or {}
    return QueryParam(
        mode=opts.get("mode", "local"),
        top_k=opts.get("top_k", 20),
        stream=stream,
        conversation_history=history,
    )


def _ollama_chat_response(content: str, model: str, done: bool = True) -> dict:
    return {
        "model": model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "message": {"role": "assistant", "content": content},
        "done": done,
    }


def _ollama_generate_response(response: str, model: str, done: bool = True) -> dict:
    return {
        "model": model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "response": response,
        "done": done,
    }


# ── Routes ───────────────────────────────────────────────────────────

@router.post("/chat")
async def ollama_chat(body: OllamaChatRequest, request: Request):
    rag = request.app.state.rag
    if not body.messages:
        return _ollama_chat_response("", body.model)

    query = body.messages[-1].content
    history = [{"role": m.role, "content": m.content} for m in body.messages[:-1]]
    param = _build_param_from_ollama(body.options, body.stream, history)
    result = await rag.aquery(query, param=param)

    if body.stream and result.is_streaming and result.response_iterator:
        async def stream_gen():
            async for chunk in result.response_iterator:
                yield json.dumps(_ollama_chat_response(chunk, body.model, done=False)) + "\n"
            yield json.dumps(_ollama_chat_response("", body.model, done=True)) + "\n"
        return StreamingResponse(stream_gen(), media_type="application/x-ndjson")

    return _ollama_chat_response(result.content, body.model)


@router.post("/generate")
async def ollama_generate(body: OllamaGenerateRequest, request: Request):
    rag = request.app.state.rag
    param = _build_param_from_ollama(body.options, body.stream, [])
    result = await rag.aquery(body.prompt, param=param)

    if body.stream and result.is_streaming and result.response_iterator:
        async def stream_gen():
            async for chunk in result.response_iterator:
                yield json.dumps(_ollama_generate_response(chunk, body.model, done=False)) + "\n"
            yield json.dumps(_ollama_generate_response("", body.model, done=True)) + "\n"
        return StreamingResponse(stream_gen(), media_type="application/x-ndjson")

    return _ollama_generate_response(result.content, body.model)


@router.get("/tags")
async def list_models(request: Request):
    rag = request.app.state.rag
    model_name = rag.llm_model_name or "semanrag"
    return {
        "models": [
            {
                "name": model_name,
                "model": model_name,
                "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "size": 0,
                "digest": "",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "semanrag",
                    "parameter_size": "N/A",
                    "quantization_level": "N/A",
                },
            }
        ]
    }
