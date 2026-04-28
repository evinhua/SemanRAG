"""Ollama LLM provider implementation."""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

import numpy as np

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore[assignment]

supports_json_mode = True
supports_tools = False
supports_vision = False


def _ensure_ollama():
    if ollama is None:
        raise ImportError("ollama is required. Install with: pip install ollama")


async def ollama_model_complete(
    model: str,
    prompt: str,
    system_prompt: str = "",
    history_messages: list[dict] | None = None,
    host: str | None = None,
    response_schema=None,
    **kwargs,
) -> str | AsyncIterator[str]:
    _ensure_ollama()
    host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    client = ollama.AsyncClient(host=host)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    request_kwargs: dict = {k: v for k, v in kwargs.items() if k not in ("keyword_extraction",)}
    if response_schema is not None:
        request_kwargs["format"] = "json"
        schema_hint = json.dumps(response_schema.model_json_schema() if hasattr(response_schema, "model_json_schema") else response_schema, indent=2)
        messages[-1]["content"] += f"\n\nRespond with valid JSON matching this schema:\n{schema_hint}"

    stream = request_kwargs.pop("stream", False)

    if stream:
        async def _stream():
            resp = await client.chat(model=model, messages=messages, stream=True, **request_kwargs)
            async for chunk in resp:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        return _stream()

    resp = await client.chat(model=model, messages=messages, **request_kwargs)
    return resp["message"]["content"]


async def ollama_embed(
    texts: list[str],
    model: str = "nomic-embed-text",
    host: str | None = None,
) -> np.ndarray:
    _ensure_ollama()
    host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    client = ollama.AsyncClient(host=host)
    resp = await client.embed(model=model, input=texts)
    return np.array(resp["embeddings"])
