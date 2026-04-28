"""LlamaIndex LLM provider implementation."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import numpy as np

try:
    from llama_index.core.llms import ChatMessage
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
except ImportError:
    ChatMessage = None  # type: ignore[assignment,misc]
    LlamaOpenAI = None  # type: ignore[assignment,misc]

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError:
    OpenAIEmbedding = None  # type: ignore[assignment,misc]

supports_json_mode = True
supports_tools = True
supports_vision = False


def _ensure_llama_index():
    if ChatMessage is None:
        raise ImportError("llama-index is required. Install with: pip install llama-index")


async def llama_index_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str = "",
    history_messages: list[dict] | None = None,
    response_schema=None,
    **kwargs,
) -> str | AsyncIterator[str]:
    _ensure_llama_index()
    llm = LlamaOpenAI(model=model, api_key=os.environ.get("OPENAI_API_KEY", ""))

    messages = []
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))
    if history_messages:
        for msg in history_messages:
            messages.append(ChatMessage(role=msg.get("role", "user"), content=msg["content"]))
    messages.append(ChatMessage(role="user", content=prompt))

    if response_schema is not None:
        import json
        schema_hint = json.dumps(response_schema.model_json_schema() if hasattr(response_schema, "model_json_schema") else response_schema, indent=2)
        messages[-1] = ChatMessage(role="user", content=prompt + f"\n\nRespond with valid JSON matching this schema:\n{schema_hint}")

    stream = kwargs.get("stream", False)

    if stream:
        async def _stream():
            resp = await llm.astream_chat(messages)
            async for chunk in resp:
                yield chunk.delta or ""
        return _stream()

    resp = await llm.achat(messages)
    return resp.message.content or ""


async def llama_index_embed(
    texts: list[str],
    model: str = "default",
) -> np.ndarray:
    if OpenAIEmbedding is None:
        raise ImportError("llama-index-embeddings-openai is required. Install with: pip install llama-index-embeddings-openai")
    embed_model = OpenAIEmbedding(model=model if model != "default" else "text-embedding-3-small", api_key=os.environ.get("OPENAI_API_KEY", ""))
    results = await embed_model.aget_text_embedding_batch(texts)
    return np.array(results)
