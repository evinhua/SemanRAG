"""vLLM (OpenAI-compatible) LLM provider implementation."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import numpy as np

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]

supports_json_mode = True
supports_tools = False
supports_vision = False


def _ensure_openai():
    if AsyncOpenAI is None:
        raise ImportError("openai is required for vLLM client. Install with: pip install openai")


async def vllm_complete(
    model: str,
    prompt: str,
    system_prompt: str = "",
    history_messages: list[dict] | None = None,
    base_url: str = "http://localhost:8000/v1",
    response_schema=None,
    **kwargs,
) -> str | AsyncIterator[str]:
    _ensure_openai()
    base_url = base_url or os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
    client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    request_kwargs: dict = {k: v for k, v in kwargs.items() if k not in ("keyword_extraction", "stream")}
    if response_schema is not None:
        schema_dict = response_schema.model_json_schema() if hasattr(response_schema, "model_json_schema") else response_schema
        request_kwargs["extra_body"] = {"guided_json": json.dumps(schema_dict)}

    stream = kwargs.get("stream", False)

    if stream:
        async def _stream():
            resp = await client.chat.completions.create(
                model=model, messages=messages, stream=True, **request_kwargs
            )
            async for chunk in resp:
                delta = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                if delta:
                    yield delta
        return _stream()

    resp = await client.chat.completions.create(
        model=model, messages=messages, **request_kwargs
    )
    return resp.choices[0].message.content or ""


async def vllm_embed(
    texts: list[str],
    model: str = "default",
    base_url: str = "http://localhost:8000/v1",
) -> np.ndarray:
    _ensure_openai()
    base_url = base_url or os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
    client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)
    resp = await client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data])
