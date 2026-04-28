"""Zhipu AI (GLM) LLM provider implementation."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import numpy as np

try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = None  # type: ignore[assignment,misc]

supports_json_mode = True
supports_tools = True
supports_vision = False


def _ensure_zhipu():
    if ZhipuAI is None:
        raise ImportError("zhipuai is required. Install with: pip install zhipuai")


async def zhipu_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str = "",
    history_messages: list[dict] | None = None,
    api_key: str | None = None,
    **kwargs,
) -> str | AsyncIterator[str]:
    _ensure_zhipu()
    import asyncio
    api_key = api_key or os.environ.get("ZHIPU_API_KEY", "")
    client = ZhipuAI(api_key=api_key)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    stream = kwargs.pop("stream", False)

    if stream:
        async def _stream():
            resp = await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.chat.completions.create(model=model, messages=messages, stream=True, **kwargs)
            )
            for chunk in resp:
                delta = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                if delta:
                    yield delta
        return _stream()

    resp = await asyncio.get_event_loop().run_in_executor(
        None, lambda: client.chat.completions.create(model=model, messages=messages, **kwargs)
    )
    return resp.choices[0].message.content or ""


async def zhipu_embedding(
    texts: list[str],
    model: str = "embedding-3",
    api_key: str | None = None,
) -> np.ndarray:
    _ensure_zhipu()
    import asyncio
    api_key = api_key or os.environ.get("ZHIPU_API_KEY", "")
    client = ZhipuAI(api_key=api_key)
    embeddings = []
    for text in texts:
        resp = await asyncio.get_event_loop().run_in_executor(
            None, lambda t=text: client.embeddings.create(model=model, input=t)
        )
        embeddings.append(resp.data[0].embedding)
    return np.array(embeddings)
