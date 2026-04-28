"""Anthropic Claude LLM provider implementation."""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

import numpy as np

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

supports_json_mode = False
supports_tools = True
supports_vision = True


def _ensure_anthropic():
    if anthropic is None:
        raise ImportError("anthropic is required. Install with: pip install anthropic")


async def anthropic_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str = "",
    history_messages: list[dict] | None = None,
    api_key: str | None = None,
    response_schema=None,
    **kwargs,
) -> str | AsyncIterator[str]:
    _ensure_anthropic()
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.AsyncAnthropic(api_key=api_key)

    messages = []
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    request_kwargs: dict = {k: v for k, v in kwargs.items() if k not in ("keyword_extraction", "stream")}
    request_kwargs.setdefault("max_tokens", 4096)

    if response_schema is not None:
        schema_dict = response_schema.model_json_schema() if hasattr(response_schema, "model_json_schema") else response_schema
        tool_def = {
            "name": "structured_output",
            "description": "Return structured data matching the schema",
            "input_schema": schema_dict,
        }
        request_kwargs["tools"] = [tool_def]
        request_kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}

    stream = kwargs.get("stream", False)

    if stream and response_schema is None:
        async def _stream():
            async with client.messages.stream(
                model=model, messages=messages, system=system_prompt or anthropic.NOT_GIVEN, **request_kwargs
            ) as resp:
                async for text in resp.text_stream:
                    yield text
        return _stream()

    resp = await client.messages.create(
        model=model, messages=messages, system=system_prompt or anthropic.NOT_GIVEN, **request_kwargs
    )

    if response_schema is not None:
        for block in resp.content:
            if block.type == "tool_use":
                return json.dumps(block.input)
    return resp.content[0].text if resp.content else ""


async def anthropic_complete(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    **kwargs,
) -> str:
    return await anthropic_complete_if_cache(model, prompt, **kwargs)


async def anthropic_embed(
    texts: list[str],
    model: str = "voyage-3",
    api_key: str | None = None,
) -> np.ndarray:
    if httpx is None:
        raise ImportError("httpx is required for Voyage embeddings. Install with: pip install httpx")
    api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": texts},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
    return np.array([d["embedding"] for d in data["data"]])
