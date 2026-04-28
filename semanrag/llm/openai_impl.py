"""OpenAI and Azure OpenAI LLM provider implementations."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import numpy as np

try:
    from openai import AsyncAzureOpenAI, AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]
    AsyncAzureOpenAI = None  # type: ignore[assignment,misc]

supports_json_mode = True
supports_tools = True
supports_vision = True


def _ensure_openai():
    if AsyncOpenAI is None:
        raise ImportError("openai is required. Install with: pip install openai")


async def openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str = "",
    history_messages: list[dict] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    response_schema=None,
    **kwargs,
) -> str | AsyncIterator[str]:
    _ensure_openai()
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = base_url or os.environ.get("OPENAI_API_BASE")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    request_kwargs: dict = {k: v for k, v in kwargs.items() if k != "keyword_extraction"}
    if response_schema is not None:
        request_kwargs["response_format"] = {"type": "json_object"}
        schema_hint = json.dumps(response_schema.model_json_schema() if hasattr(response_schema, "model_json_schema") else response_schema, indent=2)
        messages[0 if not system_prompt else 0]["content"] = (
            (messages[0]["content"] + "\n\n") if messages[0]["content"] else ""
        ) + f"Respond with valid JSON matching this schema:\n{schema_hint}"

    stream = request_kwargs.pop("stream", False)

    if stream:
        async def _stream():
            resp = await client.chat.completions.create(
                model=model, messages=messages, stream=True, **request_kwargs
            )
            async for chunk in resp:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                if delta:
                    yield delta
        return _stream()

    resp = await client.chat.completions.create(
        model=model, messages=messages, **request_kwargs
    )
    return resp.choices[0].message.content or ""


async def gpt_4o_complete(prompt: str, **kwargs) -> str:
    return await openai_complete_if_cache("gpt-4o", prompt, **kwargs)


async def gpt_4o_mini_complete(prompt: str, **kwargs) -> str:
    return await openai_complete_if_cache("gpt-4o-mini", prompt, **kwargs)


async def openai_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str | None = None,
    api_key: str | None = None,
) -> np.ndarray:
    _ensure_openai()
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = base_url or os.environ.get("OPENAI_API_BASE")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    resp = await client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data])


async def azure_openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str = "",
    history_messages: list[dict] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_version: str = "2024-02-15-preview",
    response_schema=None,
    **kwargs,
) -> str | AsyncIterator[str]:
    _ensure_openai()
    api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
    base_url = base_url or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    client = AsyncAzureOpenAI(api_key=api_key, azure_endpoint=base_url, api_version=api_version)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    request_kwargs: dict = {k: v for k, v in kwargs.items() if k != "keyword_extraction"}
    if response_schema is not None:
        request_kwargs["response_format"] = {"type": "json_object"}
        schema_hint = json.dumps(response_schema.model_json_schema() if hasattr(response_schema, "model_json_schema") else response_schema, indent=2)
        messages[0]["content"] = (
            (messages[0]["content"] + "\n\n") if messages[0]["content"] else ""
        ) + f"Respond with valid JSON matching this schema:\n{schema_hint}"

    stream = request_kwargs.pop("stream", False)

    if stream:
        async def _stream():
            resp = await client.chat.completions.create(
                model=model, messages=messages, stream=True, **request_kwargs
            )
            async for chunk in resp:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                if delta:
                    yield delta
        return _stream()

    resp = await client.chat.completions.create(
        model=model, messages=messages, **request_kwargs
    )
    return resp.choices[0].message.content or ""


async def azure_openai_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str | None = None,
    api_key: str | None = None,
    api_version: str = "2024-02-15-preview",
) -> np.ndarray:
    _ensure_openai()
    api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
    base_url = base_url or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    client = AsyncAzureOpenAI(api_key=api_key, azure_endpoint=base_url, api_version=api_version)
    resp = await client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data])
