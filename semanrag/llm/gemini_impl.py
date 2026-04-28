"""Google Gemini LLM provider implementation."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import numpy as np

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

supports_json_mode = True
supports_tools = True
supports_vision = True


def _ensure_genai():
    if genai is None:
        raise ImportError("google-genai is required. Install with: pip install google-genai")


async def gemini_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str = "",
    history_messages: list[dict] | None = None,
    api_key: str | None = None,
    response_schema=None,
    **kwargs,
) -> str | AsyncIterator[str]:
    _ensure_genai()
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY", "")
    client = genai.Client(api_key=api_key)

    contents = []
    if history_messages:
        for msg in history_messages:
            role = "model" if msg.get("role") == "assistant" else msg.get("role", "user")
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    config_kwargs = {k: v for k, v in kwargs.items() if k not in ("keyword_extraction", "stream")}
    gen_config = {}
    if system_prompt:
        gen_config["system_instruction"] = system_prompt
    if response_schema is not None:
        gen_config["response_mime_type"] = "application/json"
        if hasattr(response_schema, "model_json_schema"):
            gen_config["response_schema"] = response_schema
    gen_config.update(config_kwargs)

    stream = kwargs.get("stream", False)

    if stream:
        async def _stream():
            resp = await client.aio.models.generate_content_stream(
                model=model, contents=contents, config=types.GenerateContentConfig(**gen_config)
            )
            async for chunk in resp:
                if chunk.text:
                    yield chunk.text
        return _stream()

    resp = await client.aio.models.generate_content(
        model=model, contents=contents, config=types.GenerateContentConfig(**gen_config)
    )
    return resp.text or ""


async def gemini_model_complete(prompt: str, model: str = "gemini-2.0-flash", **kwargs) -> str:
    return await gemini_complete_if_cache(model, prompt, **kwargs)


async def gemini_embed(
    texts: list[str],
    model: str = "text-embedding-004",
    api_key: str | None = None,
) -> np.ndarray:
    _ensure_genai()
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("EMBEDDING_API_KEY", "")
    client = genai.Client(api_key=api_key)
    resp = await client.aio.models.embed_content(model=model, contents=texts)
    return np.array(resp.embeddings[0].values if hasattr(resp.embeddings[0], "values") else [e.values for e in resp.embeddings])
