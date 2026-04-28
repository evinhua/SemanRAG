"""AWS Bedrock LLM provider implementation."""

from __future__ import annotations

import json
import os
from typing import AsyncIterator

import numpy as np

try:
    import aioboto3
except ImportError:
    aioboto3 = None  # type: ignore[assignment]

supports_json_mode = True
supports_tools = True
supports_vision = True


def _ensure_aioboto3():
    if aioboto3 is None:
        raise ImportError("aioboto3 is required. Install with: pip install aioboto3")


def _is_claude(model_id: str) -> bool:
    return "anthropic" in model_id or "claude" in model_id


def _build_claude_body(prompt: str, system_prompt: str, history_messages: list[dict] | None, response_schema, **kwargs) -> dict:
    messages = []
    if history_messages:
        for msg in history_messages:
            messages.append({"role": msg.get("role", "user"), "content": msg["content"]})
    content = prompt
    if response_schema is not None:
        schema_hint = json.dumps(response_schema.model_json_schema() if hasattr(response_schema, "model_json_schema") else response_schema, indent=2)
        content += f"\n\nRespond with valid JSON matching this schema:\n{schema_hint}"
    messages.append({"role": "user", "content": content})
    body: dict = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 4096),
    }
    if system_prompt:
        body["system"] = system_prompt
    if kwargs.get("temperature") is not None:
        body["temperature"] = kwargs["temperature"]
    return body


def _build_titan_body(prompt: str, system_prompt: str, response_schema, **kwargs) -> dict:
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    if response_schema is not None:
        schema_hint = json.dumps(response_schema.model_json_schema() if hasattr(response_schema, "model_json_schema") else response_schema, indent=2)
        full_prompt += f"\n\nRespond with valid JSON matching this schema:\n{schema_hint}"
    return {
        "inputText": full_prompt,
        "textGenerationConfig": {
            "maxTokenCount": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.0),
        },
    }


async def bedrock_complete_if_cache(
    model_id: str,
    prompt: str,
    system_prompt: str = "",
    history_messages: list[dict] | None = None,
    region: str = "us-east-1",
    response_schema=None,
    **kwargs,
) -> str | AsyncIterator[str]:
    _ensure_aioboto3()
    region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    stream = kwargs.pop("stream", False)

    if _is_claude(model_id):
        body = _build_claude_body(prompt, system_prompt, history_messages, response_schema, **kwargs)
    else:
        body = _build_titan_body(prompt, system_prompt, response_schema, **kwargs)

    session = aioboto3.Session()

    if stream and _is_claude(model_id):
        async def _stream():
            async with session.client("bedrock-runtime", region_name=region) as client:
                resp = await client.invoke_model_with_response_stream(
                    modelId=model_id, body=json.dumps(body)
                )
                async for event in resp["body"]:
                    chunk = json.loads(event["chunk"]["bytes"])
                    if chunk.get("type") == "content_block_delta":
                        yield chunk.get("delta", {}).get("text", "")
        return _stream()

    async with session.client("bedrock-runtime", region_name=region) as client:
        resp = await client.invoke_model(modelId=model_id, body=json.dumps(body))
        resp_body = json.loads(await resp["body"].read())

    if _is_claude(model_id):
        return resp_body.get("content", [{}])[0].get("text", "")
    return resp_body.get("results", [{}])[0].get("outputText", "")


async def bedrock_complete(
    prompt: str,
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    **kwargs,
) -> str:
    return await bedrock_complete_if_cache(model_id, prompt, **kwargs)


async def bedrock_embed(
    texts: list[str],
    model_id: str = "amazon.titan-embed-text-v2:0",
    region: str = "us-east-1",
) -> np.ndarray:
    _ensure_aioboto3()
    region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    session = aioboto3.Session()
    embeddings = []
    async with session.client("bedrock-runtime", region_name=region) as client:
        for text in texts:
            body = json.dumps({"inputText": text})
            resp = await client.invoke_model(modelId=model_id, body=body)
            resp_body = json.loads(await resp["body"].read())
            embeddings.append(resp_body.get("embedding", []))
    return np.array(embeddings)
