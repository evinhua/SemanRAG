"""Hugging Face Transformers LLM provider implementation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment]

supports_json_mode = False
supports_tools = False
supports_vision = False

_model_cache: dict = {}


def _ensure_transformers():
    if AutoModelForCausalLM is None:
        raise ImportError("transformers is required. Install with: pip install transformers torch")


def _get_model_and_tokenizer(model_name: str, device: str):
    if model_name not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")
        model = model.to(device).eval()
        _model_cache[model_name] = (model, tokenizer)
    return _model_cache[model_name]


async def hf_model_complete(
    model_name: str,
    prompt: str,
    system_prompt: str = "",
    history_messages: list[dict] | None = None,
    device: str = "cpu",
    response_schema=None,
    **kwargs,
) -> str | AsyncIterator[str]:
    _ensure_transformers()
    model, tokenizer = _get_model_and_tokenizer(model_name, device)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    user_content = prompt
    if response_schema is not None:
        schema_hint = json.dumps(response_schema.model_json_schema() if hasattr(response_schema, "model_json_schema") else response_schema, indent=2)
        user_content += f"\n\nRespond with valid JSON matching this schema:\n{schema_hint}"
    messages.append({"role": "user", "content": user_content})

    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = "\n".join(m["content"] for m in messages)

    max_new_tokens = kwargs.get("max_tokens", 512)

    def _generate():
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=kwargs.get("temperature", 0) > 0, temperature=kwargs.get("temperature", 0.0) or 1.0)
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    result = await asyncio.get_event_loop().run_in_executor(None, _generate)
    return result


async def hf_embed(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

    def _encode():
        model = SentenceTransformer(model_name)
        return model.encode(texts, convert_to_numpy=True)

    return await asyncio.get_event_loop().run_in_executor(None, _encode)
