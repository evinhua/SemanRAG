"""SemanRAG LLM provider implementations.

All providers are lazily imported to avoid hard dependencies.
"""

from semanrag.llm.openai_impl import (
    azure_openai_complete_if_cache,
    azure_openai_embed,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    openai_complete_if_cache,
    openai_embed,
)
from semanrag.llm.ollama_impl import ollama_embed, ollama_model_complete
from semanrag.llm.gemini_impl import (
    gemini_complete_if_cache,
    gemini_embed,
    gemini_model_complete,
)
from semanrag.llm.bedrock_impl import bedrock_complete, bedrock_complete_if_cache, bedrock_embed
from semanrag.llm.anthropic_impl import anthropic_complete, anthropic_complete_if_cache, anthropic_embed
from semanrag.llm.hf_impl import hf_embed, hf_model_complete
from semanrag.llm.llama_index_impl import llama_index_complete_if_cache, llama_index_embed
from semanrag.llm.zhipu_impl import zhipu_complete_if_cache, zhipu_embedding
from semanrag.llm.vllm_impl import vllm_complete, vllm_embed

__all__ = [
    # OpenAI
    "openai_complete_if_cache",
    "gpt_4o_complete",
    "gpt_4o_mini_complete",
    "openai_embed",
    "azure_openai_complete_if_cache",
    "azure_openai_embed",
    # Ollama
    "ollama_model_complete",
    "ollama_embed",
    # Gemini
    "gemini_complete_if_cache",
    "gemini_model_complete",
    "gemini_embed",
    # Bedrock
    "bedrock_complete_if_cache",
    "bedrock_complete",
    "bedrock_embed",
    # Anthropic
    "anthropic_complete_if_cache",
    "anthropic_complete",
    "anthropic_embed",
    # Hugging Face
    "hf_model_complete",
    "hf_embed",
    # LlamaIndex
    "llama_index_complete_if_cache",
    "llama_index_embed",
    # Zhipu
    "zhipu_complete_if_cache",
    "zhipu_embedding",
    # vLLM
    "vllm_complete",
    "vllm_embed",
]
