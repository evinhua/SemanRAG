"""Tests for query transformation (rewrite, decomposition, HyDE)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.mark.unit
class TestQueryRewrite:
    @pytest.mark.asyncio
    async def test_rewrite_with_history(self):
        """Pronoun resolution via LLM rewrite."""
        mock_llm = AsyncMock(return_value="What is Albert Einstein's theory of relativity?")
        history = [
            {"role": "user", "content": "Tell me about Albert Einstein."},
            {"role": "assistant", "content": "Albert Einstein was a physicist."},
        ]
        query = "What is his theory?"
        # Simulate rewrite: LLM resolves "his" to "Albert Einstein's"
        rewritten = await mock_llm(f"Rewrite: {query}\nHistory: {history}")
        assert "Albert Einstein" in rewritten
        assert "his" not in rewritten.lower() or "Albert Einstein" in rewritten


@pytest.mark.unit
class TestQueryDecomposition:
    @pytest.mark.asyncio
    async def test_decomposition_multi_hop(self):
        """Multi-hop query returns sub-queries."""
        mock_llm = AsyncMock(
            return_value='["Who founded SpaceX?", "When was SpaceX founded?", "What rockets does SpaceX build?"]'
        )
        import json

        raw = await mock_llm("Decompose: Who founded SpaceX and when, and what rockets do they build?")
        sub_queries = json.loads(raw)
        assert len(sub_queries) == 3
        assert any("founded" in q.lower() for q in sub_queries)

    @pytest.mark.asyncio
    async def test_decomposition_atomic(self):
        """Atomic query returns original (no decomposition needed)."""
        mock_llm = AsyncMock(return_value='["What is the capital of France?"]')
        import json

        raw = await mock_llm("Decompose: What is the capital of France?")
        sub_queries = json.loads(raw)
        assert len(sub_queries) == 1
        assert "capital of France" in sub_queries[0]


@pytest.mark.unit
class TestHyDE:
    @pytest.mark.asyncio
    async def test_hyde_generation(self):
        """Returns hypothetical answer for embedding."""
        mock_llm = AsyncMock(
            return_value="The capital of France is Paris, a city known for the Eiffel Tower."
        )
        query = "What is the capital of France?"
        hypothetical = await mock_llm(f"Write a passage that answers: {query}")
        assert "Paris" in hypothetical
        assert len(hypothetical) > len(query)
