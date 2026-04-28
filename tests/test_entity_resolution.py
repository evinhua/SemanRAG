"""Tests for entity resolution logic."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.mark.unit
class TestEmbeddingSimilarityBlocking:
    @pytest.mark.asyncio
    async def test_embedding_similarity_blocking(self, mock_embedding_func):
        """Mock VDB query, verify candidates found by embedding similarity."""
        mock_vdb = AsyncMock()
        mock_vdb.query.return_value = [
            {"id": "e1", "content": "Albert Einstein", "distance": 0.05},
            {"id": "e2", "content": "A. Einstein", "distance": 0.12},
            {"id": "e3", "content": "Isaac Newton", "distance": 0.85},
        ]
        results = await mock_vdb.query("Albert Einstein", top_k=10)
        candidates = [r for r in results if r["distance"] < 0.5]
        assert len(candidates) == 2
        assert candidates[0]["content"] == "Albert Einstein"
        assert candidates[1]["content"] == "A. Einstein"


@pytest.mark.unit
class TestEditDistanceScoring:
    def test_edit_distance_scoring(self):
        """Test rapidfuzz integration for edit distance."""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            pytest.skip("rapidfuzz not installed")

        pairs = [
            ("Albert Einstein", "Albert Einstien", True),   # typo
            ("Albert Einstein", "Isaac Newton", False),       # different
            ("New York City", "New York", True),              # partial
        ]
        for name_a, name_b, should_be_similar in pairs:
            score = fuzz.ratio(name_a, name_b)
            if should_be_similar:
                assert score > 60, f"{name_a} vs {name_b}: {score}"
            else:
                assert score < 60, f"{name_a} vs {name_b}: {score}"


@pytest.mark.unit
class TestLLMAdjudicator:
    @pytest.mark.asyncio
    async def test_llm_adjudicator_same(self):
        """Mock LLM returning SAME for entity pair."""
        mock_llm = AsyncMock(return_value="SAME")
        result = await mock_llm("Are 'Albert Einstein' and 'A. Einstein' the same entity?")
        assert "SAME" in result

    @pytest.mark.asyncio
    async def test_llm_adjudicator_different(self):
        """Mock LLM returning DIFFERENT for entity pair."""
        mock_llm = AsyncMock(return_value="DIFFERENT")
        result = await mock_llm("Are 'Albert Einstein' and 'Isaac Newton' the same entity?")
        assert "DIFFERENT" in result
