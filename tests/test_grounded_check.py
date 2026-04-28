"""Tests for grounded check verifier."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from semanrag.verify.grounded_check import (
    _split_into_claims,
    grounded_check,
    retry_with_expanded_context,
)


@pytest.mark.unit
class TestGroundedCheck:
    @pytest.mark.asyncio
    async def test_all_claims_supported(self):
        """Mock verifier returning high scores."""
        mock_verifier = AsyncMock(
            return_value=json.dumps({"score": 0.95, "supporting_span": "evidence text"})
        )
        answer = "The sky is blue. Water is wet."
        contexts = [{"content": "The sky is blue and water is wet."}]
        results = await grounded_check(answer, contexts, verifier_func=mock_verifier)
        assert len(results) >= 2
        assert all(r["supported"] for r in results)
        assert all(r["score"] >= 0.5 for r in results)

    @pytest.mark.asyncio
    async def test_unsupported_claims_flagged(self):
        """Mock verifier returning low scores."""
        mock_verifier = AsyncMock(
            return_value=json.dumps({"score": 0.1, "supporting_span": ""})
        )
        answer = "Unicorns exist. They live on Mars."
        contexts = [{"content": "No evidence of unicorns."}]
        results = await grounded_check(answer, contexts, verifier_func=mock_verifier)
        assert len(results) >= 2
        assert all(not r["supported"] for r in results)

    @pytest.mark.asyncio
    async def test_retry_on_unsupported(self):
        """Verify retry logic regenerates answer for unsupported claims."""
        mock_llm = AsyncMock(return_value="Improved grounded answer.")
        check_results = [
            {"claim": "Unicorns exist.", "score": 0.1, "supporting_span": "", "supported": False},
            {"claim": "The sky is blue.", "score": 0.9, "supporting_span": "sky", "supported": True},
        ]
        contexts = [{"content": "The sky is blue. No unicorns."}]
        result = await retry_with_expanded_context(
            query="test", answer="original", check_results=check_results,
            contexts=contexts, llm_func=mock_llm,
        )
        assert result == "Improved grounded answer."
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_no_unsupported_returns_original(self):
        """If all claims supported, retry returns original answer."""
        mock_llm = AsyncMock()
        check_results = [
            {"claim": "The sky is blue.", "score": 0.9, "supporting_span": "sky", "supported": True},
        ]
        result = await retry_with_expanded_context(
            query="test", answer="original answer", check_results=check_results,
            contexts=[], llm_func=mock_llm,
        )
        assert result == "original answer"
        mock_llm.assert_not_called()


@pytest.mark.unit
class TestClaimSplitting:
    def test_claim_splitting(self):
        """Verify sentence boundary detection."""
        text = "First claim. Second claim! Third claim? Fourth."
        claims = _split_into_claims(text)
        assert len(claims) == 4
        assert claims[0] == "First claim."
        assert claims[1] == "Second claim!"
        assert claims[2] == "Third claim?"
        assert claims[3] == "Fourth."

    def test_claim_splitting_empty(self):
        assert _split_into_claims("") == []

    def test_claim_splitting_single(self):
        claims = _split_into_claims("Just one sentence.")
        assert len(claims) == 1
