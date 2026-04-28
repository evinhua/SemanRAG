"""Tests for reciprocal rank fusion."""
from __future__ import annotations

import pytest

from semanrag.utils import reciprocal_rank_fusion


@pytest.mark.unit
class TestRRFFusion:
    def test_basic_fusion(self):
        """Two lists, verify merged order."""
        list_a = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.7}, {"id": "c", "score": 0.5}]
        list_b = [{"id": "d", "score": 0.95}, {"id": "a", "score": 0.6}, {"id": "e", "score": 0.4}]
        merged = reciprocal_rank_fusion([list_a, list_b], k=60)
        ids = [r["id"] for r in merged]
        # 'a' appears in both lists so should rank first
        assert ids[0] == "a"
        assert len(merged) == 5  # 5 unique items: a, b, c, d, e
        assert len(merged) == len(set(ids))

    def test_single_list(self):
        """Single list should pass through with RRF scores."""
        items = [{"id": "x", "score": 1.0}, {"id": "y", "score": 0.5}]
        merged = reciprocal_rank_fusion([items], k=60)
        assert len(merged) == 2
        assert merged[0]["id"] == "x"
        assert merged[1]["id"] == "y"
        assert "rrf_score" in merged[0]

    def test_overlapping_items(self):
        """Same item in multiple lists gets higher score."""
        list_a = [{"id": "shared"}, {"id": "only_a"}]
        list_b = [{"id": "shared"}, {"id": "only_b"}]
        list_c = [{"id": "shared"}, {"id": "only_c"}]
        merged = reciprocal_rank_fusion([list_a, list_b, list_c], k=60)
        # 'shared' appears in all 3 lists at rank 0, so it should be first
        assert merged[0]["id"] == "shared"
        shared_score = merged[0]["rrf_score"]
        other_scores = [r["rrf_score"] for r in merged if r["id"] != "shared"]
        assert all(shared_score > s for s in other_scores)

    def test_empty_lists(self):
        """Edge case: empty input."""
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []
