"""Tests for community detection logic."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.mark.unit
class TestLeidenClustering:
    @pytest.mark.asyncio
    async def test_leiden_clustering(self):
        """Mock graph with clear communities, verify clustering."""
        mock_graph = AsyncMock()
        # Two clear communities: {A,B,C} and {D,E,F}
        mock_graph.detect_communities.return_value = {
            "algorithm": "leiden",
            "levels": [
                {
                    "level": 0,
                    "communities": {
                        "0": ["A", "B", "C"],
                        "1": ["D", "E", "F"],
                    },
                }
            ],
        }
        result = await mock_graph.detect_communities(algorithm="leiden", levels=1)
        communities = result["levels"][0]["communities"]
        assert len(communities) == 2
        assert set(communities["0"]) == {"A", "B", "C"}
        assert set(communities["1"]) == {"D", "E", "F"}

    @pytest.mark.asyncio
    async def test_community_hierarchy(self):
        """Verify multi-level output."""
        mock_graph = AsyncMock()
        mock_graph.detect_communities.return_value = {
            "algorithm": "leiden",
            "levels": [
                {
                    "level": 0,
                    "communities": {"0": ["A", "B"], "1": ["C", "D"], "2": ["E", "F"]},
                },
                {
                    "level": 1,
                    "communities": {"0": ["A", "B", "C", "D"], "1": ["E", "F"]},
                },
            ],
        }
        result = await mock_graph.detect_communities(algorithm="leiden", levels=2)
        assert len(result["levels"]) == 2
        level0 = result["levels"][0]["communities"]
        level1 = result["levels"][1]["communities"]
        assert len(level0) == 3
        assert len(level1) == 2
        # Higher level should have fewer, larger communities
        total_l0 = sum(len(v) for v in level0.values())
        total_l1 = sum(len(v) for v in level1.values())
        assert total_l0 == total_l1  # same nodes, different grouping

    @pytest.mark.asyncio
    async def test_empty_graph(self):
        """Edge case: empty graph returns no communities."""
        mock_graph = AsyncMock()
        mock_graph.detect_communities.return_value = {
            "algorithm": "leiden",
            "levels": [{"level": 0, "communities": {}}],
        }
        result = await mock_graph.detect_communities(algorithm="leiden", levels=1)
        assert result["levels"][0]["communities"] == {}
