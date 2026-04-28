"""Tests for temporal query filtering."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from semanrag.base import TemporalEdge


def _filter_edges(edges: list[TemporalEdge], snapshot_at: datetime | None) -> list[TemporalEdge]:
    """Replicate the temporal filtering logic used in graph queries."""
    if snapshot_at is None:
        return edges
    result = []
    for e in edges:
        if e.valid_from and e.valid_from > snapshot_at:
            continue
        if e.valid_to and e.valid_to < snapshot_at:
            continue
        result.append(e)
    return result


@pytest.mark.unit
class TestTemporalQueries:
    def test_snapshot_at_filters_edges(self):
        """Edges outside range excluded."""
        now = datetime(2024, 6, 15, tzinfo=UTC)
        edges = [
            TemporalEdge(source="A", valid_from=datetime(2024, 1, 1, tzinfo=UTC), valid_to=datetime(2024, 12, 31, tzinfo=UTC)),
            TemporalEdge(source="B", valid_from=datetime(2025, 1, 1, tzinfo=UTC), valid_to=None),  # future
            TemporalEdge(source="C", valid_from=datetime(2023, 1, 1, tzinfo=UTC), valid_to=datetime(2023, 12, 31, tzinfo=UTC)),  # past
        ]
        filtered = _filter_edges(edges, snapshot_at=now)
        assert len(filtered) == 1
        assert filtered[0].source == "A"

    def test_none_bounds_treated_as_unbounded(self):
        """None valid_from/valid_to means unbounded."""
        now = datetime(2024, 6, 15, tzinfo=UTC)
        edges = [
            TemporalEdge(source="A", valid_from=None, valid_to=None),
            TemporalEdge(source="B", valid_from=None, valid_to=datetime(2025, 1, 1, tzinfo=UTC)),
            TemporalEdge(source="C", valid_from=datetime(2020, 1, 1, tzinfo=UTC), valid_to=None),
        ]
        filtered = _filter_edges(edges, snapshot_at=now)
        assert len(filtered) == 3

    def test_no_snapshot_returns_all(self):
        """No snapshot_at returns all edges."""
        edges = [
            TemporalEdge(source="A", valid_from=datetime(2024, 1, 1, tzinfo=UTC)),
            TemporalEdge(source="B", valid_from=datetime(2025, 1, 1, tzinfo=UTC)),
        ]
        filtered = _filter_edges(edges, snapshot_at=None)
        assert len(filtered) == 2
