"""Tests for regression gate logic."""
from __future__ import annotations

import pytest

from semanrag.evaluation.regression_gate import compare_reports
from semanrag.evaluation.runner import EvalReport


def _make_report(metrics: dict[str, float]) -> EvalReport:
    return EvalReport(
        domain="test",
        timestamp="2024-01-01T00:00:00",
        results=[],
        aggregate_metrics=metrics,
        mode="naive",
    )


@pytest.mark.unit
class TestRegressionGate:
    def test_pass_when_above_baseline(self):
        """Current scores above baseline → pass."""
        baseline = _make_report({"f1": 0.80, "recall": 0.75})
        current = _make_report({"f1": 0.85, "recall": 0.78})
        passed, details = compare_reports(current, baseline, threshold=0.02)
        assert passed
        assert details["f1"]["status"] == "ok"
        assert details["recall"]["status"] == "ok"

    def test_fail_when_below_threshold(self):
        """Current scores below baseline by more than threshold → fail."""
        baseline = _make_report({"f1": 0.80, "recall": 0.75})
        current = _make_report({"f1": 0.70, "recall": 0.60})
        passed, details = compare_reports(current, baseline, threshold=0.02)
        assert not passed
        assert details["f1"]["status"] == "regression"
        assert details["recall"]["status"] == "regression"

    def test_threshold_boundary(self):
        """Exactly at threshold boundary → pass (not strictly below)."""
        baseline = _make_report({"f1": 0.80})
        # current = 0.78, baseline - threshold = 0.78 → not regressed (not < 0.78)
        current = _make_report({"f1": 0.78})
        passed, details = compare_reports(current, baseline, threshold=0.02)
        assert passed
        assert details["f1"]["status"] == "ok"

        # current = 0.77 < 0.78 → regressed
        current2 = _make_report({"f1": 0.77})
        passed2, details2 = compare_reports(current2, baseline, threshold=0.02)
        assert not passed2
        assert details2["f1"]["status"] == "regression"
