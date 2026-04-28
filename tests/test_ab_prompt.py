"""Tests for A/B prompt comparison."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semanrag.evaluation.runner import EvalRecord, EvalReport


@pytest.fixture()
def goldens():
    return [
        EvalRecord(id="q1", query="What is RAG?", expected_answer="Retrieval-augmented generation"),
        EvalRecord(id="q2", query="What is a knowledge graph?", expected_answer="A graph of entities"),
    ]


@pytest.fixture()
def mock_rag():
    rag = MagicMock()
    rag.user_prompt_template = "default prompt"
    return rag


@pytest.mark.asyncio
async def test_ab_comparison_returns_both_reports(goldens, mock_rag):
    """run_ab_prompt returns variant_a, variant_b, and comparison."""
    report_a = EvalReport(domain="test", timestamp="2024-01-01", aggregate_metrics={"f1": 0.8, "recall": 0.7})
    report_b = EvalReport(domain="test", timestamp="2024-01-01", aggregate_metrics={"f1": 0.85, "recall": 0.75})

    with patch("semanrag.evaluation.ab_prompt.run_eval", new_callable=AsyncMock) as mock_eval:
        mock_eval.side_effect = [report_a, report_b]

        from semanrag.evaluation.ab_prompt import run_ab_prompt

        result = await run_ab_prompt(
            goldens, mock_rag,
            prompt_a={"system_prompt": "Be concise", "mode": "naive"},
            prompt_b={"system_prompt": "Be detailed", "mode": "naive"},
        )

    assert "variant_a" in result
    assert "variant_b" in result
    assert "comparison" in result
    assert result["comparison"]["f1"]["winner"] == "b"
    assert result["comparison"]["f1"]["delta"] == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_ab_different_prompts_different_results(goldens, mock_rag):
    """Different prompts produce different evaluation results."""
    report_a = EvalReport(domain="test", timestamp="2024-01-01", aggregate_metrics={"f1": 0.6})
    report_b = EvalReport(domain="test", timestamp="2024-01-01", aggregate_metrics={"f1": 0.9})

    with patch("semanrag.evaluation.ab_prompt.run_eval", new_callable=AsyncMock) as mock_eval:
        mock_eval.side_effect = [report_a, report_b]

        from semanrag.evaluation.ab_prompt import run_ab_prompt

        result = await run_ab_prompt(
            goldens, mock_rag,
            prompt_a={"system_prompt": "prompt A"},
            prompt_b={"system_prompt": "prompt B"},
        )

    assert result["variant_a"].aggregate_metrics["f1"] != result["variant_b"].aggregate_metrics["f1"]
    assert result["comparison"]["f1"]["delta"] == pytest.approx(0.3)
