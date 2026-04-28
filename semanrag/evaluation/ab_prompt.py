"""A/B prompt testing – run same goldens with two prompt variants and compare."""

from __future__ import annotations

from typing import Any

from semanrag.evaluation.runner import EvalRecord, run_eval


async def run_ab_prompt(
    goldens: list[EvalRecord],
    rag_instance: Any,
    prompt_a: dict,
    prompt_b: dict,
) -> dict:
    # Apply prompt_a config and run
    original_prompt = getattr(rag_instance, "user_prompt_template", None)

    if "system_prompt" in prompt_a:
        rag_instance.user_prompt_template = prompt_a["system_prompt"]
    report_a = await run_eval(
        goldens, rag_instance, mode=prompt_a.get("mode", "naive")
    )
    report_a.domain = f"{report_a.domain}_variant_a"

    # Apply prompt_b config and run
    if "system_prompt" in prompt_b:
        rag_instance.user_prompt_template = prompt_b["system_prompt"]
    report_b = await run_eval(
        goldens, rag_instance, mode=prompt_b.get("mode", "naive")
    )
    report_b.domain = f"{report_b.domain}_variant_b"

    # Restore original
    if original_prompt is not None:
        rag_instance.user_prompt_template = original_prompt

    # Build comparison
    comparison: dict[str, dict] = {}
    all_keys = set(report_a.aggregate_metrics) | set(report_b.aggregate_metrics)
    for k in sorted(all_keys):
        a_val = report_a.aggregate_metrics.get(k, 0.0)
        b_val = report_b.aggregate_metrics.get(k, 0.0)
        comparison[k] = {
            "variant_a": round(a_val, 4),
            "variant_b": round(b_val, 4),
            "delta": round(b_val - a_val, 4),
            "winner": "a" if a_val > b_val else ("b" if b_val > a_val else "tie"),
        }

    return {
        "variant_a": report_a,
        "variant_b": report_b,
        "comparison": comparison,
    }
