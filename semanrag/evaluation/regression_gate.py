"""Regression gate – compare current eval report against a baseline."""

from __future__ import annotations

import json
import sys

from semanrag.evaluation.runner import EvalReport, EvalResult


def load_baseline(path: str) -> EvalReport:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    results = [EvalResult(**r) for r in data.get("results", [])]
    return EvalReport(
        domain=data["domain"],
        timestamp=data["timestamp"],
        results=results,
        aggregate_metrics=data.get("aggregate_metrics", {}),
        mode=data.get("mode", "naive"),
    )


def compare_reports(
    current: EvalReport,
    baseline: EvalReport,
    threshold: float = 0.02,
) -> tuple[bool, dict]:
    details: dict[str, dict] = {}
    passed = True
    for metric, cur_val in current.aggregate_metrics.items():
        base_val = baseline.aggregate_metrics.get(metric)
        if base_val is None:
            details[metric] = {"current": cur_val, "baseline": None, "status": "new_metric"}
            continue
        delta = cur_val - base_val
        regressed = cur_val < base_val - threshold
        if regressed:
            passed = False
        details[metric] = {
            "current": round(cur_val, 4),
            "baseline": round(base_val, 4),
            "delta": round(delta, 4),
            "status": "regression" if regressed else "ok",
        }
    return passed, details


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SemanRAG Regression Gate")
    parser.add_argument("--current", required=True, help="Path to current eval report JSON")
    parser.add_argument("--baseline", required=True, help="Path to baseline report JSON")
    parser.add_argument("--threshold", type=float, default=0.02, help="Regression threshold")
    args = parser.parse_args()

    current = load_baseline(args.current)
    baseline = load_baseline(args.baseline)
    passed, details = compare_reports(current, baseline, args.threshold)

    print(json.dumps(details, indent=2))
    if passed:
        print("✅ Regression gate PASSED")
    else:
        print("❌ Regression gate FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
