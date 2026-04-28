"""Evaluation runner – load goldens, run RAG queries, compute metrics."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semanrag.evaluation.metrics import compute_all_metrics


@dataclass
class EvalRecord:
    id: str
    query: str
    expected_answer: str
    ground_truth_contexts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    record_id: str
    query: str
    generated_answer: str
    metrics: dict[str, float] = field(default_factory=dict)
    references: list[Any] = field(default_factory=list)
    grounded_check: list[Any] = field(default_factory=list)


@dataclass
class EvalReport:
    domain: str
    timestamp: str
    results: list[EvalResult] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    mode: str = "naive"


def load_goldens(path: str) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                EvalRecord(
                    id=obj["id"],
                    query=obj["query"],
                    expected_answer=obj["expected_answer"],
                    ground_truth_contexts=obj.get("ground_truth_contexts", []),
                    tags=obj.get("tags", []),
                )
            )
    return records


async def run_eval(
    goldens: list[EvalRecord],
    rag_instance: Any,
    metrics: list[str] | None = None,
    mode: str = "naive",
) -> EvalReport:
    from semanrag.base import QueryParam

    results: list[EvalResult] = []
    for record in goldens:
        qr = await rag_instance.aquery(record.query, QueryParam(mode=mode))
        contexts = [ref.get("content", "") for ref in qr.references] if qr.references else []
        grounded = qr.grounded_check if isinstance(qr.grounded_check, list) else []

        all_metrics = compute_all_metrics(
            query=record.query,
            answer=qr.content,
            contexts=contexts,
            ground_truth=record.expected_answer,
            ground_truth_contexts=record.ground_truth_contexts,
            grounded_check=grounded,
        )
        if metrics:
            all_metrics = {k: v for k, v in all_metrics.items() if k in metrics}

        results.append(
            EvalResult(
                record_id=record.id,
                query=record.query,
                generated_answer=qr.content,
                metrics=all_metrics,
                references=qr.references,
                grounded_check=grounded,
            )
        )

    # Aggregate: mean per metric
    agg: dict[str, float] = {}
    if results:
        all_keys = {k for r in results for k in r.metrics}
        for k in sorted(all_keys):
            vals = [r.metrics[k] for r in results if k in r.metrics]
            agg[k] = sum(vals) / len(vals) if vals else 0.0

    domain = goldens[0].tags[0] if goldens and goldens[0].tags else "unknown"
    return EvalReport(
        domain=domain,
        timestamp=datetime.now(UTC).isoformat(),
        results=results,
        aggregate_metrics=agg,
        mode=mode,
    )


def save_report(report: EvalReport, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, default=str)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SemanRAG Evaluation Runner")
    parser.add_argument("--goldens", required=True, help="Path to JSONL goldens file")
    parser.add_argument("--output", default="eval_report.json", help="Output report path")
    parser.add_argument("--mode", default="naive", help="Query mode")
    parser.add_argument("--working-dir", default="./semanrag_data", help="SemanRAG working dir")
    args = parser.parse_args()

    from semanrag.semanrag import SemanRAG

    records = load_goldens(args.goldens)
    rag = SemanRAG(working_dir=args.working_dir)
    report = asyncio.run(run_eval(records, rag, mode=args.mode))
    save_report(report, args.output)
    print(f"Report saved to {args.output} — aggregate: {report.aggregate_metrics}")


if __name__ == "__main__":
    main()
