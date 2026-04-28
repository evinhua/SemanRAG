"""Evaluation metrics with RAGAS integration and heuristic fallbacks."""

from __future__ import annotations

import re


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def ragas_context_precision(
    query: str, answer: str, contexts: list[str], ground_truth: str
) -> float:
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import context_precision as cp

        ds = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth],
            }
        )
        result = ragas_evaluate(ds, metrics=[cp])
        return float(result["context_precision"])
    except Exception:
        if not contexts or not ground_truth:
            return 0.0
        gt_tokens = _tokenize(ground_truth)
        if not gt_tokens:
            return 0.0
        scores = []
        for ctx in contexts:
            ctx_tokens = _tokenize(ctx)
            overlap = len(gt_tokens & ctx_tokens)
            scores.append(overlap / len(gt_tokens) if gt_tokens else 0.0)
        return sum(scores) / len(scores)


def ragas_faithfulness(query: str, answer: str, contexts: list[str]) -> float:
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness as fm

        ds = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [""],
            }
        )
        result = ragas_evaluate(ds, metrics=[fm])
        return float(result["faithfulness"])
    except Exception:
        if not contexts or not answer:
            return 0.0
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return 0.0
        ctx_tokens: set[str] = set()
        for ctx in contexts:
            ctx_tokens |= _tokenize(ctx)
        overlap = len(answer_tokens & ctx_tokens)
        return overlap / len(answer_tokens)


def ragas_answer_relevancy(query: str, answer: str) -> float:
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy as ar

        ds = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [[]],
                "ground_truth": [""],
            }
        )
        result = ragas_evaluate(ds, metrics=[ar])
        return float(result["answer_relevancy"])
    except Exception:
        if not query or not answer:
            return 0.0
        q_tokens = _tokenize(query)
        a_tokens = _tokenize(answer)
        if not q_tokens:
            return 0.0
        return len(q_tokens & a_tokens) / len(q_tokens)


def ragas_context_recall(
    query: str,
    answer: str,
    contexts: list[str],
    ground_truth_contexts: list[str],
) -> float:
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import context_recall as cr

        ds = Dataset.from_dict(
            {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [" ".join(ground_truth_contexts)],
            }
        )
        result = ragas_evaluate(ds, metrics=[cr])
        return float(result["context_recall"])
    except Exception:
        if not ground_truth_contexts or not contexts:
            return 0.0
        gt_tokens: set[str] = set()
        for gt in ground_truth_contexts:
            gt_tokens |= _tokenize(gt)
        if not gt_tokens:
            return 0.0
        ctx_tokens: set[str] = set()
        for ctx in contexts:
            ctx_tokens |= _tokenize(ctx)
        return len(gt_tokens & ctx_tokens) / len(gt_tokens)


def grounded_check_pass_rate(grounded_checks: list[list[dict]]) -> float:
    total = 0
    passed = 0
    for checks in grounded_checks:
        for claim in checks:
            total += 1
            score = claim.get("score", claim.get("grounding_score", 0.0))
            if score >= 0.5:
                passed += 1
    return passed / total if total > 0 else 0.0


def entity_resolution_precision(
    predicted_merges: list[set[str]], gold_merges: list[set[str]]
) -> float:
    if not predicted_merges:
        return 0.0
    correct = 0
    for pred in predicted_merges:
        for gold in gold_merges:
            if pred <= gold:
                correct += 1
                break
    return correct / len(predicted_merges)


def entity_resolution_recall(
    predicted_merges: list[set[str]], gold_merges: list[set[str]]
) -> float:
    if not gold_merges:
        return 0.0
    found = 0
    for gold in gold_merges:
        for pred in predicted_merges:
            if gold <= pred:
                found += 1
                break
    return found / len(gold_merges)


def compute_all_metrics(
    query: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    ground_truth_contexts: list[str],
    grounded_check: list[dict],
) -> dict[str, float]:
    return {
        "context_precision": ragas_context_precision(query, answer, contexts, ground_truth),
        "faithfulness": ragas_faithfulness(query, answer, contexts),
        "answer_relevancy": ragas_answer_relevancy(query, answer),
        "context_recall": ragas_context_recall(query, answer, contexts, ground_truth_contexts),
        "grounded_check_pass_rate": grounded_check_pass_rate([grounded_check]),
    }
