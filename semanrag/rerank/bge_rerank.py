"""BGE reranker using FlagEmbedding."""

from __future__ import annotations

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    FlagReranker = None  # type: ignore[assignment,misc]


def bge_rerank(
    query: str,
    documents: list[dict],
    top_k: int = 5,
    model_name: str = "BAAI/bge-reranker-v2-m3",
) -> list[dict]:
    """Score query–document pairs with FlagReranker and return top-*k*."""
    if FlagReranker is None:
        raise ImportError("Install 'FlagEmbedding' to use bge_rerank.")

    reranker = FlagReranker(model_name)
    pairs = [[query, doc.get("content", "")] for doc in documents]
    scores = reranker.compute_score(pairs)
    # compute_score may return a single float when len(pairs)==1
    if isinstance(scores, (int, float)):
        scores = [scores]

    results = [
        {**doc, "rerank_score": float(score)}
        for doc, score in zip(documents, scores)
    ]
    results.sort(key=lambda d: d["rerank_score"], reverse=True)
    return results[:top_k]
