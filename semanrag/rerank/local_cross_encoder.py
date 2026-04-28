"""Local cross-encoder reranker using sentence-transformers."""

from __future__ import annotations

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None  # type: ignore[assignment,misc]


def local_cross_encoder_rerank(
    query: str,
    documents: list[dict],
    top_k: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[dict]:
    """Score query–document pairs with a local CrossEncoder and return top-*k*."""
    if CrossEncoder is None:
        raise ImportError(
            "Install 'sentence-transformers' to use local_cross_encoder_rerank."
        )

    model = CrossEncoder(model_name)
    pairs = [[query, doc.get("content", "")] for doc in documents]
    scores = model.predict(pairs)

    results = [
        {**doc, "rerank_score": float(score)}
        for doc, score in zip(documents, scores)
    ]
    results.sort(key=lambda d: d["rerank_score"], reverse=True)
    return results[:top_k]
