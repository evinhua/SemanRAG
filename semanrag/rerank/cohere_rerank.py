"""Cohere reranker using AsyncClientV2."""

from __future__ import annotations

import os

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore[assignment]


def chunk_documents_for_rerank(
    documents: list[dict], max_chunk_size: int = 4096
) -> list[dict]:
    """Split documents whose content exceeds *max_chunk_size* into smaller chunks.

    Each returned dict keeps the original document data plus ``_orig_idx``
    (index into the input list) and ``_chunk_idx``.
    """
    chunked: list[dict] = []
    for idx, doc in enumerate(documents):
        text = doc.get("content", "")
        if len(text) <= max_chunk_size:
            chunked.append({**doc, "_orig_idx": idx, "_chunk_idx": 0})
        else:
            for ci, start in enumerate(range(0, len(text), max_chunk_size)):
                chunk = {**doc, "content": text[start : start + max_chunk_size],
                         "_orig_idx": idx, "_chunk_idx": ci}
                chunked.append(chunk)
    return chunked


def aggregate_chunk_scores(chunked_results: list[dict]) -> list[dict]:
    """Aggregate per-chunk rerank scores back to original documents (max score wins)."""
    best: dict[int, dict] = {}
    for item in chunked_results:
        oidx = item["_orig_idx"]
        score = item.get("rerank_score", 0.0)
        if oidx not in best or score > best[oidx]["rerank_score"]:
            best[oidx] = {k: v for k, v in item.items()
                          if k not in ("_orig_idx", "_chunk_idx")}
            best[oidx]["rerank_score"] = score
    return sorted(best.values(), key=lambda d: d["rerank_score"], reverse=True)


async def cohere_rerank(
    query: str,
    documents: list[dict],
    top_k: int = 5,
    model: str = "rerank-v3.5",
    api_key: str | None = None,
) -> list[dict]:
    """Rerank *documents* using the Cohere Rerank API (AsyncClientV2).

    Returns up to *top_k* documents sorted by relevance, each augmented with
    a ``rerank_score`` key.
    """
    if cohere is None:
        raise ImportError("Install the 'cohere' package to use cohere_rerank.")

    key = api_key or os.environ.get("COHERE_API_KEY", "")
    client = cohere.AsyncClientV2(api_key=key)

    texts = [doc.get("content", "") for doc in documents]
    try:
        response = await client.rerank(
            query=query, documents=texts, top_n=top_k, model=model,
        )
    finally:
        await client.close()

    results: list[dict] = []
    for r in response.results:
        doc = {**documents[r.index], "rerank_score": r.relevance_score}
        results.append(doc)
    return sorted(results, key=lambda d: d["rerank_score"], reverse=True)
