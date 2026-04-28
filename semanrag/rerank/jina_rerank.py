"""Jina AI reranker via HTTP API."""

from __future__ import annotations

import os

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


async def jina_rerank(
    query: str,
    documents: list[dict],
    top_k: int = 5,
    model: str = "jina-reranker-v2-base-multilingual",
    api_key: str | None = None,
) -> list[dict]:
    """Rerank *documents* using the Jina Rerank HTTP API.

    Returns up to *top_k* documents sorted by relevance with ``rerank_score``.
    """
    if httpx is None:
        raise ImportError("Install 'httpx' to use jina_rerank.")

    key = api_key or os.environ.get("JINA_API_KEY", "")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "query": query,
        "documents": [doc.get("content", "") for doc in documents],
        "top_n": top_k,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.jina.ai/v1/rerank", headers=headers, json=payload,
        )
        resp.raise_for_status()

    results: list[dict] = []
    for r in resp.json().get("results", []):
        idx = r["index"]
        results.append({**documents[idx], "rerank_score": r["relevance_score"]})
    return sorted(results, key=lambda d: d["rerank_score"], reverse=True)
