"""Aliyun DashScope reranker."""

from __future__ import annotations

import os

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


async def ali_rerank(
    query: str,
    documents: list[dict],
    top_k: int = 5,
    model: str = "gte-rerank",
    api_key: str | None = None,
) -> list[dict]:
    """Rerank *documents* using the Aliyun DashScope rerank API.

    Returns up to *top_k* documents sorted by relevance with ``rerank_score``.
    """
    if httpx is None:
        raise ImportError("Install 'httpx' to use ali_rerank.")

    key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": {
            "query": query,
            "documents": [doc.get("content", "") for doc in documents],
        },
        "parameters": {"top_n": top_k},
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()

    results: list[dict] = []
    for r in resp.json().get("output", {}).get("results", []):
        idx = r["index"]
        results.append({**documents[idx], "rerank_score": r["relevance_score"]})
    return sorted(results, key=lambda d: d["rerank_score"], reverse=True)
