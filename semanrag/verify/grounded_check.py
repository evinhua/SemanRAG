"""Grounded-check verifier for RAG answers."""

from __future__ import annotations

import json
import re
from typing import Callable

from semanrag.prompt import PROMPTS


# ── helpers ──────────────────────────────────────────────────────────

def _split_into_claims(text: str) -> list[str]:
    """Split *text* into sentences on `.` `!` `?` followed by whitespace or end."""
    parts = re.split(r"(?<=[.!?])(?:\s|$)", text)
    return [s.strip() for s in parts if s.strip()]


def _parse_json(raw: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    # Try direct parse first, then look for first { ... }
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {"score": 0.0, "supporting_span": ""}


# ── main API ─────────────────────────────────────────────────────────

async def grounded_check(
    answer: str,
    contexts: list[dict],
    verifier_func: Callable | None = None,
    llm_func: Callable | None = None,
) -> list[dict]:
    """Verify each claim in *answer* against *contexts*.

    For every sentence (claim) the ``grounded_check`` prompt template is
    formatted with the claim and the concatenated context content, then
    *verifier_func* (or *llm_func* as fallback) is called.  The response
    is parsed for ``score`` and ``supporting_span``.

    Returns a list of dicts:
        ``{claim, score, supporting_span, supported}``
    """
    func = verifier_func or llm_func
    if func is None:
        raise ValueError("Provide verifier_func or llm_func.")

    claims = _split_into_claims(answer)
    context_text = "\n\n".join(c.get("content", "") for c in contexts)
    template = PROMPTS["grounded_check"]

    results: list[dict] = []
    for claim in claims:
        prompt = template.format(claim=claim, context=context_text)
        raw = await func(prompt)
        parsed = _parse_json(raw if isinstance(raw, str) else str(raw))
        score = float(parsed.get("score", 0.0))
        results.append({
            "claim": claim,
            "score": score,
            "supporting_span": parsed.get("supporting_span", ""),
            "supported": score >= 0.5,
        })
    return results


async def retry_with_expanded_context(
    query: str,
    answer: str,
    check_results: list[dict],
    contexts: list[dict],
    llm_func: Callable,
) -> str:
    """Re-generate the answer when some claims are unsupported.

    If every claim already has ``score >= 0.5`` the original *answer* is
    returned unchanged.  Otherwise a new prompt is built that highlights
    the unsupported claims and provides the full context, asking the LLM
    to produce an improved, fully-grounded answer.
    """
    unsupported = [r for r in check_results if r["score"] < 0.5]
    if not unsupported:
        return answer

    unsupported_text = "\n".join(f"- {r['claim']}" for r in unsupported)
    context_text = "\n\n".join(c.get("content", "") for c in contexts)

    prompt = (
        f"The following answer was generated for the query:\n\n"
        f"Query: {query}\n\n"
        f"Answer: {answer}\n\n"
        f"However, these claims could NOT be verified against the source context:\n"
        f"{unsupported_text}\n\n"
        f"Using ONLY the context below, rewrite the answer so that every claim "
        f"is supported. Remove or correct unsupported claims.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Improved answer:"
    )
    result = await llm_func(prompt)
    return result if isinstance(result, str) else str(result)
