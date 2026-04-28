"""Feedback collection routes."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/feedback", tags=["feedback"])

# In-memory store (swap for DB-backed storage in production)
_feedback_store: list[dict] = []


# ── Pydantic models ──────────────────────────────────────────────────

class RatingModel(BaseModel):
    relevance: int = Field(3, ge=1, le=5)
    accuracy: int = Field(3, ge=1, le=5)
    faithfulness: int = Field(3, ge=1, le=5)


class FeedbackRequest(BaseModel):
    query_id: str
    thumbs: Literal["up", "down"]
    rating: RatingModel | None = None
    comment: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    query_id: str
    thumbs: str
    rating: RatingModel | None = None
    comment: str | None = None
    created_at: str


# ── Routes ───────────────────────────────────────────────────────────

@router.post("", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(body: FeedbackRequest, request: Request):
    entry = {
        "feedback_id": str(uuid.uuid4()),
        "query_id": body.query_id,
        "thumbs": body.thumbs,
        "rating": body.rating.model_dump() if body.rating else None,
        "comment": body.comment,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _feedback_store.append(entry)
    return FeedbackResponse(**entry, rating=body.rating)


@router.get("")
async def list_feedback(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    total = len(_feedback_store)
    items = _feedback_store[offset : offset + limit]
    return {"feedback": items, "total": total, "offset": offset, "limit": limit}
