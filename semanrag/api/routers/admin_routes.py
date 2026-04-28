"""Admin routes – users, groups, budget, audit, eval, PII, cache."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Optional

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from semanrag.utils import TokenBudget

router = APIRouter(prefix="/admin", tags=["admin"])

# In-memory stores
_audit_log: list[dict] = []
_eval_history: list[dict] = []


def _audit(action: str, detail: str = ""):
    _audit_log.append({
        "id": str(uuid.uuid4()),
        "action": action,
        "detail": detail,
        "timestamp": datetime.now(UTC).isoformat(),
    })


# ── Pydantic models ──────────────────────────────────────────────────

class BudgetRequest(BaseModel):
    max_tokens_per_user_per_day: int = 0
    max_tokens_per_workspace_per_day: int = 0


class EvalRunRequest(BaseModel):
    dataset: Optional[str] = None
    metrics: list[str] = Field(default_factory=lambda: ["relevance", "faithfulness"])


class CachePurgeRequest(BaseModel):
    scope: str = "all"  # all | llm | vectors | lexical


# ── Routes ───────────────────────────────────────────────────────────

@router.get("/users")
async def list_users(request: Request):
    rag = request.app.state.rag
    if rag.token_budget:
        users = list(rag.token_budget._user_usage.keys())
    else:
        users = []
    return {"users": users}


@router.get("/groups")
async def list_groups(request: Request):
    # Groups are derived from ACL policies on documents
    rag = request.app.state.rag
    groups: set[str] = set()
    docs, _ = await rag.doc_status_storage.get_docs_paginated(0, 1000)
    for doc in docs:
        if doc.acl_policy and doc.acl_policy.visible_to_groups:
            groups.update(doc.acl_policy.visible_to_groups)
    return {"groups": sorted(groups)}


@router.post("/budget")
async def configure_budget(body: BudgetRequest, request: Request):
    rag = request.app.state.rag
    rag.token_budget = TokenBudget(
        max_tokens_per_user_per_day=body.max_tokens_per_user_per_day,
        max_tokens_per_workspace_per_day=body.max_tokens_per_workspace_per_day,
    )
    _audit("budget_configured", f"user={body.max_tokens_per_user_per_day}, ws={body.max_tokens_per_workspace_per_day}")
    return {"status": "configured", **body.model_dump()}


@router.get("/cost-report")
async def cost_report(request: Request):
    rag = request.app.state.rag
    if rag.token_budget is None:
        return {"user_usage": {}, "workspace_usage": {}}
    return {
        "user_usage": dict(rag.token_budget._user_usage),
        "workspace_usage": dict(rag.token_budget._workspace_usage),
    }


@router.get("/audit-log")
async def audit_log(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    total = len(_audit_log)
    items = _audit_log[offset : offset + limit]
    return {"entries": items, "total": total, "offset": offset, "limit": limit}


@router.post("/eval/run", status_code=202)
async def eval_run(body: EvalRunRequest, request: Request):
    run_id = str(uuid.uuid4())
    entry = {
        "run_id": run_id,
        "dataset": body.dataset,
        "metrics": body.metrics,
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "results": None,
    }
    _eval_history.append(entry)
    _audit("eval_run_started", run_id)
    # Actual evaluation would be dispatched as a background task
    entry["status"] = "completed"
    entry["completed_at"] = datetime.now(UTC).isoformat()
    return {"run_id": run_id, "status": "accepted"}


@router.get("/eval/history")
async def eval_history(request: Request):
    return {"runs": _eval_history}


@router.get("/pii-report")
async def pii_report(request: Request):
    rag = request.app.state.rag
    docs, _ = await rag.doc_status_storage.get_docs_paginated(0, 10000)
    findings = []
    for doc in docs:
        if doc.pii_findings:
            findings.append({"doc_id": doc.id, "file_path": doc.file_path, "pii_findings": doc.pii_findings})
    return {"total_docs_with_pii": len(findings), "documents": findings}


@router.post("/cache/purge")
async def purge_cache(body: CachePurgeRequest, request: Request):
    rag = request.app.state.rag
    await rag.clear_cache(scope=body.scope)
    _audit("cache_purged", body.scope)
    return {"status": "purged", "scope": body.scope}
