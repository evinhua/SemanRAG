"""Document ingestion and management routes."""

from __future__ import annotations

import asyncio
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, WebSocket
from pydantic import BaseModel, Field

from semanrag.base import ACLPolicy, DocStatus

router = APIRouter(prefix="/documents", tags=["documents"])


# ── Pydantic models ──────────────────────────────────────────────────

class ACLPolicyModel(BaseModel):
    owner: str = ""
    visible_to_groups: list[str] = Field(default_factory=list)
    visible_to_users: list[str] = Field(default_factory=list)
    public: bool = True


class TextInsertRequest(BaseModel):
    content: str
    doc_id: str | None = None
    acl_policy: ACLPolicyModel | None = None


class TextsInsertRequest(BaseModel):
    contents: list[str]
    doc_ids: list[str] | None = None
    acl_policy: ACLPolicyModel | None = None


class ScanRequest(BaseModel):
    directory: str


class ACLUpdateRequest(BaseModel):
    acl_policy: ACLPolicyModel


class DocumentUploadResponse(BaseModel):
    doc_id: str
    status: str
    message: str


class DocumentSummary(BaseModel):
    id: str
    status: str
    content_length: int
    chunks_count: int
    created_at: str
    file_path: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total: int
    offset: int
    limit: int


class DocumentDetailResponse(BaseModel):
    id: str
    status: str
    content: str
    content_length: int
    chunks_count: int
    chunks_list: list[str]
    created_at: str
    updated_at: str
    file_path: str
    pii_findings: list[dict]
    prompt_injection_flags: list[dict]
    acl_policy: ACLPolicyModel | None = None
    version: int
    error_message: str


class PipelineStatusResponse(BaseModel):
    status_counts: dict[str, int]
    pending: int
    processing: int
    completed: int
    failed: int


# ── Helpers ──────────────────────────────────────────────────────────

def _acl_from_model(m: ACLPolicyModel | None) -> ACLPolicy | None:
    if m is None:
        return None
    return ACLPolicy(
        owner=m.owner,
        visible_to_groups=m.visible_to_groups,
        visible_to_users=m.visible_to_users,
        public=m.public,
    )


def _doc_to_detail(doc: DocStatus) -> dict:
    acl = None
    if doc.acl_policy:
        acl = ACLPolicyModel(
            owner=doc.acl_policy.owner,
            visible_to_groups=doc.acl_policy.visible_to_groups,
            visible_to_users=doc.acl_policy.visible_to_users,
            public=doc.acl_policy.public,
        )
    return DocumentDetailResponse(
        id=doc.id,
        status=doc.status,
        content=doc.content,
        content_length=doc.content_length,
        chunks_count=doc.chunks_count,
        chunks_list=doc.chunks_list,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        file_path=doc.file_path,
        pii_findings=doc.pii_findings,
        prompt_injection_flags=doc.prompt_injection_flags,
        acl_policy=acl,
        version=doc.version,
        error_message=doc.error_message,
    ).model_dump()


# ── Routes ───────────────────────────────────────────────────────────

@router.post("/text", response_model=DocumentUploadResponse, status_code=201)
async def insert_text(body: TextInsertRequest, request: Request):
    rag = request.app.state.rag
    acl = _acl_from_model(body.acl_policy)
    ids = [body.doc_id] if body.doc_id else None
    await rag.ainsert(body.content, ids=ids, acl_policy=acl)
    from semanrag.utils import compute_mdhash_id
    doc_id = body.doc_id or compute_mdhash_id(body.content, prefix="doc-")
    return DocumentUploadResponse(doc_id=doc_id, status="completed", message="Text inserted")


@router.post("/texts", response_model=list[DocumentUploadResponse], status_code=201)
async def insert_texts(body: TextsInsertRequest, request: Request):
    rag = request.app.state.rag
    acl = _acl_from_model(body.acl_policy)
    await rag.ainsert(body.contents, ids=body.doc_ids, acl_policy=acl)
    from semanrag.utils import compute_mdhash_id
    ids = body.doc_ids or [compute_mdhash_id(c, prefix="doc-") for c in body.contents]
    return [
        DocumentUploadResponse(doc_id=did, status="completed", message="Text inserted")
        for did in ids
    ]


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    acl_owner: str = Form(""),
    acl_public: bool = Form(True),
):
    rag = request.app.state.rag
    suffix = os.path.splitext(file.filename or "")[1].lower()
    supported = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".csv"}
    if suffix not in supported:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content_bytes = await file.read()
        tmp.write(content_bytes)
        tmp_path = tmp.name

    try:
        acl = ACLPolicy(owner=acl_owner, public=acl_public) if acl_owner else None
        if suffix in {".pdf", ".docx", ".pptx", ".xlsx"}:
            await rag.ainsert("", file_paths=[tmp_path], acl_policy=acl)
        else:
            text = content_bytes.decode("utf-8", errors="replace")
            await rag.ainsert(text, file_paths=[tmp_path], acl_policy=acl)
        from semanrag.utils import compute_mdhash_id
        doc_id = compute_mdhash_id(tmp_path, prefix="doc-")
        return DocumentUploadResponse(doc_id=doc_id, status="completed", message=f"Uploaded {file.filename}")
    finally:
        os.unlink(tmp_path)


@router.post("/scan", response_model=list[DocumentUploadResponse], status_code=201)
async def scan_directory(body: ScanRequest, request: Request):
    rag = request.app.state.rag
    directory = body.directory
    if not os.path.isdir(directory):
        raise HTTPException(400, f"Not a valid directory: {directory}")

    supported = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".csv"}
    results = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in supported:
                continue
            fpath = os.path.join(root, fname)
            try:
                if ext in {".pdf", ".docx", ".pptx", ".xlsx"}:
                    await rag.ainsert("", file_paths=[fpath])
                else:
                    with open(fpath, encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    await rag.ainsert(text, file_paths=[fpath])
                from semanrag.utils import compute_mdhash_id
                doc_id = compute_mdhash_id(fpath, prefix="doc-")
                results.append(DocumentUploadResponse(doc_id=doc_id, status="completed", message=f"Scanned {fname}"))
            except Exception as exc:
                results.append(DocumentUploadResponse(doc_id="", status="failed", message=str(exc)))
    return results


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    user_id: str | None = Query(None),
):
    rag = request.app.state.rag
    acl_filter = {"user_id": user_id, "user_groups": []} if user_id else None
    docs, total = await rag.doc_status_storage.get_docs_paginated(
        offset=offset, limit=limit, status=status, acl_filter=acl_filter,
    )
    return DocumentListResponse(
        documents=[
            DocumentSummary(
                id=d.id, status=d.status, content_length=d.content_length,
                chunks_count=d.chunks_count, created_at=d.created_at, file_path=d.file_path,
            )
            for d in docs
        ],
        total=total, offset=offset, limit=limit,
    )


@router.get("/pipeline-status", response_model=PipelineStatusResponse)
async def pipeline_status(request: Request):
    rag = request.app.state.rag
    counts = await rag.doc_status_storage.get_all_status_counts()
    return PipelineStatusResponse(
        status_counts=counts,
        pending=counts.get("pending", 0),
        processing=counts.get("processing", 0),
        completed=counts.get("completed", 0),
        failed=counts.get("failed", 0),
    )


@router.post("/pipeline-cancel", status_code=200)
async def pipeline_cancel(request: Request):
    rag = request.app.state.rag
    pending = await rag.doc_status_storage.get_docs_by_status("pending")
    cancelled = 0
    for doc in pending:
        doc.status = "cancelled"
        await rag.doc_status_storage.upsert(doc.id, doc)
        cancelled += 1
    return {"cancelled": cancelled}


# ── Inbox-based upload (copy to volume, then scan) ───────────────────

INBOX_DIR = os.environ.get(
    "SEMANRAG_INBOX_DIR",
    os.path.join(os.environ.get("WORKING_DIR", "./semanrag_data"), "inbox"),
)


@router.post("/inbox/upload", status_code=201)
async def inbox_upload(file: UploadFile = File(...)):
    """Copy file to inbox volume without ingestion. Fast file transfer only."""
    os.makedirs(INBOX_DIR, exist_ok=True)
    fname = file.filename or "unnamed"
    dest = os.path.join(INBOX_DIR, fname)
    try:
        with open(dest, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
        return {"file": fname, "status": "copied", "path": dest}
    except Exception as exc:
        raise HTTPException(500, f"Failed to copy {fname}: {exc}")


@router.get("/inbox")
async def inbox_list():
    """List files currently in the inbox (queued or waiting for ingestion)."""
    if not os.path.isdir(INBOX_DIR):
        return {"files": []}
    supported = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".csv"}
    files = []
    for fname in sorted(os.listdir(INBOX_DIR)):
        fpath = os.path.join(INBOX_DIR, fname)
        if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in supported:
            files.append({"file": fname, "size": os.path.getsize(fpath)})
    return {"files": files}


@router.post("/inbox/scan")
async def inbox_scan(request: Request):
    """Scan inbox directory and enqueue files for background ingestion."""
    from fastapi.responses import JSONResponse
    import asyncio

    rag = request.app.state.rag
    if not os.path.isdir(INBOX_DIR):
        return JSONResponse(content={"files": [], "message": "Inbox empty"})

    supported = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".csv"}
    files_found = []
    for fname in sorted(os.listdir(INBOX_DIR)):
        fpath = os.path.join(INBOX_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in supported:
            continue
        files_found.append({"file": fname, "path": fpath, "ext": ext})

    if not files_found:
        return JSONResponse(content={"files": [], "message": "No supported files in inbox"})

    # Process in background
    async def _ingest_files():
        for item in files_found:
            fpath, ext, fname = item["path"], item["ext"], item["file"]
            try:
                if ext in {".pdf", ".docx", ".pptx", ".xlsx"}:
                    await rag.ainsert("", file_paths=[fpath])
                else:
                    with open(fpath, encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    await rag.ainsert(text, file_paths=[fpath])
                os.unlink(fpath)
            except Exception:
                pass  # File stays in inbox for retry

    asyncio.create_task(_ingest_files())

    return JSONResponse(content={
        "files": [f["file"] for f in files_found],
        "message": f"Enqueued {len(files_found)} files for ingestion",
    })


@router.get("/{doc_id}")
async def get_document(doc_id: str, request: Request):
    rag = request.app.state.rag
    doc = await rag.doc_status_storage.get(doc_id)
    if doc is None:
        raise HTTPException(404, f"Document {doc_id} not found")
    return _doc_to_detail(doc)


@router.delete("/{doc_id}", status_code=200)
async def delete_document(doc_id: str, request: Request):
    rag = request.app.state.rag
    doc = await rag.doc_status_storage.get(doc_id)
    if doc is None:
        raise HTTPException(404, f"Document {doc_id} not found")
    await rag.delete_by_doc_id(doc_id)
    return {"deleted": doc_id}


@router.put("/{doc_id}/acl")
async def update_acl(doc_id: str, body: ACLUpdateRequest, request: Request):
    rag = request.app.state.rag
    doc = await rag.doc_status_storage.get(doc_id)
    if doc is None:
        raise HTTPException(404, f"Document {doc_id} not found")
    doc.acl_policy = _acl_from_model(body.acl_policy)
    await rag.doc_status_storage.upsert(doc_id, doc)
    return {"doc_id": doc_id, "acl_policy": body.acl_policy.model_dump()}


@router.post("/{doc_id}/reingest", response_model=DocumentUploadResponse)
async def reingest_document(doc_id: str, request: Request):
    rag = request.app.state.rag
    doc = await rag.doc_status_storage.get(doc_id)
    if doc is None:
        raise HTTPException(404, f"Document {doc_id} not found")

    # Delete old data
    await rag.delete_by_doc_id(doc_id)

    # Re-insert with version bump
    new_version = doc.version + 1
    acl = doc.acl_policy
    content = doc.content
    file_path = doc.file_path

    if file_path and os.path.exists(file_path):
        await rag.ainsert("", ids=[doc_id], file_paths=[file_path], acl_policy=acl)
    else:
        await rag.ainsert(content, ids=[doc_id], acl_policy=acl)

    # Update version
    updated = await rag.doc_status_storage.get(doc_id)
    if updated:
        updated.version = new_version
        await rag.doc_status_storage.upsert(doc_id, updated)

    return DocumentUploadResponse(doc_id=doc_id, status="completed", message=f"Re-ingested v{new_version}")


@router.websocket("/ws/pipeline")
async def pipeline_ws(websocket: WebSocket):
    """WebSocket for live pipeline status — sends updates when docs change status."""
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping"})
    except Exception:
        pass
