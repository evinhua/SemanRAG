"""Tests for document status chunk preservation during pipeline failures."""

from __future__ import annotations

import asyncio

import pytest

from semanrag.base import DocStatus
from semanrag.kg.json_doc_status_impl import JsonDocStatusStorage


@pytest.fixture()
def storage(tmp_working_dir):
    cfg = {"working_dir": str(tmp_working_dir)}
    s = JsonDocStatusStorage(cfg, namespace="test", workspace="default")
    asyncio.get_event_loop().run_until_complete(s.initialize())
    return s


@pytest.mark.asyncio
async def test_pipeline_failure_preserves_chunks(storage):
    """Simulate failure mid-pipeline; chunks from successful steps are preserved."""
    doc = DocStatus(id="doc1", status="processing", chunks_list=["c1", "c2"])
    await storage.upsert("doc1", doc)

    # Simulate a failure during further processing
    with pytest.raises(RuntimeError):
        doc.chunks_list.append("c3")
        raise RuntimeError("Embedding service unavailable")

    # Verify chunks from before failure are still in storage
    recovered = await storage.get("doc1")
    assert recovered is not None
    assert recovered.chunks_list == ["c1", "c2"]
    assert recovered.status == "processing"


@pytest.mark.asyncio
async def test_retry_after_failure(storage):
    """Verify retry picks up from last good state."""
    doc = DocStatus(id="doc2", status="processing", chunks_list=["c1"], error_message="timeout")
    await storage.upsert("doc2", doc)

    # Retry: load existing state and continue
    existing = await storage.get("doc2")
    assert existing is not None
    assert existing.chunks_list == ["c1"]

    # Continue processing from last good state
    existing.chunks_list.append("c2")
    existing.status = "completed"
    existing.error_message = ""
    await storage.upsert("doc2", existing)

    final = await storage.get("doc2")
    assert final.status == "completed"
    assert final.chunks_list == ["c1", "c2"]


@pytest.mark.asyncio
async def test_status_transitions(storage):
    """Verify pending -> processing -> completed transitions."""
    doc = DocStatus(id="doc3", status="pending")
    await storage.upsert("doc3", doc)

    result = await storage.get("doc3")
    assert result.status == "pending"

    doc.status = "processing"
    await storage.upsert("doc3", doc)
    result = await storage.get("doc3")
    assert result.status == "processing"

    doc.status = "completed"
    await storage.upsert("doc3", doc)
    result = await storage.get("doc3")
    assert result.status == "completed"
