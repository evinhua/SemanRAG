"""Tests for InMemoryBM25Storage."""

from __future__ import annotations

import asyncio
import os

import pytest

from semanrag.kg.inmemory_bm25_impl import BM25_AVAILABLE, InMemoryBM25Storage

pytestmark = pytest.mark.skipif(not BM25_AVAILABLE, reason="rank_bm25 not installed")


@pytest.fixture()
def bm25_storage(tmp_working_dir):
    cfg = {"working_dir": str(tmp_working_dir)}
    storage = InMemoryBM25Storage(cfg, namespace="test", workspace="default")
    asyncio.get_event_loop().run_until_complete(storage.initialize())
    return storage


@pytest.mark.asyncio
async def test_inmemory_upsert_and_search(bm25_storage):
    """Upsert documents and verify BM25 search returns relevant results."""
    await bm25_storage.upsert({
        "d1": {"content": "knowledge graphs represent entities and relations"},
        "d2": {"content": "machine learning models require training data"},
        "d3": {"content": "graph neural networks combine graphs with deep learning"},
    })

    results = await bm25_storage.search_bm25("knowledge graph entities", top_k=2)
    assert len(results) > 0
    assert results[0]["id"] == "d1"


@pytest.mark.asyncio
async def test_inmemory_delete_rebuilds_index(bm25_storage):
    """Deleting documents rebuilds the BM25 index."""
    await bm25_storage.upsert({
        "d1": {"content": "alpha beta gamma"},
        "d2": {"content": "delta epsilon zeta"},
    })

    await bm25_storage.delete(["d1"])

    assert "d1" not in bm25_storage._documents
    assert len(bm25_storage._doc_ids) == 1
    assert bm25_storage._doc_ids[0] == "d2"

    results = await bm25_storage.search_bm25("alpha beta", top_k=5)
    # d1 was deleted, should not appear
    result_ids = [r["id"] for r in results]
    assert "d1" not in result_ids


@pytest.mark.asyncio
async def test_empty_corpus_returns_empty(bm25_storage):
    """Searching an empty corpus returns empty results."""
    results = await bm25_storage.search_bm25("anything", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_snapshot_persistence(tmp_working_dir):
    """Verify finalize saves and initialize loads snapshot."""
    cfg = {"working_dir": str(tmp_working_dir)}
    storage = InMemoryBM25Storage(cfg, namespace="persist", workspace="default")
    await storage.initialize()

    await storage.upsert({
        "d1": {"content": "persistent document one"},
        "d2": {"content": "persistent document two"},
    })
    await storage.finalize()

    # Verify snapshot file exists
    assert os.path.exists(storage._snapshot_path)

    # Create new instance and load
    storage2 = InMemoryBM25Storage(cfg, namespace="persist", workspace="default")
    await storage2.initialize()

    assert len(storage2._documents) == 2
    results = await storage2.search_bm25("persistent document", top_k=2)
    assert len(results) == 2
