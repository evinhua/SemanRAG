"""Tests for FAISS sidecar metadata consistency and ACL filtering."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


@pytest.fixture()
def faiss_storage(tmp_working_dir):
    """Create a FaissVectorDBStorage instance with mocked faiss."""
    with patch("semanrag.kg.faiss_impl.FAISS_AVAILABLE", True), \
         patch("semanrag.kg.faiss_impl.faiss") as mock_faiss:
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_index.is_trained = True
        mock_faiss.IndexFlatIP = MagicMock(return_value=mock_index)
        mock_faiss.read_index = MagicMock(return_value=mock_index)
        mock_faiss.write_index = MagicMock()

        from semanrag.kg.faiss_impl import FaissVectorDBStorage

        embedding_func = MagicMock()
        embedding_func.embedding_dim = 128
        embedding_func.max_token_size = 512

        storage = FaissVectorDBStorage(
            global_config={"working_dir": str(tmp_working_dir)},
            namespace="test",
            workspace="default",
            embedding_func=embedding_func,
        )
        # Manually set up internal state
        storage._index = mock_index
        storage._meta = {
            0: {"__id__": "doc1", "content": "hello", "acl_public": True},
            1: {"__id__": "doc2", "content": "world", "acl_public": False, "acl_owner": "user_a"},
            2: {"__id__": "doc3", "content": "secret", "acl_public": False, "acl_owner": "user_b",
                "acl_visible_to_groups": ["admin"]},
        }
        storage._id_to_pos = {"doc1": 0, "doc2": 1, "doc3": 2}
        storage._next_pos = 3
        yield storage


@pytest.mark.asyncio
async def test_delete_removes_from_sidecar(faiss_storage):
    """Deleting a document removes it from the sidecar metadata."""
    await faiss_storage.delete(["doc1"])

    assert "doc1" not in faiss_storage._id_to_pos
    assert 0 not in faiss_storage._meta
    # Other entries remain
    assert "doc2" in faiss_storage._id_to_pos
    assert 1 in faiss_storage._meta


@pytest.mark.asyncio
async def test_query_filters_by_acl(faiss_storage):
    """Query with ACL filter excludes documents the user cannot access."""
    rng = np.random.default_rng(42)
    mock_embedding = rng.standard_normal((1, 128)).astype(np.float32)
    faiss_storage.embedding_func = AsyncMock(return_value=mock_embedding)

    # Mock FAISS search to return all 3 positions
    faiss_storage._index.ntotal = 3
    faiss_storage._index.search = MagicMock(
        return_value=(
            np.array([[0.9, 0.8, 0.7]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )
    )

    # user_a can see doc1 (public) and doc2 (owner), but not doc3
    results = await faiss_storage.query("test", top_k=10, acl_filter={"user_id": "user_a", "user_groups": []})
    result_ids = [r["__id__"] for r in results]
    assert "doc1" in result_ids
    assert "doc2" in result_ids
    assert "doc3" not in result_ids

    # admin group can see doc1 (public) and doc3 (group), but not doc2
    results = await faiss_storage.query("test", top_k=10, acl_filter={"user_id": "other", "user_groups": ["admin"]})
    result_ids = [r["__id__"] for r in results]
    assert "doc1" in result_ids
    assert "doc3" in result_ids
    assert "doc2" not in result_ids
