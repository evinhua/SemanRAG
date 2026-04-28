"""Tests for Qdrant migration (mock qdrant_client)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture()
def mock_qdrant_client():
    client = AsyncMock()
    collections_response = MagicMock()
    collections_response.collections = []
    client.get_collections = AsyncMock(return_value=collections_response)
    client.create_collection = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_collection_creation(mock_qdrant_client):
    """Verify create_collection is called when collection doesn't exist."""
    with patch("semanrag.kg.qdrant_impl.AsyncQdrantClient", return_value=mock_qdrant_client):
        from semanrag.kg.qdrant_impl import QdrantVectorDBStorage

        embedding_func = MagicMock()
        embedding_func.embedding_dim = 128
        embedding_func.max_token_size = 512

        storage = QdrantVectorDBStorage.__new__(QdrantVectorDBStorage)
        storage._url = "http://localhost:6333"
        storage._api_key = None
        storage._prefer_grpc = False
        storage._collection_name = "test_collection"
        storage._dim = 128
        storage._client = None
        storage._namespace = "test"
        storage._workspace = "default"
        storage.embedding_func = embedding_func

        storage._client = mock_qdrant_client
        await storage.initialize()

        mock_qdrant_client.get_collections.assert_called_once()
        mock_qdrant_client.create_collection.assert_called_once()
        call_kwargs = mock_qdrant_client.create_collection.call_args[1]
        assert call_kwargs["collection_name"] == "test_collection"


def test_batch_size_estimation():
    """Verify batch size estimation constants."""
    from semanrag.kg.qdrant_impl import _POINT_OVERHEAD, _MAX_BATCH_BYTES

    dim = 128
    vector_bytes = dim * 4  # float32
    point_size = vector_bytes + _POINT_OVERHEAD
    estimated_batch = _MAX_BATCH_BYTES // point_size

    assert estimated_batch > 0
    assert _POINT_OVERHEAD == 128
    assert _MAX_BATCH_BYTES == 32 * 1024 * 1024
