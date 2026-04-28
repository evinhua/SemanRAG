"""Tests for batch embedding logic."""
from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest


async def batch_embed(texts: list[str], embed_func, batch_size: int = 32) -> np.ndarray:
    """Batch embedding helper that respects batch_size."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = await embed_func(batch)
        all_embeddings.append(embs)
    return np.vstack(all_embeddings) if all_embeddings else np.array([])


@pytest.mark.unit
class TestBatchEmbeddings:
    @pytest.mark.asyncio
    async def test_batch_size_respected(self, mock_embedding_func):
        """Mock embedding func, verify batch calls."""
        call_log = []

        async def tracked_embed(texts):
            call_log.append(len(texts))
            return np.random.randn(len(texts), 128).astype(np.float32)

        texts = [f"text_{i}" for i in range(100)]
        result = await batch_embed(texts, tracked_embed, batch_size=32)

        assert result.shape == (100, 128)
        assert call_log == [32, 32, 32, 4]  # 100 / 32 = 3 full + 1 partial

    @pytest.mark.asyncio
    async def test_single_item_batch(self, mock_embedding_func):
        """Edge case: single item."""
        async def single_embed(texts):
            return np.random.randn(len(texts), 128).astype(np.float32)

        result = await batch_embed(["single text"], single_embed, batch_size=32)
        assert result.shape == (1, 128)
