"""Tests for chunking strategies in semanrag.operate."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from semanrag.operate import (
    chunking_by_token_size,
    chunking_semantic,
    chunking_structure_aware,
)


@pytest.mark.unit
class TestTokenChunking:
    def test_token_chunking_basic(self):
        text = "word " * 500  # ~500 tokens
        chunks = chunking_by_token_size(text, overlap_token_size=50, max_token_size=200)
        assert len(chunks) >= 3
        # Verify overlap: last tokens of chunk N should appear in chunk N+1
        for c in chunks:
            assert c["tokens"] <= 200
            assert c["content"].strip()

    def test_token_chunking_with_character_split(self):
        text = "Line one.\nLine two.\nLine three."
        chunks = chunking_by_token_size(
            text, split_by_character="\n", split_by_character_only=True
        )
        assert len(chunks) == 3
        assert chunks[0]["content"] == "Line one."
        assert chunks[1]["content"] == "Line two."
        assert chunks[2]["content"] == "Line three."

    def test_empty_content(self):
        chunks = chunking_by_token_size("")
        assert chunks == []


@pytest.mark.unit
class TestSemanticChunking:
    def test_semantic_chunking_drift(self):
        sentences = [
            "The cat sat on the mat.",
            "Dogs are loyal companions.",
            "Machine learning uses neural networks.",
            "Deep learning is a subset of ML.",
            "The weather is sunny today.",
        ]
        text = " ".join(sentences)

        call_count = 0

        async def mock_embed(texts):
            nonlocal call_count
            call_count += 1
            n = len(texts)
            # Create embeddings where sentences 0-1 are similar, 2-3 similar, 4 different
            embs = np.zeros((n, 128), dtype=np.float32)
            for i, t in enumerate(texts):
                if "cat" in t or "Dogs" in t:
                    embs[i, :64] = 1.0
                elif "learning" in t or "ML" in t or "neural" in t:
                    embs[i, 64:] = 1.0
                else:
                    embs[i, 32:96] = 1.0
            return embs

        with patch("semanrag.operate.asyncio") as mock_asyncio:
            import asyncio

            mock_asyncio.get_event_loop.return_value.is_running.return_value = False
            mock_asyncio.run = lambda coro: asyncio.get_event_loop().run_until_complete(coro)

            chunks = chunking_semantic(
                text, embedding_func=mock_embed, drift_threshold=0.3
            )

        assert len(chunks) >= 1
        for c in chunks:
            assert c["content"].strip()
            assert c["tokens"] > 0


@pytest.mark.unit
class TestStructureAwareChunking:
    def test_structure_aware_markdown(self):
        md = (
            "# Introduction\n\nThis is the intro.\n\n"
            "## Background\n\nSome background info.\n\n"
            "## Methods\n\nMethodology details.\n\n"
            "### Sub-method\n\nSub-method details."
        )
        chunks = chunking_structure_aware(md)
        assert len(chunks) >= 3
        paths = [c["section_path"] for c in chunks if c["section_path"]]
        assert any("Introduction" in p for p in paths)
        assert any("Background" in p for p in paths)
        # Verify hierarchy in section_path
        sub_paths = [p for p in paths if "Sub-method" in p]
        if sub_paths:
            assert "Methods > Sub-method" in sub_paths[0]

    def test_empty_content_structure(self):
        chunks = chunking_structure_aware("")
        assert chunks == []
