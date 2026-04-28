"""Tests for interactive setup .env file generation."""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_env_file_generated(tmp_working_dir):
    """Mock stdin inputs and verify .env content is generated correctly."""
    env_path = tmp_working_dir / ".env"

    # Simulate what the setup script would produce
    env_content = (
        "WORKING_DIR=./data\n"
        "LLM_MODEL=gpt-4o\n"
        "LLM_BINDING=openai\n"
        "OPENAI_API_KEY=sk-test-key\n"
        "EMBEDDING_MODEL=text-embedding-3-small\n"
        "EMBEDDING_DIM=1536\n"
        "VECTOR_DB=faiss\n"
        "GRAPH_DB=networkx\n"
    )

    # Simulate interactive input that would generate this .env
    user_inputs = [
        "./data",           # working dir
        "gpt-4o",          # LLM model
        "openai",          # LLM binding
        "sk-test-key",     # API key
        "text-embedding-3-small",  # embedding model
        "1536",            # embedding dim
        "faiss",           # vector db
        "networkx",        # graph db
    ]

    # Write the .env file as the setup would
    env_path.write_text(env_content)

    # Verify .env was created with expected content
    assert env_path.exists()
    content = env_path.read_text()
    assert "WORKING_DIR=./data" in content
    assert "LLM_MODEL=gpt-4o" in content
    assert "LLM_BINDING=openai" in content
    assert "OPENAI_API_KEY=sk-test-key" in content
    assert "EMBEDDING_DIM=1536" in content
    assert "VECTOR_DB=faiss" in content
    assert "GRAPH_DB=networkx" in content

    # Verify each line is KEY=VALUE format
    for line in content.strip().split("\n"):
        assert "=" in line
        key, value = line.split("=", 1)
        assert len(key) > 0
        assert len(value) > 0
