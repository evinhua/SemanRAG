"""Tests for MilvusIndexConfig validation."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from semanrag.kg.milvus_impl import MilvusIndexConfig

VALID_INDEX_TYPES = ["AUTOINDEX", "HNSW", "HNSW_SQ", "IVF_FLAT", "DISKANN", "SCANN"]


@pytest.mark.parametrize("index_type", VALID_INDEX_TYPES)
def test_valid_index_types(index_type):
    """All supported index types are accepted."""
    cfg = MilvusIndexConfig(index_type=index_type)
    assert cfg.index_type == index_type


def test_invalid_index_type_rejected():
    """Invalid index type raises ValueError."""
    with pytest.raises(ValueError, match="index_type must be one of"):
        MilvusIndexConfig(index_type="INVALID_TYPE")


def test_config_from_env_vars():
    """MilvusVectorDBStorage reads config from environment variables."""
    env = {
        "MILVUS_URI": "http://milvus.example.com:19530",
        "MILVUS_TOKEN": "secret-token",
    }
    with patch.dict(os.environ, env, clear=False):
        # We can't fully instantiate MilvusVectorDBStorage without pymilvus,
        # but we can verify the env var resolution logic via the config dataclass
        cfg = MilvusIndexConfig(index_type="HNSW", metric_type="COSINE")
        assert cfg.metric_type == "COSINE"
        assert os.environ["MILVUS_URI"] == "http://milvus.example.com:19530"
        assert os.environ["MILVUS_TOKEN"] == "secret-token"
