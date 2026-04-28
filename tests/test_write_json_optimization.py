"""Tests for write_json and SanitizingJSONEncoder."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np
import pytest

from semanrag.utils import SanitizingJSONEncoder, write_json


@pytest.fixture()
def output_path(tmp_working_dir):
    return str(tmp_working_dir / "output.json")


def test_unicode_sanitization(output_path):
    """Non-ASCII characters are preserved in output."""
    data = {"name": "日本語テスト", "emoji": "🚀", "accented": "café"}
    write_json(data, output_path)
    with open(output_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["name"] == "日本語テスト"
    assert loaded["emoji"] == "🚀"
    assert loaded["accented"] == "café"


def test_datetime_serialization(output_path):
    """datetime objects are serialized to ISO format."""
    dt = datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC)
    data = {"created_at": dt}
    write_json(data, output_path)
    with open(output_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["created_at"] == "2024-06-15T12:30:00+00:00"


def test_numpy_array_serialization(output_path):
    """numpy arrays are serialized as lists."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    data = {"embedding": arr}
    write_json(data, output_path)
    with open(output_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["embedding"] == [1.0, 2.0, 3.0]


def test_atomic_write(tmp_working_dir):
    """Verify write uses .tmp then os.replace pattern."""
    output_path = str(tmp_working_dir / "atomic.json")
    data = {"key": "value"}

    with patch("semanrag.utils.os.replace", wraps=os.replace) as mock_replace:
        write_json(data, output_path)
        mock_replace.assert_called_once()
        args = mock_replace.call_args[0]
        assert args[0].endswith(".tmp")
        assert args[1] == output_path

    # Verify final file is correct
    with open(output_path, encoding="utf-8") as f:
        assert json.load(f) == {"key": "value"}
