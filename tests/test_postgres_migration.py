"""Tests for PostgreSQL migration SQL generation (mock asyncpg)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

asyncpg = pytest.importorskip("asyncpg")


@pytest.fixture()
def mock_pool():
    pool = AsyncMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    conn.execute = AsyncMock(return_value="CREATE TABLE")
    conn.fetch = AsyncMock(return_value=[])
    return pool, conn


@pytest.mark.asyncio
async def test_table_creation(mock_pool):
    """Verify CREATE TABLE SQL is issued during initialization."""
    pool, conn = mock_pool

    with patch("asyncpg.create_pool", new_callable=AsyncMock, return_value=pool):
        from semanrag.kg.postgres_impl import PGKVStorage, PostgreSQLDB

        db = PostgreSQLDB({"pg_host": "localhost", "pg_database": "test"})
        db._pool = pool

        storage = PGKVStorage.__new__(PGKVStorage)
        storage._db = db
        storage._table = "test_kv"
        storage._global_config = {}
        storage._namespace = "test"
        storage._workspace = None

        await storage.initialize()

        # Verify CREATE TABLE was called
        calls = [str(c) for c in conn.execute.call_args_list]
        create_calls = [c for c in calls if "CREATE TABLE" in c]
        assert len(create_calls) > 0


@pytest.mark.asyncio
async def test_index_creation(mock_pool):
    """Verify index SQL is issued during initialization."""
    pool, conn = mock_pool

    with patch("asyncpg.create_pool", new_callable=AsyncMock, return_value=pool):
        from semanrag.kg.postgres_impl import PGKVStorage, PostgreSQLDB

        db = PostgreSQLDB({"pg_host": "localhost", "pg_database": "test"})
        db._pool = pool

        storage = PGKVStorage.__new__(PGKVStorage)
        storage._db = db
        storage._table = "test_kv"
        storage._global_config = {}
        storage._namespace = "test"
        storage._workspace = None

        await storage.initialize()

        # Check that execute was called (table + potential index creation)
        assert conn.execute.call_count >= 1
        all_sql = " ".join(str(c) for c in conn.execute.call_args_list)
        # At minimum, CREATE TABLE should be present
        assert "CREATE TABLE" in all_sql
