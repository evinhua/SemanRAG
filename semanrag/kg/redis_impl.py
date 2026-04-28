from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict

from semanrag.base import ACLPolicy, BaseKVStorage, DocStatus, DocStatusStorage

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError as _import_err:
    REDIS_AVAILABLE = False
    _redis_error = _import_err


def _get_redis_config(global_config: dict) -> dict:
    """Resolve Redis config from env vars then global_config."""
    return {
        "host": os.environ.get("REDIS_HOST") or global_config.get("redis_host", "localhost"),
        "port": int(os.environ.get("REDIS_PORT") or global_config.get("redis_port", 6379)),
        "db": int(os.environ.get("REDIS_DB") or global_config.get("redis_db", 0)),
        "password": os.environ.get("REDIS_PASSWORD") or global_config.get("redis_password") or None,
        "ssl": (os.environ.get("REDIS_SSL", "").lower() in ("1", "true"))
        or global_config.get("redis_ssl", False),
    }


def _make_client(cfg: dict) -> "aioredis.Redis":
    return aioredis.Redis(
        host=cfg["host"],
        port=cfg["port"],
        db=cfg["db"],
        password=cfg["password"],
        ssl=cfg["ssl"],
        decode_responses=True,
    )


# ---------------------------------------------------------------------------
# RedisKVStorage
# ---------------------------------------------------------------------------
class RedisKVStorage(BaseKVStorage):
    """Redis hash-backed key-value storage. One hash per namespace, workspace-prefixed keys."""

    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
    ) -> None:
        super().__init__(global_config, namespace, workspace)
        if not REDIS_AVAILABLE:
            raise ImportError(
                f"redis is required but not installed: {_redis_error}. "
                "Install with: pip install redis"
            )
        self._cfg = _get_redis_config(global_config)
        self._redis: aioredis.Redis | None = None
        ws = self._workspace or "default"
        self._hash_key = f"semanrag:kv:{ws}:{self._namespace}"
        self._key_prefix = f"{ws}:"

    def _fk(self, key: str) -> str:
        """Full key with workspace prefix."""
        return f"{self._key_prefix}{key}"

    async def initialize(self) -> None:
        self._redis = _make_client(self._cfg)
        await self._redis.ping()
        logger.info("RedisKVStorage connected: hash=%s", self._hash_key)

    async def finalize(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    async def get_by_id(self, id: str) -> dict | None:
        raw = await self._redis.hget(self._hash_key, self._fk(id))
        return json.loads(raw) if raw else None

    async def get_by_ids(self, ids: list[str]) -> list[dict | None]:
        if not ids:
            return []
        keys = [self._fk(i) for i in ids]
        values = await self._redis.hmget(self._hash_key, *keys)
        return [json.loads(v) if v else None for v in values]

    async def filter_keys(self, data: set[str]) -> set[str]:
        if not data:
            return set()
        pipe = self._redis.pipeline()
        ordered = list(data)
        for k in ordered:
            pipe.hexists(self._hash_key, self._fk(k))
        results = await pipe.execute()
        return {k for k, exists in zip(ordered, results) if exists}

    async def upsert(self, data: dict[str, dict]) -> None:
        if not data:
            return
        pipe = self._redis.pipeline()
        for k, v in data.items():
            pipe.hset(self._hash_key, self._fk(k), json.dumps(v, ensure_ascii=False))
        await pipe.execute()

    async def delete(self, ids: list[str]) -> None:
        if ids:
            keys = [self._fk(i) for i in ids]
            await self._redis.hdel(self._hash_key, *keys)

    async def drop(self) -> None:
        await self._redis.delete(self._hash_key)

    async def index_done_callback(self) -> None:
        # Redis persists via RDB/AOF; explicit BGSAVE not required.
        pass


# ---------------------------------------------------------------------------
# RedisDocStatusStorage
# ---------------------------------------------------------------------------
class RedisDocStatusStorage(DocStatusStorage):
    """Redis hash for doc status with sorted-set index for status lookups."""

    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
    ) -> None:
        super().__init__(global_config, namespace, workspace)
        if not REDIS_AVAILABLE:
            raise ImportError(
                f"redis is required but not installed: {_redis_error}. "
                "Install with: pip install redis"
            )
        self._cfg = _get_redis_config(global_config)
        self._redis: aioredis.Redis | None = None
        ws = self._workspace or "default"
        self._hash_key = f"semanrag:docstatus:{ws}:{self._namespace}"
        self._status_set_prefix = f"semanrag:docstatus_idx:{ws}:{self._namespace}"

    def _status_set_key(self, status: str) -> str:
        return f"{self._status_set_prefix}:{status}"

    @staticmethod
    def _to_json(status: DocStatus) -> str:
        d = asdict(status)
        if status.acl_policy is not None:
            d["acl_policy"] = asdict(status.acl_policy)
        return json.dumps(d, ensure_ascii=False)

    @staticmethod
    def _from_json(raw: str) -> DocStatus:
        d = json.loads(raw)
        acl = d.get("acl_policy")
        d["acl_policy"] = ACLPolicy(**acl) if isinstance(acl, dict) else None
        return DocStatus(**d)

    async def initialize(self) -> None:
        self._redis = _make_client(self._cfg)
        await self._redis.ping()
        logger.info("RedisDocStatusStorage connected: hash=%s", self._hash_key)

    async def finalize(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    async def get(self, doc_id: str) -> DocStatus | None:
        raw = await self._redis.hget(self._hash_key, doc_id)
        return self._from_json(raw) if raw else None

    async def upsert(self, doc_id: str, status: DocStatus) -> None:
        # Remove old status index entry
        old_raw = await self._redis.hget(self._hash_key, doc_id)
        if old_raw:
            old = json.loads(old_raw)
            old_status = old.get("status", "pending")
            await self._redis.srem(self._status_set_key(old_status), doc_id)

        await self._redis.hset(self._hash_key, doc_id, self._to_json(status))
        await self._redis.sadd(self._status_set_key(status.status), doc_id)

    async def delete(self, doc_id: str) -> None:
        raw = await self._redis.hget(self._hash_key, doc_id)
        if raw:
            old = json.loads(raw)
            await self._redis.srem(self._status_set_key(old.get("status", "pending")), doc_id)
            await self._redis.hdel(self._hash_key, doc_id)

    async def get_status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        cursor = "0"
        while True:
            cursor, items = await self._redis.hscan(self._hash_key, cursor=cursor, count=200)
            for _key, val in items.items():
                s = json.loads(val).get("status", "pending")
                counts[s] = counts.get(s, 0) + 1
            if cursor == "0" or cursor == 0:
                break
        return counts

    async def get_all_status_counts(self) -> dict[str, int]:
        return await self.get_status_counts()

    async def get_docs_by_status(self, status: str) -> list[DocStatus]:
        doc_ids = await self._redis.smembers(self._status_set_key(status))
        if not doc_ids:
            return []
        values = await self._redis.hmget(self._hash_key, *doc_ids)
        return [self._from_json(v) for v in values if v]

    async def get_docs_paginated(
        self,
        offset: int,
        limit: int,
        status: str | None = None,
        acl_filter: dict | None = None,
    ) -> tuple[list[DocStatus], int]:
        # Collect all matching docs via HSCAN
        docs: list[DocStatus] = []
        cursor = "0"
        while True:
            cursor, items = await self._redis.hscan(self._hash_key, cursor=cursor, count=200)
            for _key, val in items.items():
                doc = self._from_json(val)
                if status is not None and doc.status != status:
                    continue
                if acl_filter:
                    uid = acl_filter.get("user_id", "")
                    groups = acl_filter.get("user_groups", [])
                    if doc.acl_policy and not doc.acl_policy.can_access(uid, groups):
                        continue
                docs.append(doc)
            if cursor == "0" or cursor == 0:
                break
        total = len(docs)
        return docs[offset : offset + limit], total

    async def get_doc_by_file_path(self, file_path: str) -> DocStatus | None:
        cursor = "0"
        while True:
            cursor, items = await self._redis.hscan(self._hash_key, cursor=cursor, count=200)
            for _key, val in items.items():
                d = json.loads(val)
                if d.get("file_path") == file_path:
                    return self._from_json(val)
            if cursor == "0" or cursor == 0:
                break
        return None
