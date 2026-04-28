from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

from semanrag.base import ACLPolicy, DocStatus, DocStatusStorage


class JsonDocStatusStorage(DocStatusStorage):
    """JSON file-backed document status storage with ACL-aware filtering."""

    def __init__(
        self,
        global_config: dict,
        namespace: str,
        workspace: str | None = None,
    ) -> None:
        super().__init__(global_config, namespace, workspace)
        self._working_dir = global_config.get("working_dir", "./data")
        self._data: dict[str, dict] | None = None
        self._lock = asyncio.Lock()

    @property
    def _file_path(self) -> str:
        return os.path.join(self._working_dir, f"{self.full_namespace}_doc_status.json")

    def _ensure_dir(self) -> None:
        Path(self._file_path).parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, dict]:
        if self._data is not None:
            return self._data
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}
        return self._data

    def _save(self) -> None:
        self._ensure_dir()
        tmp = self._file_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)
        os.replace(tmp, self._file_path)

    @staticmethod
    def _to_dict(status: DocStatus) -> dict:
        d = asdict(status)
        if status.acl_policy is not None:
            d["acl_policy"] = asdict(status.acl_policy)
        return d

    @staticmethod
    def _from_dict(d: dict) -> DocStatus:
        acl = d.get("acl_policy")
        if isinstance(acl, dict):
            d = {**d, "acl_policy": ACLPolicy(**acl)}
        else:
            d = {**d, "acl_policy": None}
        return DocStatus(**d)

    async def get(self, doc_id: str) -> DocStatus | None:
        raw = self._load().get(doc_id)
        return self._from_dict(raw) if raw else None

    async def upsert(self, doc_id: str, status: DocStatus) -> None:
        async with self._lock:
            self._load()[doc_id] = self._to_dict(status)
            self._save()

    async def delete(self, doc_id: str) -> None:
        async with self._lock:
            if self._load().pop(doc_id, None) is not None:
                self._save()

    async def get_status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self._load().values():
            s = d.get("status", "pending")
            counts[s] = counts.get(s, 0) + 1
        return counts

    async def get_all_status_counts(self) -> dict[str, int]:
        return await self.get_status_counts()

    async def get_docs_by_status(self, status: str) -> list[DocStatus]:
        return [
            self._from_dict(d)
            for d in self._load().values()
            if d.get("status") == status
        ]

    async def get_docs_paginated(
        self,
        offset: int,
        limit: int,
        status: str | None = None,
        acl_filter: dict | None = None,
    ) -> tuple[list[DocStatus], int]:
        docs = list(self._load().values())
        if status is not None:
            docs = [d for d in docs if d.get("status") == status]
        if acl_filter is not None:
            uid = acl_filter.get("user_id", "")
            groups = acl_filter.get("user_groups", [])
            filtered = []
            for d in docs:
                acl_raw = d.get("acl_policy")
                if acl_raw is None or not isinstance(acl_raw, dict):
                    filtered.append(d)
                    continue
                if ACLPolicy(**acl_raw).can_access(uid, groups):
                    filtered.append(d)
            docs = filtered
        total = len(docs)
        page = [self._from_dict(d) for d in docs[offset : offset + limit]]
        return page, total

    async def get_doc_by_file_path(self, file_path: str) -> DocStatus | None:
        for d in self._load().values():
            if d.get("file_path") == file_path:
                return self._from_dict(d)
        return None

    async def initialize(self) -> None:
        self._ensure_dir()
        self._load()

    async def finalize(self) -> None:
        async with self._lock:
            if self._data is not None:
                self._save()
