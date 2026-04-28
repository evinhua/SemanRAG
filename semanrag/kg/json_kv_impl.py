from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from semanrag.base import BaseKVStorage


class JsonKVStorage(BaseKVStorage):
    """File-based JSON key-value storage with lazy loading and atomic writes."""

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
        return os.path.join(self._working_dir, f"{self.full_namespace}.json")

    def _ensure_dir(self) -> None:
        Path(self._file_path).parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, dict]:
        if self._data is not None:
            return self._data
        try:
            with open(self._file_path, encoding="utf-8") as f:
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

    async def get_by_id(self, id: str) -> dict | None:
        return self._load().get(id)

    async def get_by_ids(self, ids: list[str]) -> list[dict | None]:
        data = self._load()
        return [data.get(i) for i in ids]

    async def filter_keys(self, data: set[str]) -> set[str]:
        return data & set(self._load().keys())

    async def upsert(self, data: dict[str, dict]) -> None:
        async with self._lock:
            self._load().update(data)

    async def delete(self, ids: list[str]) -> None:
        async with self._lock:
            store = self._load()
            for i in ids:
                store.pop(i, None)

    async def drop(self) -> None:
        async with self._lock:
            self._data = {}
            try:
                os.remove(self._file_path)
            except FileNotFoundError:
                pass

    async def initialize(self) -> None:
        self._ensure_dir()
        # Legacy cache migration: check for old-format file without workspace prefix
        if self._workspace:
            legacy_path = os.path.join(self._working_dir, f"{self._namespace}.json")
            if os.path.exists(legacy_path) and not os.path.exists(self._file_path):
                try:
                    with open(legacy_path, encoding="utf-8") as f:
                        legacy_data = json.load(f)
                    self._data = legacy_data
                    self._save()
                except (json.JSONDecodeError, OSError):
                    pass
        self._load()

    async def finalize(self) -> None:
        async with self._lock:
            if self._data is not None:
                self._save()

    async def index_done_callback(self) -> None:
        async with self._lock:
            if self._data is not None:
                self._save()
