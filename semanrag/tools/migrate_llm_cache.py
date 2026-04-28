"""Migrate LLM response cache between storage backends (JSON ↔ Redis)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


async def _migrate_json_to_redis(working_dir: str, redis_url: str) -> int:
    try:
        import redis.asyncio as aioredis
    except ImportError:
        print("Error: redis not installed. pip install redis", file=sys.stderr)
        return 1

    client = aioredis.from_url(redis_url)
    total = 0
    for cache_file in Path(working_dir).rglob("*_llm_response_cache.json"):
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        prefix = cache_file.stem
        pipe = client.pipeline()
        for key, val in data.items():
            pipe.set(f"semanrag:cache:{prefix}:{key}", json.dumps(val))
            total += 1
        await pipe.execute()
        print(f"  Migrated {len(data)} entries from {cache_file.name}")

    await client.aclose()
    print(f"\nTotal: {total} entries migrated to Redis")
    return 0


async def _migrate_redis_to_json(working_dir: str, redis_url: str) -> int:
    try:
        import redis.asyncio as aioredis
    except ImportError:
        print("Error: redis not installed. pip install redis", file=sys.stderr)
        return 1

    client = aioredis.from_url(redis_url)
    keys = [k async for k in client.scan_iter("semanrag:cache:*")]

    buckets: dict[str, dict] = {}
    for key in keys:
        key_str = key.decode() if isinstance(key, bytes) else key
        parts = key_str.split(":", 3)
        if len(parts) < 4:
            continue
        prefix = parts[2]
        cache_key = parts[3]
        val = await client.get(key)
        if val:
            buckets.setdefault(prefix, {})[cache_key] = json.loads(val)

    out_dir = Path(working_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for prefix, data in buckets.items():
        out_path = out_dir / f"{prefix}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        total += len(data)
        print(f"  Wrote {len(data)} entries to {out_path.name}")

    await client.aclose()
    print(f"\nTotal: {total} entries migrated to JSON")
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Migrate LLM cache between backends")
    parser.add_argument("direction", choices=["json-to-redis", "redis-to-json"])
    parser.add_argument("--working-dir", default=os.environ.get("WORKING_DIR", "./data"))
    parser.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    args = parser.parse_args()

    if args.direction == "json-to-redis":
        return asyncio.run(_migrate_json_to_redis(args.working_dir, args.redis_url))
    else:
        return asyncio.run(_migrate_redis_to_json(args.working_dir, args.redis_url))


if __name__ == "__main__":
    sys.exit(main())
