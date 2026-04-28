"""Scan for stale entities — not updated in N days."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from dotenv import load_dotenv


async def _scan(working_dir: str, workspace: str | None, days: int, remove: bool) -> int:
    from semanrag.semanrag import SemanRAG

    rag = SemanRAG(working_dir=working_dir, workspace=workspace)
    await rag.initialize_storages()

    cutoff = time.time() - (days * 86400)
    try:
        all_labels = await rag.graph_storage.get_all_labels()
        stale: list[tuple[str, float]] = []

        for label in all_labels:
            node = await rag.graph_storage.get_node(label)
            if not node:
                continue
            updated = node.get("updated_at", node.get("created_at", 0))
            if isinstance(updated, str):
                try:
                    from datetime import datetime
                    updated = datetime.fromisoformat(updated).timestamp()
                except (ValueError, TypeError):
                    updated = 0
            if updated < cutoff:
                stale.append((label, updated))

        if not stale:
            print(f"No entities stale for >{days} days.")
            return 0

        stale.sort(key=lambda x: x[1])
        print(f"Found {len(stale)} stale entities (>{days} days):")
        for label, ts in stale[:50]:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts > 0 else "unknown"
            print(f"  - {label}: last updated {dt}")
        if len(stale) > 50:
            print(f"  ... and {len(stale) - 50} more")

        if remove:
            for label, _ in stale:
                await rag.graph_storage.delete_node(label)
            print(f"\nRemoved {len(stale)} stale entities.")
    finally:
        await rag.finalize_storages()

    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Scan for stale entities")
    parser.add_argument("--working-dir", default=os.environ.get("WORKING_DIR", "./data"))
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--days", type=int, default=90, help="Stale threshold in days")
    parser.add_argument("--remove", action="store_true", help="Remove stale entities")
    args = parser.parse_args()
    return asyncio.run(_scan(args.working_dir, args.workspace, args.days, args.remove))


if __name__ == "__main__":
    sys.exit(main())
