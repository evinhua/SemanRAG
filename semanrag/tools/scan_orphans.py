"""Scan for orphan entities — nodes with no connected edges."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv


async def _scan(working_dir: str, workspace: str | None, remove: bool) -> int:
    from semanrag.semanrag import SemanRAG

    rag = SemanRAG(working_dir=working_dir, workspace=workspace)
    await rag.initialize_storages()

    try:
        all_labels = await rag.graph_storage.get_all_labels()
        orphans: list[str] = []

        for label in all_labels:
            edges = await rag.graph_storage.get_node_edges(label)
            if not edges:
                orphans.append(label)

        if not orphans:
            print("No orphan entities found.")
            return 0

        print(f"Found {len(orphans)} orphan entities:")
        for o in orphans:
            node = await rag.graph_storage.get_node(o)
            desc = (node.get("description", "") or "")[:80] if node else ""
            print(f"  - {o}: {desc}")

        if remove:
            for o in orphans:
                await rag.graph_storage.delete_node(o)
            print(f"\nRemoved {len(orphans)} orphan entities.")
    finally:
        await rag.finalize_storages()

    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Scan for orphan entities")
    parser.add_argument("--working-dir", default=os.environ.get("WORKING_DIR", "./data"))
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--remove", action="store_true", help="Remove orphan entities")
    args = parser.parse_args()
    return asyncio.run(_scan(args.working_dir, args.workspace, args.remove))


if __name__ == "__main__":
    sys.exit(main())
