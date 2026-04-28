"""Run entity resolution sweep — merge duplicate/similar entities."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv


async def _resolve(working_dir: str, workspace: str | None, threshold: float, dry_run: bool) -> int:
    from semanrag.semanrag import SemanRAG

    rag = SemanRAG(working_dir=working_dir, workspace=workspace)
    await rag.initialize_storages()

    try:
        from semanrag.operate import resolve_entities

        merge_groups = await resolve_entities(
            rag.graph_storage,
            rag._entities_vdb,
            rag._global_config,
            similarity_threshold=threshold,
        )

        if not merge_groups:
            print("No duplicate entities found.")
            return 0

        print(f"Found {len(merge_groups)} merge groups:")
        for canonical, aliases in merge_groups:
            print(f"  {canonical} ← {', '.join(aliases)}")

        if dry_run:
            print(f"\n[dry-run] Would merge {sum(len(a) for _, a in merge_groups)} entities into {len(merge_groups)}.")
        else:
            print(f"\nMerged {sum(len(a) for _, a in merge_groups)} entities into {len(merge_groups)} canonical forms.")
    except ImportError:
        print("Entity resolution requires the full SemanRAG installation.", file=sys.stderr)
        return 1
    finally:
        await rag.finalize_storages()

    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run entity resolution sweep")
    parser.add_argument("--working-dir", default=os.environ.get("WORKING_DIR", "./data"))
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold for merging")
    parser.add_argument("--dry-run", action="store_true", help="Show merges without applying")
    args = parser.parse_args()
    return asyncio.run(_resolve(args.working_dir, args.workspace, args.threshold, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
