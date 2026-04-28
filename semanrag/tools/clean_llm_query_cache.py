"""Clean LLM response cache files by age or type."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean SemanRAG LLM cache")
    parser.add_argument("--working-dir", default="./data", help="SemanRAG working directory")
    parser.add_argument("--max-age-days", type=int, default=30, help="Remove entries older than N days")
    parser.add_argument("--type", choices=["all", "query", "extract", "summary"], default="all",
                        help="Cache type to clean")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed")
    args = parser.parse_args()

    working_dir = Path(args.working_dir)
    if not working_dir.exists():
        print(f"Working directory not found: {working_dir}", file=sys.stderr)
        return 1

    patterns = {
        "all": ["*_llm_response_cache.json", "*_query_cache.json", "*_extract_cache.json", "*_summary_cache.json"],
        "query": ["*_llm_response_cache.json", "*_query_cache.json"],
        "extract": ["*_extract_cache.json"],
        "summary": ["*_summary_cache.json"],
    }

    cutoff = time.time() - (args.max_age_days * 86400)
    total_removed = 0
    total_kept = 0

    for pattern in patterns[args.type]:
        for cache_file in working_dir.rglob(pattern):
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            if not isinstance(data, dict):
                continue

            original_count = len(data)
            cleaned = {
                k: v for k, v in data.items()
                if isinstance(v, dict) and v.get("timestamp", time.time()) >= cutoff
            }
            removed = original_count - len(cleaned)
            total_removed += removed
            total_kept += len(cleaned)

            if removed > 0:
                if args.dry_run:
                    print(f"[dry-run] {cache_file}: would remove {removed}/{original_count} entries")
                else:
                    cache_file.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"[cleaned] {cache_file}: removed {removed}/{original_count} entries")

    action = "Would remove" if args.dry_run else "Removed"
    print(f"\n{action} {total_removed} stale entries, kept {total_kept}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
