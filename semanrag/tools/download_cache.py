"""Download tiktoken encoding cache files for offline use."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

TIKTOKEN_ENCODINGS = {
    "cl100k_base": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    "o200k_base": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
}


def main() -> int:
    cache_dir = Path.home() / ".cache" / "semanrag" / "tiktoken"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for name, url in TIKTOKEN_ENCODINGS.items():
        filename = hashlib.sha1(url.encode()).hexdigest()
        dest = cache_dir / filename
        if dest.exists():
            print(f"[skip] {name} already cached at {dest}")
            continue
        print(f"[download] {name} → {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[ok] {name} ({dest.stat().st_size:,} bytes)")
        except Exception as exc:
            print(f"[error] {name}: {exc}", file=sys.stderr)
            return 1

    print(f"\nSet TIKTOKEN_CACHE_DIR={cache_dir} to use offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
