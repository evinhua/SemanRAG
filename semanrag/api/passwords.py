"""SemanRAG password hashing utilities and CLI entry point."""

from __future__ import annotations

import getpass
import sys

try:
    import bcrypt as _bcrypt
except ImportError:
    _bcrypt = None  # type: ignore[assignment]


def hash_password(password: str) -> str:
    if _bcrypt is None:
        raise ImportError("bcrypt is required. Install with: pip install bcrypt")
    return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    if _bcrypt is None:
        raise ImportError("bcrypt is required. Install with: pip install bcrypt")
    return _bcrypt.checkpw(plain.encode(), hashed.encode())


def main() -> None:
    """CLI entry point: prompt for a password and print its bcrypt hash."""
    try:
        password = getpass.getpass("Password: ")
        if not password:
            print("Error: empty password", file=sys.stderr)
            sys.exit(1)
        print(hash_password(password))
    except KeyboardInterrupt:
        print()
        sys.exit(130)


if __name__ == "__main__":
    main()
