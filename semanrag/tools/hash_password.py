"""Prompt for a password and print its bcrypt hash."""

from __future__ import annotations

import getpass
import sys


def main() -> int:
    try:
        import bcrypt
    except ImportError:
        print("Error: bcrypt not installed. pip install bcrypt", file=sys.stderr)
        return 1

    password = getpass.getpass("Enter password: ")
    if not password:
        print("Error: empty password", file=sys.stderr)
        return 1
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Error: passwords do not match", file=sys.stderr)
        return 1

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    print(f"\nBcrypt hash:\n{hashed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
