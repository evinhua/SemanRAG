"""Verify SemanRAG setup: storage connectivity, model availability."""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv


async def _check() -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    # 1. Check working directory
    working_dir = os.environ.get("WORKING_DIR", "./data")
    exists = os.path.isdir(working_dir)
    results.append(("Working directory", exists, working_dir if exists else f"{working_dir} (not found)"))

    # 2. Check LLM config
    model = os.environ.get("LLM_MODEL", "")
    binding = os.environ.get("LLM_BINDING", "")
    has_llm = bool(model and binding)
    results.append(("LLM config", has_llm, f"{binding}/{model}" if has_llm else "LLM_MODEL or LLM_BINDING not set"))

    # 3. Check embedding config
    emb_model = os.environ.get("EMBEDDING_MODEL", "")
    emb_binding = os.environ.get("EMBEDDING_BINDING", "")
    has_emb = bool(emb_model and emb_binding)
    results.append(("Embedding config", has_emb,
                     f"{emb_binding}/{emb_model}" if has_emb else "EMBEDDING_MODEL or EMBEDDING_BINDING not set"))

    # 4. Check storage backends
    for name, env_key, default in [
        ("KG storage", "KG_STORAGE", "networkx"),
        ("Vector storage", "VECTOR_STORAGE", "nano-vectordb"),
        ("Doc storage", "DOC_STORAGE", "json"),
    ]:
        val = os.environ.get(env_key, default)
        results.append((name, True, val))

    # 5. Try to initialize SemanRAG
    try:
        from semanrag.semanrag import SemanRAG

        rag = SemanRAG(working_dir=working_dir)
        await rag.initialize_storages()
        await rag.finalize_storages()
        results.append(("SemanRAG init", True, "OK"))
    except Exception as exc:
        results.append(("SemanRAG init", False, str(exc)[:120]))

    # 6. Check optional services
    for name, env_key in [
        ("Redis", "REDIS_URL"),
        ("Neo4j", "NEO4J_URI"),
        ("PostgreSQL", "POSTGRES_DSN"),
    ]:
        val = os.environ.get(env_key, "")
        if val:
            results.append((name, True, f"configured ({val[:40]}...)"))
        else:
            results.append((name, True, "not configured (optional)"))

    return results


def main() -> int:
    load_dotenv()
    results = asyncio.run(_check())

    print("SemanRAG Initialization Check")
    print("=" * 60)
    all_ok = True
    for name, ok, detail in results:
        status = "✓" if ok else "✗"
        if not ok:
            all_ok = False
        print(f"  {status} {name:25s} {detail}")
    print("=" * 60)
    print("Result:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
