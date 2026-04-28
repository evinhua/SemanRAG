"""SemanRAG unified CLI — argparse-based, no extra dependencies."""

from __future__ import annotations

import argparse
import asyncio
import configparser
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _load_config() -> None:
    """Load .env and config.ini from CWD or project root."""
    for candidate in [Path.cwd(), Path(__file__).resolve().parents[2]]:
        env_path = candidate / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            break
    for candidate in [Path.cwd(), Path(__file__).resolve().parents[2]]:
        ini_path = candidate / "config.ini"
        if ini_path.exists():
            cfg = configparser.ConfigParser()
            cfg.read(ini_path)
            for section in cfg.sections():
                for key, val in cfg.items(section):
                    env_key = f"{section.upper()}_{key.upper()}"
                    os.environ.setdefault(env_key, val)
            break


def _make_rag(workspace: str | None = None):
    from semanrag.semanrag import SemanRAG

    working_dir = os.environ.get("WORKING_DIR", "./data")
    rag = SemanRAG(working_dir=working_dir, workspace=workspace)
    return rag


async def _init_rag(workspace: str | None = None):
    rag = _make_rag(workspace)
    await rag.initialize_storages()
    return rag


# ── Subcommand handlers ──────────────────────────────────────────────


async def _cmd_query(args: argparse.Namespace) -> int:
    from semanrag.base import QueryParam

    rag = await _init_rag(args.workspace)
    try:
        param = QueryParam(mode=args.mode)
        if args.snapshot_at:
            from datetime import datetime

            param.snapshot_at = datetime.fromisoformat(args.snapshot_at)
        if args.user:
            param.user_id = args.user
        result = await rag.aquery(args.query, param=param)
        print(result.content if result.content else "")
    finally:
        await rag.finalize_storages()
    return 0


async def _cmd_ingest(args: argparse.Namespace) -> int:
    rag = await _init_rag(args.workspace)
    try:
        path = Path(args.path)
        if path.is_file():
            files = [path]
        elif path.is_dir() and args.recursive:
            files = sorted(path.rglob("*"))
            files = [f for f in files if f.is_file()]
        elif path.is_dir():
            files = sorted(f for f in path.iterdir() if f.is_file())
        else:
            print(f"Error: {path} not found", file=sys.stderr)
            return 1

        acl_policy = None
        if args.acl_owner:
            from semanrag.base import ACLPolicy

            acl_policy = ACLPolicy(owner=args.acl_owner)

        for fp in files:
            suffix = fp.suffix.lower()
            if suffix in {".pdf", ".docx", ".pptx", ".xlsx"}:
                await rag.ainsert("", file_paths=[str(fp)], acl_policy=acl_policy)
            else:
                text = fp.read_text(encoding="utf-8", errors="replace")
                await rag.ainsert(text, file_paths=[str(fp)], acl_policy=acl_policy)
            print(f"Ingested: {fp}")
    finally:
        await rag.finalize_storages()
    return 0


async def _cmd_graph_export(args: argparse.Namespace) -> int:
    rag = await _init_rag(args.workspace)
    try:
        output = args.output or f"export.{args.format}"
        await rag.export_data(output, file_format=args.format)
        print(f"Exported graph to {output}")
    finally:
        await rag.finalize_storages()
    return 0


async def _cmd_eval_run(args: argparse.Namespace) -> int:
    try:
        from semanrag.evaluation.runner import run_evaluation
    except ImportError:
        print("Error: evaluation extras not installed. pip install semanrag[evaluation]", file=sys.stderr)
        return 1
    rag = await _init_rag(args.workspace)
    try:
        await run_evaluation(rag, domain=args.domain, baseline_path=args.baseline)
    finally:
        await rag.finalize_storages()
    return 0


async def _cmd_admin_budget(args: argparse.Namespace) -> int:
    from semanrag.utils import logger

    user = args.user or "default"
    max_tokens = args.max_tokens or 100000
    budget_file = Path(os.environ.get("WORKING_DIR", "./data")) / "budgets.json"
    budget_file.parent.mkdir(parents=True, exist_ok=True)

    import json

    budgets: dict = {}
    if budget_file.exists():
        budgets = json.loads(budget_file.read_text())
    budgets[user] = {"max_tokens": max_tokens}
    budget_file.write_text(json.dumps(budgets, indent=2))
    logger.info("Set budget for user=%s max_tokens=%d", user, max_tokens)
    print(f"Budget set: {user} → {max_tokens} tokens")
    return 0


async def _cmd_admin_cache_purge(args: argparse.Namespace) -> int:
    working_dir = Path(os.environ.get("WORKING_DIR", "./data"))
    scope = args.scope
    purged = 0
    patterns = {
        "all": ["*_llm_response_cache.json", "*_query_cache.json", "*_extract_cache.json", "*_summary_cache.json"],
        "query": ["*_query_cache.json", "*_llm_response_cache.json"],
        "extract": ["*_extract_cache.json"],
        "summary": ["*_summary_cache.json"],
    }
    for pattern in patterns.get(scope, patterns["all"]):
        for f in working_dir.rglob(pattern):
            f.unlink()
            purged += 1
    print(f"Purged {purged} cache file(s) (scope={scope})")
    return 0


async def _cmd_admin_users_list(args: argparse.Namespace) -> int:
    users_file = Path(os.environ.get("WORKING_DIR", "./data")) / "users.json"
    if not users_file.exists():
        print("No users file found.")
        return 0
    import json

    users = json.loads(users_file.read_text())
    for u in users if isinstance(users, list) else users.values():
        name = u.get("username", u.get("name", "unknown"))
        role = u.get("role", "user")
        print(f"  {name:20s}  role={role}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. pip install semanrag[api]", file=sys.stderr)
        return 1
    host = args.host or os.environ.get("API_HOST", "0.0.0.0")
    port = args.port or int(os.environ.get("API_PORT", "9621"))
    workers = args.workers or int(os.environ.get("API_WORKERS", "1"))
    uvicorn.run(
        "semanrag.api.semanrag_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )
    return 0


# ── Parser construction ──────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semanrag", description="SemanRAG CLI")
    sub = parser.add_subparsers(dest="command")

    # query
    p_query = sub.add_parser("query", help="Run a RAG query")
    p_query.add_argument("query", help="Query text")
    p_query.add_argument("--mode", choices=["local", "global", "hybrid", "naive", "mix", "community", "bypass"],
                         default="local")
    p_query.add_argument("--snapshot-at", default=None, help="ISO timestamp for temporal snapshot")
    p_query.add_argument("--user", default=None, help="User ID for ACL filtering")
    p_query.add_argument("--workspace", default=None)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents")
    p_ingest.add_argument("path", help="File or directory path")
    p_ingest.add_argument("--acl-owner", default=None)
    p_ingest.add_argument("--workspace", default=None)
    p_ingest.add_argument("--recursive", action="store_true")

    # graph export
    p_graph = sub.add_parser("graph", help="Graph operations")
    p_graph_sub = p_graph.add_subparsers(dest="graph_command")
    p_export = p_graph_sub.add_parser("export", help="Export knowledge graph")
    p_export.add_argument("--format", choices=["csv", "json", "graphml", "rdf", "cypher"], default="csv")
    p_export.add_argument("--output", default=None)
    p_export.add_argument("--workspace", default=None)

    # eval run
    p_eval = sub.add_parser("eval", help="Evaluation commands")
    p_eval_sub = p_eval.add_subparsers(dest="eval_command")
    p_eval_run = p_eval_sub.add_parser("run", help="Run evaluation")
    p_eval_run.add_argument("--domain", choices=["agriculture", "cs", "legal", "finance", "mixed"], default="mixed")
    p_eval_run.add_argument("--baseline", default=None)
    p_eval_run.add_argument("--workspace", default=None)

    # admin
    p_admin = sub.add_parser("admin", help="Admin commands")
    p_admin_sub = p_admin.add_subparsers(dest="admin_command")

    p_budget = p_admin_sub.add_parser("budget", help="Set token budget")
    p_budget.add_argument("--user", default=None)
    p_budget.add_argument("--max-tokens", type=int, default=None)

    p_cache = p_admin_sub.add_parser("cache", help="Cache operations")
    p_cache_sub = p_cache.add_subparsers(dest="cache_command")
    p_purge = p_cache_sub.add_parser("purge", help="Purge cache")
    p_purge.add_argument("--scope", choices=["all", "query", "extract", "summary"], default="all")

    p_users = p_admin_sub.add_parser("users", help="User management")
    p_users_sub = p_users.add_subparsers(dest="users_command")
    p_users_sub.add_parser("list", help="List users")

    # serve
    p_serve = sub.add_parser("serve", help="Start API server")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--workers", type=int, default=None)

    return parser


# ── Entry point ──────────────────────────────────────────────────────


def main() -> int:
    _load_config()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "serve":
        return _cmd_serve(args)

    if args.command == "query":
        return asyncio.run(_cmd_query(args))

    if args.command == "ingest":
        return asyncio.run(_cmd_ingest(args))

    if args.command == "graph":
        if getattr(args, "graph_command", None) == "export":
            return asyncio.run(_cmd_graph_export(args))
        parser.parse_args(["graph", "--help"])
        return 1

    if args.command == "eval":
        if getattr(args, "eval_command", None) == "run":
            return asyncio.run(_cmd_eval_run(args))
        parser.parse_args(["eval", "--help"])
        return 1

    if args.command == "admin":
        admin_cmd = getattr(args, "admin_command", None)
        if admin_cmd == "budget":
            return asyncio.run(_cmd_admin_budget(args))
        if admin_cmd == "cache":
            if getattr(args, "cache_command", None) == "purge":
                return asyncio.run(_cmd_admin_cache_purge(args))
        if admin_cmd == "users":
            if getattr(args, "users_command", None) == "list":
                return asyncio.run(_cmd_admin_users_list(args))
        parser.parse_args(["admin", "--help"])
        return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
