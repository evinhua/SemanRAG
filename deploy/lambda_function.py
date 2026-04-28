"""AWS Lambda handler for SemanRAG with S3-backed knowledge graph storage.

Deploy as a Lambda function with an API Gateway trigger.
Environment variables: LLM_MODEL, LLM_BINDING, LLM_API_KEY, EMBEDDING_MODEL,
EMBEDDING_BINDING, S3_BUCKET, S3_PREFIX, WORKING_DIR (defaults to /tmp/semanrag).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import Any

import boto3

S3_BUCKET = os.environ.get("S3_BUCKET", "semanrag-data")
S3_PREFIX = os.environ.get("S3_PREFIX", "kg/")
WORKING_DIR = os.environ.get("WORKING_DIR", "/tmp/semanrag")

s3 = boto3.client("s3")


def _sync_from_s3() -> None:
    """Download KG data from S3 to local working dir."""
    os.makedirs(WORKING_DIR, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            local_path = os.path.join(WORKING_DIR, key[len(S3_PREFIX):])
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(S3_BUCKET, key, local_path)


def _sync_to_s3() -> None:
    """Upload modified KG data back to S3."""
    for root, _dirs, files in os.walk(WORKING_DIR):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, WORKING_DIR)
            s3_key = S3_PREFIX + rel
            s3.upload_file(local_path, S3_BUCKET, s3_key)


async def _handle_query(body: dict) -> dict:
    from semanrag.base import QueryParam
    from semanrag.semanrag import SemanRAG

    rag = SemanRAG(working_dir=WORKING_DIR)
    await rag.initialize_storages()
    try:
        param = QueryParam(mode=body.get("mode", "local"))
        result = await rag.aquery(body["query"], param=param)
        return {"content": result.content, "latency_ms": result.latency_ms}
    finally:
        await rag.finalize_storages()


async def _handle_ingest(body: dict) -> dict:
    from semanrag.semanrag import SemanRAG

    rag = SemanRAG(working_dir=WORKING_DIR)
    await rag.initialize_storages()
    try:
        content = body.get("content", "")
        if isinstance(content, list):
            await rag.ainsert(content)
        else:
            await rag.ainsert(content)
        return {"status": "ingested"}
    finally:
        await rag.finalize_storages()


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point."""
    _sync_from_s3()

    # Parse request
    http_method = event.get("httpMethod", event.get("requestContext", {}).get("http", {}).get("method", "POST"))
    body_str = event.get("body", "{}")
    body = json.loads(body_str) if isinstance(body_str, str) else body_str
    path = event.get("path", event.get("rawPath", "/query"))

    try:
        if "/ingest" in path:
            result = asyncio.get_event_loop().run_until_complete(_handle_ingest(body))
            _sync_to_s3()
        elif "/query" in path:
            result = asyncio.get_event_loop().run_until_complete(_handle_query(body))
        elif "/health" in path:
            result = {"status": "ok"}
        else:
            return {"statusCode": 404, "body": json.dumps({"error": "not found"})}

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result, ensure_ascii=False),
        }
    except Exception as exc:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(exc)}),
        }
