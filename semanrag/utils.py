"""SemanRAG utility functions used by operate.py, semanrag.py, and storage backends."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from functools import wraps
from typing import Any, Callable, Protocol, runtime_checkable

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore[assignment]

from semanrag.base import BaseKVStorage

# ── context var for trace_id propagation ──────────────────────────────────
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


# ═══════════════════════════════════════════════════════════════════════════
# 1. compute_mdhash_id
# ═══════════════════════════════════════════════════════════════════════════
def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + hashlib.md5(content.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# 2. EmbeddingFunc
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: Callable

    async def __call__(self, texts: list[str]) -> Any:
        return await self.func(texts)


# ═══════════════════════════════════════════════════════════════════════════
# 3. wrap_embedding_func_with_attrs
# ═══════════════════════════════════════════════════════════════════════════
def wrap_embedding_func_with_attrs(embedding_dim: int, max_token_size: int):
    def decorator(func):
        func.embedding_dim = embedding_dim
        func.max_token_size = max_token_size
        return func
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# 4. TokenizerInterface
# ═══════════════════════════════════════════════════════════════════════════
@runtime_checkable
class TokenizerInterface(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


# ═══════════════════════════════════════════════════════════════════════════
# 5. TiktokenTokenizer
# ═══════════════════════════════════════════════════════════════════════════
class TiktokenTokenizer:
    def __init__(self, model_name: str = "gpt-4o") -> None:
        if tiktoken is None:
            raise ImportError(
                "tiktoken is required for TiktokenTokenizer. "
                "Install it with: pip install tiktoken"
            )
        self._enc = tiktoken.encoding_for_model(model_name)

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)


# ═══════════════════════════════════════════════════════════════════════════
# 6. truncate_list_by_token_size
# ═══════════════════════════════════════════════════════════════════════════
def truncate_list_by_token_size(
    items: list,
    key: str,
    max_token_size: int,
    tokenizer: TokenizerInterface | None = None,
) -> list:
    if tokenizer is None:
        tokenizer = TiktokenTokenizer()
    result: list = []
    total = 0
    for item in items:
        tokens = len(tokenizer.encode(item[key]))
        if total + tokens > max_token_size:
            break
        total += tokens
        result.append(item)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 7. pack_user_ass_to_openai_messages
# ═══════════════════════════════════════════════════════════════════════════
def pack_user_ass_to_openai_messages(
    user_msg: str, assistant_msg: str
) -> list[dict]:
    return [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 8. CacheData
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class CacheData:
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    chunk_id: str = ""
    user_id: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# 9. handle_cache
# ═══════════════════════════════════════════════════════════════════════════
async def handle_cache(
    hashing_kv: BaseKVStorage | None,
    args_hash: str,
    prompt: str,
    cache_type: str,
) -> CacheData | None:
    if hashing_kv is None:
        return None
    cached = await hashing_kv.get_by_id(args_hash)
    if cached is None:
        return None
    return CacheData(
        content=cached.get("content", ""),
        timestamp=cached.get("timestamp", ""),
        chunk_id=cached.get("chunk_id", ""),
        user_id=cached.get("user_id", ""),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 10. save_to_cache
# ═══════════════════════════════════════════════════════════════════════════
async def save_to_cache(
    hashing_kv: BaseKVStorage | None,
    args_hash: str,
    content: str,
    prompt: str,
    cache_type: str,
    chunk_id: str = "",
    user_id: str = "",
) -> None:
    if hashing_kv is None:
        return
    cache_data = CacheData(
        content=content,
        chunk_id=chunk_id,
        user_id=user_id,
    )
    await hashing_kv.upsert({args_hash: asdict(cache_data)})


# ═══════════════════════════════════════════════════════════════════════════
# 11. use_llm_func_with_cache
# ═══════════════════════════════════════════════════════════════════════════
async def use_llm_func_with_cache(
    prompt: str,
    llm_func: Callable,
    system_prompt: str = "",
    llm_response_cache: BaseKVStorage | None = None,
    cache_type: str = "default",
    chunk_id: str = "",
    user_id: str = "",
) -> tuple[str, str]:
    args_hash = compute_mdhash_id(prompt + system_prompt)
    cached = await handle_cache(llm_response_cache, args_hash, prompt, cache_type)
    if cached is not None:
        return cached.content, cached.timestamp

    result = await llm_func(prompt, system_prompt=system_prompt)
    timestamp = datetime.now(UTC).isoformat()
    await save_to_cache(
        llm_response_cache, args_hash, result, prompt, cache_type,
        chunk_id=chunk_id, user_id=user_id,
    )
    return result, timestamp


# ═══════════════════════════════════════════════════════════════════════════
# 12. priority_limit_async_func_call
# ═══════════════════════════════════════════════════════════════════════════
def priority_limit_async_func_call(
    func: Callable, max_concurrent: int = 4, priority: int = 0
) -> Callable:
    semaphore = asyncio.Semaphore(max_concurrent)
    queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
    metrics = {"queue_depth": 0, "total_wait_time": 0.0, "call_count": 0}

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        event = asyncio.Event()
        metrics["queue_depth"] += 1
        await queue.put((priority, time.monotonic(), event))

        # Drain from queue in priority order
        _, enqueue_time, my_event = await queue.get()
        wait_start = time.monotonic()
        async with semaphore:
            wait_time = time.monotonic() - wait_start
            metrics["total_wait_time"] += wait_time
            metrics["call_count"] += 1
            metrics["queue_depth"] -= 1
            try:
                return await func(*args, **kwargs)
            finally:
                my_event.set()

    wrapper.metrics = metrics  # type: ignore[attr-defined]
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════
# 13. TokenTracker
# ═══════════════════════════════════════════════════════════════════════════
class TokenTracker:
    def __init__(self) -> None:
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0

    async def __aenter__(self) -> TokenTracker:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass

    def track(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens

    def report(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 14. BudgetExceededError
# ═══════════════════════════════════════════════════════════════════════════
class BudgetExceededError(Exception):
    pass


# ═══════════════════════════════════════════════════════════════════════════
# 15. TokenBudget
# ═══════════════════════════════════════════════════════════════════════════
class TokenBudget:
    def __init__(
        self,
        max_tokens_per_user_per_day: int = 0,
        max_tokens_per_workspace_per_day: int = 0,
    ) -> None:
        self.max_tokens_per_user_per_day = max_tokens_per_user_per_day
        self.max_tokens_per_workspace_per_day = max_tokens_per_workspace_per_day
        self._user_usage: dict[str, int] = {}
        self._workspace_usage: dict[str, int] = {}

    def check(self, user_id: str, workspace: str, tokens_needed: int) -> None:
        if self.max_tokens_per_user_per_day > 0:
            current_user = self._user_usage.get(user_id, 0)
            if current_user + tokens_needed > self.max_tokens_per_user_per_day:
                raise BudgetExceededError(
                    f"User '{user_id}' would exceed daily token budget "
                    f"({current_user + tokens_needed}/{self.max_tokens_per_user_per_day})"
                )
        if self.max_tokens_per_workspace_per_day > 0:
            current_ws = self._workspace_usage.get(workspace, 0)
            if current_ws + tokens_needed > self.max_tokens_per_workspace_per_day:
                raise BudgetExceededError(
                    f"Workspace '{workspace}' would exceed daily token budget "
                    f"({current_ws + tokens_needed}/{self.max_tokens_per_workspace_per_day})"
                )

    def record(self, user_id: str, workspace: str, tokens_used: int) -> None:
        self._user_usage[user_id] = self._user_usage.get(user_id, 0) + tokens_used
        self._workspace_usage[workspace] = (
            self._workspace_usage.get(workspace, 0) + tokens_used
        )

    def reset_daily(self) -> None:
        self._user_usage.clear()
        self._workspace_usage.clear()


# ═══════════════════════════════════════════════════════════════════════════
# 16 & 17. setup_logger + module-level logger
# ═══════════════════════════════════════════════════════════════════════════
class SafeStreamHandler(logging.StreamHandler):
    """Stream handler that gracefully handles encoding errors."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                stream.write(msg.encode("utf-8", errors="replace").decode("ascii", errors="replace") + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


class _JsonLikeFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        tid = trace_id_var.get("")
        log_obj = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if tid:
            log_obj["trace_id"] = tid
        if record.exc_info and record.exc_info[1]:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logger(name: str = "semanrag") -> logging.Logger:
    _logger = logging.getLogger(name)
    if not _logger.handlers:
        handler = SafeStreamHandler(sys.stderr)
        handler.setFormatter(_JsonLikeFormatter())
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
    return _logger


logger = setup_logger()


# ═══════════════════════════════════════════════════════════════════════════
# 20. SanitizingJSONEncoder
# ═══════════════════════════════════════════════════════════════════════════
class SanitizingJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, set):
            return list(o)
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="replace")
        if np is not None and isinstance(o, (np.ndarray,)):
            return o.tolist()
        if np is not None and isinstance(o, (np.integer,)):
            return int(o)
        if np is not None and isinstance(o, (np.floating,)):
            return float(o)
        # Pydantic models
        if hasattr(o, "model_dump"):
            return o.model_dump()
        if hasattr(o, "dict"):
            return o.dict()
        return super().default(o)


# ═══════════════════════════════════════════════════════════════════════════
# 18. write_json
# ═══════════════════════════════════════════════════════════════════════════
def write_json(data: Any, path: str) -> None:
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=SanitizingJSONEncoder, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


# ═══════════════════════════════════════════════════════════════════════════
# 19. load_json
# ═══════════════════════════════════════════════════════════════════════════
def load_json(path: str) -> dict | list:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# 21. safe_vdb_operation_with_exception
# ═══════════════════════════════════════════════════════════════════════════
def safe_vdb_operation_with_exception(func: Callable, max_retries: int = 3) -> Callable:
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"VDB operation {func.__name__} failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {wait}s: {e}"
                    )
                    await asyncio.sleep(wait)
        raise last_exc  # type: ignore[misc]

    return wrapper


# ═══════════════════════════════════════════════════════════════════════════
# 22. reciprocal_rank_fusion
# ═══════════════════════════════════════════════════════════════════════════
def reciprocal_rank_fusion(
    result_lists: list[list[dict]], k: int = 60
) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            items[item_id] = item
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [{**items[i], "rrf_score": scores[i]} for i in sorted_ids]


# ═══════════════════════════════════════════════════════════════════════════
# 23. detect_pii
# ═══════════════════════════════════════════════════════════════════════════
def detect_pii(text: str, policy: str = "flag") -> list[dict]:
    try:
        from presidio_analyzer import AnalyzerEngine
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=text, language="en")
        return [
            {
                "type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": r.score,
                "text": text[r.start : r.end],
            }
            for r in results
        ]
    except ImportError:
        return []


# ═══════════════════════════════════════════════════════════════════════════
# 24. detect_prompt_injection
# ═══════════════════════════════════════════════════════════════════════════
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"system\s*prompt",
    r"you\s+are\s+now",
    r"act\s+as\s+if",
    r"pretend\s+you\s+are",
    r"override\s+(your\s+)?instructions",
    r"new\s+instructions",
    r"forget\s+(all\s+)?previous",
    r"do\s+not\s+follow\s+(your\s+)?instructions",
    r"reveal\s+(your\s+)?(system|initial)\s+prompt",
]


def detect_prompt_injection(text: str) -> dict:
    matched: list[str] = []
    lower = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lower):
            matched.append(pattern)
    risk_score = min(len(matched) / max(len(_INJECTION_PATTERNS), 1), 1.0)
    return {"risk_score": risk_score, "patterns_matched": matched}


# ═══════════════════════════════════════════════════════════════════════════
# 25. sanitize_output
# ═══════════════════════════════════════════════════════════════════════════
_DEFAULT_FORBIDDEN = [
    "You are",
    "System:",
    "Instructions:",
    "<<SYS>>",
    "[INST]",
    "### System",
    "### Instructions",
]


def sanitize_output(
    text: str, forbidden_patterns: list[str] | None = None
) -> str:
    patterns = forbidden_patterns or _DEFAULT_FORBIDDEN
    result = text
    for pat in patterns:
        idx = result.find(pat)
        if idx != -1:
            # Remove from the pattern to end of that line
            end = result.find("\n", idx)
            if end == -1:
                result = result[:idx].rstrip()
            else:
                result = result[:idx] + result[end + 1 :]
    return result.strip()


# ═══════════════════════════════════════════════════════════════════════════
# 26. otel_span
# ═══════════════════════════════════════════════════════════════════════════
def otel_span(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                from opentelemetry import trace
                tracer = trace.get_tracer("semanrag")
                with tracer.start_as_current_span(name):
                    return await func(*args, **kwargs)
            except ImportError:
                return await func(*args, **kwargs)
        return wrapper
    return decorator
