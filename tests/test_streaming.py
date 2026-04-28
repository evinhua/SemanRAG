"""Tests for streaming response formats (SSE and WebSocket)."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest


async def format_sse_event(data: str, event: str = "message") -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {data}\n\n"


async def stream_sse(chunks: list[str]):
    """Simulate SSE streaming."""
    for chunk in chunks:
        yield await format_sse_event(json.dumps({"content": chunk}))
    yield await format_sse_event("[DONE]", event="done")


async def stream_websocket(ws, chunks: list[str]):
    """Simulate WebSocket streaming."""
    for chunk in chunks:
        await ws.send_json({"type": "chunk", "content": chunk})
    await ws.send_json({"type": "done"})


@pytest.mark.unit
class TestStreaming:
    @pytest.mark.asyncio
    async def test_sse_streaming_format(self):
        """Verify SSE event format."""
        chunks = ["Hello", " world", "!"]
        events = []
        async for event in stream_sse(chunks):
            events.append(event)

        assert len(events) == 4  # 3 chunks + 1 done
        # Verify SSE format
        assert events[0].startswith("event: message\n")
        assert "data: " in events[0]
        assert events[0].endswith("\n\n")
        # Parse data
        data_line = events[0].split("data: ")[1].strip()
        parsed = json.loads(data_line)
        assert parsed["content"] == "Hello"
        # Verify done event
        assert "event: done" in events[-1]
        assert "[DONE]" in events[-1]

    @pytest.mark.asyncio
    async def test_websocket_streaming(self):
        """Verify WebSocket message format."""
        mock_ws = AsyncMock()
        chunks = ["Part 1", "Part 2", "Part 3"]
        await stream_websocket(mock_ws, chunks)

        assert mock_ws.send_json.call_count == 4  # 3 chunks + done
        calls = [c.args[0] for c in mock_ws.send_json.call_args_list]
        assert calls[0] == {"type": "chunk", "content": "Part 1"}
        assert calls[1] == {"type": "chunk", "content": "Part 2"}
        assert calls[2] == {"type": "chunk", "content": "Part 3"}
        assert calls[3] == {"type": "done"}
