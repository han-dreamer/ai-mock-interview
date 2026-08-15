"""Per-task streaming context used to forward LLM deltas to the transport layer."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from typing import Any, Awaitable, Callable

StreamSink = Callable[[dict[str, Any]], Awaitable[None]]


class SessionEventBus:
    """In-process fan-out bus for one interview session.

    The graph publishes events even when no WebSocket is connected. Subscribers
    receive a bounded queue so a slow browser cannot block the graph forever.
    The final LangGraph state remains the source of truth for reconnects.
    """

    def __init__(self, max_queue_size: int = 256) -> None:
        self._max_queue_size = max_queue_size
        self._sequence = 0
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=self._max_queue_size,
        )
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    async def publish(self, event: dict[str, Any]) -> None:
        self._sequence += 1
        item = {
            "sequence": self._sequence,
            "event": dict(event),
        }
        for queue in tuple(self._subscribers):
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                # A final "end" event carries the complete text, so dropping
                # an old delta is recoverable and cannot corrupt checkpoint state.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(item)
                except asyncio.QueueFull:
                    pass

_stream_sink: ContextVar[StreamSink | None] = ContextVar(
    "interview_stream_sink",
    default=None,
)


def set_stream_sink(sink: StreamSink | None) -> Token[StreamSink | None]:
    """Install a task-local stream sink for the current graph invocation."""
    return _stream_sink.set(sink)


def reset_stream_sink(token: Token[StreamSink | None]) -> None:
    """Restore the previous task-local stream sink."""
    _stream_sink.reset(token)


def streaming_enabled() -> bool:
    return _stream_sink.get() is not None


async def emit_stream_event(event: dict[str, Any]) -> None:
    """Forward an event when the current invocation has a stream sink."""
    sink = _stream_sink.get()
    if sink is not None:
        await sink(event)


async def emit_status_event(stage: str, message: str) -> None:
    """Publish a user-visible workflow progress update when streaming is active."""
    await emit_stream_event(
        {
            "type": "status",
            "stage": stage,
            "message": message,
        }
    )
