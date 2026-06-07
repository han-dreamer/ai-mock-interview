"""LangGraph checkpointer lifecycle management."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from app.config import settings

logger = logging.getLogger(__name__)

_checkpointer: Any | None = None
_checkpointer_cm: Any | None = None


async def init_checkpointer() -> Any:
    """Initialize the configured LangGraph checkpointer.

    Local development and tests default to MemorySaver. Docker/production can set
    CHECKPOINTER_BACKEND=postgres to persist LangGraph checkpoints in PostgreSQL.
    """
    global _checkpointer, _checkpointer_cm

    if _checkpointer is not None:
        return _checkpointer

    backend = settings.checkpointer_backend.strip().lower()
    if backend in {"", "memory"}:
        _checkpointer = MemorySaver()
        logger.info("LangGraph checkpointer initialized: memory")
        return _checkpointer

    if backend != "postgres":
        raise ValueError(f"Unsupported CHECKPOINTER_BACKEND: {settings.checkpointer_backend}")

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        raise RuntimeError(
            "CHECKPOINTER_BACKEND=postgres requires langgraph-checkpoint-postgres "
            "and psycopg dependencies."
        ) from exc

    _checkpointer_cm = AsyncPostgresSaver.from_conn_string(settings.postgres_url)
    _checkpointer = await _checkpointer_cm.__aenter__()
    await _checkpointer.setup()
    logger.info("LangGraph checkpointer initialized: postgres")
    return _checkpointer


def get_checkpointer() -> Any:
    """Return the initialized checkpointer, lazily falling back to memory."""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
        logger.info("LangGraph checkpointer lazily initialized: memory")
    return _checkpointer


async def close_checkpointer() -> None:
    """Close any managed checkpointer resources."""
    global _checkpointer, _checkpointer_cm
    if _checkpointer_cm is not None:
        await _checkpointer_cm.__aexit__(None, None, None)
    _checkpointer = None
    _checkpointer_cm = None


def reset_checkpointer_for_tests() -> None:
    """Reset module state for tests that need a fresh in-memory checkpointer."""
    global _checkpointer, _checkpointer_cm
    _checkpointer = None
    _checkpointer_cm = None
