"""Redis distributed locks for cross-worker coordination."""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from app.cache.redis_client import get_redis
from app.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def redis_lock(key: str, ttl_ms: int | None = None) -> AsyncIterator[None]:
    """Acquire a best-effort Redis lock using SET NX PX.

    When Redis is unavailable, the context manager yields normally so the
    existing in-process asyncio lock remains the fallback.
    """
    client = await get_redis()
    if client is None:
        yield
        return

    token = str(uuid.uuid4())
    ttl = ttl_ms or settings.redis_lock_ttl_ms
    try:
        acquired = bool(await client.set(key, token, nx=True, px=ttl))
    except Exception:
        logger.warning("Redis lock failed open: key=%s", key, exc_info=True)
        yield
        return

    if not acquired:
        raise RuntimeError("This session is already processing an answer. Please wait.")

    try:
        yield
    finally:
        try:
            current = await client.get(key)
            if current == token:
                await client.delete(key)
        except Exception:
            logger.debug("Failed to release Redis lock: key=%s", key, exc_info=True)


def session_answer_lock(session_id: str):
    return redis_lock(f"lock:session:{session_id}:answer")
