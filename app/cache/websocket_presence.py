"""Short-lived WebSocket presence stored in Redis."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from app.cache.redis_client import get_redis
from app.config import settings

logger = logging.getLogger(__name__)


def _key(session_id: str) -> str:
    return f"ws:session:{session_id}:online"


async def mark_online(session_id: str, user_id: str | None = None) -> None:
    client = await get_redis()
    if client is None:
        return
    payload = {
        "session_id": session_id,
        "user_id": user_id or "",
        "connected_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        await client.set(_key(session_id), json.dumps(payload), ex=settings.ws_presence_ttl_seconds)
    except Exception:
        logger.debug("Failed to mark WebSocket online: session=%s", session_id, exc_info=True)


async def refresh_online(session_id: str) -> None:
    client = await get_redis()
    if client is None:
        return
    try:
        await client.expire(_key(session_id), settings.ws_presence_ttl_seconds)
    except Exception:
        logger.debug("Failed to refresh WebSocket presence: session=%s", session_id, exc_info=True)


async def mark_offline(session_id: str) -> None:
    client = await get_redis()
    if client is None:
        return
    try:
        await client.delete(_key(session_id))
    except Exception:
        logger.debug("Failed to mark WebSocket offline: session=%s", session_id, exc_info=True)


async def is_online(session_id: str) -> bool:
    client = await get_redis()
    if client is None:
        return False
    try:
        return bool(await client.exists(_key(session_id)))
    except Exception:
        logger.debug("Failed to read WebSocket presence: session=%s", session_id, exc_info=True)
        return False
