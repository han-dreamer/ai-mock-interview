"""Optional Redis client singleton."""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover - exercised when dependency is absent locally
    redis = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_client: Any | None = None
_logged_unavailable = False


async def get_redis() -> Any | None:
    """Return a connected Redis client, or None when Redis is disabled/unavailable."""
    global _client, _logged_unavailable

    if not settings.redis_enabled:
        return None
    if redis is None:
        if not _logged_unavailable:
            logger.warning("Redis is enabled but the redis package is not installed.")
            _logged_unavailable = True
        return None
    if _client is not None:
        return _client

    try:
        client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=settings.redis_socket_timeout_seconds,
            socket_connect_timeout=settings.redis_socket_timeout_seconds,
        )
        await client.ping()
        _client = client
        logger.info("Redis connected: %s", settings.redis_url)
        return _client
    except Exception:
        if not _logged_unavailable:
            logger.warning("Redis is enabled but unavailable; Redis features will fail open.")
            _logged_unavailable = True
        return None


async def close_redis() -> None:
    """Close the Redis client during application shutdown."""
    global _client
    if _client is None:
        return
    try:
        await _client.aclose()
    except Exception:
        logger.debug("Failed to close Redis client cleanly.", exc_info=True)
    finally:
        _client = None
