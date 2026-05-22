"""Simple fixed-window Redis rate limiter."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

from app.cache.redis_client import get_redis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int
    key: str


def _key(name: str, identity: str, window_seconds: int) -> str:
    window_id = int(time.time() // window_seconds)
    identity_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    safe_name = "".join(ch if ch.isalnum() or ch in ":-_" else "_" for ch in name)
    return f"rl:{safe_name}:{identity_hash}:{window_id}"


async def check_rate_limit(
    name: str,
    identity: str,
    limit: int,
    window_seconds: int,
) -> RateLimitResult:
    """Check a fixed-window limit with INCR + EXPIRE.

    Redis errors fail open because rate limiting should protect the demo service
    without taking down the interview flow.
    """
    if limit <= 0 or window_seconds <= 0:
        return RateLimitResult(True, limit, 0, 0, "")

    client = await get_redis()
    key = _key(name, identity, window_seconds)
    if client is None:
        return RateLimitResult(True, limit, limit, 0, key)

    try:
        count = await client.incr(key)
        if count == 1:
            await client.expire(key, window_seconds + 1)
        ttl = await client.ttl(key)
        retry_after = max(int(ttl), 1) if ttl and ttl > 0 else window_seconds
        remaining = max(limit - int(count), 0)
        return RateLimitResult(
            allowed=int(count) <= limit,
            limit=limit,
            remaining=remaining,
            retry_after_seconds=retry_after,
            key=key,
        )
    except Exception:
        logger.warning("Redis rate limit failed open: name=%s", name, exc_info=True)
        return RateLimitResult(True, limit, limit, 0, key)
