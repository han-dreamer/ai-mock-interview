"""Lightweight Redis cache for session metadata, snapshots, and reports."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.cache.redis_client import get_redis
from app.config import settings
from app.models.interview import ChatMessage, InterviewSession

logger = logging.getLogger(__name__)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    return value


def _last_interviewer_message(state: dict[str, Any]) -> str | None:
    for msg in reversed(state.get("conversation_history", []) or []):
        if isinstance(msg, ChatMessage) and msg.role == "interviewer":
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "interviewer":
            return msg.get("content", "")
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _set_json(key: str, payload: dict[str, Any], ttl_seconds: int) -> None:
    client = await get_redis()
    if client is None:
        return
    try:
        await client.set(key, json.dumps(payload, ensure_ascii=False, default=str), ex=ttl_seconds)
    except Exception:
        logger.debug("Failed to write Redis cache key=%s", key, exc_info=True)


async def _get_json(key: str) -> dict[str, Any] | None:
    client = await get_redis()
    if client is None:
        return None
    try:
        raw = await client.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        logger.debug("Failed to read Redis cache key=%s", key, exc_info=True)
        return None


async def save_session_meta(session: InterviewSession, mode: str) -> None:
    await _set_json(
        f"session:{session.session_id}:meta",
        {
            "session_id": session.session_id,
            "mode": mode,
            "session": _dump(session),
            "updated_at": _now(),
        },
        settings.session_cache_ttl_seconds,
    )


async def save_session_snapshot(
    session: InterviewSession,
    state: dict[str, Any],
    mode: str,
    graph_started: bool,
) -> None:
    payload = {
        "session_id": session.session_id,
        "mode": mode,
        "graph_started": graph_started,
        "session": _dump(session),
        "state": {
            "interview_complete": bool(state.get("interview_complete", False)),
            "current_round": state.get("current_round", 1),
            "current_question_index": state.get("current_question_index", 0),
            "follow_up_count": state.get("follow_up_count", 0),
            "question_count": len(state.get("question_plan", []) or []),
            "assessment_count": len(state.get("assessments", []) or []),
            "last_interviewer_message": _last_interviewer_message(state),
        },
        "updated_at": _now(),
    }
    await _set_json(
        f"session:{session.session_id}:snapshot",
        payload,
        settings.session_cache_ttl_seconds,
    )


async def get_session_snapshot(session_id: str) -> dict[str, Any] | None:
    return await _get_json(f"session:{session_id}:snapshot")


async def save_session_report(session_id: str, report: Any) -> None:
    await _set_json(
        f"session:{session_id}:report",
        {
            "session_id": session_id,
            "report": _dump(report),
            "updated_at": _now(),
        },
        settings.report_cache_ttl_seconds,
    )


async def get_session_report(session_id: str) -> dict[str, Any] | None:
    return await _get_json(f"session:{session_id}:report")
