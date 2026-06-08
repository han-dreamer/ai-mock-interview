"""WebSocket endpoint that drives the LangGraph interview in real time."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.cache.rate_limiter import check_rate_limit
from app.cache.websocket_presence import mark_offline, mark_online, refresh_online
from app.config import settings
from app.models.interview import ChatMessage
from app.models.ws_protocol import (
    ServerError,
    ServerFollowUp,
    ServerInterviewEnd,
    ServerQuestion,
    ServerReport,
    ServerStatus,
)
from app.security import decode_access_token
from app.services.user_repository import get_user_by_id
from app.services.session_manager import get_session_manager

router = APIRouter()
logger = logging.getLogger(__name__)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    return value


def _get_last_interviewer_message(state: dict) -> str | None:
    history = state.get("conversation_history", [])
    for msg in reversed(history):
        if isinstance(msg, ChatMessage) and msg.role == "interviewer":
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "interviewer":
            return msg.get("content", "")
    return None


def _get_question_meta(state: dict) -> tuple[int, int, list[str], str]:
    """Return (current_index, total, skill_tags, difficulty) for the current question."""
    plan = state.get("question_plan", []) or []
    idx = int(state.get("current_question_index", 0) or 0)
    if idx < len(plan):
        question = plan[idx]
        skill_tags = getattr(question, "skill_tags", None)
        difficulty = getattr(question, "difficulty", None)
        if isinstance(question, dict):
            skill_tags = question.get("skill_tags", skill_tags)
            difficulty = question.get("difficulty", difficulty)
        return idx, len(plan), skill_tags or [], difficulty or ""
    return idx, len(plan), [], ""


def _report_from_state(session_id: str, state: dict) -> Any:
    mgr = get_session_manager()
    return (
        state.get("practice_report")
        or state.get("professional_report")
        or state.get("final_report")
        or mgr.get_report_for_session(session_id)
    )


async def _send_current_turn(websocket: WebSocket, state: dict) -> bool:
    interviewer_msg = _get_last_interviewer_message(state)
    if not interviewer_msg:
        await _send(websocket, ServerStatus(stage="waiting", message="Waiting for next question."))
        return False

    idx, total, tags, diff = _get_question_meta(state)
    follow_up_count = int(state.get("follow_up_count", 0) or 0)
    if follow_up_count > 0:
        await _send(
            websocket,
            ServerFollowUp(
                question_index=idx + 1,
                follow_up_number=follow_up_count,
                content=interviewer_msg,
            ),
        )
    else:
        await _send(
            websocket,
            ServerQuestion(
                question_index=idx + 1,
                total_questions=total,
                content=interviewer_msg,
                skill_tags=tags,
                difficulty=diff,
            ),
        )
    return True


async def _send_report_if_ready(websocket: WebSocket, session_id: str, state: dict) -> bool:
    if not state.get("interview_complete"):
        return False
    report = _report_from_state(session_id, state)
    await _send(websocket, ServerInterviewEnd())
    if report:
        await _send(websocket, ServerReport(data=_dump(report)))
    return True


async def _ensure_started(websocket: WebSocket, session_id: str) -> dict | None:
    mgr = get_session_manager()
    state = mgr.get_last_state(session_id)
    if mgr.has_graph_started(session_id) and state:
        await _send(websocket, ServerStatus(stage="resumed", message="Interview session resumed."))
        return state

    await _send(
        websocket,
        ServerStatus(stage="analyzing_jd", message="Analyzing JD and preparing questions."),
    )
    try:
        state = await mgr.start_interview_graph(session_id)
    except Exception as exc:
        logger.exception("Graph start failed for %s", session_id)
        await _send(websocket, ServerError(message=f"Failed to start interview: {exc}"))
        return None

    await _send(
        websocket,
        ServerStatus(
            stage="questions_ready",
            message=f"Prepared {len(state.get('question_plan', []) or [])} interview questions.",
        ),
    )
    return state


@router.websocket("/interview/{session_id}")
async def interview_websocket(websocket: WebSocket, session_id: str):
    """Full interview WebSocket endpoint.

    The endpoint remains backward compatible with the older client: connecting to a
    pending session starts the graph automatically. Reconnecting to an existing
    session resumes from the latest checkpoint instead of starting a duplicate run.
    """
    raw_token = websocket.query_params.get("token") or websocket.query_params.get("auth_token")
    if not raw_token:
        await websocket.close(code=4401, reason="Authentication required")
        return
    try:
        payload = decode_access_token(raw_token)
        current_user = await get_user_by_id(str(payload["sub"]))
    except Exception:
        await websocket.close(code=4401, reason="Invalid authentication token")
        return
    if not current_user or not current_user.is_active:
        await websocket.close(code=4401, reason="User not found or inactive")
        return

    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    if session.user_id != current_user.id:
        await websocket.close(code=4403, reason="Session access denied")
        return

    await websocket.accept()
    await mark_online(session_id, session.user_id)
    logger.info("WS connected: session=%s", session_id)

    try:
        current_turn_delivered = False
        state = await _ensure_started(websocket, session_id)
        if state is None:
            return
        if await _send_report_if_ready(websocket, session_id, state):
            return
        current_turn_delivered = await _send_current_turn(websocket, state)

        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await _send(websocket, ServerError(message="Invalid JSON"))
                continue

            msg_type = msg.get("type", "")

            if msg_type in {"ping"}:
                await refresh_online(session_id)
                await _send(websocket, ServerStatus(stage="pong", message="ok"))
                continue

            if msg_type in {"start", "start_interview"}:
                await refresh_online(session_id)
                if current_turn_delivered:
                    continue
                state = await _ensure_started(websocket, session_id)
                report_sent = (
                    state is not None
                    and await _send_report_if_ready(websocket, session_id, state)
                )
                if state is not None and not report_sent and not current_turn_delivered:
                    current_turn_delivered = await _send_current_turn(websocket, state)
                continue

            if msg_type in {"stop", "end_interview"}:
                await refresh_online(session_id)
                await _send(
                    websocket,
                    ServerStatus(
                        stage="evaluating",
                        message="Stopping interview and generating report.",
                    ),
                )
                try:
                    state = await mgr.stop_interview(session_id)
                except Exception as exc:
                    logger.exception("Stop interview failed for %s", session_id)
                    await _send(websocket, ServerError(message=f"Stop interview failed: {exc}"))
                    continue
                await _send_report_if_ready(websocket, session_id, state)
                break

            if msg_type != "answer":
                await _send(
                    websocket,
                    ServerError(message=f"Expected 'answer' message, got '{msg_type}'"),
                )
                continue

            if not current_turn_delivered:
                state = mgr.get_last_state(session_id)
                current_turn_delivered = await _send_current_turn(websocket, state)
                if current_turn_delivered:
                    await _send(
                        websocket,
                        ServerError(
                            message=(
                                "The current question was just sent. "
                                "Please answer after it is displayed."
                            )
                        ),
                    )
                    continue
                await _send(
                    websocket,
                    ServerError(message="No interview question is ready yet. Please wait."),
                )
                continue

            answer_text = msg.get("content", "").strip()
            if not answer_text:
                await _send(websocket, ServerError(message="Answer cannot be empty"))
                continue
            await refresh_online(session_id)

            limit_identity = f"session:{session_id}:user:{session.user_id}"
            rate_limit = await check_rate_limit(
                "interview:answer:ws",
                limit_identity,
                settings.rate_limit_answer_per_minute,
                60,
            )
            if not rate_limit.allowed:
                await _send(
                    websocket,
                    ServerError(
                        message=(
                            "Too many answers. "
                            f"Please retry after {rate_limit.retry_after_seconds} seconds."
                        )
                    ),
                )
                continue

            await _send(
                websocket,
                ServerStatus(
                    stage="processing",
                    message="Assessing answer and deciding next step.",
                ),
            )

            try:
                state = await mgr.submit_answer(session_id, answer_text)
            except Exception as exc:
                logger.exception("Graph resume failed for %s", session_id)
                await _send(websocket, ServerError(message=f"Processing error: {exc}"))
                continue

            if await _send_report_if_ready(websocket, session_id, state):
                break
            current_turn_delivered = await _send_current_turn(websocket, state)

    except WebSocketDisconnect:
        logger.info("WS disconnected: session=%s", session_id)
    except Exception:
        logger.exception("Unexpected error in WS session=%s", session_id)
    finally:
        await mark_offline(session_id)


async def _send(ws: WebSocket, msg) -> None:
    """Send a Pydantic model as JSON over the WebSocket."""
    await ws.send_json(msg.model_dump(mode="json"))
