"""Persistence helpers for interview session metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.models.interview import AnswerAssessment, ChatMessage, InterviewSession
from app.models.resume import ResumeParseResult
from app.services.database import get_pool, session_store_enabled

logger = logging.getLogger(__name__)


@dataclass
class PersistedSession:
    session: InterviewSession
    mode: str
    resume_text: str = ""
    resume_parse_result: ResumeParseResult | None = None
    graph_started: bool = False
    last_state: dict[str, Any] | None = None
    persisted_assessment_count: int = 0
    final_memory_saved: bool = False
    error_message: str | None = None


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    return value


def _session_from_row(row: dict[str, Any]) -> PersistedSession:
    conversation = [
        msg if isinstance(msg, ChatMessage) else ChatMessage.model_validate(msg)
        for msg in (row.get("conversation_history") or [])
    ]
    assessments = [
        item if isinstance(item, AnswerAssessment) else AnswerAssessment.model_validate(item)
        for item in (row.get("assessments") or [])
    ]
    session = InterviewSession(
        session_id=row["session_id"],
        user_id=row["user_id"],
        jd_text=row["jd_text"],
        status=row["status"],
        current_question_index=row.get("current_question_index") or 0,
        follow_up_count=row.get("follow_up_count") or 0,
        max_follow_ups=row.get("max_follow_ups") or 2,
        conversation_history=conversation,
        assessments=assessments,
    )
    raw_resume = row.get("resume_parse_result")
    resume_parse_result = (
        ResumeParseResult.model_validate(raw_resume)
        if raw_resume
        else None
    )
    return PersistedSession(
        session=session,
        mode=row["mode"],
        resume_text=row.get("resume_text") or "",
        resume_parse_result=resume_parse_result,
        graph_started=bool(row.get("graph_started")),
        last_state=row.get("last_state") or {},
        persisted_assessment_count=row.get("persisted_assessment_count") or 0,
        final_memory_saved=bool(row.get("final_memory_saved")),
        error_message=row.get("error_message"),
    )


async def save_session_record(
    *,
    session: InterviewSession,
    mode: str,
    resume_text: str = "",
    resume_parse_result: ResumeParseResult | None = None,
    graph_started: bool = False,
    last_state: dict[str, Any] | None = None,
    persisted_assessment_count: int = 0,
    final_memory_saved: bool = False,
    error_message: str | None = None,
) -> None:
    """Upsert one interview session metadata row."""
    if not session_store_enabled():
        return
    pool = get_pool()
    if pool is None:
        logger.debug("Postgres session store enabled but pool is not initialized")
        return

    from psycopg.types.json import Jsonb

    completed_expr = "NOW()" if session.status == "completed" else "NULL"
    async with pool.connection() as conn:
        await conn.execute(
            f"""
            INSERT INTO interview_sessions (
                session_id,
                user_id,
                mode,
                jd_text,
                max_follow_ups,
                status,
                current_question_index,
                follow_up_count,
                graph_started,
                resume_text,
                resume_parse_result,
                conversation_history,
                assessments,
                last_state,
                persisted_assessment_count,
                final_memory_saved,
                error_message,
                completed_at
            )
            VALUES (
                %(session_id)s,
                %(user_id)s,
                %(mode)s,
                %(jd_text)s,
                %(max_follow_ups)s,
                %(status)s,
                %(current_question_index)s,
                %(follow_up_count)s,
                %(graph_started)s,
                %(resume_text)s,
                %(resume_parse_result)s,
                %(conversation_history)s,
                %(assessments)s,
                %(last_state)s,
                %(persisted_assessment_count)s,
                %(final_memory_saved)s,
                %(error_message)s,
                {completed_expr}
            )
            ON CONFLICT (session_id) DO UPDATE SET
                user_id = EXCLUDED.user_id,
                mode = EXCLUDED.mode,
                jd_text = EXCLUDED.jd_text,
                max_follow_ups = EXCLUDED.max_follow_ups,
                status = EXCLUDED.status,
                current_question_index = EXCLUDED.current_question_index,
                follow_up_count = EXCLUDED.follow_up_count,
                graph_started = EXCLUDED.graph_started,
                resume_text = EXCLUDED.resume_text,
                resume_parse_result = EXCLUDED.resume_parse_result,
                conversation_history = EXCLUDED.conversation_history,
                assessments = EXCLUDED.assessments,
                last_state = EXCLUDED.last_state,
                persisted_assessment_count = EXCLUDED.persisted_assessment_count,
                final_memory_saved = EXCLUDED.final_memory_saved,
                error_message = EXCLUDED.error_message,
                updated_at = NOW(),
                completed_at = CASE
                    WHEN EXCLUDED.status = 'completed'
                    THEN COALESCE(interview_sessions.completed_at, NOW())
                    ELSE interview_sessions.completed_at
                END
            """,
            {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "mode": mode,
                "jd_text": session.jd_text,
                "max_follow_ups": session.max_follow_ups,
                "status": session.status,
                "current_question_index": session.current_question_index,
                "follow_up_count": session.follow_up_count,
                "graph_started": graph_started,
                "resume_text": resume_text,
                "resume_parse_result": (
                    Jsonb(_dump(resume_parse_result)) if resume_parse_result else None
                ),
                "conversation_history": Jsonb(_dump(session.conversation_history)),
                "assessments": Jsonb(_dump(session.assessments)),
                "last_state": Jsonb(_dump(last_state or {})),
                "persisted_assessment_count": persisted_assessment_count,
                "final_memory_saved": final_memory_saved,
                "error_message": error_message,
            },
        )


async def load_session_record(session_id: str) -> PersistedSession | None:
    if not session_store_enabled():
        return None
    pool = get_pool()
    if pool is None:
        logger.debug("Postgres session store enabled but pool is not initialized")
        return None

    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM interview_sessions WHERE session_id = %s",
            (session_id,),
        )
        row = await cursor.fetchone()
    if not row:
        return None
    return _session_from_row(row)
