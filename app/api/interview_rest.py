"""REST endpoints for interview operations."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from app.cache.rate_limiter import check_rate_limit
from app.cache.session_cache import save_session_meta
from app.config import settings
from app.models.interview import ChatMessage
from app.models.user import UserInDB
from app.security import get_current_user
from app.services.session_manager import get_session_manager
from app.utils.file_parser import parse_resume_with_metadata

router = APIRouter()


class StartInterviewRequest(BaseModel):
    jd_text: str = Field(..., min_length=10, description="Job description text")
    max_follow_ups: int = Field(default=2, ge=0, le=5)
    mode: str = Field(default="practice", description="Interview mode: practice or professional")


class StartInterviewResponse(BaseModel):
    session_id: str
    message: str
    websocket_url: str
    mode: str


class SubmitAnswerRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Candidate answer text")


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    return value


def _last_interviewer_message(state: dict) -> str | None:
    for msg in reversed(state.get("conversation_history", []) or []):
        if isinstance(msg, ChatMessage) and msg.role == "interviewer":
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "interviewer":
            return msg.get("content", "")
    return None


def _question_meta(state: dict) -> tuple[int, int, list[str], str]:
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


def _current_turn_payload(state: dict) -> dict | None:
    message = _last_interviewer_message(state)
    if not message:
        return None
    idx, total, tags, difficulty = _question_meta(state)
    follow_up_count = int(state.get("follow_up_count", 0) or 0)
    payload = {
        "question_index": idx + 1,
        "content": message,
        "kind": "follow_up" if follow_up_count > 0 else "question",
    }
    if follow_up_count > 0:
        payload["follow_up_number"] = follow_up_count
    else:
        payload.update(
            {
                "total_questions": total,
                "skill_tags": tags,
                "difficulty": difficulty,
            }
        )
    return payload


def _rate_limit_identity(
    request: Request,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
) -> str:
    host = request.client.host if request.client else "unknown"
    if session_id:
        return f"session:{session_id}"
    if user_id:
        return f"user:{user_id}"
    return f"ip:{host}"


def _assert_session_owner(session, current_user: UserInDB) -> None:
    if session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Session access denied")


async def _enforce_rate_limit(
    name: str,
    identity: str,
    limit: int,
    window_seconds: int,
) -> None:
    result = await check_rate_limit(name, identity, limit, window_seconds)
    if result.allowed:
        return
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=f"Too many requests. Please retry after {result.retry_after_seconds} seconds.",
        headers={"Retry-After": str(result.retry_after_seconds)},
    )


def _report_from_state_or_manager(session_id: str, state: dict) -> Any:
    mgr = get_session_manager()
    return (
        state.get("practice_report")
        or state.get("professional_report")
        or state.get("final_report")
        or mgr.get_report_for_session(session_id)
    )


def _interview_payload(session_id: str, state: dict) -> dict:
    mgr = get_session_manager()
    session = mgr.get_session(session_id)
    report = _report_from_state_or_manager(session_id, state)
    return {
        "session": _dump(session),
        "state": {
            "interview_complete": bool(state.get("interview_complete", False)),
            "current_round": state.get("current_round", 1),
            "question_count": len(state.get("question_plan", []) or []),
            "assessment_count": len(state.get("assessments", []) or []),
        },
        "next": _current_turn_payload(state),
        "report": _dump(report) if report else None,
    }


async def _store_and_parse_resume(session_id: str, upload: UploadFile):
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Resume file name is required")
    safe_name = Path(upload.filename).name
    upload_dir = settings.upload_root / "resumes" / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / safe_name
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="Resume file is empty")
    file_path.write_bytes(content)
    return await parse_resume_with_metadata(file_path)


@router.post("/start", response_model=StartInterviewResponse)
async def start_interview(
    req: StartInterviewRequest,
    request: Request,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Create a new interview session. Returns a session_id for WebSocket connection."""
    await _enforce_rate_limit(
        "interview:start",
        _rate_limit_identity(request, user_id=current_user.id),
        settings.rate_limit_start_per_minute,
        60,
    )
    mgr = get_session_manager()
    session_id = str(uuid.uuid4())
    mode = req.mode if req.mode in ("practice", "professional") else "practice"
    session = mgr.create_session(
        session_id=session_id,
        jd_text=req.jd_text,
        max_follow_ups=req.max_follow_ups,
        mode=mode,
        user_id=current_user.id,
    )
    await mgr.persist_session(session_id)
    await save_session_meta(session, mode)
    return StartInterviewResponse(
        session_id=session_id,
        message="Session created. Connect via WebSocket to begin the interview.",
        websocket_url=f"/api/ws/interview/{session_id}",
        mode=mode,
    )


@router.post("/start-with-resume")
async def start_interview_with_resume(
    request: Request,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    jd_text: str = Form(..., min_length=10),
    resume_file: UploadFile = File(...),
    max_follow_ups: int = Form(default=2, ge=0, le=5),
    mode: str = Form(default="professional"),
):
    """Create a session and attach a parsed resume in one multipart request."""
    identity = _rate_limit_identity(request, user_id=current_user.id)
    await _enforce_rate_limit(
        "interview:start",
        identity,
        settings.rate_limit_start_per_minute,
        60,
    )
    await _enforce_rate_limit(
        "interview:resume",
        identity,
        settings.rate_limit_resume_per_hour,
        3600,
    )
    mgr = get_session_manager()
    session_id = str(uuid.uuid4())
    normalized_mode = mode if mode in ("practice", "professional") else "professional"
    parse_result = await _store_and_parse_resume(session_id, resume_file)
    session = mgr.create_session(
        session_id=session_id,
        jd_text=jd_text,
        max_follow_ups=max_follow_ups,
        mode=normalized_mode,
        resume_text=parse_result.normalized_text,
        resume_parse_result=parse_result,
        user_id=current_user.id,
    )
    await mgr.persist_session(session_id)
    await save_session_meta(session, normalized_mode)
    return {
        "session_id": session_id,
        "message": "Session created with parsed resume.",
        "websocket_url": f"/api/ws/interview/{session_id}",
        "mode": normalized_mode,
        "resume": _dump(parse_result),
    }


@router.get("/sessions")
async def list_sessions(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    limit: int = 50,
    offset: int = 0,
):
    """List interview sessions owned by the current user."""
    normalized_limit = max(1, min(limit, 100))
    normalized_offset = max(0, offset)
    mgr = get_session_manager()
    sessions = await mgr.list_user_sessions(
        current_user.id,
        limit=normalized_limit,
        offset=normalized_offset,
    )
    return {
        "items": sessions,
        "limit": normalized_limit,
        "offset": normalized_offset,
    }


@router.get("/session/{session_id}")
async def get_session(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Get the current state of an interview session."""
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    return session


@router.post("/session/{session_id}/resume")
async def upload_resume(
    session_id: str,
    request: Request,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
    resume_file: UploadFile = File(...),
):
    """Upload and parse a resume for an existing session before interview start."""
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    await _enforce_rate_limit(
        "interview:resume",
        _rate_limit_identity(request, user_id=session.user_id, session_id=session_id),
        settings.rate_limit_resume_per_hour,
        3600,
    )
    try:
        parse_result = await _store_and_parse_resume(session_id, resume_file)
        mgr.set_resume(
            session_id=session_id,
            resume_text=parse_result.normalized_text,
            resume_parse_result=parse_result,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return {
        "session_id": session_id,
        "resume": _dump(parse_result),
        "message": "Resume parsed and attached to session.",
    }


@router.post("/session/{session_id}/start")
async def start_session_graph(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Start the LangGraph interview and return the first interviewer turn."""
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    try:
        state = await mgr.start_interview_graph(session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start interview: {exc}") from exc
    return _interview_payload(session_id, state)


@router.post("/session/{session_id}/answer")
async def submit_session_answer(
    session_id: str,
    req: SubmitAnswerRequest,
    request: Request,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Submit one answer through REST and return the next turn or final report."""
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    await _enforce_rate_limit(
        "interview:answer",
        _rate_limit_identity(request, user_id=session.user_id, session_id=session_id),
        settings.rate_limit_answer_per_minute,
        60,
    )
    answer = req.content.strip()
    if not answer:
        raise HTTPException(status_code=400, detail="Answer cannot be empty")
    try:
        state = await mgr.submit_answer(session_id, answer)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process answer: {exc}") from exc
    return _interview_payload(session_id, state)


@router.post("/session/{session_id}/stop")
async def stop_session(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Stop a running interview and generate an early report when possible."""
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    try:
        state = await mgr.stop_interview(session_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to stop interview: {exc}") from exc
    return _interview_payload(session_id, state)


@router.get("/session/{session_id}/state")
async def get_session_state(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Return lightweight session metadata plus the latest LangGraph state snapshot."""
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    state = mgr.get_last_state(session_id)
    return {
        "session": _dump(session),
        "graph_started": mgr.has_graph_started(session_id),
        "state": _dump(state),
        "next": _current_turn_payload(state),
        "report": _dump(_report_from_state_or_manager(session_id, state)),
    }


@router.get("/report/{session_id}")
async def get_report(
    session_id: str,
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    """Get the final interview report for a completed session."""
    mgr = get_session_manager()
    session = await mgr.ensure_session_loaded(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, current_user)
    if session.status != "completed":
        raise HTTPException(status_code=400, detail="Interview not yet completed")
    report = mgr.get_report_for_session(session_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report
