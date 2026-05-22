"""WebSocket message protocol — typed models for client/server communication."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Client → Server messages ────────────────────────────────────────

class ClientMessage(BaseModel):
    """Envelope for all client-to-server WebSocket messages."""

    type: Literal["start", "start_interview", "answer", "stop", "end_interview", "ping"]
    content: str = ""


# ── Server → Client messages ────────────────────────────────────────

class ServerStatus(BaseModel):
    type: Literal["status"] = "status"
    stage: str
    message: str


class ServerQuestion(BaseModel):
    type: Literal["question"] = "question"
    question_index: int
    total_questions: int
    content: str
    skill_tags: list[str] = Field(default_factory=list)
    difficulty: str = ""


class ServerFollowUp(BaseModel):
    type: Literal["follow_up"] = "follow_up"
    question_index: int
    follow_up_number: int
    content: str


class ServerQuestionStream(BaseModel):
    """Incremental streaming chunk for a question/follow-up."""

    type: Literal["stream_chunk"] = "stream_chunk"
    chunk: str
    done: bool = False


class ServerReport(BaseModel):
    type: Literal["report"] = "report"
    data: dict[str, Any]


class ServerError(BaseModel):
    type: Literal["error"] = "error"
    message: str


class ServerInterviewEnd(BaseModel):
    type: Literal["interview_end"] = "interview_end"
    message: str = "The interview has concluded. Your report is being generated."
