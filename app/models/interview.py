"""Data models for interview session state and assessment."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnswerAssessment(BaseModel):
    """Assessment of a single candidate answer (internal, not shown to candidate)."""

    question_id: int
    score: int = Field(..., ge=1, le=10, description="Score on 1-10 scale")
    covered_points: list[str] = Field(default_factory=list, description="Reference points covered")
    missed_points: list[str] = Field(default_factory=list, description="Reference points missed")
    should_follow_up: bool = Field(default=False)
    follow_up_reason: str | None = Field(default=None)


class ChatMessage(BaseModel):
    role: Literal["interviewer", "candidate", "system"]
    content: str


class InterviewSession(BaseModel):
    """Metadata for a running interview session."""

    session_id: str
    user_id: str = "local-user"
    jd_text: str
    status: Literal[
        "pending",
        "analyzing",
        "interviewing",
        "evaluating",
        "completed",
        "failed",
    ] = "pending"
    current_question_index: int = 0
    follow_up_count: int = 0
    max_follow_ups: int = 2
    conversation_history: list[ChatMessage] = Field(default_factory=list)
    assessments: list[AnswerAssessment] = Field(default_factory=list)
