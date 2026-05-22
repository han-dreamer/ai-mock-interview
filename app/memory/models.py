"""Structured long-term memory models.

The implementation keeps storage compact: one generic MemoryItem table for
facts/events/summaries, plus one SkillMemory table for the user skill model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryType(StrEnum):
    PROFILE = "profile"
    RESUME_PROJECT = "resume_project"
    INTERVIEW_EPISODE = "interview_episode"
    SESSION_REFLECTION = "session_reflection"
    PREFERENCE = "preference"
    STRATEGY = "strategy"


class MasteryLevel(StrEnum):
    STRONG = "strong"
    NORMAL = "normal"
    WEAK = "weak"
    UNKNOWN = "unknown"


class MemoryItem(BaseModel):
    id: str
    user_id: str = "local-user"
    memory_type: MemoryType | str
    content: str
    structured: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    source: str = "system"
    source_id: str | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SkillMemory(BaseModel):
    id: str
    user_id: str = "local-user"
    skill_name: str
    category: str = "unknown"
    attempts: int = 0
    avg_score: float = 0.0
    recent_score: float = 0.0
    mastery_level: MasteryLevel | str = MasteryLevel.UNKNOWN
    strengths: list[str] = Field(default_factory=list)
    weak_points: list[str] = Field(default_factory=list)
    evidence_memory_ids: list[str] = Field(default_factory=list)
    next_practice_priority: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MemoryContext(BaseModel):
    user_id: str = "local-user"
    profile_items: list[MemoryItem] = Field(default_factory=list)
    resume_items: list[MemoryItem] = Field(default_factory=list)
    recent_reflections: list[MemoryItem] = Field(default_factory=list)
    weak_skills: list[SkillMemory] = Field(default_factory=list)
    relevant_episodes: list[MemoryItem] = Field(default_factory=list)
    semantic_memories: list[MemoryItem] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(
            [
                self.profile_items,
                self.resume_items,
                self.recent_reflections,
                self.weak_skills,
                self.relevant_episodes,
                self.semantic_memories,
            ]
        )
