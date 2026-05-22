"""Data models for interview questions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QuestionItem(BaseModel):
    id: int = Field(..., description="Question sequence number (1-based)")
    content: str = Field(..., description="The interview question text")
    skill_tags: list[str] = Field(..., description="Skills this question tests")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Difficulty level")
    reference_points: list[str] = Field(
        ..., min_length=1, description="Key points a good answer should cover"
    )
    follow_up_directions: list[str] = Field(
        default_factory=list, description="Possible follow-up question directions"
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="Reference question IDs adapted from the RAG question bank",
    )
    source_categories: list[str] = Field(
        default_factory=list,
        description="Question bank categories for the adapted references",
    )


class QuestionPlan(BaseModel):
    """The full interview question plan produced by the Question Planner agent."""

    questions: list[QuestionItem] = Field(..., min_length=1, max_length=10)
    total_estimated_minutes: int = Field(
        default=20, description="Estimated total interview duration"
    )
