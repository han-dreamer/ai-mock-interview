"""Data models for JD analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SkillItem(BaseModel):
    name: str = Field(..., description="Skill name, e.g. 'Python', 'System Design'")
    category: Literal["language", "framework", "system_design", "soft_skill", "domain"] = Field(
        ..., description="Skill classification"
    )
    weight: float = Field(..., ge=0.0, le=1.0, description="Importance weight")
    is_required: bool = Field(..., description="True if explicitly required in JD")


class SkillMatrix(BaseModel):
    position_title: str = Field(..., description="Job title extracted from JD")
    experience_level: Literal["intern", "junior", "mid", "senior"] = Field(
        ..., description="Expected experience level"
    )
    skills: list[SkillItem] = Field(..., min_length=1, description="Extracted skill list")

    @property
    def required_skills(self) -> list[SkillItem]:
        return [s for s in self.skills if s.is_required]

    @property
    def skill_names(self) -> list[str]:
        return [s.name for s in self.skills]
