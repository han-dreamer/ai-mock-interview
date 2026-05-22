"""Data models for resume parsing and structured representation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


LINK_TYPES = {
    "github",
    "gitee",
    "gitlab",
    "blog",
    "portfolio",
    "linkedin",
    "email",
    "phone",
    "other",
}


def _empty_if_none(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _list_if_none(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _str_list(value: Any) -> list[str]:
    items = _list_if_none(value)
    return [str(item).strip() for item in items if item is not None and str(item).strip()]


class ResumeLink(BaseModel):
    """A link or contact item extracted from the resume text."""

    url: str = Field(..., description="The extracted URL, email, or phone number")
    type: Literal[
        "github",
        "gitee",
        "gitlab",
        "blog",
        "portfolio",
        "linkedin",
        "email",
        "phone",
        "other",
    ] = Field(..., description="Link category")
    source_text: str | None = Field(
        default=None,
        description="Optional nearby source text or matched snippet",
    )

    @field_validator("url", mode="before")
    @classmethod
    def _coerce_url(cls, value: Any) -> str:
        return _empty_if_none(value)

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, value: Any) -> str:
        if value is None:
            return "other"
        value = str(value).strip().lower()
        return value if value in LINK_TYPES else "other"


class ResumeParseMetadata(BaseModel):
    """Metadata produced by the resume parsing pipeline."""

    file_name: str = Field(default="", description="Original file name")
    file_type: str = Field(default="", description="Lowercase file suffix")
    parser: str = Field(default="", description="Parser backend used")
    page_count: int | None = Field(default=None, description="PDF page count if available")
    raw_char_count: int = Field(default=0, description="Raw extracted text length")
    normalized_char_count: int = Field(default=0, description="Normalized text length")


class ResumeParseResult(BaseModel):
    """Output of the lightweight resume parsing pipeline."""

    raw_text: str = Field(default="", description="Text directly extracted from the file")
    normalized_text: str = Field(
        default="",
        description="Cleaned text passed to the LLM resume analyst",
    )
    links: list[ResumeLink] = Field(
        default_factory=list,
        description="Links and contact items extracted before LLM analysis",
    )
    metadata: ResumeParseMetadata = Field(default_factory=ResumeParseMetadata)
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal parsing issues or uncertainty signals",
    )


class ProjectInfo(BaseModel):
    """A project extracted from the candidate's resume."""

    name: str = Field(..., description="Project name")
    tech_stack: list[str] = Field(..., description="Technologies used in this project")
    description: str = Field(..., description="Brief description of what the project does")
    key_contributions: list[str] = Field(
        ..., description="Candidate's specific contributions and responsibilities"
    )
    potential_deep_dive_points: list[str] = Field(
        ..., description="Technical points worth probing in an interview"
    )

    @field_validator("name", "description", mode="before")
    @classmethod
    def _coerce_text_fields(cls, value: Any) -> str:
        return _empty_if_none(value)

    @field_validator(
        "tech_stack",
        "key_contributions",
        "potential_deep_dive_points",
        mode="before",
    )
    @classmethod
    def _coerce_string_lists(cls, value: Any) -> list[str]:
        return _str_list(value)


class ResumeProfile(BaseModel):
    """Structured representation of a candidate's resume."""

    name: str = Field(default="", description="Candidate name")
    education: str = Field(default="", description="Education background (school, major, degree)")
    skills: list[str] = Field(default_factory=list, description="Technical skills listed")
    projects: list[ProjectInfo] = Field(
        default_factory=list, description="Projects from the resume"
    )
    experience: list[str] = Field(
        default_factory=list, description="Work or internship experience summaries"
    )
    highlights: list[str] = Field(
        default_factory=list,
        description="Resume highlights or notable points worth discussing",
    )
    links: list[ResumeLink] = Field(
        default_factory=list,
        description="Links and contact items extracted from the resume",
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Unclear or risky points that interviewers should verify",
    )
    parse_warnings: list[str] = Field(
        default_factory=list,
        description="Warnings from the file parsing stage",
    )
    summary: str = Field(
        default="",
        description="One-paragraph overall impression of the candidate's background",
    )

    @field_validator("name", "education", "summary", mode="before")
    @classmethod
    def _coerce_text_fields(cls, value: Any) -> str:
        return _empty_if_none(value)

    @field_validator(
        "skills",
        "experience",
        "highlights",
        "concerns",
        "parse_warnings",
        mode="before",
    )
    @classmethod
    def _coerce_string_lists(cls, value: Any) -> list[str]:
        return _str_list(value)

    @field_validator("projects", "links", mode="before")
    @classmethod
    def _coerce_object_lists(cls, value: Any) -> list[Any]:
        return _list_if_none(value)


class ResumeJDMatch(BaseModel):
    """Lightweight match between a resume profile and a JD skill matrix."""

    matched_skills: list[str] = Field(
        default_factory=list,
        description="JD skills that appear in the resume profile",
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description="Important JD skills not clearly found in the resume",
    )
    relevant_projects: list[str] = Field(
        default_factory=list,
        description="Resume projects most relevant to the JD",
    )
    interview_focus: list[str] = Field(
        default_factory=list,
        description="Suggested focus areas for resume-aware interview questions",
    )
    project_skill_map: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Mapping from project name to matched JD skills",
    )
