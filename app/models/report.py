"""Data models for the evaluation report."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SkillScore(BaseModel):
    skill_name: str
    score: int = Field(..., ge=1, le=10)
    evidence: str = Field(..., description="Specific evidence from the transcript")


class ReportHighlight(BaseModel):
    point: str = Field(..., description="Strength or area for improvement")
    evidence: str = Field(..., description="Supporting evidence from transcript")


class InterviewReport(BaseModel):
    """Final structured evaluation report (used for single-round evaluation)."""

    skill_scores: list[SkillScore] = Field(..., description="Per-skill dimension scores")
    overall_score: float = Field(..., ge=1.0, le=10.0, description="Weighted average score")
    grade: Literal["A", "B", "C", "D"] = Field(..., description="Overall grade")
    strengths: list[ReportHighlight] = Field(
        ..., min_length=1, max_length=5, description="Top strengths"
    )
    improvements: list[ReportHighlight] = Field(
        ..., min_length=1, max_length=5, description="Top areas for improvement"
    )
    overall_assessment: str = Field(
        ..., description="2-3 sentence summary of candidate performance"
    )


class RoundScore(BaseModel):
    """Score summary for a single interview round."""

    round_name: str = Field(..., description="Round name, e.g. '一面（技术深度）'")
    score: float = Field(..., ge=1.0, le=10.0, description="Round score")
    grade: Literal["A", "B", "C", "D"] = Field(..., description="Round grade")
    summary: str = Field(..., description="1-2 sentence round summary in Chinese")


class ProfessionalReport(BaseModel):
    """Comprehensive evaluation report for dual-round professional interview."""

    round_scores: list[RoundScore] = Field(
        ..., min_length=1, max_length=2, description="Per-round score summaries"
    )
    skill_scores: list[SkillScore] = Field(
        ..., description="Consolidated per-skill scores across both rounds"
    )
    overall_score: float = Field(
        ..., ge=1.0, le=10.0, description="Weighted average across both rounds"
    )
    grade: Literal["A", "B", "C", "D"] = Field(..., description="Overall grade")
    technical_depth_score: float = Field(
        ..., ge=1.0, le=10.0, description="Round 1 technical depth score"
    )
    technical_breadth_score: float = Field(
        ..., ge=1.0, le=10.0, description="Round 2 technical breadth score"
    )
    strengths: list[ReportHighlight] = Field(
        ..., min_length=1, max_length=5, description="Top strengths across both rounds"
    )
    improvements: list[ReportHighlight] = Field(
        ..., min_length=1, max_length=5, description="Top areas for improvement"
    )
    overall_assessment: str = Field(
        ..., description="3-5 sentence comprehensive assessment in Chinese, covering both rounds"
    )
    hiring_recommendation: str = Field(
        ..., description="Brief hiring recommendation: 强烈推荐/推荐/待定/不推荐, with one-line reason"
    )


# ── Round 1 Summary (intermediate feedback in professional mode) ──


class Round1Summary(BaseModel):
    """Intermediate summary after Round 1 of a professional interview."""

    round1_score: float = Field(..., ge=1.0, le=10.0, description="Round 1 overall score")
    round1_grade: Literal["A", "B", "C", "D"] = Field(..., description="Round 1 grade")
    technical_depth_assessment: str = Field(
        ..., description="2-3 sentence assessment of the candidate's technical depth"
    )
    project_understanding: str = Field(
        ..., description="1-2 sentence assessment of how well they know their own projects"
    )
    strengths_observed: list[str] = Field(
        ..., min_length=1, max_length=3, description="Key strengths shown in Round 1"
    )
    areas_to_probe: list[str] = Field(
        ..., min_length=1, max_length=3,
        description="Weak areas or topics to probe further in Round 2"
    )
    proceed_to_round2: bool = Field(
        default=True, description="Whether the candidate should proceed to Round 2"
    )
    round1_feedback: str = Field(
        ..., description="Brief feedback message to show the candidate between rounds (in Chinese)"
    )


# ── Practice Mode Report ──────────────────────────────────────────


class MissedKnowledge(BaseModel):
    """A knowledge point the candidate missed or answered incorrectly."""

    question: str = Field(..., description="The interview question text")
    score: int = Field(..., ge=1, le=10, description="Score for this question")
    missed_points: list[str] = Field(..., description="Key points the candidate missed")
    reference_answer: str = Field(
        ..., description="Concise reference answer covering all key points, in Chinese"
    )


class PracticeReport(BaseModel):
    """Evaluation report for practice mode — includes reference answers for learning."""

    overall_score: float = Field(..., ge=1.0, le=10.0, description="Weighted average score")
    grade: Literal["A", "B", "C", "D"] = Field(..., description="Overall grade")
    total_questions: int = Field(..., description="Total questions answered")
    skill_scores: list[SkillScore] = Field(..., description="Per-skill dimension scores")
    missed_knowledge: list[MissedKnowledge] = Field(
        ..., description="Questions with missed points and reference answers"
    )
    study_suggestions: list[str] = Field(
        ..., min_length=1, max_length=5,
        description="Prioritized study suggestions based on weak areas"
    )
    overall_assessment: str = Field(
        ..., description="2-3 sentence encouraging summary with clear next steps"
    )
