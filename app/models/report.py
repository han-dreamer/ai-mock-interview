"""Data models for the evaluation report."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SkillScore(BaseModel):
    skill_name: str
    score: int = Field(..., ge=1, le=10)
    evidence: str = Field(..., description="来自面试记录的具体评分依据，必须使用简体中文")


class ReportHighlight(BaseModel):
    point: str = Field(..., description="优势点或改进点标题，必须使用简体中文")
    evidence: str = Field(..., description="来自面试记录的支撑证据，必须使用简体中文")


class InterviewReport(BaseModel):
    """Final structured evaluation report used for single-round evaluation."""

    skill_scores: list[SkillScore] = Field(..., description="Per-skill dimension scores")
    overall_score: float = Field(..., ge=1.0, le=10.0, description="Weighted average score")
    grade: Literal["A", "B", "C", "D"] = Field(..., description="Overall grade")
    strengths: list[ReportHighlight] = Field(
        ..., min_length=1, max_length=5, description="Top strengths, must be Chinese"
    )
    improvements: list[ReportHighlight] = Field(
        ..., min_length=1, max_length=5, description="Top areas for improvement, must be Chinese"
    )
    overall_assessment: str = Field(
        ..., description="2-3 句候选人整体表现总结，必须使用简体中文"
    )


class RoundScore(BaseModel):
    """Score summary for a single interview round."""

    round_name: str = Field(..., description="轮次名称，例如“一面（技术深度）”")
    score: float = Field(..., ge=1.0, le=10.0, description="Round score")
    grade: Literal["A", "B", "C", "D"] = Field(..., description="Round grade")
    summary: str = Field(..., description="1-2 句轮次表现总结，必须使用简体中文")


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
        ..., min_length=1, max_length=5, description="Top strengths, must be Chinese"
    )
    improvements: list[ReportHighlight] = Field(
        ..., min_length=1, max_length=5, description="Top areas for improvement, must be Chinese"
    )
    overall_assessment: str = Field(
        ..., description="3-5 句综合评价，覆盖两轮面试表现，必须使用简体中文"
    )
    hiring_recommendation: str = Field(
        ..., description="简短录用建议：强烈推荐/推荐/待定/不推荐，并附一行中文理由"
    )


class Round1Summary(BaseModel):
    """Intermediate summary after Round 1 of a professional interview."""

    round1_score: float = Field(..., ge=1.0, le=10.0, description="Round 1 overall score")
    round1_grade: Literal["A", "B", "C", "D"] = Field(..., description="Round 1 grade")
    technical_depth_assessment: str = Field(
        ..., description="2-3 句技术深度评价，必须使用简体中文"
    )
    project_understanding: str = Field(
        ..., description="1-2 句项目理解评价，必须使用简体中文"
    )
    strengths_observed: list[str] = Field(
        ..., min_length=1, max_length=3, description="一面体现出的主要优势，必须使用简体中文"
    )
    areas_to_probe: list[str] = Field(
        ..., min_length=1, max_length=3, description="二面需要继续追问的薄弱点，必须使用简体中文"
    )
    proceed_to_round2: bool = Field(
        default=True, description="Whether the candidate should proceed to Round 2"
    )
    round1_feedback: str = Field(..., description="展示给候选人的一面反馈，必须使用简体中文")


class MissedKnowledge(BaseModel):
    """A knowledge point the candidate missed or answered incorrectly."""

    question: str = Field(..., description="面试题正文，必须使用简体中文")
    score: int = Field(..., ge=1, le=10, description="Score for this question")
    missed_points: list[str] = Field(..., description="候选人遗漏的关键点，必须使用简体中文")
    reference_answer: str = Field(..., description="覆盖关键点的参考答案，必须使用简体中文")


class PracticeReport(BaseModel):
    """Evaluation report for practice mode with reference answers for learning."""

    overall_score: float = Field(..., ge=1.0, le=10.0, description="Weighted average score")
    grade: Literal["A", "B", "C", "D"] = Field(..., description="Overall grade")
    total_questions: int = Field(..., description="Total questions answered")
    skill_scores: list[SkillScore] = Field(..., description="Per-skill dimension scores")
    missed_knowledge: list[MissedKnowledge] = Field(
        ..., description="Questions with missed points and reference answers"
    )
    study_suggestions: list[str] = Field(
        ..., min_length=1, max_length=5, description="基于薄弱点的学习建议，必须使用简体中文"
    )
    overall_assessment: str = Field(
        ..., description="2-3 句鼓励性总结和后续行动建议，必须使用简体中文"
    )
