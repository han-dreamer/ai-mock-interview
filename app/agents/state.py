"""LangGraph shared state definition for the interview workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from app.models.interview import AnswerAssessment, ChatMessage
from app.models.jd import SkillMatrix
from app.models.question import QuestionItem
from app.models.report import InterviewReport, PracticeReport, ProfessionalReport
from app.models.resume import ResumeJDMatch, ResumeParseResult, ResumeProfile


def _append_messages(
    existing: list[ChatMessage], new: list[ChatMessage]
) -> list[ChatMessage]:
    """Reducer that appends new messages to the conversation history."""
    return (existing or []) + (new or [])


class InterviewState(TypedDict, total=False):
    """Shared state flowing through the LangGraph interview graph.

    Each agent reads the fields it needs and writes the fields it owns.
    Using total=False so we can build the state incrementally.
    """

    # --- Mode ---
    interview_mode: Literal["practice", "professional"]

    # --- Input ---
    user_id: str
    jd_text: str
    resume_text: str
    resume_parse_result: ResumeParseResult

    # --- Long-term memory ---
    memory_context: str
    retrieved_memories: dict

    # --- Resume Analyst output (professional mode) ---
    resume_profile: ResumeProfile
    resume_jd_match: ResumeJDMatch

    # --- JD Analyst output ---
    skill_matrix: SkillMatrix

    # --- Question Planner output ---
    question_plan: list[QuestionItem]

    # --- Interview loop state ---
    current_question_index: int
    follow_up_count: int
    max_follow_ups: int
    conversation_history: Annotated[list[ChatMessage], _append_messages]
    current_candidate_answer: str

    # --- Assessment state ---
    assessments: Annotated[list[AnswerAssessment], operator.add]

    # --- Two-round state (professional mode) ---
    current_round: int
    round1_question_plan: list[QuestionItem]
    round1_assessments: list[AnswerAssessment]
    round1_summary_text: str
    round2_question_plan: list[QuestionItem]

    # --- Evaluator output (professional mode) ---
    final_report: InterviewReport
    professional_report: ProfessionalReport

    # --- Evaluator output (practice mode) ---
    practice_report: PracticeReport

    # --- Control flow ---
    interview_complete: bool
