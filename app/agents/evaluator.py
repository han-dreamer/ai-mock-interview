"""Evaluator Agents — produce structured reports for both professional and practice modes.

Also includes the round1_summary node for intermediate feedback in dual-round mode.
"""

from __future__ import annotations

import logging

from app.agents.state import InterviewState
from app.llm.client import get_llm_client
from app.llm.prompts import (
    EVALUATOR_SYSTEM,
    PRACTICE_EVALUATOR_SYSTEM,
    PROFESSIONAL_EVALUATOR_SYSTEM,
    ROUND1_SUMMARY_SYSTEM,
)
from app.models.interview import ChatMessage
from app.models.report import InterviewReport, PracticeReport, ProfessionalReport, Round1Summary

logger = logging.getLogger(__name__)


# ── Shared formatting helpers ────────────────────────────────────────


def _format_transcript(state: InterviewState) -> str:
    lines = []
    for msg in state.get("conversation_history", []):
        if isinstance(msg, ChatMessage):
            role, content = msg.role, msg.content
        elif isinstance(msg, dict):
            role, content = msg.get("role", "?"), msg.get("content", "")
        else:
            continue
        if role == "interviewer":
            lines.append(f"[Interviewer]: {content}")
        elif role == "candidate":
            lines.append(f"[Candidate]: {content}")
    return "\n\n".join(lines)


def _format_assessments(state: InterviewState) -> str:
    lines = []
    for a in state.get("assessments", []):
        lines.append(
            f"Q{a.question_id}: score={a.score}/10, "
            f"covered={a.covered_points}, missed={a.missed_points}"
        )
    return "\n".join(lines) if lines else "No per-question assessments available."


def _format_skill_matrix(state: InterviewState) -> str:
    sm = state.get("skill_matrix")
    if not sm:
        return "Skill matrix not available."
    lines = [f"Position: {sm.position_title} ({sm.experience_level})"]
    for s in sm.skills:
        lines.append(f"  - {s.name}: weight={s.weight:.1f}, required={s.is_required}")
    return "\n".join(lines)


def _format_question_plan(state: InterviewState) -> str:
    """Include question content + reference points so the evaluator can write answers."""
    plan = state.get("question_plan", [])
    if not plan:
        return "No question plan available."
    lines = []
    for q in plan:
        lines.append(
            f"Q{q.id} [{q.difficulty}]: {q.content}\n"
            f"  Reference points: {', '.join(q.reference_points)}"
        )
    return "\n".join(lines)


# ── Professional Mode Evaluator ──────────────────────────────────────


def _format_round1_data(state: InterviewState) -> str:
    """Format Round 1 assessments for the final evaluator."""
    r1_assessments = state.get("round1_assessments", [])
    r1_plan = state.get("round1_question_plan", [])
    if not r1_assessments:
        return ""

    lines = ["## Round 1 (Technical Depth) Assessments"]
    for a in r1_assessments:
        lines.append(
            f"Q{a.question_id}: score={a.score}/10, "
            f"covered={a.covered_points}, missed={a.missed_points}"
        )

    summary_raw = state.get("round1_summary_text", "")
    if summary_raw:
        try:
            summary = Round1Summary.model_validate_json(summary_raw)
            lines.append(
                f"\nRound 1 Summary: score={summary.round1_score:.1f} ({summary.round1_grade})"
                f"\n  Depth: {summary.technical_depth_assessment}"
                f"\n  Projects: {summary.project_understanding}"
            )
        except Exception:
            pass

    return "\n".join(lines)


async def evaluate_interview(state: InterviewState) -> dict:
    """LangGraph node: produce a professional-mode evaluation report.

    In dual-round mode, uses ProfessionalReport for a comprehensive dual-round report.
    Falls back to InterviewReport for single-round or early-stop scenarios.
    """
    transcript = _format_transcript(state)
    assessments_text = _format_assessments(state)
    skill_context = _format_skill_matrix(state)
    round1_data = _format_round1_data(state)

    current_round = state.get("current_round", 1)
    is_dual_round = bool(round1_data) and current_round >= 2

    logger.info(
        "Evaluating interview (professional, round=%d, dual=%s): %d messages, %d assessments",
        current_round,
        is_dual_round,
        len(state.get("conversation_history", [])),
        len(state.get("assessments", [])),
    )

    user_content = f"## Skill Matrix\n{skill_context}\n\n"

    if is_dual_round:
        user_content += f"{round1_data}\n\n"
        r1_count = len(state.get("round1_assessments", []))
        all_assessments = state.get("assessments", [])
        r2_assessments = all_assessments[r1_count:]
        if r2_assessments:
            r2_lines = []
            for a in r2_assessments:
                r2_lines.append(
                    f"Q{a.question_id}: score={a.score}/10, "
                    f"covered={a.covered_points}, missed={a.missed_points}"
                )
            user_content += "## Round 2 (Technical Breadth) Assessments\n" + "\n".join(r2_lines) + "\n\n"

        user_content += (
            f"## Full Interview Transcript (Both Rounds)\n{transcript}\n\n"
            f"请基于以上两轮面试信息生成完整的双轮面试评价报告。"
            f"报告中所有候选人可见文本必须使用简体中文。"
        )

        llm = get_llm_client()
        report = await llm.chat_structured(
            messages=[
                {"role": "system", "content": PROFESSIONAL_EVALUATOR_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            response_model=ProfessionalReport,
            temperature=0.3,
        )

        logger.info(
            "Professional evaluation complete: overall=%.1f (%s), depth=%.1f, breadth=%.1f, rec=%s",
            report.overall_score, report.grade,
            report.technical_depth_score, report.technical_breadth_score,
            report.hiring_recommendation,
        )
        return {"professional_report": report, "interview_complete": True}

    user_content += f"## Per-Question Assessments\n{assessments_text}\n\n"
    user_content += (
        f"## Full Interview Transcript\n{transcript}\n\n"
        f"请基于以上内容生成完整面试评价报告。"
        f"报告中所有候选人可见文本必须使用简体中文。"
    )

    llm = get_llm_client()
    report = await llm.chat_structured(
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_model=InterviewReport,
        temperature=0.3,
    )

    logger.info(
        "Evaluation complete: overall=%.1f (%s), %d skill scores",
        report.overall_score, report.grade, len(report.skill_scores),
    )
    return {"final_report": report, "interview_complete": True}


# ── Round 1 Summary (intermediate node in dual-round professional mode) ──


async def summarize_round1(state: InterviewState) -> dict:
    """LangGraph node: produce an intermediate summary after Round 1.

    Saves Round 1 data (question plan + assessments) and generates a
    summary that guides Round 2 question planning.
    """
    transcript = _format_transcript(state)
    assessments_text = _format_assessments(state)
    skill_context = _format_skill_matrix(state)
    question_context = _format_question_plan(state)

    num_assessed = len(state.get("assessments", []))
    logger.info("Summarizing Round 1: %d assessments", num_assessed)

    user_content = (
        f"## Skill Matrix\n{skill_context}\n\n"
        f"## Round 1 Question Plan\n{question_context}\n\n"
        f"## Per-Question Assessments\n{assessments_text}\n\n"
        f"## Full Round 1 Transcript\n{transcript}\n\n"
        f"Total questions asked in Round 1: {num_assessed}\n\n"
        f"请生成一面阶段总结。所有候选人可见文本必须使用简体中文。"
    )

    llm = get_llm_client()
    summary = await llm.chat_structured(
        messages=[
            {"role": "system", "content": ROUND1_SUMMARY_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_model=Round1Summary,
        temperature=0.3,
    )

    logger.info(
        "Round 1 summary: score=%.1f (%s), proceed=%s",
        summary.round1_score, summary.round1_grade, summary.proceed_to_round2,
    )

    return {
        "round1_question_plan": list(state.get("question_plan", [])),
        "round1_assessments": list(state.get("assessments", [])),
        "round1_summary_text": summary.model_dump_json(),
        "current_round": 2,
        "current_question_index": 0,
        "follow_up_count": 0,
        "conversation_history": [
            ChatMessage(
                role="interviewer",
                content=(
                    f"---\n\n**一面结束 | Round 1 Complete**\n\n"
                    f"{summary.round1_feedback}\n\n"
                    f"得分: {summary.round1_score:.1f}/10 ({summary.round1_grade})\n\n"
                    f"---\n\n现在进入二面（技术广度），准备好了吗？"
                ),
            )
        ],
    }


# ── Practice Mode Evaluator ──────────────────────────────────────────


async def evaluate_practice(state: InterviewState) -> dict:
    """LangGraph node: produce a practice-mode report with reference answers."""
    transcript = _format_transcript(state)
    assessments_text = _format_assessments(state)
    skill_context = _format_skill_matrix(state)
    question_context = _format_question_plan(state)

    num_answered = len(state.get("assessments", []))
    logger.info(
        "Evaluating practice session: %d questions answered, %d assessments",
        num_answered, len(state.get("assessments", [])),
    )

    user_content = (
        f"## Skill Matrix\n{skill_context}\n\n"
        f"## Question Plan (with reference points)\n{question_context}\n\n"
        f"## Per-Question Assessments\n{assessments_text}\n\n"
        f"## Full Interview Transcript\n{transcript}\n\n"
        f"Total questions answered: {num_answered}\n\n"
        f"请基于以上内容生成练习模式评估报告。"
        f"所有候选人可见文本必须使用简体中文。"
        f"对于得分低于 8 分的问题，请给出覆盖遗漏点的中文参考答案。"
    )

    llm = get_llm_client()
    report = await llm.chat_structured(
        messages=[
            {"role": "system", "content": PRACTICE_EVALUATOR_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_model=PracticeReport,
        temperature=0.3,
    )

    logger.info(
        "Practice evaluation complete: overall=%.1f (%s), %d missed knowledge items",
        report.overall_score, report.grade, len(report.missed_knowledge),
    )
    return {"practice_report": report, "interview_complete": True}
