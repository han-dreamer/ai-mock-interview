"""Interviewer Agent — conducts the interview: ask, follow-up, assess each answer."""

from __future__ import annotations

import logging

from app.agents.state import InterviewState
from app.llm.client import get_llm_client
from app.llm.prompts import ANSWER_ASSESSOR_SYSTEM, INTERVIEWER_SYSTEM
from app.models.interview import AnswerAssessment, ChatMessage

logger = logging.getLogger(__name__)


def _current_question(state: InterviewState):
    idx = state["current_question_index"]
    plan = state["question_plan"]
    if idx < len(plan):
        return plan[idx]
    return None


def _build_conversation_context(state: InterviewState) -> list[dict]:
    """Convert conversation history into OpenAI message format."""
    msgs: list[dict] = []
    for m in state.get("conversation_history", []):
        if isinstance(m, ChatMessage):
            role = "assistant" if m.role == "interviewer" else "user"
            msgs.append({"role": role, "content": m.content})
        elif isinstance(m, dict):
            role = "assistant" if m.get("role") == "interviewer" else "user"
            msgs.append({"role": role, "content": m.get("content", "")})
    return msgs


# ─────────────────────────────────────────────────────────────────────
# Node 1: Ask a question
# ─────────────────────────────────────────────────────────────────────
async def ask_question(state: InterviewState) -> dict:
    """LangGraph node: ask the current question to the candidate.

    Reads:  question_plan, current_question_index, conversation_history
    Writes: conversation_history (append interviewer message)
    """
    question = _current_question(state)
    if not question:
        return {"interview_complete": True}

    llm = get_llm_client()
    max_follow_ups = state.get("max_follow_ups", 2)
    system_prompt = INTERVIEWER_SYSTEM.format(max_follow_ups=max_follow_ups)

    conversation = _build_conversation_context(state)
    conversation.append({
        "role": "user",
        "content": (
            f"[INTERNAL — not visible to candidate] "
            f"Now ask Question #{question.id}: {question.content}\n"
            f"Skills tested: {', '.join(question.skill_tags)}\n"
            f"Difficulty: {question.difficulty}\n"
            f"Just ask the question naturally, as a human interviewer would."
        ),
    })

    response = await llm.chat(
        messages=[{"role": "system", "content": system_prompt}] + conversation,
        temperature=0.7,
    )

    logger.info("Interviewer asks Q%d: %s", question.id, response[:80])

    return {
        "conversation_history": [ChatMessage(role="interviewer", content=response)],
        "follow_up_count": 0,
    }


# ─────────────────────────────────────────────────────────────────────
# Node 2: Ask a follow-up question
# ─────────────────────────────────────────────────────────────────────
async def ask_follow_up(state: InterviewState) -> dict:
    """LangGraph node: ask a targeted follow-up based on the latest assessment.

    Reads:  question_plan, current_question_index, assessments, conversation_history
    Writes: conversation_history (append follow-up), follow_up_count
    """
    question = _current_question(state)
    assessments = state.get("assessments", [])
    latest_assessment = assessments[-1] if assessments else None

    llm = get_llm_client()
    max_follow_ups = state.get("max_follow_ups", 2)
    system_prompt = INTERVIEWER_SYSTEM.format(max_follow_ups=max_follow_ups)

    missed_hint = ""
    if latest_assessment and latest_assessment.missed_points:
        missed_hint = (
            f"\n[INTERNAL] The candidate missed these points: "
            f"{', '.join(latest_assessment.missed_points[:3])}. "
            f"Follow-up reason: {latest_assessment.follow_up_reason or 'explore deeper'}. "
            f"Ask a natural follow-up that guides them toward these areas."
        )

    follow_up_dirs = ""
    if question and question.follow_up_directions:
        follow_up_dirs = (
            f"\n[INTERNAL] Possible follow-up directions: "
            f"{', '.join(question.follow_up_directions)}"
        )

    conversation = _build_conversation_context(state)
    conversation.append({
        "role": "user",
        "content": (
            f"[INTERNAL] The candidate's answer was incomplete. "
            f"Please ask a targeted follow-up question to dig deeper."
            f"{missed_hint}{follow_up_dirs}"
        ),
    })

    response = await llm.chat(
        messages=[{"role": "system", "content": system_prompt}] + conversation,
        temperature=0.7,
    )

    logger.info("Interviewer follow-up #%d: %s", state.get("follow_up_count", 0) + 1, response[:80])

    return {
        "conversation_history": [ChatMessage(role="interviewer", content=response)],
        "follow_up_count": state.get("follow_up_count", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────
# Node 3: Assess the candidate's answer (internal, not shown to user)
# ─────────────────────────────────────────────────────────────────────
async def assess_answer(state: InterviewState) -> dict:
    """LangGraph node: evaluate the candidate's latest answer.

    Reads:  current_candidate_answer, question_plan, current_question_index,
            follow_up_count, max_follow_ups
    Writes: assessments (append), conversation_history (append candidate msg)
    """
    question = _current_question(state)
    answer = state.get("current_candidate_answer", "")

    if not question:
        return {"interview_complete": True}

    max_follow_ups = state.get("max_follow_ups", 2)
    follow_up_count = state.get("follow_up_count", 0)

    llm = get_llm_client()
    system_prompt = ANSWER_ASSESSOR_SYSTEM.format(max_follow_ups=max_follow_ups)

    user_content = (
        f"Question (#{question.id}, {question.difficulty}):\n{question.content}\n\n"
        f"Reference answer key points:\n"
        + "\n".join(f"- {p}" for p in question.reference_points)
        + f"\n\nCandidate's answer:\n{answer}\n\n"
        f"Current follow-up count for this question: {follow_up_count}/{max_follow_ups}\n\n"
        f"Produce a structured assessment."
    )

    assessment = await llm.chat_structured(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_model=AnswerAssessment,
        temperature=0.2,
    )
    assessment.question_id = question.id

    # Override follow-up decision if we've hit the limit
    if follow_up_count >= max_follow_ups:
        assessment.should_follow_up = False
        assessment.follow_up_reason = None

    logger.info(
        "Assessment Q%d: score=%d, follow_up=%s (covered=%d, missed=%d)",
        question.id,
        assessment.score,
        assessment.should_follow_up,
        len(assessment.covered_points),
        len(assessment.missed_points),
    )

    return {
        "assessments": [assessment],
        "conversation_history": [ChatMessage(role="candidate", content=answer)],
    }


# ─────────────────────────────────────────────────────────────────────
# Routing function: decide what happens after assessment
# ─────────────────────────────────────────────────────────────────────
def route_after_assessment(state: InterviewState) -> str:
    """Conditional edge: decide whether to follow up, move to next question, or finish.

    For practice mode, returns: "ask_follow_up", "ask_question", or "evaluate".
    For professional mode with dual rounds, returns:
      - "ask_follow_up", "ask_question" during a round
      - "summarize_round1" when Round 1 ends
      - "evaluate" when Round 2 ends
    """
    assessments = state.get("assessments", [])
    if not assessments:
        return "evaluate"

    latest = assessments[-1]
    question_plan = state.get("question_plan", [])
    current_idx = state.get("current_question_index", 0)

    if latest.should_follow_up:
        logger.info("Route → follow up (score=%d, reason=%s)", latest.score, latest.follow_up_reason)
        return "ask_follow_up"

    if current_idx < len(question_plan) - 1:
        logger.info("Route → next question (Q%d done, moving to Q%d)", current_idx + 1, current_idx + 2)
        return "ask_question"

    mode = state.get("interview_mode", "practice")
    current_round = state.get("current_round", 1)

    if mode == "professional" and current_round == 1:
        logger.info("Route → summarize_round1 (Round 1 complete, %d questions)", len(question_plan))
        return "summarize_round1"

    logger.info("Route → evaluate (all questions done, round=%d)", current_round)
    return "evaluate"


# ─────────────────────────────────────────────────────────────────────
# Helper node: advance to next question
# ─────────────────────────────────────────────────────────────────────
async def advance_question(state: InterviewState) -> dict:
    """Increment current_question_index and reset follow_up_count."""
    new_idx = state.get("current_question_index", 0) + 1
    return {
        "current_question_index": new_idx,
        "follow_up_count": 0,
    }
