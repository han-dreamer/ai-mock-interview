"""LangGraph interview workflow — the orchestration core.

Supports two modes:
  - "practice":     JD → questions → interview loop → practice report
  - "professional": JD + resume → parallel_analyze (resume & JD concurrently)
                    → Round 1 (depth) → round1_summary
                    → Round 2 (breadth) → professional report

Practice flow:
    START → analyze_jd → plan_questions → ask_question
        → [interrupt] → assess_answer → route
            ├─ ask_follow_up → [interrupt] → assess_answer → route
            ├─ advance_question → ask_question → ...
            └─ evaluate_practice → END

Professional flow (dual-round):
    START → parallel_analyze → plan_questions_round1 → ask_question
        → [interrupt] → assess_answer → route
            ├─ ask_follow_up → [interrupt] → assess_answer → route
            ├─ advance_question → ask_question → ...
            └─ summarize_round1 → plan_questions_round2 → ask_question
                → ... (same interview loop) ...
                └─ evaluate_interview → END
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from app.agents.evaluator import evaluate_interview, evaluate_practice, summarize_round1
from app.agents.interviewer import (
    advance_question,
    ask_follow_up,
    ask_question,
    assess_answer,
    route_after_assessment,
)
from app.agents.jd_analyst import analyze_jd
from app.agents.parallel_prep import parallel_analyze
from app.agents.question_planner import (
    plan_questions,
    plan_questions_round2,
    plan_questions_with_resume,
)
from app.agents.state import InterviewState

logger = logging.getLogger(__name__)


def build_practice_graph():
    """Build a graph for practice mode (JD-only, no resume)."""
    graph = StateGraph(InterviewState)

    graph.add_node("analyze_jd", analyze_jd)
    graph.add_node("plan_questions", plan_questions)
    graph.add_node("ask_question", ask_question)
    graph.add_node("ask_follow_up", ask_follow_up)
    graph.add_node("assess_answer", assess_answer)
    graph.add_node("advance_question", advance_question)
    graph.add_node("evaluate_practice", evaluate_practice)

    graph.set_entry_point("analyze_jd")
    graph.add_edge("analyze_jd", "plan_questions")
    graph.add_edge("plan_questions", "ask_question")
    graph.add_edge("ask_question", "assess_answer")
    graph.add_edge("ask_follow_up", "assess_answer")
    graph.add_conditional_edges(
        "assess_answer",
        route_after_assessment,
        {
            "ask_follow_up": "ask_follow_up",
            "ask_question": "advance_question",
            "evaluate": "evaluate_practice",
        },
    )
    graph.add_edge("advance_question", "ask_question")
    graph.add_edge("evaluate_practice", END)

    return graph


def build_professional_graph():
    """Build a graph for professional mode (JD + resume, dual-round).

    Round 1: Technical depth — project deep-dive + fundamentals
    Round 2: Technical breadth — system design + latest AI tech

    Key optimization: analyze_resume and analyze_jd run in parallel
    via the parallel_analyze node, saving ~15-30s of startup time.
    """
    graph = StateGraph(InterviewState)

    # Parallel preparation (resume + JD analysis concurrently)
    graph.add_node("parallel_analyze", parallel_analyze)
    graph.add_node("plan_questions_round1", plan_questions_with_resume)

    # Shared interview loop nodes (reused for both rounds)
    graph.add_node("ask_question", ask_question)
    graph.add_node("ask_follow_up", ask_follow_up)
    graph.add_node("assess_answer", assess_answer)
    graph.add_node("advance_question", advance_question)

    # Round transition
    graph.add_node("summarize_round1", summarize_round1)
    graph.add_node("plan_questions_round2", plan_questions_round2)

    # Final evaluation
    graph.add_node("evaluate_interview", evaluate_interview)

    # ── Edges ──
    graph.set_entry_point("parallel_analyze")
    graph.add_edge("parallel_analyze", "plan_questions_round1")
    graph.add_edge("plan_questions_round1", "ask_question")

    # Interview loop (shared by both rounds)
    graph.add_edge("ask_question", "assess_answer")
    graph.add_edge("ask_follow_up", "assess_answer")
    graph.add_conditional_edges(
        "assess_answer",
        route_after_assessment,
        {
            "ask_follow_up": "ask_follow_up",
            "ask_question": "advance_question",
            "summarize_round1": "summarize_round1",
            "evaluate": "evaluate_interview",
        },
    )
    graph.add_edge("advance_question", "ask_question")

    # Round 1 → Round 2 transition
    graph.add_edge("summarize_round1", "plan_questions_round2")
    graph.add_edge("plan_questions_round2", "ask_question")

    graph.add_edge("evaluate_interview", END)

    return graph


# Singleton compiled graphs
_practice_graph = None
_professional_graph = None


def get_interview_graph(mode: str = "practice"):
    global _practice_graph, _professional_graph
    if mode == "professional":
        if _professional_graph is None:
            _professional_graph = build_professional_graph().compile(
                interrupt_before=["assess_answer"],
            )
            logger.info("Professional graph compiled (dual-round with resume analysis)")
        return _professional_graph
    else:
        if _practice_graph is None:
            _practice_graph = build_practice_graph().compile(
                interrupt_before=["assess_answer"],
            )
            logger.info("Practice graph compiled")
        return _practice_graph
