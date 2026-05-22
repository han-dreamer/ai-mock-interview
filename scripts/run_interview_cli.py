"""Interactive CLI to run a full mock interview through the LangGraph pipeline.

Usage:
    python -m scripts.run_interview_cli
    python -m scripts.run_interview_cli --jd data/sample_jds/ai_engineer.txt

This script drives the compiled graph manually:
1. Starts the graph → it runs analyze_jd → plan_questions → ask_question
2. The graph pauses at interrupt_before("assess_answer")
3. We print the interviewer's question, get user input
4. Inject the answer into state and resume
5. Repeat until the interview is complete and the report is generated
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("interview_cli")

from langgraph.checkpoint.memory import MemorySaver

from app.agents.state import InterviewState
from app.agents.evaluator import evaluate_interview
from app.agents.interviewer import (
    advance_question,
    ask_follow_up,
    ask_question,
    assess_answer,
    route_after_assessment,
)
from app.agents.jd_analyst import analyze_jd
from app.agents.question_planner import plan_questions
from app.models.interview import ChatMessage

from langgraph.graph import END, StateGraph


def build_cli_graph():
    """Build graph with MemorySaver checkpointer for the CLI session."""
    graph = StateGraph(InterviewState)

    graph.add_node("analyze_jd", analyze_jd)
    graph.add_node("plan_questions", plan_questions)
    graph.add_node("ask_question", ask_question)
    graph.add_node("ask_follow_up", ask_follow_up)
    graph.add_node("assess_answer", assess_answer)
    graph.add_node("advance_question", advance_question)
    graph.add_node("evaluate_interview", evaluate_interview)

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
            "evaluate": "evaluate_interview",
        },
    )
    graph.add_edge("advance_question", "ask_question")
    graph.add_edge("evaluate_interview", END)

    checkpointer = MemorySaver()
    return graph.compile(interrupt_before=["assess_answer"], checkpointer=checkpointer)


def get_last_interviewer_message(state: dict) -> str | None:
    """Extract the last interviewer message from conversation history."""
    history = state.get("conversation_history", [])
    for msg in reversed(history):
        if isinstance(msg, ChatMessage) and msg.role == "interviewer":
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "interviewer":
            return msg.get("content", "")
    return None


async def main():
    jd_path = PROJECT_ROOT / "data" / "sample_jds" / "ai_engineer.txt"
    for arg in sys.argv[1:]:
        if arg.startswith("--jd"):
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                jd_path = Path(sys.argv[idx + 1])

    if not jd_path.exists():
        print(f"JD file not found: {jd_path}")
        return

    jd_text = jd_path.read_text(encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"  AI Mock Interview CLI")
    print(f"  JD: {jd_path.name}")
    print(f"{'=' * 60}\n")

    graph = build_cli_graph()
    config = {"configurable": {"thread_id": "cli-session-1"}}

    initial_state: InterviewState = {
        "jd_text": jd_text,
        "max_follow_ups": 2,
        "current_question_index": 0,
        "follow_up_count": 0,
        "conversation_history": [],
        "assessments": [],
        "interview_complete": False,
    }

    print("[System] Analyzing JD and preparing questions...\n")

    # First invocation: runs analyze_jd → plan_questions → ask_question → interrupt
    result = await graph.ainvoke(initial_state, config)

    question_msg = get_last_interviewer_message(result)
    if question_msg:
        print(f"[Interviewer]: {question_msg}\n")

    # Interview loop
    while True:
        answer = input("[You]: ").strip()
        if not answer:
            print("(Please type your answer, or 'quit' to exit)")
            continue
        if answer.lower() in ("quit", "exit", "q"):
            print("\n[System] Interview terminated by user.")
            break

        # Inject the candidate's answer and resume
        current_state = graph.get_state(config)
        graph.update_state(
            config,
            {"current_candidate_answer": answer},
        )

        result = await graph.ainvoke(None, config)

        # Check if interview is complete
        if result.get("interview_complete"):
            report = result.get("final_report")
            if report:
                print(f"\n{'=' * 60}")
                print("  INTERVIEW REPORT")
                print(f"{'=' * 60}\n")
                print(f"Overall Score: {report.overall_score:.1f}/10  Grade: {report.grade}\n")
                print("Skill Scores:")
                for ss in report.skill_scores:
                    bar = "█" * ss.score + "░" * (10 - ss.score)
                    print(f"  {ss.skill_name:20s} {bar} {ss.score}/10")
                    print(f"    Evidence: {ss.evidence[:100]}")
                print(f"\nStrengths:")
                for s in report.strengths:
                    print(f"  ✓ {s.point}")
                    print(f"    {s.evidence[:100]}")
                print(f"\nAreas for Improvement:")
                for imp in report.improvements:
                    print(f"  → {imp.point}")
                    print(f"    {imp.evidence[:100]}")
                print(f"\nOverall Assessment:\n  {report.overall_assessment}")
            break

        # Print next question / follow-up
        question_msg = get_last_interviewer_message(result)
        if question_msg:
            print(f"\n[Interviewer]: {question_msg}\n")
        else:
            print("\n[System] Waiting for next action...\n")

    print(f"\n{'=' * 60}")
    print("  Interview session complete. Thank you!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
