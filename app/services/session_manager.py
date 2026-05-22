"""Interview session manager bridges frontend with LangGraph execution.

Supports two modes:
  - practice:     Quick practice with reference answers in the report
  - professional: Dual-round interview (Round 1: depth, Round 2: breadth)
                  with resume analysis and detailed evaluation
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.cache.locks import session_answer_lock
from app.cache.session_cache import save_session_report, save_session_snapshot
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
from app.memory.service import get_memory_service
from app.models.interview import AnswerAssessment, ChatMessage, InterviewSession
from app.models.question import QuestionItem
from app.models.report import InterviewReport, PracticeReport, ProfessionalReport
from app.models.resume import ResumeParseResult

logger = logging.getLogger(__name__)


@dataclass
class _SessionData:
    """Internal bookkeeping for a live interview session."""

    session: InterviewSession
    mode: Literal["practice", "professional"] = "practice"
    resume_text: str = ""
    resume_parse_result: ResumeParseResult | None = None
    report: InterviewReport | None = None
    professional_report: ProfessionalReport | None = None
    practice_report: PracticeReport | None = None
    user_id: str = "local-user"
    persisted_assessment_count: int = 0
    final_memory_saved: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    graph_started: bool = False
    last_state: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


class SessionManager:
    """Manages interview sessions and drives the LangGraph execution.

    Each session gets its own LangGraph thread_id so checkpoint state
    is isolated. Separate graphs are compiled for practice and professional modes.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionData] = {}
        self._checkpointer = MemorySaver()
        self._practice_graph = self._build_graph("practice")
        self._professional_graph = self._build_graph("professional")

    # Graph construction

    def _build_graph(self, mode: str):
        graph = StateGraph(InterviewState)

        if mode == "professional":
            graph.add_node("parallel_analyze", parallel_analyze)
            graph.add_node("plan_questions_round1", plan_questions_with_resume)

            graph.add_node("ask_question", ask_question)
            graph.add_node("ask_follow_up", ask_follow_up)
            graph.add_node("assess_answer", assess_answer)
            graph.add_node("advance_question", advance_question)

            graph.add_node("summarize_round1", summarize_round1)
            graph.add_node("plan_questions_round2", plan_questions_round2)
            graph.add_node("evaluate", evaluate_interview)

            graph.set_entry_point("parallel_analyze")
            graph.add_edge("parallel_analyze", "plan_questions_round1")
            graph.add_edge("plan_questions_round1", "ask_question")

            graph.add_edge("ask_question", "assess_answer")
            graph.add_edge("ask_follow_up", "assess_answer")
            graph.add_conditional_edges(
                "assess_answer",
                route_after_assessment,
                {
                    "ask_follow_up": "ask_follow_up",
                    "ask_question": "advance_question",
                    "summarize_round1": "summarize_round1",
                    "evaluate": "evaluate",
                },
            )
            graph.add_edge("advance_question", "ask_question")

            graph.add_edge("summarize_round1", "plan_questions_round2")
            graph.add_edge("plan_questions_round2", "ask_question")
            graph.add_edge("evaluate", END)
        else:
            graph.add_node("analyze_jd", analyze_jd)
            graph.add_node("plan_questions", plan_questions)
            graph.add_node("evaluate", evaluate_practice)

            graph.add_node("ask_question", ask_question)
            graph.add_node("ask_follow_up", ask_follow_up)
            graph.add_node("assess_answer", assess_answer)
            graph.add_node("advance_question", advance_question)

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
                    "evaluate": "evaluate",
                },
            )
            graph.add_edge("advance_question", "ask_question")
            graph.add_edge("evaluate", END)

        compiled = graph.compile(
            interrupt_before=["assess_answer"],
            checkpointer=self._checkpointer,
        )
        logger.info("SessionManager: %s graph compiled", mode)
        return compiled

    def _get_graph(self, mode: str):
        return self._practice_graph if mode == "practice" else self._professional_graph

    # Session CRUD

    def create_session(
        self,
        session_id: str,
        jd_text: str,
        max_follow_ups: int = 2,
        mode: Literal["practice", "professional"] = "practice",
        resume_text: str = "",
        resume_parse_result: ResumeParseResult | None = None,
        user_id: str = "local-user",
    ) -> InterviewSession:
        session = InterviewSession(
            session_id=session_id,
            user_id=user_id,
            jd_text=jd_text,
            max_follow_ups=max_follow_ups,
        )
        data = _SessionData(
            session=session,
            mode=mode,
            resume_text=resume_text,
            resume_parse_result=resume_parse_result,
            user_id=user_id,
        )
        self._sessions[session_id] = data
        logger.info(
            "Session created: %s (mode=%s, resume=%d chars)",
            session_id,
            mode,
            len(resume_text),
        )
        return session

    def get_session(self, session_id: str) -> InterviewSession | None:
        data = self._sessions.get(session_id)
        return data.session if data else None

    def get_session_mode(self, session_id: str) -> str:
        data = self._sessions.get(session_id)
        return data.mode if data else "practice"

    def update_session_status(self, session_id: str, status: str) -> None:
        data = self._sessions.get(session_id)
        if data:
            data.session.status = status

    def set_resume(
        self,
        session_id: str,
        resume_text: str,
        resume_parse_result: ResumeParseResult | None = None,
    ) -> None:
        """Attach parsed resume content before the graph starts."""
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")
        if data.graph_started:
            raise RuntimeError("Resume cannot be changed after the interview has started")
        data.resume_text = resume_text
        data.resume_parse_result = resume_parse_result

    def get_resume_parse_result(self, session_id: str) -> ResumeParseResult | None:
        data = self._sessions.get(session_id)
        return data.resume_parse_result if data else None

    def has_graph_started(self, session_id: str) -> bool:
        data = self._sessions.get(session_id)
        return bool(data and data.graph_started)

    def save_report(self, session_id: str, report: InterviewReport) -> None:
        data = self._sessions.get(session_id)
        if data:
            data.report = report
            data.session.status = "completed"

    def save_professional_report(self, session_id: str, report: ProfessionalReport) -> None:
        data = self._sessions.get(session_id)
        if data:
            data.professional_report = report
            data.session.status = "completed"

    def save_practice_report(self, session_id: str, report: PracticeReport) -> None:
        data = self._sessions.get(session_id)
        if data:
            data.practice_report = report
            data.session.status = "completed"

    def get_report(self, session_id: str) -> InterviewReport | None:
        data = self._sessions.get(session_id)
        return data.report if data else None

    def get_professional_report(self, session_id: str) -> ProfessionalReport | None:
        data = self._sessions.get(session_id)
        return data.professional_report if data else None

    def get_practice_report(self, session_id: str) -> PracticeReport | None:
        data = self._sessions.get(session_id)
        return data.practice_report if data else None

    def get_report_for_session(
        self,
        session_id: str,
    ) -> PracticeReport | ProfessionalReport | InterviewReport | None:
        data = self._sessions.get(session_id)
        if not data:
            return None
        if data.mode == "practice":
            return data.practice_report or data.report
        return data.professional_report or data.report

    def get_last_state(self, session_id: str) -> dict[str, Any]:
        data = self._sessions.get(session_id)
        if not data:
            return {}
        graph_state = self.get_graph_state(session_id)
        if graph_state:
            self._sync_session_from_state(session_id, graph_state)
            return graph_state
        return data.last_state

    def _sync_session_from_state(self, session_id: str, state: dict | None) -> None:
        """Mirror LangGraph state into lightweight session metadata."""
        if not state:
            return
        data = self._sessions.get(session_id)
        if not data:
            return

        state_dict = dict(state)
        data.last_state = state_dict

        session = data.session
        session.current_question_index = int(
            state_dict.get("current_question_index", session.current_question_index) or 0
        )
        session.follow_up_count = int(
            state_dict.get("follow_up_count", session.follow_up_count) or 0
        )

        history = []
        for raw_msg in state_dict.get("conversation_history", []) or []:
            try:
                if isinstance(raw_msg, ChatMessage):
                    history.append(raw_msg)
                else:
                    history.append(ChatMessage.model_validate(raw_msg))
            except Exception:
                logger.debug("Skipped invalid chat message in session=%s: %r", session_id, raw_msg)
        session.conversation_history = history

        assessments = []
        for raw_assessment in state_dict.get("assessments", []) or []:
            try:
                assessments.append(
                    raw_assessment
                    if isinstance(raw_assessment, AnswerAssessment)
                    else AnswerAssessment.model_validate(raw_assessment)
                )
            except Exception:
                logger.debug(
                    "Skipped invalid assessment in session=%s: %r",
                    session_id,
                    raw_assessment,
                )
        session.assessments = assessments

        if state_dict.get("interview_complete"):
            session.status = "completed"

    async def _cache_runtime_state(self, session_id: str, state: dict | None) -> None:
        if not state:
            return
        data = self._sessions.get(session_id)
        if not data:
            return
        await save_session_snapshot(
            data.session,
            dict(state),
            mode=data.mode,
            graph_started=data.graph_started,
        )
        report = (
            state.get("practice_report")
            or state.get("professional_report")
            or state.get("final_report")
            or self.get_report_for_session(session_id)
        )
        if report:
            await save_session_report(session_id, report)

    # LangGraph execution helpers

    def _config(self, session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    async def start_interview_graph(self, session_id: str) -> dict:
        """Kick off the graph.

        Practice:     analyze_jd -> plan_questions -> ask_question -> interrupt
        Professional: analyze_resume -> analyze_jd -> plan_round1 -> ask_question -> interrupt
        """
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")

        async with data.lock:
            if data.graph_started:
                state = self.get_graph_state(session_id) or data.last_state
                self._sync_session_from_state(session_id, state)
                await self._cache_runtime_state(session_id, state)
                return state

            session = data.session
            mode = data.mode
            graph = self._get_graph(mode)
            memory = get_memory_service()
            semantic_query = self._build_memory_query(session.jd_text, data.resume_text)
            memory_context = await memory.abuild_context(
                data.user_id,
                semantic_query=semantic_query,
            )
            memory_context_text = memory.format_context(memory_context)

            initial_state: InterviewState = {
                "interview_mode": mode,
                "user_id": data.user_id,
                "jd_text": session.jd_text,
                "memory_context": memory_context_text,
                "retrieved_memories": memory_context.model_dump(mode="json"),
                "max_follow_ups": session.max_follow_ups,
                "current_question_index": 0,
                "follow_up_count": 0,
                "conversation_history": [],
                "assessments": [],
                "interview_complete": False,
                "current_round": 1,
            }

            if mode == "professional" and data.resume_text:
                initial_state["resume_text"] = data.resume_text
            if mode == "professional" and data.resume_parse_result:
                initial_state["resume_parse_result"] = data.resume_parse_result

            self.update_session_status(session_id, "analyzing")
            try:
                result = await graph.ainvoke(initial_state, self._config(session_id))
                data.graph_started = True
                if result.get("resume_profile"):
                    try:
                        saved = memory.save_resume_profile(
                            data.user_id,
                            session_id,
                            result.get("resume_profile"),
                        )
                        await memory.aindex_memory_items(saved)
                    except Exception:
                        logger.exception("Failed to save resume memory for session=%s", session_id)
                self._sync_session_from_state(session_id, result)
                if not result.get("interview_complete"):
                    self.update_session_status(session_id, "interviewing")
                await self._cache_runtime_state(session_id, result)
                return result
            except Exception as exc:
                data.error_message = str(exc)
                self.update_session_status(session_id, "failed")
                raise

    async def submit_answer(self, session_id: str, answer: str) -> dict:
        """Inject candidate answer and resume the graph."""
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")

        async with session_answer_lock(session_id):
            async with data.lock:
                if not data.graph_started:
                    raise RuntimeError("Interview graph has not started")

                mode = data.mode
                graph = self._get_graph(mode)
                config = self._config(session_id)

                self.update_session_status(session_id, "evaluating")
                try:
                    graph.update_state(config, {"current_candidate_answer": answer})
                    result = await graph.ainvoke(None, config)
                    await self._persist_new_assessment_memories(session_id, result, answer)

                    if result.get("interview_complete"):
                        if mode == "practice":
                            practice_report = result.get("practice_report")
                            if practice_report:
                                self.save_practice_report(session_id, practice_report)
                        else:
                            pro_report = result.get("professional_report")
                            if pro_report:
                                self.save_professional_report(session_id, pro_report)
                            else:
                                report = result.get("final_report")
                                if report:
                                    self.save_report(session_id, report)
                        await self._persist_final_memory(session_id, result)

                    self._sync_session_from_state(session_id, result)
                    if not result.get("interview_complete"):
                        self.update_session_status(session_id, "interviewing")
                    await self._cache_runtime_state(session_id, result)
                    return result
                except Exception as exc:
                    data.error_message = str(exc)
                    self.update_session_status(session_id, "failed")
                    raise

    async def stop_interview(self, session_id: str) -> dict:
        """Stop the interview early and generate a report with whatever has been answered.

        Works by directly calling the appropriate evaluator with the current state.
        """
        data = self._sessions.get(session_id)
        if not data:
            raise ValueError(f"Session {session_id} not found")

        async with data.lock:
            mode = data.mode
            graph = self._get_graph(mode)
            config = self._config(session_id)

            try:
                snapshot = graph.get_state(config)
                current_state = snapshot.values if snapshot else data.last_state
            except Exception:
                current_state = data.last_state

            current_state = dict(current_state or {})
            if not current_state or not current_state.get("assessments"):
                has_r1 = bool(current_state.get("round1_assessments"))
                if not has_r1:
                    result = {
                        "interview_complete": True,
                        "practice_report": None,
                        "professional_report": None,
                        "final_report": None,
                    }
                    self.update_session_status(session_id, "completed")
                    self._sync_session_from_state(session_id, result)
                    await self._cache_runtime_state(session_id, result)
                    return result

            logger.info(
                "Stopping interview early: %s (mode=%s, round=%d, %d assessments so far)",
                session_id,
                mode,
                current_state.get("current_round", 1),
                len(current_state.get("assessments", [])),
            )

            self.update_session_status(session_id, "evaluating")
            try:
                if mode == "practice":
                    result = await evaluate_practice(current_state)
                    practice_report = result.get("practice_report")
                    if practice_report:
                        self.save_practice_report(session_id, practice_report)
                else:
                    result = await evaluate_interview(current_state)
                    pro_report = result.get("professional_report")
                    if pro_report:
                        self.save_professional_report(session_id, pro_report)
                    else:
                        report = result.get("final_report")
                        if report:
                            self.save_report(session_id, report)

                result["interview_complete"] = True
                merged_state = dict(current_state)
                merged_state.update(result)
                await self._persist_new_assessment_memories(session_id, merged_state, "")
                await self._persist_final_memory(session_id, merged_state)
                self._sync_session_from_state(session_id, merged_state)
                self.update_session_status(session_id, "completed")
                await self._cache_runtime_state(session_id, merged_state)
                return merged_state
            except Exception as exc:
                data.error_message = str(exc)
                self.update_session_status(session_id, "failed")
                raise

    def get_graph_state(self, session_id: str) -> dict | None:
        mode = self.get_session_mode(session_id)
        graph = self._get_graph(mode)
        try:
            snapshot = graph.get_state(self._config(session_id))
            return snapshot.values if snapshot else None
        except Exception:
            return None

    # Long-term memory helpers

    def _build_memory_query(self, jd_text: str, resume_text: str = "") -> str:
        parts = [
            "Interview personalization query.",
            "Job description:",
            jd_text[:1200],
        ]
        if resume_text:
            parts.extend(["Candidate resume:", resume_text[:1000]])
        return "\n".join(parts)

    def _candidate_question(self, state: dict, question_id: int) -> QuestionItem | None:
        plans = [
            state.get("question_plan", []),
            state.get("round1_question_plan", []),
            state.get("round2_question_plan", []),
        ]
        for plan in plans:
            for raw_question in plan or []:
                question = (
                    raw_question
                    if isinstance(raw_question, QuestionItem)
                    else QuestionItem.model_validate(raw_question)
                )
                if question.id == question_id:
                    return question
        return None

    async def _persist_new_assessment_memories(
        self,
        session_id: str,
        state: dict,
        latest_answer: str,
    ) -> None:
        data = self._sessions.get(session_id)
        if not data:
            return

        assessments = state.get("assessments", []) or []
        if data.persisted_assessment_count >= len(assessments):
            return

        memory = get_memory_service()
        mode = data.mode
        for raw_assessment in assessments[data.persisted_assessment_count:]:
            try:
                assessment = raw_assessment
                if not hasattr(assessment, "question_id"):
                    from app.models.interview import AnswerAssessment

                    assessment = AnswerAssessment.model_validate(raw_assessment)

                question = self._candidate_question(state, assessment.question_id)
                item = memory.save_assessment_episode(
                    user_id=data.user_id,
                    session_id=session_id,
                    assessment=assessment,
                    question=question,
                    answer=latest_answer,
                    mode=mode,
                )
                await memory.aindex_memory_item(item)
                data.persisted_assessment_count += 1
            except Exception:
                logger.exception(
                    "Failed to persist assessment memory: session=%s index=%d",
                    session_id,
                    data.persisted_assessment_count,
                )

    async def _persist_final_memory(self, session_id: str, state: dict) -> None:
        data = self._sessions.get(session_id)
        if not data or data.final_memory_saved:
            return

        report = (
            state.get("practice_report")
            or state.get("professional_report")
            or state.get("final_report")
        )
        try:
            item = get_memory_service().save_session_reflection(
                user_id=data.user_id,
                session_id=session_id,
                mode=data.mode,
                report=report,
                assessments=state.get("assessments", []) or [],
            )
            if item:
                await get_memory_service().aindex_memory_item(item)
            data.final_memory_saved = True
        except Exception:
            logger.exception("Failed to persist final memory: session=%s", session_id)


# Module-level singleton
_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
