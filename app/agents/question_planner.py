"""Question Planner Agent — builds interview question plans from skill matrix + RAG.

Supports three planning strategies:
  - Standard:     Based on skill matrix + RAG (for practice mode)
  - Resume-aware: Based on skill matrix + resume profile + RAG (professional Round 1)
  - Round 2:      Based on skill matrix + Round 1 summary (professional Round 2)
"""

from __future__ import annotations

import logging

from app.agents.state import InterviewState
from app.llm.client import get_llm_client
from app.llm.prompts import (
    QUESTION_PLANNER_ROUND2_SYSTEM,
    QUESTION_PLANNER_SYSTEM,
    QUESTION_PLANNER_WITH_RESUME_SYSTEM,
)
from app.models.question import QuestionPlan
from app.models.report import Round1Summary
from app.models.resume import ResumeJDMatch, ResumeProfile
from app.rag.postprocess import (
    diversify_results,
    hydrate_parent_results,
    metadata_list,
    source_category_map,
)
from app.rag.query_builder import RetrievalPurpose, build_retrieval_queries
from app.rag.reranker import rerank_results
from app.rag.retriever import get_retriever
from app.resume.matcher import match_resume_to_jd

logger = logging.getLogger(__name__)


def _format_skill_matrix(state: InterviewState) -> str:
    sm = state["skill_matrix"]
    lines = [f"Position: {sm.position_title} ({sm.experience_level})"]
    lines.append("Skills:")
    for s in sm.skills:
        req = "REQUIRED" if s.is_required else "nice-to-have"
        lines.append(f"  - {s.name} [{s.category}] weight={s.weight:.1f} ({req})")
    return "\n".join(lines)


def _format_retrieved_questions(results: list) -> str:
    if not results:
        return "No reference questions retrieved. Generate all questions from scratch."
    lines = ["Retrieved reference questions from the question bank:"]
    for i, r in enumerate(results, 1):
        meta = r.metadata
        content = meta.get("content", r.content[:200])
        difficulty = meta.get("difficulty", "?")
        category = meta.get("category", "?")
        tags = metadata_list(meta, "skill_tags")
        ref_points = metadata_list(meta, "reference_points")
        chunk_ids = metadata_list(meta, "source_chunk_ids")
        chunk_types = metadata_list(meta, "source_chunk_types")
        matched_chunks = metadata_list(meta, "matched_chunk_texts")
        lines.append(
            f"\n--- Reference #{i} ---\n"
            f"source_id: {r.id}\n"
            f"category: {category}\n"
            f"difficulty: {difficulty}\n"
            f"tags: {tags}\n"
            f"matched_chunk_types: {chunk_types}\n"
            f"matched_chunk_ids: {chunk_ids[:5]}\n"
            f"retrieval_source: {r.source}\n"
            f"fusion_score: {r.score:.4f}\n"
            f"Q: {content}\n"
            f"Key points: {ref_points}\n"
            f"Most relevant retrieved chunks: {matched_chunks[:3]}"
        )
    return "\n".join(lines)


def _source_tracking_instruction() -> str:
    return (
        "When you adapt or are inspired by a retrieved reference, fill source_ids "
        "with the exact source_id values above. Only use source_ids from the retrieved "
        "references. Leave source_ids empty for fully original questions."
    )


def _format_memory_context(state: InterviewState) -> str:
    memory_context = state.get("memory_context", "").strip()
    if not memory_context:
        return "No long-term memory is available for this user yet."
    return memory_context


async def _retrieve_reference_questions(
    state: InterviewState,
    purpose: RetrievalPurpose,
    final_top_k: int,
) -> list:
    queries = build_retrieval_queries(state, purpose=purpose)
    logger.info(
        "RAG queries for %s: %s",
        purpose,
        [f"{q.purpose}:{q.text}" for q in queries],
    )
    retriever = get_retriever()
    retrieved = await retriever.retrieve_multi(
        queries,
        top_k_per_query=8,
        final_top_k=final_top_k * 4,
    )
    reranked = rerank_results(state, retrieved, purpose=purpose)
    hydrated = hydrate_parent_results(reranked, top_k=final_top_k + 6)
    return diversify_results(
        hydrated,
        top_k=final_top_k,
        max_per_category=6,
        max_per_difficulty=8,
    )


def _clean_question_sources(plan: QuestionPlan, retrieved: list) -> QuestionPlan:
    valid_sources = source_category_map(retrieved)
    for question in plan.questions:
        cleaned_ids: list[str] = []
        for source_id in question.source_ids:
            normalized = str(source_id).strip()
            if normalized in valid_sources and normalized not in cleaned_ids:
                cleaned_ids.append(normalized)
        question.source_ids = cleaned_ids
        question.source_categories = sorted(
            {valid_sources[source_id] for source_id in cleaned_ids if valid_sources[source_id]}
        )
    return plan


async def plan_questions(state: InterviewState) -> dict:
    """LangGraph node: generate a question plan based on skill matrix and RAG results.

    Reads:  state["skill_matrix"]
    Writes: state["question_plan"], state["current_question_index"],
            state["follow_up_count"], state["max_follow_ups"]
    """
    skill_matrix = state["skill_matrix"]
    logger.info("Planning questions for %d skills...", len(skill_matrix.skills))

    try:
        retrieved = await _retrieve_reference_questions(state, purpose="practice", final_top_k=15)
    except Exception as e:
        logger.warning("RAG retrieval failed (%s), using LLM-only generation", e)
        retrieved = []

    user_content = (
        f"{_format_skill_matrix(state)}\n\n"
        f"{_format_memory_context(state)}\n\n"
        f"{_format_retrieved_questions(retrieved)}\n\n"
        f"{_source_tracking_instruction()}\n\n"
        "请基于以上技能矩阵和参考题，生成 5-8 道面试题。"
        "题目正文、参考要点和追问方向必须使用简体中文；技术名词可以保留英文。"
        "请结合长期记忆强调薄弱技能，并避免重复候选人已经表现较好的题目。"
    )

    llm = get_llm_client()
    plan = await llm.chat_structured(
        messages=[
            {"role": "system", "content": QUESTION_PLANNER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_model=QuestionPlan,
        temperature=0.5,
    )
    plan = _clean_question_sources(plan, retrieved)

    logger.info(
        "Question plan ready: %d questions, est. %d min",
        len(plan.questions),
        plan.total_estimated_minutes,
    )
    for q in plan.questions:
        logger.debug("  Q%d [%s] %s", q.id, q.difficulty, q.content[:60])

    return {
        "question_plan": plan.questions,
        "current_question_index": 0,
        "follow_up_count": 0,
        "max_follow_ups": state.get("max_follow_ups", 2),
    }


def _format_resume_profile(profile: ResumeProfile) -> str:
    """Format a ResumeProfile into a readable string for the LLM."""
    lines = [
        f"Name: {profile.name}",
        f"Education: {profile.education}",
        f"Skills: {', '.join(profile.skills)}",
    ]
    if profile.experience:
        lines.append("Experience:")
        for exp in profile.experience:
            lines.append(f"  - {exp}")
    if profile.projects:
        lines.append("Projects:")
        for p in profile.projects:
            lines.append(f"  [{p.name}] Tech: {', '.join(p.tech_stack)}")
            lines.append(f"    Description: {p.description}")
            lines.append(f"    Contributions: {'; '.join(p.key_contributions)}")
            lines.append(f"    Deep-dive points: {'; '.join(p.potential_deep_dive_points)}")
    if profile.highlights:
        lines.append("Highlights:")
        for h in profile.highlights:
            lines.append(f"  - {h}")
    if profile.links:
        lines.append("Links/contact items:")
        for link in profile.links:
            lines.append(f"  - [{link.type}] {link.url}")
    if profile.concerns:
        lines.append("Concerns to verify:")
        for concern in profile.concerns:
            lines.append(f"  - {concern}")
    if profile.parse_warnings:
        lines.append("Parser warnings:")
        for warning in profile.parse_warnings:
            lines.append(f"  - {warning}")
    lines.append(f"Overall: {profile.summary}")
    return "\n".join(lines)


def _format_resume_jd_match(match: ResumeJDMatch) -> str:
    """Format lightweight resume-JD matching signals for the LLM."""
    lines = [
        f"Matched JD skills: {', '.join(match.matched_skills) or 'None'}",
        f"Missing or unclear JD skills: {', '.join(match.missing_skills) or 'None'}",
        f"Relevant projects: {', '.join(match.relevant_projects) or 'None'}",
    ]
    if match.project_skill_map:
        lines.append("Project-skill map:")
        for project_name, skills in match.project_skill_map.items():
            lines.append(f"  - {project_name}: {', '.join(skills)}")
    if match.interview_focus:
        lines.append("Suggested interview focus:")
        for focus in match.interview_focus:
            lines.append(f"  - {focus}")
    return "\n".join(lines)


async def plan_questions_with_resume(state: InterviewState) -> dict:
    """LangGraph node: generate a resume-aware question plan for professional mode.

    Uses the candidate's resume profile to create questions that probe
    their actual experience and project depth.

    Reads:  skill_matrix, resume_profile
    Writes: question_plan, current_question_index, follow_up_count, max_follow_ups
    """
    skill_matrix = state["skill_matrix"]
    resume_profile = state.get("resume_profile")

    logger.info(
        "Planning resume-aware questions for %d skills, %d projects...",
        len(skill_matrix.skills),
        len(resume_profile.projects) if resume_profile else 0,
    )

    resume_context = ""
    resume_jd_match = None
    match_context = ""
    if resume_profile:
        resume_jd_match = match_resume_to_jd(resume_profile, skill_matrix)
        resume_context = f"\n\n## Candidate Resume Profile\n{_format_resume_profile(resume_profile)}"
        match_context = (
            "\n\n## Resume-JD Match Signals\n"
            f"{_format_resume_jd_match(resume_jd_match)}"
        )

    retrieval_state = dict(state)
    if resume_jd_match:
        retrieval_state["resume_jd_match"] = resume_jd_match

    try:
        retrieved = await _retrieve_reference_questions(
            retrieval_state,  # type: ignore[arg-type]
            purpose="round1",
            final_top_k=15,
        )
    except Exception as e:
        logger.warning("RAG retrieval failed (%s), using LLM-only generation", e)
        retrieved = []

    user_content = (
        f"{_format_skill_matrix(state)}"
        f"{resume_context}"
        f"{match_context}\n\n"
        f"{_format_memory_context(state)}\n\n"
        f"{_format_retrieved_questions(retrieved)}\n\n"
        f"{_source_tracking_instruction()}\n\n"
        "请基于以上技能矩阵、候选人简历和参考题，生成 6-8 道一面技术深度题。"
        "题目正文、参考要点和追问方向必须使用简体中文；技术名词可以保留英文。"
        "至少一半题目应围绕候选人的项目和经历展开。"
        "请结合长期记忆回访薄弱技能、历史项目风险和曾经遗漏的面试点。"
    )

    llm = get_llm_client()
    plan = await llm.chat_structured(
        messages=[
            {"role": "system", "content": QUESTION_PLANNER_WITH_RESUME_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_model=QuestionPlan,
        temperature=0.5,
    )
    plan = _clean_question_sources(plan, retrieved)

    logger.info(
        "Resume-aware question plan ready: %d questions, est. %d min",
        len(plan.questions),
        plan.total_estimated_minutes,
    )

    result = {
        "question_plan": plan.questions,
        "current_question_index": 0,
        "follow_up_count": 0,
        "max_follow_ups": state.get("max_follow_ups", 2),
    }
    if resume_jd_match:
        result["resume_jd_match"] = resume_jd_match
    return result


def _format_round1_summary(state: InterviewState) -> str:
    """Deserialize and format the Round 1 summary for the Round 2 planner."""
    raw = state.get("round1_summary_text", "")
    if not raw:
        return "Round 1 summary not available."
    try:
        summary = Round1Summary.model_validate_json(raw)
    except Exception:
        return f"Round 1 summary (raw): {raw[:500]}"
    lines = [
        f"Round 1 Score: {summary.round1_score:.1f}/10 ({summary.round1_grade})",
        f"Technical Depth: {summary.technical_depth_assessment}",
        f"Project Understanding: {summary.project_understanding}",
        f"Strengths: {', '.join(summary.strengths_observed)}",
        f"Areas to Probe in Round 2: {', '.join(summary.areas_to_probe)}",
    ]
    return "\n".join(lines)


async def plan_questions_round2(state: InterviewState) -> dict:
    """LangGraph node: generate Round 2 (technical breadth) questions.

    Uses the Round 1 summary to target weak areas and broaden coverage.

    Reads:  skill_matrix, round1_summary_text
    Writes: question_plan, round2_question_plan, current_question_index,
            follow_up_count, max_follow_ups
    """
    skill_matrix = state["skill_matrix"]
    logger.info(
        "Planning Round 2 (breadth) questions for %d skills...",
        len(skill_matrix.skills),
    )

    try:
        retrieved = await _retrieve_reference_questions(state, purpose="round2", final_top_k=12)
    except Exception as e:
        logger.warning("RAG retrieval failed (%s), using LLM-only generation", e)
        retrieved = []

    round1_context = _format_round1_summary(state)

    user_content = (
        f"{_format_skill_matrix(state)}\n\n"
        f"## Round 1 Summary\n{round1_context}\n\n"
        f"{_format_memory_context(state)}\n\n"
        f"{_format_retrieved_questions(retrieved)}\n\n"
        f"{_source_tracking_instruction()}\n\n"
        "请基于以上技能矩阵、一面表现总结和参考题，生成 4-6 道二面技术广度题。"
        "题目正文、参考要点和追问方向必须使用简体中文；技术名词可以保留英文。"
        "重点考察技术广度、系统设计、架构取舍和新兴 AI 技术。"
        "长期记忆只作为个性化信号，本场一面证据优先级更高。"
    )

    llm = get_llm_client()
    plan = await llm.chat_structured(
        messages=[
            {"role": "system", "content": QUESTION_PLANNER_ROUND2_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_model=QuestionPlan,
        temperature=0.5,
    )
    plan = _clean_question_sources(plan, retrieved)

    logger.info(
        "Round 2 question plan ready: %d questions, est. %d min",
        len(plan.questions),
        plan.total_estimated_minutes,
    )

    return {
        "question_plan": plan.questions,
        "round2_question_plan": plan.questions,
        "current_question_index": 0,
        "follow_up_count": 0,
        "max_follow_ups": state.get("max_follow_ups", 2),
    }
