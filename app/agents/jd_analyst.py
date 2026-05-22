"""JD Analyst Agent — parses a job description into a structured skill matrix."""

from __future__ import annotations

import logging

from app.agents.state import InterviewState
from app.llm.client import get_llm_client
from app.llm.prompts import JD_ANALYST_SYSTEM
from app.models.jd import SkillMatrix

logger = logging.getLogger(__name__)


async def analyze_jd(state: InterviewState) -> dict:
    """LangGraph node: analyze JD text and produce a SkillMatrix.

    Reads:  state["jd_text"]
    Writes: state["skill_matrix"]
    """
    jd_text = state["jd_text"]
    logger.info("Analyzing JD (%d chars)...", len(jd_text))

    llm = get_llm_client()
    skill_matrix = await llm.chat_structured(
        messages=[
            {"role": "system", "content": JD_ANALYST_SYSTEM},
            {"role": "user", "content": f"Please analyze the following job description:\n\n{jd_text}"},
        ],
        response_model=SkillMatrix,
        temperature=0.3,
    )

    logger.info(
        "JD analysis complete: position='%s', level='%s', %d skills extracted",
        skill_matrix.position_title,
        skill_matrix.experience_level,
        len(skill_matrix.skills),
    )
    for s in skill_matrix.skills:
        logger.debug(
            "  - %s (%s, weight=%.2f, required=%s)",
            s.name, s.category, s.weight, s.is_required,
        )

    return {"skill_matrix": skill_matrix}
