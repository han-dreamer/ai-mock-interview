"""Parallel preparation node — runs resume analysis and JD analysis concurrently.

In professional mode, analyze_resume and analyze_jd are independent:
  - analyze_resume reads resume_text, writes resume_profile
  - analyze_jd reads jd_text, writes skill_matrix

Running them in parallel instead of sequentially saves ~15-30 seconds.
"""

from __future__ import annotations

import asyncio
import logging

from app.agents.jd_analyst import analyze_jd
from app.agents.resume_analyst import analyze_resume
from app.agents.state import InterviewState

logger = logging.getLogger(__name__)


async def parallel_analyze(state: InterviewState) -> dict:
    """LangGraph node: run resume analysis and JD analysis concurrently.

    Replaces the sequential analyze_resume → analyze_jd pipeline.
    """
    logger.info("Starting parallel analysis (resume + JD)...")

    resume_task = asyncio.create_task(analyze_resume(state))
    jd_task = asyncio.create_task(analyze_jd(state))

    resume_result, jd_result = await asyncio.gather(resume_task, jd_task)

    merged = {}
    merged.update(resume_result)
    merged.update(jd_result)

    logger.info(
        "Parallel analysis complete: resume_profile=%s, skill_matrix=%s",
        bool(merged.get("resume_profile")),
        bool(merged.get("skill_matrix")),
    )
    return merged
