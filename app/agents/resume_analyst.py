"""Resume Analyst Agent — parses raw resume text into a structured profile.

Used in professional mode to enable resume-based interview questions.
"""

from __future__ import annotations

import logging

from app.agents.state import InterviewState
from app.llm.client import get_llm_client
from app.llm.prompts import RESUME_ANALYST_SYSTEM
from app.models.resume import ResumeLink, ResumeParseResult, ResumeProfile

logger = logging.getLogger(__name__)


async def analyze_resume(state: InterviewState) -> dict:
    """LangGraph node: parse resume text into a structured ResumeProfile.

    Reads:  resume_text
    Writes: resume_profile
    """
    parse_result = state.get("resume_parse_result")
    if isinstance(parse_result, dict):
        parse_result = ResumeParseResult.model_validate(parse_result)

    resume_text = (
        parse_result.normalized_text
        if isinstance(parse_result, ResumeParseResult) and parse_result.normalized_text
        else state.get("resume_text", "")
    )
    if not resume_text:
        logger.warning("No resume text provided, creating empty profile")
        return {"resume_profile": ResumeProfile()}

    logger.info("Analyzing resume (%d chars)...", len(resume_text))

    parse_context = ""
    if isinstance(parse_result, ResumeParseResult):
        links_text = "\n".join(
            f"- {link.type}: {link.url}" for link in parse_result.links
        ) or "- None"
        warnings_text = (
            "\n".join(f"- {warning}" for warning in parse_result.warnings) or "- None"
        )
        parse_context = (
            "Parser metadata:\n"
            f"- file_name: {parse_result.metadata.file_name}\n"
            f"- file_type: {parse_result.metadata.file_type}\n"
            f"- parser: {parse_result.metadata.parser}\n"
            f"- raw_char_count: {parse_result.metadata.raw_char_count}\n"
            f"- normalized_char_count: {parse_result.metadata.normalized_char_count}\n\n"
            f"Deterministically extracted links/contact items:\n{links_text}\n\n"
            f"Parser warnings:\n{warnings_text}\n\n"
        )

    llm = get_llm_client()
    profile = await llm.chat_structured(
        messages=[
            {"role": "system", "content": RESUME_ANALYST_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"{parse_context}"
                    f"Please analyze the following resume and extract a structured profile:\n\n"
                    f"--- RESUME START ---\n{resume_text}\n--- RESUME END ---"
                ),
            },
        ],
        response_model=ResumeProfile,
        temperature=0.2,
    )

    if isinstance(parse_result, ResumeParseResult):
        existing_links = {(link.type, link.url.lower()) for link in profile.links}
        merged_links: list[ResumeLink] = list(profile.links)
        for link in parse_result.links:
            key = (link.type, link.url.lower())
            if key not in existing_links:
                merged_links.append(link)
                existing_links.add(key)
        profile.links = merged_links

        existing_warnings = set(profile.parse_warnings)
        for warning in parse_result.warnings:
            if warning not in existing_warnings:
                profile.parse_warnings.append(warning)
                existing_warnings.add(warning)

    logger.info(
        "Resume analysis complete: name=%s, %d skills, %d projects, %d experience items",
        profile.name,
        len(profile.skills),
        len(profile.projects),
        len(profile.experience),
    )

    return {"resume_profile": profile}
