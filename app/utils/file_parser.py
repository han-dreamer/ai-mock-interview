"""Compatibility wrappers for resume file parsing."""

from __future__ import annotations

from pathlib import Path

from app.models.resume import ResumeParseResult
from app.resume.parser import (
    parse_image_with_vision as _parse_image_with_vision,
    parse_pdf as _parse_pdf,
    parse_resume_document,
)


async def parse_pdf(file_path: str | Path) -> str:
    """Extract text from a PDF resume using the parser backend."""
    raw_text, _metadata = await _parse_pdf(file_path)
    return raw_text


async def parse_image_with_vision(file_path: str | Path) -> str:
    """Extract resume text from an image using the parser backend."""
    raw_text, _metadata = await _parse_image_with_vision(file_path)
    return raw_text


async def parse_resume_with_metadata(file_path: str | Path) -> ResumeParseResult:
    """Parse a resume and return normalized text plus metadata."""
    return await parse_resume_document(file_path)


async def parse_resume_file(file_path: str | Path) -> str:
    """Auto-detect file type and return normalized resume text."""
    result = await parse_resume_document(file_path)
    return result.normalized_text
