"""Text cleanup helpers for resume parsing."""

from __future__ import annotations

import re

MAX_RESUME_CHARS = 30000

_PAGE_PATTERNS = [
    re.compile(r"^\s*page\s+\d+\s+(of|/)\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*第\s*\d+\s*页\s*(共\s*\d+\s*页)?\s*$"),
    re.compile(r"^\s*[-_ ]*\d+\s*[-_ ]*$"),
]


def normalize_resume_text(text: str, max_chars: int = MAX_RESUME_CHARS) -> tuple[str, list[str]]:
    """Normalize extracted resume text while preserving section boundaries."""
    warnings: list[str] = []
    if not text:
        return "", ["No text was extracted from the resume."]

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\xa0", " ").replace("\u3000", " ")
    normalized = normalized.replace("\t", " ")

    cleaned_lines: list[str] = []
    blank_seen = False
    for raw_line in normalized.split("\n"):
        line = re.sub(r"[ ]{2,}", " ", raw_line).strip()
        if any(pattern.match(line) for pattern in _PAGE_PATTERNS):
            continue

        if not line:
            if not blank_seen:
                cleaned_lines.append("")
            blank_seen = True
            continue

        cleaned_lines.append(line)
        blank_seen = False

    normalized = "\n".join(cleaned_lines).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    if len(normalized) > max_chars:
        warnings.append(
            f"Resume text was truncated from {len(normalized)} to {max_chars} characters."
        )
        normalized = normalized[:max_chars].rstrip()

    if len(normalized) < 200:
        warnings.append(
            "Extracted resume text is short; the file may be scanned, image-heavy, or incomplete."
        )

    return normalized, warnings
