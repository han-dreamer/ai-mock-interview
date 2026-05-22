"""Rule-based link and contact extraction for resumes."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from app.models.resume import ResumeLink

_URL_RE = re.compile(
    r"(?P<url>(?:https?://|www\.)[^\s<>()\[\]{}\"']+|"
    r"(?:github|gitee|gitlab|linkedin)\.com/[^\s<>()\[\]{}\"']+)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"(?P<email>[\w.+-]+@[\w-]+(?:\.[\w-]+)+)")
_PHONE_RE = re.compile(
    r"(?P<phone>(?:\+?86[- ]?)?1[3-9]\d{9}|(?:\+\d{1,3}[- ]?)?\d{3,4}[- ]?\d{4}[- ]?\d{4})"
)
_TRAILING_PUNCTUATION = ".,;:!?)]}>，。；：！？）】》"


def _clean_match(value: str) -> str:
    return value.strip().strip(_TRAILING_PUNCTUATION)


def _host(value: str) -> str:
    candidate = value if "://" in value else f"https://{value}"
    return urlparse(candidate).netloc.lower()


def classify_url(value: str) -> str:
    """Classify a URL-like value into resume-specific categories."""
    host = _host(value)
    lower_value = value.lower()

    if "github.com" in host:
        return "github"
    if "gitee.com" in host:
        return "gitee"
    if "gitlab.com" in host:
        return "gitlab"
    if "linkedin.com" in host:
        return "linkedin"
    if any(token in host for token in ("blog", "csdn", "juejin", "cnblogs", "medium")):
        return "blog"
    if any(token in lower_value for token in ("portfolio", "homepage", "个人主页")):
        return "portfolio"
    return "other"


def extract_resume_links(text: str) -> list[ResumeLink]:
    """Extract URLs, emails, and phone numbers with simple deterministic rules."""
    links: list[ResumeLink] = []
    seen: set[tuple[str, str]] = set()

    def add(value: str, link_type: str, source_text: str | None = None) -> None:
        value = _clean_match(value)
        if not value:
            return
        key = (link_type, value.lower())
        if key in seen:
            return
        seen.add(key)
        links.append(ResumeLink(url=value, type=link_type, source_text=source_text or value))

    for match in _URL_RE.finditer(text):
        value = _clean_match(match.group("url"))
        add(value, classify_url(value), match.group(0))

    for match in _EMAIL_RE.finditer(text):
        add(match.group("email"), "email", match.group(0))

    for match in _PHONE_RE.finditer(text):
        add(match.group("phone"), "phone", match.group(0))

    return links
