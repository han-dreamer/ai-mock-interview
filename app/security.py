"""Lightweight access controls for public trial deployments."""

from __future__ import annotations

from fastapi import Request

from app.config import settings


ACCESS_HEADER = "x-access-code"


def access_control_enabled() -> bool:
    return bool(settings.app_access_token.strip())


def is_valid_access_token(token: str | None) -> bool:
    expected = settings.app_access_token.strip()
    if not expected:
        return True
    return (token or "").strip() == expected


def token_from_request(request: Request) -> str:
    header_token = request.headers.get(ACCESS_HEADER, "")
    if header_token:
        return header_token
    authorization = request.headers.get("authorization", "")
    prefix = "Bearer "
    if authorization.startswith(prefix):
        return authorization[len(prefix):]
    return ""
