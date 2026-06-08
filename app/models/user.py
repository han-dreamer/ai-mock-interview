"""User and authentication models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class UserPublic(BaseModel):
    id: str
    username: str
    display_name: str = ""
    role: str = "user"
    created_at: datetime | None = None
    last_login_at: datetime | None = None


class UserInDB(UserPublic):
    password_hash: str
    is_active: bool = True
    updated_at: datetime | None = None


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)
    display_name: str = Field(default="", max_length=64)


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
