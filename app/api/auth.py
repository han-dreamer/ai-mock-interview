"""Authentication endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from psycopg.errors import UniqueViolation

from app.models.user import AuthTokenResponse, LoginRequest, RegisterRequest, UserInDB, UserPublic
from app.security import (
    create_access_token,
    hash_password,
    new_user_id,
    normalize_username,
    to_public_user,
    verify_password,
    get_current_user,
)
from app.services.user_repository import (
    create_user,
    get_user_by_username,
    update_last_login,
)

router = APIRouter()


@router.post("/register", response_model=AuthTokenResponse)
async def register(req: RegisterRequest):
    username = normalize_username(req.username)
    if not username:
        raise HTTPException(status_code=400, detail="Username is required.")

    existing = await get_user_by_username(username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already exists.")

    try:
        user = await create_user(
            user_id=new_user_id(),
            username=username,
            password_hash=hash_password(req.password),
            display_name=req.display_name.strip() or username,
        )
    except UniqueViolation as exc:
        raise HTTPException(status_code=409, detail="Username already exists.") from exc

    return AuthTokenResponse(
        access_token=create_access_token(user),
        user=to_public_user(user),
    )


@router.post("/login", response_model=AuthTokenResponse)
async def login(req: LoginRequest):
    username = normalize_username(req.username)
    user = await get_user_by_username(username)
    if not user or not user.is_active or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    await update_last_login(user.id, datetime.now(timezone.utc))
    user = await get_user_by_username(username) or user
    return AuthTokenResponse(
        access_token=create_access_token(user),
        user=to_public_user(user),
    )


@router.get("/me", response_model=UserPublic)
async def me(current_user: Annotated[UserInDB, Depends(get_current_user)]):
    return to_public_user(current_user)
