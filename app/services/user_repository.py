"""PostgreSQL repository for application users."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.models.user import UserInDB
from app.services.database import get_pool


def _pool_or_raise():
    pool = get_pool()
    if pool is None:
        raise RuntimeError("User repository requires initialized PostgreSQL database pool.")
    return pool


def _row_to_user(row: dict[str, Any]) -> UserInDB:
    return UserInDB(
        id=row["id"],
        username=row["username"],
        password_hash=row["password_hash"],
        display_name=row.get("display_name") or "",
        role=row.get("role") or "user",
        is_active=bool(row.get("is_active", True)),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        last_login_at=row.get("last_login_at"),
    )


async def create_user(
    *,
    user_id: str,
    username: str,
    password_hash: str,
    display_name: str = "",
) -> UserInDB:
    pool = _pool_or_raise()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO users (
                id,
                username,
                password_hash,
                display_name
            )
            VALUES (%s, %s, %s, %s)
            RETURNING *
            """,
            (user_id, username, password_hash, display_name),
        )
        row = await cursor.fetchone()
    return _row_to_user(row)


async def get_user_by_username(username: str) -> UserInDB | None:
    pool = _pool_or_raise()
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT * FROM users WHERE lower(username) = lower(%s)",
            (username,),
        )
        row = await cursor.fetchone()
    return _row_to_user(row) if row else None


async def get_user_by_id(user_id: str) -> UserInDB | None:
    pool = _pool_or_raise()
    async with pool.connection() as conn:
        cursor = await conn.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        row = await cursor.fetchone()
    return _row_to_user(row) if row else None


async def update_last_login(user_id: str, when: datetime) -> None:
    pool = _pool_or_raise()
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE users
            SET last_login_at = %s, updated_at = NOW()
            WHERE id = %s
            """,
            (when, user_id),
        )
