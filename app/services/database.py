"""PostgreSQL connection lifecycle for business persistence."""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_pool: Any | None = None


def session_store_enabled() -> bool:
    return settings.session_store_backend.strip().lower() == "postgres"


async def init_database() -> Any | None:
    """Initialize PostgreSQL pool and business tables when enabled."""
    global _pool
    if not session_store_enabled():
        logger.info("Business session store initialized: memory")
        return None
    if _pool is not None:
        return _pool

    try:
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
    except ImportError as exc:
        raise RuntimeError(
            "SESSION_STORE_BACKEND=postgres requires psycopg[binary,pool]."
        ) from exc

    _pool = AsyncConnectionPool(
        conninfo=settings.postgres_url,
        min_size=settings.postgres_pool_min_size,
        max_size=settings.postgres_pool_max_size,
        kwargs={"autocommit": True, "row_factory": dict_row},
        open=False,
    )
    await _pool.open()
    await setup_database()
    logger.info("Business session store initialized: postgres")
    return _pool


def get_pool() -> Any | None:
    return _pool


async def setup_database() -> None:
    """Create business persistence tables.

    LangGraph checkpoint tables are managed separately by AsyncPostgresSaver.setup().
    """
    if _pool is None:
        return
    async with _pool.connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interview_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                jd_text TEXT NOT NULL,
                max_follow_ups INTEGER NOT NULL,
                status TEXT NOT NULL,
                current_question_index INTEGER NOT NULL DEFAULT 0,
                follow_up_count INTEGER NOT NULL DEFAULT 0,
                graph_started BOOLEAN NOT NULL DEFAULT FALSE,
                resume_text TEXT NOT NULL DEFAULT '',
                resume_parse_result JSONB,
                conversation_history JSONB NOT NULL DEFAULT '[]'::jsonb,
                assessments JSONB NOT NULL DEFAULT '[]'::jsonb,
                last_state JSONB NOT NULL DEFAULT '{}'::jsonb,
                persisted_assessment_count INTEGER NOT NULL DEFAULT 0,
                final_memory_saved BOOLEAN NOT NULL DEFAULT FALSE,
                error_message TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMPTZ
            )
            """
        )
        await conn.execute(
            """
            ALTER TABLE interview_sessions
            ADD COLUMN IF NOT EXISTS persisted_assessment_count INTEGER NOT NULL DEFAULT 0
            """
        )
        await conn.execute(
            """
            ALTER TABLE interview_sessions
            ADD COLUMN IF NOT EXISTS final_memory_saved BOOLEAN NOT NULL DEFAULT FALSE
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interview_sessions_user_updated
            ON interview_sessions (user_id, updated_at DESC)
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_interview_sessions_status
            ON interview_sessions (status)
            """
        )


async def close_database() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
    _pool = None


def reset_database_for_tests() -> None:
    global _pool
    _pool = None
