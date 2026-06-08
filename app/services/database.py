"""PostgreSQL connection lifecycle for durable application persistence."""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_pool: Any | None = None


def session_store_enabled() -> bool:
    return settings.session_store_backend.strip().lower() == "postgres"


def memory_store_enabled() -> bool:
    return settings.memory_store_backend.strip().lower() == "postgres"


def memory_vector_enabled() -> bool:
    return settings.memory_vector_backend.strip().lower() == "pgvector"


def postgres_enabled() -> bool:
    return session_store_enabled() or memory_store_enabled() or memory_vector_enabled()


async def init_database() -> Any | None:
    """Initialize PostgreSQL pool and durable application tables when enabled."""
    global _pool
    if not postgres_enabled():
        logger.info("Application database initialized: memory/sqlite/chroma")
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
    logger.info("Application database initialized: postgres")
    return _pool


def get_pool() -> Any | None:
    return _pool


async def setup_database() -> None:
    """Create durable application tables.

    LangGraph checkpoint tables are managed separately by AsyncPostgresSaver.setup().
    """
    if _pool is None:
        return
    async with _pool.connection() as conn:
        if memory_vector_enabled():
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                role TEXT NOT NULL DEFAULT 'user',
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_login_at TIMESTAMPTZ
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_users_created_at
            ON users (created_at DESC)
            """
        )

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

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                structured JSONB NOT NULL DEFAULT '{}'::jsonb,
                tags TEXT[] NOT NULL DEFAULT '{}'::text[],
                source TEXT NOT NULL,
                source_id TEXT,
                importance DOUBLE PRECISION NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_items_user_type_updated
            ON memory_items (user_id, memory_type, updated_at DESC)
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_items_user_tags
            ON memory_items USING GIN (tags)
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS skill_memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                skill_name TEXT NOT NULL,
                category TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                avg_score DOUBLE PRECISION NOT NULL,
                recent_score DOUBLE PRECISION NOT NULL,
                mastery_level TEXT NOT NULL,
                strengths JSONB NOT NULL DEFAULT '[]'::jsonb,
                weak_points JSONB NOT NULL DEFAULT '[]'::jsonb,
                evidence_memory_ids TEXT[] NOT NULL DEFAULT '{}'::text[],
                next_practice_priority DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                UNIQUE (user_id, skill_name)
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_skill_memories_user_priority
            ON skill_memories (user_id, next_practice_priority DESC, updated_at DESC)
            """
        )

        if memory_vector_enabled():
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT PRIMARY KEY REFERENCES memory_items(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    embedding vector NOT NULL,
                    embedding_model TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_embeddings_user_type
                ON memory_embeddings (user_id, memory_type)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_embeddings_user_updated
                ON memory_embeddings (user_id, updated_at DESC)
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
