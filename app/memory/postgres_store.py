"""PostgreSQL-backed long-term memory storage."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from app.memory.models import MemoryItem, SkillMemory
from app.services.database import get_pool


def _run_sync(awaitable):
    """Run a small async DB operation from the existing synchronous memory API."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError(
        "PostgresMemoryStore synchronous methods cannot run inside an active event loop. "
        "Use async wrappers before enabling this backend in that path."
    )


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _pool_or_raise():
    pool = get_pool()
    if pool is None:
        raise RuntimeError("PostgreSQL memory store requires initialized database pool.")
    return pool


class PostgresMemoryStore:
    """Persistent store for cross-session memories in PostgreSQL."""

    def upsert_memory_item(self, item: MemoryItem) -> MemoryItem:
        return _run_sync(self.aupsert_memory_item(item))

    async def aupsert_memory_item(self, item: MemoryItem) -> MemoryItem:
        from psycopg.types.json import Jsonb

        pool = _pool_or_raise()
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO memory_items (
                    id,
                    user_id,
                    memory_type,
                    content,
                    structured,
                    tags,
                    source,
                    source_id,
                    importance,
                    confidence,
                    created_at,
                    updated_at
                )
                VALUES (
                    %(id)s,
                    %(user_id)s,
                    %(memory_type)s,
                    %(content)s,
                    %(structured)s,
                    %(tags)s,
                    %(source)s,
                    %(source_id)s,
                    %(importance)s,
                    %(confidence)s,
                    %(created_at)s,
                    %(updated_at)s
                )
                ON CONFLICT (id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    memory_type = EXCLUDED.memory_type,
                    content = EXCLUDED.content,
                    structured = EXCLUDED.structured,
                    tags = EXCLUDED.tags,
                    source = EXCLUDED.source,
                    source_id = EXCLUDED.source_id,
                    importance = EXCLUDED.importance,
                    confidence = EXCLUDED.confidence,
                    updated_at = EXCLUDED.updated_at
                """,
                {
                    "id": item.id,
                    "user_id": item.user_id,
                    "memory_type": str(item.memory_type),
                    "content": item.content,
                    "structured": Jsonb(item.structured),
                    "tags": list(item.tags),
                    "source": item.source,
                    "source_id": item.source_id,
                    "importance": item.importance,
                    "confidence": item.confidence,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                },
            )
        return item

    def list_memory_items(
        self,
        user_id: str,
        memory_types: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        return _run_sync(self.alist_memory_items(user_id, memory_types, tags, limit))

    async def alist_memory_items(
        self,
        user_id: str,
        memory_types: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        clauses = ["user_id = %(user_id)s"]
        params: dict[str, Any] = {"user_id": user_id, "limit": limit}
        if memory_types:
            clauses.append("memory_type = ANY(%(memory_types)s)")
            params["memory_types"] = memory_types
        if tags:
            clauses.append("tags && %(tags)s")
            params["tags"] = tags

        pool = _pool_or_raise()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT *
                FROM memory_items
                WHERE {' AND '.join(clauses)}
                ORDER BY importance DESC, updated_at DESC
                LIMIT %(limit)s
                """,
                params,
            )
            rows = await cursor.fetchall()
        return [self._row_to_memory_item(row) for row in rows]

    def get_memory_item(self, memory_id: str) -> MemoryItem | None:
        return _run_sync(self.aget_memory_item(memory_id))

    async def aget_memory_item(self, memory_id: str) -> MemoryItem | None:
        pool = _pool_or_raise()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM memory_items WHERE id = %s",
                (memory_id,),
            )
            row = await cursor.fetchone()
        return self._row_to_memory_item(row) if row else None

    def get_skill_memory(self, user_id: str, skill_name: str) -> SkillMemory | None:
        return _run_sync(self.aget_skill_memory(user_id, skill_name))

    async def aget_skill_memory(self, user_id: str, skill_name: str) -> SkillMemory | None:
        pool = _pool_or_raise()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT *
                FROM skill_memories
                WHERE user_id = %s AND lower(skill_name) = lower(%s)
                """,
                (user_id, skill_name),
            )
            row = await cursor.fetchone()
        return self._row_to_skill_memory(row) if row else None

    def upsert_skill_memory(self, memory: SkillMemory) -> SkillMemory:
        return _run_sync(self.aupsert_skill_memory(memory))

    async def aupsert_skill_memory(self, memory: SkillMemory) -> SkillMemory:
        from psycopg.types.json import Jsonb

        pool = _pool_or_raise()
        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO skill_memories (
                    id,
                    user_id,
                    skill_name,
                    category,
                    attempts,
                    avg_score,
                    recent_score,
                    mastery_level,
                    strengths,
                    weak_points,
                    evidence_memory_ids,
                    next_practice_priority,
                    created_at,
                    updated_at
                )
                VALUES (
                    %(id)s,
                    %(user_id)s,
                    %(skill_name)s,
                    %(category)s,
                    %(attempts)s,
                    %(avg_score)s,
                    %(recent_score)s,
                    %(mastery_level)s,
                    %(strengths)s,
                    %(weak_points)s,
                    %(evidence_memory_ids)s,
                    %(next_practice_priority)s,
                    %(created_at)s,
                    %(updated_at)s
                )
                ON CONFLICT (user_id, skill_name) DO UPDATE SET
                    category = EXCLUDED.category,
                    attempts = EXCLUDED.attempts,
                    avg_score = EXCLUDED.avg_score,
                    recent_score = EXCLUDED.recent_score,
                    mastery_level = EXCLUDED.mastery_level,
                    strengths = EXCLUDED.strengths,
                    weak_points = EXCLUDED.weak_points,
                    evidence_memory_ids = EXCLUDED.evidence_memory_ids,
                    next_practice_priority = EXCLUDED.next_practice_priority,
                    updated_at = EXCLUDED.updated_at
                """,
                {
                    "id": memory.id,
                    "user_id": memory.user_id,
                    "skill_name": memory.skill_name,
                    "category": memory.category,
                    "attempts": memory.attempts,
                    "avg_score": memory.avg_score,
                    "recent_score": memory.recent_score,
                    "mastery_level": str(memory.mastery_level),
                    "strengths": Jsonb(memory.strengths),
                    "weak_points": Jsonb(memory.weak_points),
                    "evidence_memory_ids": list(memory.evidence_memory_ids),
                    "next_practice_priority": memory.next_practice_priority,
                    "created_at": memory.created_at,
                    "updated_at": memory.updated_at,
                },
            )
        return memory

    def list_skill_memories(
        self,
        user_id: str,
        limit: int = 10,
        weak_first: bool = True,
    ) -> list[SkillMemory]:
        return _run_sync(self.alist_skill_memories(user_id, limit, weak_first))

    async def alist_skill_memories(
        self,
        user_id: str,
        limit: int = 10,
        weak_first: bool = True,
    ) -> list[SkillMemory]:
        order = (
            "next_practice_priority DESC, recent_score ASC, updated_at DESC"
            if weak_first
            else "updated_at DESC"
        )
        pool = _pool_or_raise()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT *
                FROM skill_memories
                WHERE user_id = %s
                ORDER BY {order}
                LIMIT %s
                """,
                (user_id, limit),
            )
            rows = await cursor.fetchall()
        return [self._row_to_skill_memory(row) for row in rows]

    def _row_to_memory_item(self, row: dict[str, Any]) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            user_id=row["user_id"],
            memory_type=row["memory_type"],
            content=row["content"],
            structured=row.get("structured") or {},
            tags=list(row.get("tags") or []),
            source=row["source"],
            source_id=row.get("source_id"),
            importance=row["importance"],
            confidence=row["confidence"],
            created_at=_parse_dt(row.get("created_at")),
            updated_at=_parse_dt(row.get("updated_at")),
        )

    def _row_to_skill_memory(self, row: dict[str, Any]) -> SkillMemory:
        return SkillMemory(
            id=row["id"],
            user_id=row["user_id"],
            skill_name=row["skill_name"],
            category=row["category"],
            attempts=row["attempts"],
            avg_score=row["avg_score"],
            recent_score=row["recent_score"],
            mastery_level=row["mastery_level"],
            strengths=list(row.get("strengths") or []),
            weak_points=list(row.get("weak_points") or []),
            evidence_memory_ids=list(row.get("evidence_memory_ids") or []),
            next_practice_priority=row["next_practice_priority"],
            created_at=_parse_dt(row.get("created_at")),
            updated_at=_parse_dt(row.get("updated_at")),
        )
