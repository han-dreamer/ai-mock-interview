"""SQLite-backed long-term memory storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.memory.models import MemoryItem, SkillMemory


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _from_json(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class MemoryStore:
    """Small persistent store for cross-session memories."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else settings.memory_db_file
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    structured_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_id TEXT,
                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_items_user_type_updated
                ON memory_items(user_id, memory_type, updated_at DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    avg_score REAL NOT NULL,
                    recent_score REAL NOT NULL,
                    mastery_level TEXT NOT NULL,
                    strengths_json TEXT NOT NULL,
                    weak_points_json TEXT NOT NULL,
                    evidence_memory_ids_json TEXT NOT NULL,
                    next_practice_priority REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, skill_name)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_skill_memories_user_priority
                ON skill_memories(user_id, next_practice_priority DESC, updated_at DESC)
                """
            )

    def upsert_memory_item(self, item: MemoryItem) -> MemoryItem:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_items (
                    id, user_id, memory_type, content, structured_json, tags_json,
                    source, source_id, importance, confidence, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content=excluded.content,
                    structured_json=excluded.structured_json,
                    tags_json=excluded.tags_json,
                    source=excluded.source,
                    source_id=excluded.source_id,
                    importance=excluded.importance,
                    confidence=excluded.confidence,
                    updated_at=excluded.updated_at
                """,
                (
                    item.id,
                    item.user_id,
                    str(item.memory_type),
                    item.content,
                    _to_json(item.structured),
                    _to_json(item.tags),
                    item.source,
                    item.source_id,
                    item.importance,
                    item.confidence,
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                ),
            )
        return item

    def list_memory_items(
        self,
        user_id: str,
        memory_types: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if memory_types:
            placeholders = ", ".join("?" for _ in memory_types)
            clauses.append(f"memory_type IN ({placeholders})")
            params.extend(memory_types)

        sql = (
            "SELECT * FROM memory_items "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY importance DESC, updated_at DESC "
            "LIMIT ?"
        )
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        items = [self._row_to_memory_item(row) for row in rows]
        if not tags:
            return items
        tag_set = {t.lower() for t in tags}
        return [
            item
            for item in items
            if tag_set.intersection({tag.lower() for tag in item.tags})
        ][:limit]

    def get_memory_item(self, memory_id: str) -> MemoryItem | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_items WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return self._row_to_memory_item(row) if row else None

    def get_skill_memory(self, user_id: str, skill_name: str) -> SkillMemory | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM skill_memories
                WHERE user_id = ? AND lower(skill_name) = lower(?)
                """,
                (user_id, skill_name),
            ).fetchone()
        return self._row_to_skill_memory(row) if row else None

    def upsert_skill_memory(self, memory: SkillMemory) -> SkillMemory:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO skill_memories (
                    id, user_id, skill_name, category, attempts, avg_score, recent_score,
                    mastery_level, strengths_json, weak_points_json,
                    evidence_memory_ids_json, next_practice_priority, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, skill_name) DO UPDATE SET
                    category=excluded.category,
                    attempts=excluded.attempts,
                    avg_score=excluded.avg_score,
                    recent_score=excluded.recent_score,
                    mastery_level=excluded.mastery_level,
                    strengths_json=excluded.strengths_json,
                    weak_points_json=excluded.weak_points_json,
                    evidence_memory_ids_json=excluded.evidence_memory_ids_json,
                    next_practice_priority=excluded.next_practice_priority,
                    updated_at=excluded.updated_at
                """,
                (
                    memory.id,
                    memory.user_id,
                    memory.skill_name,
                    memory.category,
                    memory.attempts,
                    memory.avg_score,
                    memory.recent_score,
                    str(memory.mastery_level),
                    _to_json(memory.strengths),
                    _to_json(memory.weak_points),
                    _to_json(memory.evidence_memory_ids),
                    memory.next_practice_priority,
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                ),
            )
        return memory

    def list_skill_memories(
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
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM skill_memories
                WHERE user_id = ?
                ORDER BY {order}
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [self._row_to_skill_memory(row) for row in rows]

    def _row_to_memory_item(self, row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            user_id=row["user_id"],
            memory_type=row["memory_type"],
            content=row["content"],
            structured=_from_json(row["structured_json"], {}),
            tags=_from_json(row["tags_json"], []),
            source=row["source"],
            source_id=row["source_id"],
            importance=row["importance"],
            confidence=row["confidence"],
            created_at=_parse_dt(row["created_at"]) or datetime.now(timezone.utc),
            updated_at=_parse_dt(row["updated_at"]) or datetime.now(timezone.utc),
        )

    def _row_to_skill_memory(self, row: sqlite3.Row) -> SkillMemory:
        return SkillMemory(
            id=row["id"],
            user_id=row["user_id"],
            skill_name=row["skill_name"],
            category=row["category"],
            attempts=row["attempts"],
            avg_score=row["avg_score"],
            recent_score=row["recent_score"],
            mastery_level=row["mastery_level"],
            strengths=_from_json(row["strengths_json"], []),
            weak_points=_from_json(row["weak_points_json"], []),
            evidence_memory_ids=_from_json(row["evidence_memory_ids_json"], []),
            next_practice_priority=row["next_practice_priority"],
            created_at=_parse_dt(row["created_at"]) or datetime.now(timezone.utc),
            updated_at=_parse_dt(row["updated_at"]) or datetime.now(timezone.utc),
        )


def get_memory_store():
    backend = settings.memory_store_backend.strip().lower()
    if backend in {"", "sqlite"}:
        return MemoryStore()
    if backend == "postgres":
        from app.memory.postgres_store import PostgresMemoryStore

        return PostgresMemoryStore()
    raise ValueError(f"Unsupported MEMORY_STORE_BACKEND: {settings.memory_store_backend}")
