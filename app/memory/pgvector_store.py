"""pgvector-backed semantic index for long-term memory items."""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.memory.models import MemoryItem
from app.rag.embeddings import EmbeddingService, get_embedding_service
from app.services.database import get_pool

logger = logging.getLogger(__name__)


def _pool_or_raise():
    pool = get_pool()
    if pool is None:
        raise RuntimeError("pgvector memory store requires initialized database pool.")
    return pool


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


class PgVectorMemoryVectorStore:
    """Semantic index for MemoryItem content using PostgreSQL pgvector."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self._embedding = embedding_service or get_embedding_service()

    async def upsert_item(self, item: MemoryItem) -> None:
        embedding = await self._embedding.embed_query(item.content)
        pool = _pool_or_raise()
        from psycopg.types.json import Jsonb

        async with pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO memory_embeddings (
                    memory_id,
                    user_id,
                    memory_type,
                    content,
                    metadata,
                    embedding,
                    embedding_model,
                    updated_at
                )
                VALUES (
                    %(memory_id)s,
                    %(user_id)s,
                    %(memory_type)s,
                    %(content)s,
                    %(metadata)s,
                    %(embedding)s::vector,
                    %(embedding_model)s,
                    %(updated_at)s
                )
                ON CONFLICT (memory_id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    memory_type = EXCLUDED.memory_type,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    updated_at = EXCLUDED.updated_at
                """,
                {
                    "memory_id": item.id,
                    "user_id": item.user_id,
                    "memory_type": str(item.memory_type),
                    "content": item.content,
                    "metadata": Jsonb(self._metadata(item)),
                    "embedding": _vector_literal(embedding),
                    "embedding_model": settings.embedding_model,
                    "updated_at": item.updated_at,
                },
            )

    async def semantic_search(
        self,
        user_id: str,
        query: str,
        memory_types: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict]:
        if not query.strip():
            return []

        query_embedding = await self._embedding.embed_query(query)
        params: dict[str, Any] = {
            "user_id": user_id,
            "query_embedding": _vector_literal(query_embedding),
            "limit": limit,
        }
        clauses = ["user_id = %(user_id)s"]
        if memory_types:
            clauses.append("memory_type = ANY(%(memory_types)s)")
            params["memory_types"] = memory_types

        pool = _pool_or_raise()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT
                    memory_id,
                    content,
                    metadata,
                    memory_type,
                    embedding <=> %(query_embedding)s::vector AS distance
                FROM memory_embeddings
                WHERE {' AND '.join(clauses)}
                ORDER BY embedding <=> %(query_embedding)s::vector
                LIMIT %(limit)s
                """,
                params,
            )
            rows = await cursor.fetchall()

        items: list[dict] = []
        for row in rows:
            distance = float(row["distance"])
            items.append(
                {
                    "id": row["memory_id"],
                    "document": row["content"],
                    "metadata": row.get("metadata") or {"memory_type": row["memory_type"]},
                    "distance": distance,
                    "score": 1.0 - distance,
                }
            )
        return items

    def _metadata(self, item: MemoryItem) -> dict:
        return {
            "user_id": item.user_id,
            "memory_type": str(item.memory_type),
            "source": item.source,
            "source_id": item.source_id or "",
            "tags_text": ",".join(item.tags),
            "importance": float(item.importance),
            "confidence": float(item.confidence),
            "updated_at": item.updated_at.isoformat(),
        }
