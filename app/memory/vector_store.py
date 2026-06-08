"""Semantic vector index for long-term memory items."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.memory.models import MemoryItem
from app.rag.embeddings import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)

MEMORY_COLLECTION_NAME = "long_term_memories"


class MemoryVectorStore:
    """ChromaDB-backed semantic index for MemoryItem content."""

    def __init__(
        self,
        persist_dir: str | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        persist_path = persist_dir or settings.chroma_persist_dir
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self._embedding = embedding_service or get_embedding_service()
        self._client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "MemoryVectorStore ready: collection '%s' has %d documents",
            MEMORY_COLLECTION_NAME,
            self._collection.count(),
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    async def upsert_item(self, item: MemoryItem) -> None:
        embedding = await self._embedding.embed_query(item.content)
        self._collection.upsert(
            ids=[item.id],
            documents=[item.content],
            embeddings=[embedding],
            metadatas=[self._metadata(item)],
        )

    async def semantic_search(
        self,
        user_id: str,
        query: str,
        memory_types: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict]:
        if not query.strip() or self.count == 0:
            return []

        query_embedding = await self._embedding.embed_query(query)
        fetch_n = max(limit * 4, limit)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_n,
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"],
        )

        items: list[dict] = []
        valid_types = set(memory_types or [])
        for i, memory_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i] or {}
            if valid_types and metadata.get("memory_type") not in valid_types:
                continue
            distance = float(results["distances"][0][i])
            items.append(
                {
                    "id": memory_id,
                    "document": results["documents"][0][i],
                    "metadata": metadata,
                    "distance": distance,
                    "score": 1.0 - distance,
                }
            )
            if len(items) >= limit:
                break
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


_memory_vector_store: MemoryVectorStore | None = None


def get_memory_vector_store() -> MemoryVectorStore:
    global _memory_vector_store
    if _memory_vector_store is None:
        backend = settings.memory_vector_backend.strip().lower()
        if backend in {"", "chroma"}:
            _memory_vector_store = MemoryVectorStore()
        elif backend == "pgvector":
            from app.memory.pgvector_store import PgVectorMemoryVectorStore

            _memory_vector_store = PgVectorMemoryVectorStore()  # type: ignore[assignment]
        else:
            raise ValueError(f"Unsupported MEMORY_VECTOR_BACKEND: {settings.memory_vector_backend}")
    return _memory_vector_store
