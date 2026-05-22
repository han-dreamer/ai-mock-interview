"""ChromaDB vector store wrapper for the interview question bank."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "interview_questions"


class VectorStore:
    """Thin wrapper around ChromaDB with persistent storage."""

    def __init__(self, persist_dir: str | None = None) -> None:
        persist_path = persist_dir or settings.chroma_persist_dir
        Path(persist_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready — collection '%s' has %d documents",
            COLLECTION_NAME,
            self._collection.count(),
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """Add documents with pre-computed embeddings."""
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info("Upserted %d documents into '%s'", len(ids), COLLECTION_NAME)

    def query_by_embedding(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """Query by embedding vector, return ranked results."""
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        items: list[dict] = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return items

    def query_by_text(
        self,
        query_text: str,
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """Query using ChromaDB's built-in embedding (if configured)."""
        kwargs: dict = {
            "query_texts": [query_text],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        items: list[dict] = []
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return items

    def get_all_documents(self) -> dict:
        """Retrieve all documents (for BM25 index building)."""
        return self._collection.get(include=["documents", "metadatas"])

    def delete_collection(self) -> None:
        """Drop and recreate the collection."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' reset", COLLECTION_NAME)


_default_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store
