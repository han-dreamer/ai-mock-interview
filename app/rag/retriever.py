"""Hybrid retriever combining vector search and BM25 keyword search with RRF fusion."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Any

from rank_bm25 import BM25Okapi

from app.rag.embeddings import EmbeddingService, get_embedding_service
from app.rag.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievedQuestion:
    """A single retrieved question with fusion score."""

    id: str
    content: str
    metadata: dict
    score: float
    source: str = ""  # "vector" | "bm25" | "hybrid"


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25: split on non-word chars, lowercase."""
    return [t.lower() for t in re.split(r"\W+", text) if t]


class HybridRetriever:
    """Two-stage retriever: vector + BM25, fused with Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ 1 / (k + rank_i(d))
    where k is a constant (default 60) and rank_i is the rank from source i.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        rrf_k: int = 60,
    ) -> None:
        self._store = vector_store or get_vector_store()
        self._embedding = embedding_service or get_embedding_service()
        self._rrf_k = rrf_k

        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict] = []
        self._bm25_built = False

    def build_bm25_index(self) -> None:
        """Build BM25 index from all documents in the vector store."""
        all_docs = self._store.get_all_documents()
        if not all_docs["ids"]:
            logger.warning("No documents in vector store, BM25 index empty")
            self._bm25 = None
            self._bm25_docs = []
            self._bm25_built = False
            return

        self._bm25_docs = []
        corpus: list[list[str]] = []

        for i, doc_id in enumerate(all_docs["ids"]):
            doc_text = all_docs["documents"][i]
            meta = all_docs["metadatas"][i]
            self._bm25_docs.append({
                "id": doc_id,
                "document": doc_text,
                "metadata": meta,
            })
            corpus.append(_tokenize(doc_text))

        self._bm25 = BM25Okapi(corpus)
        self._bm25_built = True
        logger.info("BM25 index built with %d documents", len(corpus))

    def _ensure_bm25_index(self) -> None:
        """Build BM25 once on first production retrieval."""
        if self._bm25_built or self._store.count == 0:
            return
        self.build_bm25_index()

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        where: dict | None = None,
    ) -> list[RetrievedQuestion]:
        """Hybrid retrieval with RRF fusion.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            vector_weight: Weight for vector search ranks in RRF.
            bm25_weight: Weight for BM25 ranks in RRF.
            where: Optional ChromaDB metadata filter.
        """
        fetch_n = top_k * 3
        self._ensure_bm25_index()

        # --- Vector search ---
        query_embedding = await self._embedding.embed_query(query)
        vector_results = self._store.query_by_embedding(
            query_embedding=query_embedding,
            n_results=fetch_n,
            where=where,
        )

        # --- BM25 search ---
        bm25_results = self._bm25_search(query, n_results=fetch_n, where=where)

        # --- RRF fusion ---
        fused = self._rrf_fuse(
            vector_results, bm25_results, vector_weight, bm25_weight
        )

        results = sorted(fused.values(), key=lambda x: x.score, reverse=True)[:top_k]
        logger.info(
            "Hybrid retrieval for '%s': %d results (vector=%d, bm25=%d)",
            query[:50],
            len(results),
            len(vector_results),
            len(bm25_results),
        )
        return results

    async def retrieve_multi(
        self,
        queries: list[Any],
        top_k_per_query: int = 8,
        final_top_k: int = 15,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        where: dict | None = None,
    ) -> list[RetrievedQuestion]:
        """Run multiple retrieval queries and fuse duplicate references.

        This keeps the implementation lightweight: each query uses the existing
        hybrid retriever, then duplicate question ids receive accumulated scores.
        """
        fused: dict[str, RetrievedQuestion] = {}
        for query in queries:
            text, priority = self._query_text_and_priority(query)
            if not text:
                continue
            results = await self.retrieve(
                text,
                top_k=top_k_per_query,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                where=where,
            )
            for item in results:
                weighted_score = item.score * priority
                if item.id not in fused:
                    fused[item.id] = replace(item, score=weighted_score)
                    continue
                fused[item.id].score += weighted_score
                if item.source not in fused[item.id].source.split("+"):
                    fused[item.id].source = f"{fused[item.id].source}+{item.source}"

        results = sorted(fused.values(), key=lambda x: x.score, reverse=True)[:final_top_k]
        logger.info(
            "Multi-query retrieval: %d queries, %d fused results",
            len(queries),
            len(results),
        )
        return results

    async def retrieve_vector_only(
        self,
        query: str,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[RetrievedQuestion]:
        """Pure vector search (fallback when BM25 index is empty)."""
        query_embedding = await self._embedding.embed_query(query)
        results = self._store.query_by_embedding(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where,
        )
        return [
            RetrievedQuestion(
                id=r["id"],
                content=r["document"],
                metadata=r["metadata"],
                score=1.0 - r["distance"],
                source="vector",
            )
            for r in results
        ]

    def _bm25_search(
        self,
        query: str,
        n_results: int,
        where: dict | None = None,
    ) -> list[dict]:
        """BM25 keyword search."""
        if self._bm25 is None or not self._bm25_docs:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                break
            doc = self._bm25_docs[idx]
            if where and not _metadata_matches_where(doc["metadata"], where):
                continue
            results.append({
                "id": doc["id"],
                "document": doc["document"],
                "metadata": doc["metadata"],
                "bm25_score": float(scores[idx]),
            })
            if len(results) >= n_results:
                break
        return results

    def _rrf_fuse(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        vector_weight: float,
        bm25_weight: float,
    ) -> dict[str, RetrievedQuestion]:
        """Reciprocal Rank Fusion across two result lists."""
        k = self._rrf_k
        fused: dict[str, RetrievedQuestion] = {}

        for rank, item in enumerate(vector_results):
            doc_id = item["id"]
            rrf_score = vector_weight / (k + rank + 1)
            if doc_id not in fused:
                fused[doc_id] = RetrievedQuestion(
                    id=doc_id,
                    content=item["document"],
                    metadata=item["metadata"],
                    score=0.0,
                    source="vector",
                )
            fused[doc_id].score += rrf_score

        for rank, item in enumerate(bm25_results):
            doc_id = item["id"]
            rrf_score = bm25_weight / (k + rank + 1)
            if doc_id not in fused:
                fused[doc_id] = RetrievedQuestion(
                    id=doc_id,
                    content=item["document"],
                    metadata=item["metadata"],
                    score=0.0,
                    source="bm25",
                )
                fused[doc_id].score += rrf_score
            else:
                fused[doc_id].score += rrf_score
                fused[doc_id].source = "hybrid"

        return fused

    @staticmethod
    def _query_text_and_priority(query: Any) -> tuple[str, float]:
        if hasattr(query, "text"):
            return str(query.text), float(getattr(query, "priority", 1.0))
        return str(query), 1.0


def _metadata_matches_where(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    """Small subset of Chroma-style metadata filters for BM25 parity."""
    for key, expected in where.items():
        actual = metadata.get(key)
        if isinstance(expected, dict):
            if "$eq" in expected and actual != expected["$eq"]:
                return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            if "$in" in expected and actual not in expected["$in"]:
                return False
            if "$nin" in expected and actual in expected["$nin"]:
                return False
            continue
        if actual != expected:
            return False
    return True


_default_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = HybridRetriever()
    return _default_retriever
