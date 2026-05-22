"""Quick verification script for the hybrid retrieval system.

Usage:
    python -m scripts.test_retrieval
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.retriever import HybridRetriever  # noqa: E402
from app.rag.vector_store import get_vector_store  # noqa: E402
from app.rag.embeddings import get_embedding_service  # noqa: E402


TEST_QUERIES = [
    "Python 异步编程 asyncio",
    "分布式系统设计 缓存",
    "RAG 检索增强生成",
    "LangGraph Agent 框架",
    "Docker 容器化部署",
]


async def main() -> None:
    store = get_vector_store()
    print(f"\nVector store has {store.count} documents\n")

    if store.count == 0:
        print("ERROR: Vector store is empty. Run 'python -m scripts.init_vector_store' first.")
        return

    retriever = HybridRetriever(
        vector_store=store,
        embedding_service=get_embedding_service(),
    )
    retriever.build_bm25_index()

    for query in TEST_QUERIES:
        print(f"{'=' * 70}")
        print(f"Query: {query}")
        print(f"{'=' * 70}")

        results = await retriever.retrieve(query, top_k=5)
        for i, r in enumerate(results, 1):
            print(f"  #{i} [{r.source:6s}] (score={r.score:.4f}) [{r.metadata.get('category', '?')}] "
                  f"{r.metadata.get('content', r.content[:80])}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
