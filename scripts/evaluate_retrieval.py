"""Lightweight retrieval evaluation for the question-bank RAG system.

Usage:
    python -m scripts.evaluate_retrieval
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.postprocess import hydrate_parent_results, result_parent_id  # noqa: E402
from app.rag.retriever import RetrievedQuestion, get_retriever  # noqa: E402
from app.rag.vector_store import get_vector_store  # noqa: E402

GOLDEN_PATH = PROJECT_ROOT / "data" / "eval" / "retrieval_golden.json"


def load_cases(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def metric_row(results: list[RetrievedQuestion], expected_ids: list[str]) -> dict[str, float]:
    retrieved_ids = _unique_parent_ids(results)
    expected = set(expected_ids)
    top5 = retrieved_ids[:5]
    top10 = retrieved_ids[:10]

    hit_at_5 = 1.0 if expected.intersection(top5) else 0.0
    recall_at_5 = len(expected.intersection(top5)) / max(len(expected), 1)
    mrr_at_10 = 0.0
    for rank, doc_id in enumerate(top10, 1):
        if doc_id in expected:
            mrr_at_10 = 1.0 / rank
            break

    return {
        "hit@5": hit_at_5,
        "recall@5": recall_at_5,
        "mrr@10": mrr_at_10,
    }


def _unique_parent_ids(results: list[RetrievedQuestion]) -> list[str]:
    seen: set[str] = set()
    parent_ids: list[str] = []
    for result in results:
        parent_id = result_parent_id(result)
        if parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent_ids.append(parent_id)
    return parent_ids


def print_summary(name: str, rows: list[dict[str, float]]) -> None:
    print(f"\n{name}")
    print("-" * len(name))
    for metric in ("hit@5", "recall@5", "mrr@10"):
        print(f"{metric:9s}: {mean(row[metric] for row in rows):.3f}")


async def evaluate() -> None:
    store = get_vector_store()
    if store.count == 0:
        print("Vector store is empty. Run: python -m scripts.init_vector_store")
        return

    retriever = get_retriever()
    cases = load_cases()
    vector_rows: list[dict[str, float]] = []
    hybrid_rows: list[dict[str, float]] = []
    multi_rows: list[dict[str, float]] = []
    hydrated_rows: list[dict[str, float]] = []

    print(f"Loaded {len(cases)} golden retrieval cases")
    print(f"Vector store documents: {store.count}\n")
    print(f"{'case':28s} {'vector':>10s} {'hybrid':>10s} {'multi':>10s} {'parent':>10s}")
    print("-" * 76)

    for case in cases:
        query = case["query"]
        queries = case.get("queries") or [query]
        expected_ids = case["expected_ids"]

        vector_results = await retriever.retrieve_vector_only(query, top_k=10)
        hybrid_results = await retriever.retrieve(query, top_k=10)
        multi_results = await retriever.retrieve_multi(
            queries,
            top_k_per_query=8,
            final_top_k=10,
        )
        hydrated_results = hydrate_parent_results(multi_results, top_k=10)

        vector_metric = metric_row(vector_results, expected_ids)
        hybrid_metric = metric_row(hybrid_results, expected_ids)
        multi_metric = metric_row(multi_results, expected_ids)
        hydrated_metric = metric_row(hydrated_results, expected_ids)
        vector_rows.append(vector_metric)
        hybrid_rows.append(hybrid_metric)
        multi_rows.append(multi_metric)
        hydrated_rows.append(hydrated_metric)

        print(
            f"{case['name'][:28]:28s} "
            f"{vector_metric['hit@5']:10.0f} "
            f"{hybrid_metric['hit@5']:10.0f} "
            f"{multi_metric['hit@5']:10.0f} "
            f"{hydrated_metric['hit@5']:10.0f}"
        )

    print_summary("Vector-only", vector_rows)
    print_summary("Hybrid", hybrid_rows)
    print_summary("Multi-query hybrid", multi_rows)
    print_summary("Parent-hydrated multi-query hybrid", hydrated_rows)


if __name__ == "__main__":
    asyncio.run(evaluate())
