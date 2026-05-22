"""Small post-processing helpers for retrieved RAG references."""

from __future__ import annotations

from copy import copy
import json
from typing import Any


def metadata_list(metadata: dict[str, Any], key: str) -> list[str]:
    """Read a list-like metadata value stored either as JSON or a native list."""
    value = metadata.get(key, [])
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        if parsed:
            return [str(parsed)]
    return []


def result_category(result: Any) -> str:
    return str(getattr(result, "metadata", {}).get("category", "") or "")


def result_difficulty(result: Any) -> str:
    return str(getattr(result, "metadata", {}).get("difficulty", "") or "")


def result_parent_id(result: Any) -> str:
    metadata = getattr(result, "metadata", {})
    return str(
        metadata.get("parent_id")
        or metadata.get("question_id")
        or getattr(result, "id", "")
    )


def result_chunk_id(result: Any) -> str:
    metadata = getattr(result, "metadata", {})
    return str(metadata.get("chunk_id") or getattr(result, "id", ""))


def deduplicate_results(results: list[Any]) -> list[Any]:
    seen: set[str] = set()
    unique: list[Any] = []
    for result in results:
        result_id = str(getattr(result, "id", ""))
        if not result_id or result_id in seen:
            continue
        seen.add(result_id)
        unique.append(result)
    return unique


def filter_results(
    results: list[Any],
    target_categories: set[str] | None = None,
    target_difficulties: set[str] | None = None,
) -> list[Any]:
    """Filter retrieved results by optional category and difficulty hints."""
    filtered: list[Any] = []
    for result in results:
        category = result_category(result)
        difficulty = result_difficulty(result)
        if target_categories and category not in target_categories:
            continue
        if target_difficulties and difficulty not in target_difficulties:
            continue
        filtered.append(result)
    return filtered


def diversify_results(
    results: list[Any],
    top_k: int,
    max_per_category: int = 6,
    max_per_difficulty: int = 8,
) -> list[Any]:
    """Keep high-scoring results while avoiding one category dominating the prompt."""
    unique = deduplicate_results(results)
    selected: list[Any] = []
    skipped: list[Any] = []
    category_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}

    for result in unique:
        category = result_category(result) or "unknown"
        difficulty = result_difficulty(result) or "unknown"
        if (
            category_counts.get(category, 0) >= max_per_category
            or difficulty_counts.get(difficulty, 0) >= max_per_difficulty
        ):
            skipped.append(result)
            continue
        selected.append(result)
        category_counts[category] = category_counts.get(category, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        if len(selected) >= top_k:
            return selected

    for result in skipped:
        if len(selected) >= top_k:
            break
        selected.append(result)
    return selected


def hydrate_parent_results(results: list[Any], top_k: int) -> list[Any]:
    """Aggregate retrieved child chunks back to parent question references."""
    groups: dict[str, dict[str, Any]] = {}

    for result in results:
        parent_id = result_parent_id(result)
        if not parent_id:
            continue

        group = groups.setdefault(
            parent_id,
            {
                "best": result,
                "score": 0.0,
                "chunk_ids": [],
                "chunk_types": [],
                "chunk_texts": [],
                "sources": set(),
            },
        )
        group["score"] += float(getattr(result, "score", 0.0))
        if float(getattr(result, "score", 0.0)) > float(getattr(group["best"], "score", 0.0)):
            group["best"] = result

        chunk_id = result_chunk_id(result)
        chunk_type = str(getattr(result, "metadata", {}).get("chunk_type", "") or "")
        chunk_text = str(getattr(result, "metadata", {}).get("chunk_text", "") or "")
        if chunk_id and chunk_id not in group["chunk_ids"]:
            group["chunk_ids"].append(chunk_id)
        if chunk_type and chunk_type not in group["chunk_types"]:
            group["chunk_types"].append(chunk_type)
        if chunk_text and chunk_text not in group["chunk_texts"]:
            group["chunk_texts"].append(chunk_text)
        source = str(getattr(result, "source", "") or "")
        if source:
            group["sources"].add(source)

    hydrated: list[Any] = []
    for parent_id, group in groups.items():
        best = copy(group["best"])
        best.id = parent_id
        best.score = group["score"]
        best.source = "+".join(sorted(group["sources"])) or getattr(best, "source", "")

        metadata = dict(getattr(best, "metadata", {}))
        metadata["parent_id"] = parent_id
        metadata["question_id"] = parent_id
        metadata["source_chunk_ids"] = json.dumps(group["chunk_ids"], ensure_ascii=False)
        metadata["source_chunk_types"] = json.dumps(group["chunk_types"], ensure_ascii=False)
        metadata["matched_chunk_texts"] = json.dumps(group["chunk_texts"][:5], ensure_ascii=False)
        metadata["retrieved_chunk_count"] = len(group["chunk_ids"])
        best.metadata = metadata
        best.content = metadata.get("content", getattr(best, "content", ""))
        hydrated.append(best)

    return sorted(hydrated, key=lambda item: item.score, reverse=True)[:top_k]


def source_category_map(results: list[Any]) -> dict[str, str]:
    """Map retrieved source ids to their categories for planner output cleanup."""
    mapping: dict[str, str] = {}
    for result in results:
        result_id = str(getattr(result, "id", ""))
        if result_id:
            mapping[result_id] = result_category(result)
    return mapping
