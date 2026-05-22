"""Shared helpers for turning retrieved RAG results into grounded context text."""

from __future__ import annotations

from typing import Any

from app.rag.postprocess import metadata_list, result_parent_id


def build_retrieval_context(
    result: Any,
    *,
    max_reference_points: int = 5,
    max_follow_ups: int = 3,
    max_source_chunk_ids: int = 5,
    max_matched_chunks: int = 5,
) -> str:
    """Serialize a retrieved question into a compact, evidence-rich context block."""

    metadata = getattr(result, "metadata", {})
    content = str(metadata.get("content") or getattr(result, "content", "") or "")
    lines: list[str] = [
        f"Source ID: {result_parent_id(result)}",
        f"Category: {metadata.get('category', '')}",
        f"Difficulty: {metadata.get('difficulty', '')}",
    ]

    skill_tags = metadata_list(metadata, "skill_tags")
    if skill_tags:
        lines.append("Skill tags: " + "; ".join(skill_tags))

    chunk_type = str(metadata.get("chunk_type") or "")
    if chunk_type:
        lines.append(f"Chunk type: {chunk_type}")

    chunk_strategy = str(metadata.get("chunk_strategy") or "")
    if chunk_strategy:
        lines.append(f"Chunk strategy: {chunk_strategy}")
        lines.append(
            "Context header purpose: category, difficulty, skill tags, chunk type, "
            "and parent question make isolated child chunks easier for embedding "
            "and keyword retrieval to interpret."
        )

    if content:
        lines.append(f"Question or topic: {content}")

    reference_points = metadata_list(metadata, "reference_points")
    if reference_points:
        lines.append("Reference points: " + "; ".join(reference_points[:max_reference_points]))

    follow_ups = metadata_list(metadata, "follow_up_directions")
    if follow_ups:
        lines.append("Follow-up directions: " + "; ".join(follow_ups[:max_follow_ups]))

    source_chunk_ids = metadata_list(metadata, "source_chunk_ids")
    if source_chunk_ids:
        lines.append("Source chunk ids: " + "; ".join(source_chunk_ids[:max_source_chunk_ids]))

    source_chunk_types = metadata_list(metadata, "source_chunk_types")
    if source_chunk_types:
        lines.append("Source chunk types: " + "; ".join(source_chunk_types))

    matched_chunk_texts = metadata_list(metadata, "matched_chunk_texts")
    if matched_chunk_texts:
        lines.append("Matched chunks: " + "; ".join(matched_chunk_texts[:max_matched_chunks]))

    retrieved_chunk_count = metadata.get("retrieved_chunk_count")
    if retrieved_chunk_count is not None:
        lines.append(f"Retrieved chunk count: {retrieved_chunk_count}")

    return "\n".join(lines)
