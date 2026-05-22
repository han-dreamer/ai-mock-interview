"""Structure-aware chunking for the interview question bank."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RagChunk:
    """A child chunk indexed for retrieval, with parent question metadata."""

    id: str
    text: str
    metadata: dict[str, Any]


def build_question_chunks(
    question: dict[str, Any],
    category: str,
    source_file: str,
) -> list[RagChunk]:
    """Split one structured question record into retrieval-friendly chunks.

    The retrieval unit is a child chunk, while the generation unit remains the
    parent question. This improves recall without losing the full question
    context when the planner consumes retrieved references.
    """

    question_id = str(question["id"])
    content = str(question.get("content", ""))
    skill_tags = [str(tag) for tag in question.get("skill_tags", [])]
    difficulty = str(question.get("difficulty", "medium"))
    reference_points = [str(item) for item in question.get("reference_points", [])]
    follow_ups = [str(item) for item in question.get("follow_up_directions", [])]

    base_metadata = {
        "parent_id": question_id,
        "question_id": question_id,
        "category": category,
        "difficulty": difficulty,
        "skill_tags": json.dumps(skill_tags, ensure_ascii=False),
        "content": content,
        "reference_points": json.dumps(reference_points, ensure_ascii=False),
        "follow_up_directions": json.dumps(follow_ups, ensure_ascii=False),
        "source_file": source_file,
        "chunk_strategy": "parent_child_structure_aware_v1",
    }

    chunks: list[RagChunk] = []
    chunks.append(
        _make_chunk(
            question_id=question_id,
            chunk_type="question_stem",
            text=content,
            category=category,
            skill_tags=skill_tags,
            difficulty=difficulty,
            base_metadata=base_metadata,
        )
    )

    summary_text = _question_summary(content, reference_points, follow_ups)
    if summary_text:
        chunks.append(
            _make_chunk(
                question_id=question_id,
                chunk_type="question_summary",
                text=summary_text,
                category=category,
                skill_tags=skill_tags,
                difficulty=difficulty,
                base_metadata=base_metadata,
            )
        )

    for index, point in enumerate(reference_points, 1):
        chunks.append(
            _make_chunk(
                question_id=question_id,
                chunk_type="answer_point",
                text=point,
                category=category,
                skill_tags=skill_tags,
                difficulty=difficulty,
                base_metadata=base_metadata,
                index=index,
            )
        )

    for index, follow_up in enumerate(follow_ups, 1):
        chunks.append(
            _make_chunk(
                question_id=question_id,
                chunk_type="follow_up",
                text=follow_up,
                category=category,
                skill_tags=skill_tags,
                difficulty=difficulty,
                base_metadata=base_metadata,
                index=index,
            )
        )

    for index, generated_query in enumerate(_generated_queries(content, skill_tags, category), 1):
        chunks.append(
            _make_chunk(
                question_id=question_id,
                chunk_type="generated_query",
                text=generated_query,
                category=category,
                skill_tags=skill_tags,
                difficulty=difficulty,
                base_metadata=base_metadata,
                index=index,
            )
        )

    return chunks


def _make_chunk(
    question_id: str,
    chunk_type: str,
    text: str,
    category: str,
    skill_tags: list[str],
    difficulty: str,
    base_metadata: dict[str, Any],
    index: int | None = None,
) -> RagChunk:
    chunk_id = _chunk_id(question_id, chunk_type, index)
    metadata = dict(base_metadata)
    metadata.update({
        "chunk_id": chunk_id,
        "chunk_type": chunk_type,
        "chunk_index": index or 0,
        "chunk_text": text,
    })
    return RagChunk(
        id=chunk_id,
        text=_contextualize_chunk(
            text=text,
            question_id=question_id,
            chunk_type=chunk_type,
            category=category,
            skill_tags=skill_tags,
            difficulty=difficulty,
            parent_question=base_metadata["content"],
        ),
        metadata=metadata,
    )


def _contextualize_chunk(
    text: str,
    question_id: str,
    chunk_type: str,
    category: str,
    skill_tags: list[str],
    difficulty: str,
    parent_question: str,
) -> str:
    """Add a compact context header so each child chunk is self-describing."""
    return "\n".join([
        f"Question ID: {question_id}",
        f"Category: {category}",
        f"Difficulty: {difficulty}",
        f"Skill tags: {', '.join(skill_tags)}",
        f"Chunk type: {chunk_type}",
        f"Parent question: {parent_question}",
        f"Chunk text: {text}",
    ])


def _question_summary(
    content: str,
    reference_points: list[str],
    follow_ups: list[str],
) -> str:
    parts = [content]
    if reference_points:
        parts.append("Key answer points: " + "; ".join(reference_points[:6]))
    if follow_ups:
        parts.append("Follow-up directions: " + "; ".join(follow_ups[:3]))
    return "\n".join(part for part in parts if part)


def _generated_queries(
    content: str,
    skill_tags: list[str],
    category: str,
) -> list[str]:
    skills = ", ".join(skill_tags[:4]) if skill_tags else category
    queries = [
        f"Interview question about {skills}",
        f"How to evaluate a candidate on {skills}",
        f"Deep dive follow-up for {skills}",
    ]
    if _contains_any(skill_tags + [content, category], {"rag", "retrieval", "embedding"}):
        queries.append("RAG retrieval chunking reranking evaluation interview question")
    if _contains_any(skill_tags + [content, category], {"agent", "langgraph", "tool"}):
        queries.append("AI Agent LangGraph planning memory tools interview question")
    if _contains_any(skill_tags + [content, category], {"system", "design", "api", "websocket"}):
        queries.append("System design scalability reliability API interview question")
    return _dedupe(queries)[:5]


def _contains_any(values: list[str], needles: set[str]) -> bool:
    text = " ".join(values).lower()
    return any(needle in text for needle in needles)


def _chunk_id(question_id: str, chunk_type: str, index: int | None) -> str:
    suffix = chunk_type if index is None else f"{chunk_type}_{index}"
    return f"{_safe_id(question_id)}__{suffix}"


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        normalized = " ".join(item.split())
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique
