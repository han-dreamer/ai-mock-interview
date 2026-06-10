"""Small grounded QA chain used by RAG evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.llm.client import get_llm_client
from app.rag.context_builder import build_retrieval_context
from app.models.jd import SkillItem, SkillMatrix
from app.rag.postprocess import hydrate_parent_results, result_parent_id
from app.rag.query_builder import RetrievalQuery
from app.rag.reranker import rerank_results
from app.rag.retriever import RetrievedQuestion, get_retriever


RagVariant = Literal["vector", "hybrid", "multi", "full"]


@dataclass
class RagQAResult:
    question: str
    answer: str
    contexts: list[str]
    source_ids: list[str]
    retrieved: list[RetrievedQuestion]


async def answer_with_rag(
    question: str,
    tags: list[str] | None = None,
    variant: RagVariant = "full",
    top_k: int = 5,
) -> RagQAResult:
    """Answer a QA prompt using the project's RAG pipeline."""

    tags = tags or []
    retrieved = await retrieve_for_qa(question, tags=tags, variant=variant, top_k=top_k)
    contexts = [_context_from_result(item) for item in retrieved]
    answer = await _generate_grounded_answer(question, contexts)
    return RagQAResult(
        question=question,
        answer=answer,
        contexts=contexts,
        source_ids=[result_parent_id(item) for item in retrieved],
        retrieved=retrieved,
    )


async def retrieve_for_qa(
    question: str,
    tags: list[str] | None = None,
    variant: RagVariant = "full",
    top_k: int = 5,
) -> list[RetrievedQuestion]:
    """Retrieve contexts for QA evaluation using a selected pipeline variant."""

    tags = tags or []
    retriever = get_retriever()

    if variant == "vector":
        results = await retriever.retrieve_vector_only(question, top_k=top_k * 4)
        return hydrate_parent_results(results, top_k=top_k)

    if variant == "hybrid":
        results = await retriever.retrieve(question, top_k=top_k * 4)
        return hydrate_parent_results(results, top_k=top_k)

    queries = _qa_queries(question, tags)
    results = await retriever.retrieve_multi(
        queries,
        top_k_per_query=8,
        final_top_k=top_k * 5,
    )

    if variant == "multi":
        return hydrate_parent_results(results, top_k=top_k)

    state = _qa_state(tags)
    reranked = rerank_results(state, results, purpose="practice")
    return hydrate_parent_results(reranked, top_k=top_k)


def _qa_queries(question: str, tags: list[str]) -> list[RetrievalQuery]:
    tag_text = " ".join(tags[:5])
    queries = [
        RetrievalQuery(question, "qa_question", 1.0),
    ]
    if tag_text:
        queries.append(RetrievalQuery(f"{question} {tag_text}", "qa_tags", 0.95))
        queries.append(RetrievalQuery(f"{tag_text} interview explanation evaluation", "qa_domain", 0.75))
    return queries


def _qa_state(tags: list[str]) -> dict:
    skills = [
        SkillItem(
            name=tag,
            category=_skill_category(tag),
            weight=0.8,
            is_required=True,
        )
        for tag in tags[:8]
    ] or [
        SkillItem(
            name="RAG",
            category="domain",
            weight=0.8,
            is_required=True,
        )
    ]
    return {
        "skill_matrix": SkillMatrix(
            position_title="RAG QA Evaluation",
            experience_level="junior",
            skills=skills,
        )
    }


def _skill_category(tag: str) -> str:
    normalized = tag.lower()
    if any(term in normalized for term in ("fastapi", "langgraph", "websocket")):
        return "framework"
    if any(term in normalized for term in ("system", "architecture", "design")):
        return "system_design"
    if any(term in normalized for term in ("python", "llm", "rag", "agent", "retrieval")):
        return "domain"
    return "domain"


async def _generate_grounded_answer(question: str, contexts: list[str]) -> str:
    llm = get_llm_client()
    context_text = "\n\n".join(
        f"[Context {index}]\n{context}"
        for index, context in enumerate(contexts, 1)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict grounded RAG QA assistant. Answer only with facts "
                "that are explicitly stated in the retrieved contexts. Prefer the "
                "`Reference points`, `Question or topic`, and metadata fields over "
                "general knowledge. Do not add best practices, examples, risks, or "
                "implementation details unless they appear in the contexts. If a "
                "detail is not supported, omit it. Start with a direct answer that "
                "uses the key terms from the question; do not start with phrases "
                "like `Based on the retrieved contexts`. Do not cite context numbers "
                "or mention that you are using retrieved contexts. For simple "
                "definition or explanation questions, prefer one short paragraph. "
                "For comparison, distinction, multi-hop, or debugging questions, do "
                "not collapse the answer into one sentence; use 2-4 concise bullets "
                "or short sentences and cover each supported side of the distinction "
                "with retrieved evidence. For debugging questions, include "
                "retrieval-side signals such as query-document alignment, chunking, "
                "top-k candidates, scores, source metadata, and context coverage "
                "when they appear in the contexts; include generation-side signals "
                "such as ignored evidence or unsupported claims when they appear in "
                "the contexts."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nRetrieved contexts:\n{context_text}",
        },
    ]
    return await llm.chat(messages, temperature=0.1)


def _context_from_result(item: RetrievedQuestion) -> str:
    return build_retrieval_context(item)
