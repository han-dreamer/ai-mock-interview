from pathlib import Path
from uuid import uuid4

import pytest

from app.memory.service import MemoryService
from app.memory.store import MemoryStore
from app.models.interview import AnswerAssessment
from app.models.question import QuestionItem
from app.models.report import PracticeReport, SkillScore


class FakeMemoryVectorStore:
    def __init__(self):
        self.items = {}

    async def upsert_item(self, item):
        self.items[item.id] = item

    async def semantic_search(self, user_id, query, memory_types=None, limit=5):
        valid_types = set(memory_types or [])
        query_terms = {term.lower() for term in query.split()}
        results = []
        for item in self.items.values():
            if item.user_id != user_id:
                continue
            if valid_types and str(item.memory_type) not in valid_types:
                continue
            content_terms = {term.lower().strip(".,;:") for term in item.content.split()}
            overlap = query_terms.intersection(content_terms)
            if not overlap:
                continue
            results.append(
                {
                    "id": item.id,
                    "document": item.content,
                    "metadata": {"memory_type": str(item.memory_type)},
                    "distance": 0.2,
                    "score": 0.8,
                }
            )
        return results[:limit]


def test_memory_service_persists_episode_skill_and_reflection():
    db_dir = Path("memory_data")
    db_dir.mkdir(exist_ok=True)
    service = MemoryService(MemoryStore(db_dir / f"test_memory_{uuid4().hex}.db"))
    question = QuestionItem(
        id=1,
        content="How would you evaluate a RAG system?",
        skill_tags=["RAG", "Evaluation"],
        difficulty="medium",
        reference_points=["Recall@K", "RAGAS"],
    )
    assessment = AnswerAssessment(
        question_id=1,
        score=5,
        covered_points=["Mentioned hybrid retrieval"],
        missed_points=["Recall@K", "RAGAS"],
        should_follow_up=True,
    )

    episode = service.save_assessment_episode(
        user_id="u1",
        session_id="s1",
        assessment=assessment,
        question=question,
        answer="I would inspect retrieved documents manually.",
        mode="practice",
    )
    report = PracticeReport(
        overall_score=5.0,
        grade="C",
        total_questions=1,
        skill_scores=[SkillScore(skill_name="RAG", score=5, evidence="Missed RAG metrics")],
        missed_knowledge=[],
        study_suggestions=["Study Recall@K and RAGAS"],
        overall_assessment="Needs stronger evaluation methodology.",
    )
    reflection = service.save_session_reflection("u1", "s1", "practice", report, [assessment])

    context = service.build_context("u1", tags=["RAG"])
    formatted = service.format_context(context)

    assert episode.memory_type == "interview_episode"
    assert reflection is not None
    assert context.weak_skills
    assert {skill.skill_name for skill in context.weak_skills} >= {"RAG", "Evaluation"}
    assert context.relevant_episodes
    assert "Weak Skills To Prioritize" in formatted
    assert "Recall@K" in formatted


@pytest.mark.asyncio
async def test_memory_service_semantic_recall_uses_vector_index():
    db_dir = Path("memory_data")
    db_dir.mkdir(exist_ok=True)
    vector_store = FakeMemoryVectorStore()
    service = MemoryService(
        MemoryStore(db_dir / f"test_semantic_memory_{uuid4().hex}.db"),
        vector_store=vector_store,
    )
    question = QuestionItem(
        id=2,
        content="Explain structured output failure handling.",
        skill_tags=["Structured Output"],
        difficulty="medium",
        reference_points=["schema validation", "retry", "fallback"],
    )
    assessment = AnswerAssessment(
        question_id=2,
        score=6,
        covered_points=["schema validation"],
        missed_points=["retry", "fallback"],
        should_follow_up=False,
    )

    episode = service.save_assessment_episode(
        user_id="u2",
        session_id="s2",
        assessment=assessment,
        question=question,
        answer="Use Pydantic schema validation.",
        mode="practice",
    )
    await service.aindex_memory_item(episode)

    context = await service.abuild_context(
        "u2",
        semantic_query="structured output retry fallback",
    )

    assert context.semantic_memories
    assert context.semantic_memories[0].id == episode.id
    assert context.semantic_memories[0].structured["_semantic_score"] == 0.8
