import pytest

from app.models.interview import ChatMessage, InterviewSession
from app.services.session_manager import SessionManager
from app.services.session_repository import PersistedSession


class FakeAsyncGraph:
    def __init__(self):
        self.updated = False

    async def aupdate_state(self, config, values):
        self.updated = True
        assert config["configurable"]["thread_id"] == "async-session"
        assert values == {"current_candidate_answer": "answer text"}

    def update_state(self, *_args, **_kwargs):
        raise AssertionError("sync update_state must not be used with async graph")

    async def ainvoke(self, *_args, **_kwargs):
        return {
            "interview_complete": False,
            "current_question_index": 0,
            "follow_up_count": 0,
            "conversation_history": [],
            "assessments": [],
        }


@pytest.mark.asyncio
async def test_session_manager_restores_persisted_metadata(monkeypatch):
    session = InterviewSession(
        session_id="persisted-session",
        user_id="student-1",
        jd_text="Python FastAPI LangGraph developer position",
        status="interviewing",
        current_question_index=1,
        follow_up_count=0,
        max_follow_ups=2,
        conversation_history=[
            ChatMessage(role="interviewer", content="Please introduce your project.")
        ],
    )

    async def fake_load_session_record(session_id: str):
        assert session_id == session.session_id
        return PersistedSession(
            session=session,
            mode="professional",
            resume_text="Candidate resume text",
            resume_parse_result=None,
            graph_started=True,
            last_state={"current_question_index": 1},
            persisted_assessment_count=1,
            final_memory_saved=True,
        )

    monkeypatch.setattr(
        "app.services.session_manager.load_session_record",
        fake_load_session_record,
    )

    manager = SessionManager()

    restored = await manager.ensure_session_loaded(session.session_id)

    assert restored == session
    assert manager.get_session(session.session_id) == session
    assert manager.get_session_mode(session.session_id) == "professional"
    assert manager.has_graph_started(session.session_id) is True
    assert (await manager.get_last_state(session.session_id))["current_question_index"] == 1

    data = manager._sessions[session.session_id]
    assert data.persisted_assessment_count == 1
    assert data.final_memory_saved is True


@pytest.mark.asyncio
async def test_session_manager_public_persist_delegates_to_repository(monkeypatch):
    captured = {}

    async def fake_save_session_record(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "app.services.session_manager.save_session_record",
        fake_save_session_record,
    )

    manager = SessionManager()
    session = manager.create_session(
        session_id="new-session",
        user_id="student-2",
        jd_text="Python FastAPI LangGraph developer position",
        mode="practice",
    )

    await manager.persist_session(session.session_id)

    assert captured["session"] == session
    assert captured["mode"] == "practice"
    assert captured["resume_text"] == ""
    assert captured["graph_started"] is False
    assert captured["persisted_assessment_count"] == 0
    assert captured["final_memory_saved"] is False


@pytest.mark.asyncio
async def test_session_manager_submit_answer_uses_async_update_state(monkeypatch):
    async def fake_save_session_record(**_kwargs):
        return None

    monkeypatch.setattr(
        "app.services.session_manager.save_session_record",
        fake_save_session_record,
    )

    manager = SessionManager()
    session = manager.create_session(
        session_id="async-session",
        user_id="student-3",
        jd_text="Python FastAPI LangGraph developer position",
        mode="practice",
    )
    data = manager._sessions[session.session_id]
    data.graph_started = True
    graph = FakeAsyncGraph()
    manager._practice_graph = graph

    result = await manager.submit_answer(session.session_id, "answer text")

    assert graph.updated is True
    assert result["interview_complete"] is False
