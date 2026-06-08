from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api import interview_rest, interview_ws
from app.main import app
from app.models.interview import ChatMessage, InterviewSession
from app.models.resume import ResumeParseMetadata, ResumeParseResult
from app.models.user import UserInDB
from app.security import create_access_token, get_current_user


def _fake_user(user_id="user-1") -> UserInDB:
    return UserInDB(
        id=user_id,
        username=f"{user_id}@example.test",
        password_hash="hash",
        display_name=user_id,
        role="user",
        is_active=True,
    )


def _client_with_user(user: UserInDB | None = None) -> TestClient:
    current_user = user or _fake_user()

    async def override_current_user():
        return current_user

    app.dependency_overrides[get_current_user] = override_current_user
    return TestClient(app)


def _clear_overrides():
    app.dependency_overrides.clear()


def _ws_url(session_id: str, user: UserInDB | None = None) -> str:
    token = create_access_token(user or _fake_user())
    return f"/api/ws/interview/{session_id}?token={token}"


async def _fake_get_user_by_id(user_id: str):
    return _fake_user(user_id)


class FakeInterviewManager:
    def __init__(self):
        self.sessions = {}
        self.graph_started = False
        self.resume_text = ""
        self.report = {
            "overall_score": 8.0,
            "grade": "B",
            "overall_assessment": "The candidate showed solid backend engineering thinking.",
        }
        self.last_state = {}

    def create_session(
        self,
        session_id,
        jd_text,
        max_follow_ups=2,
        mode="practice",
        resume_text="",
        resume_parse_result=None,
        user_id="local-user",
    ):
        session = InterviewSession(
            session_id=session_id,
            user_id=user_id,
            jd_text=jd_text,
            max_follow_ups=max_follow_ups,
        )
        self.sessions[session_id] = session
        self.mode = mode
        self.resume_text = resume_text
        self.resume_parse_result = resume_parse_result
        return session

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    async def ensure_session_loaded(self, session_id):
        return self.get_session(session_id)

    async def persist_session(self, _session_id):
        return None

    def get_session_mode(self, _session_id):
        return getattr(self, "mode", "practice")

    def has_graph_started(self, _session_id):
        return self.graph_started

    def get_last_state(self, _session_id):
        return self.last_state

    def set_resume(self, session_id, resume_text, resume_parse_result=None):
        if self.graph_started:
            raise RuntimeError("Resume cannot be changed after the interview has started")
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        self.resume_text = resume_text
        self.resume_parse_result = resume_parse_result

    def get_report_for_session(self, _session_id):
        return self.report

    async def list_user_sessions(self, user_id, limit=50, offset=0):
        items = []
        for session in self.sessions.values():
            if session.user_id != user_id:
                continue
            items.append(
                {
                    "session_id": session.session_id,
                    "title": session.jd_text[:36],
                    "mode": getattr(self, "mode", "practice"),
                    "status": session.status,
                    "graph_started": self.graph_started,
                    "has_report": session.status == "completed",
                    "question_count": len(self.last_state.get("question_plan", []) or []),
                    "assessment_count": len(self.last_state.get("assessments", []) or []),
                    "current_question_index": session.current_question_index,
                    "max_follow_ups": session.max_follow_ups,
                    "jd_preview": session.jd_text[:180],
                    "overall_score": self.report["overall_score"] if session.status == "completed" else None,
                    "grade": self.report["grade"] if session.status == "completed" else None,
                    "error_message": None,
                    "created_at": None,
                    "updated_at": None,
                    "completed_at": None,
                }
            )
        return items[offset : offset + limit]

    async def start_interview_graph(self, session_id):
        self.graph_started = True
        self.sessions[session_id].status = "interviewing"
        self.last_state = {
            "interview_complete": False,
            "current_question_index": 0,
            "follow_up_count": 0,
            "current_round": 1,
            "question_plan": [
                {
                    "skill_tags": ["FastAPI", "LangGraph"],
                    "difficulty": "medium",
                }
            ],
            "conversation_history": [
                ChatMessage(
                    role="interviewer",
                    content="Explain how your FastAPI layer drives the Agent workflow.",
                )
            ],
            "assessments": [],
        }
        return self.last_state

    async def submit_answer(self, session_id, answer):
        self.sessions[session_id].status = "completed"
        self.last_state = {
            "interview_complete": True,
            "practice_report": self.report,
            "current_round": 1,
            "question_plan": [],
            "conversation_history": [
                ChatMessage(role="interviewer", content="Explain FastAPI."),
                ChatMessage(role="candidate", content=answer),
            ],
            "assessments": [{"question_id": 1, "score": 8}],
        }
        return self.last_state

    async def stop_interview(self, session_id):
        self.sessions[session_id].status = "completed"
        self.last_state = {
            "interview_complete": True,
            "practice_report": self.report,
            "current_round": 1,
            "question_plan": [],
            "conversation_history": [],
            "assessments": [],
        }
        return self.last_state


def test_rest_interview_contract(monkeypatch):
    manager = FakeInterviewManager()
    monkeypatch.setattr(interview_rest, "get_session_manager", lambda: manager)
    client = _client_with_user()

    try:
        created = client.post(
            "/api/interview/start",
            json={
                "jd_text": "Python FastAPI LangGraph AI application developer position",
                "mode": "practice",
                "max_follow_ups": 1,
            },
        )
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        assert manager.sessions[session_id].user_id == "user-1"

        started = client.post(f"/api/interview/session/{session_id}/start")
        assert started.status_code == 200
        assert started.json()["next"]["kind"] == "question"
        assert started.json()["next"]["skill_tags"] == ["FastAPI", "LangGraph"]

        answered = client.post(
            f"/api/interview/session/{session_id}/answer",
            json={"content": "I use FastAPI endpoints to invoke the session manager."},
        )
        assert answered.status_code == 200
        assert answered.json()["state"]["interview_complete"] is True
        assert answered.json()["report"]["overall_score"] == 8.0

        report = client.get(f"/api/interview/report/{session_id}")
        assert report.status_code == 200
        assert report.json()["grade"] == "B"
    finally:
        _clear_overrides()


def test_resume_upload_contract(monkeypatch):
    manager = FakeInterviewManager()
    monkeypatch.setattr(interview_rest, "get_session_manager", lambda: manager)

    async def fake_parse(_session_id, upload):
        return ResumeParseResult(
            raw_text="Python FastAPI LangGraph",
            normalized_text="Python FastAPI LangGraph",
            metadata=ResumeParseMetadata(
                file_name=upload.filename or "resume.pdf",
                file_type=".pdf",
                parser="fake",
                raw_char_count=24,
                normalized_char_count=24,
            ),
        )

    monkeypatch.setattr(interview_rest, "_store_and_parse_resume", fake_parse)
    client = _client_with_user()

    try:
        created = client.post(
            "/api/interview/start",
            json={"jd_text": "Python FastAPI LangGraph AI application developer position"},
        )
        session_id = created.json()["session_id"]

        uploaded = client.post(
            f"/api/interview/session/{session_id}/resume",
            files={"resume_file": ("resume.pdf", b"%PDF fake", "application/pdf")},
        )

        assert uploaded.status_code == 200
        assert manager.resume_text == "Python FastAPI LangGraph"
        assert uploaded.json()["resume"]["metadata"]["parser"] == "fake"
    finally:
        _clear_overrides()


def test_websocket_interview_contract(monkeypatch):
    manager = FakeInterviewManager()
    session = manager.create_session(
        session_id="ws-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-1",
    )
    monkeypatch.setattr(interview_ws, "get_session_manager", lambda: manager)
    monkeypatch.setattr(interview_ws, "get_user_by_id", _fake_get_user_by_id)
    client = TestClient(app)

    with client.websocket_connect(_ws_url(session.session_id)) as ws:
        assert ws.receive_json()["type"] == "status"
        assert ws.receive_json()["stage"] == "questions_ready"
        question = ws.receive_json()
        assert question["type"] == "question"
        assert question["skill_tags"] == ["FastAPI", "LangGraph"]

        ws.send_json({"type": "answer", "content": "I keep the graph state in SessionManager."})
        assert ws.receive_json()["stage"] == "processing"
        assert ws.receive_json()["type"] == "interview_end"
        report = ws.receive_json()
        assert report["type"] == "report"
        assert report["data"]["grade"] == "B"


def test_websocket_start_handshake_does_not_duplicate_question(monkeypatch):
    manager = FakeInterviewManager()
    session = manager.create_session(
        session_id="ws-start-handshake-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-1",
    )
    monkeypatch.setattr(interview_ws, "get_session_manager", lambda: manager)
    monkeypatch.setattr(interview_ws, "get_user_by_id", _fake_get_user_by_id)
    client = TestClient(app)

    with client.websocket_connect(_ws_url(session.session_id)) as ws:
        assert ws.receive_json()["type"] == "status"
        assert ws.receive_json()["stage"] == "questions_ready"
        assert ws.receive_json()["type"] == "question"

        ws.send_json({"type": "start_interview"})
        ws.send_json({"type": "answer", "content": "I keep the graph state in SessionManager."})
        assert ws.receive_json()["stage"] == "processing"
        assert ws.receive_json()["type"] == "interview_end"


def test_rest_rejects_cross_user_session(monkeypatch):
    manager = FakeInterviewManager()
    session = manager.create_session(
        session_id="other-user-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-2",
    )
    monkeypatch.setattr(interview_rest, "get_session_manager", lambda: manager)
    client = _client_with_user(_fake_user("user-1"))

    try:
        response = client.get(f"/api/interview/session/{session.session_id}")
        assert response.status_code == 403
    finally:
        _clear_overrides()


def test_rest_lists_only_current_user_sessions(monkeypatch):
    manager = FakeInterviewManager()
    own = manager.create_session(
        session_id="own-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-1",
    )
    own.status = "completed"
    manager.create_session(
        session_id="other-session",
        jd_text="Java backend developer position",
        user_id="user-2",
    )
    manager.last_state = {
        "question_plan": [{"skill_tags": ["FastAPI"]}],
        "assessments": [{"question_id": 1, "score": 8}],
    }
    monkeypatch.setattr(interview_rest, "get_session_manager", lambda: manager)
    client = _client_with_user(_fake_user("user-1"))

    try:
        response = client.get("/api/interview/sessions")
        assert response.status_code == 200
        items = response.json()["items"]
        assert [item["session_id"] for item in items] == ["own-session"]
        assert items[0]["has_report"] is True
        assert items[0]["overall_score"] == 8.0
        assert items[0]["grade"] == "B"
    finally:
        _clear_overrides()


def test_websocket_requires_authentication(monkeypatch):
    manager = FakeInterviewManager()
    session = manager.create_session(
        session_id="ws-auth-required-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-1",
    )
    monkeypatch.setattr(interview_ws, "get_session_manager", lambda: manager)
    client = TestClient(app)

    try:
        with client.websocket_connect(f"/api/ws/interview/{session.session_id}"):
            raise AssertionError("WebSocket should reject a missing auth token")
    except WebSocketDisconnect as exc:
        assert exc.code == 4401


def test_websocket_rejects_cross_user_session(monkeypatch):
    manager = FakeInterviewManager()
    session = manager.create_session(
        session_id="ws-cross-user-session",
        jd_text="Python FastAPI LangGraph AI application developer position",
        user_id="user-2",
    )
    monkeypatch.setattr(interview_ws, "get_session_manager", lambda: manager)
    monkeypatch.setattr(interview_ws, "get_user_by_id", _fake_get_user_by_id)
    client = TestClient(app)

    try:
        with client.websocket_connect(_ws_url(session.session_id, _fake_user("user-1"))):
            raise AssertionError("WebSocket should reject cross-user access")
    except WebSocketDisconnect as exc:
        assert exc.code == 4403
