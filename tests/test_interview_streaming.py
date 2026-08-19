from __future__ import annotations

import asyncio

import pytest

from app.agents import interviewer
from app.models.question import QuestionItem
from app.services.session_manager import SessionManager
from app.streaming import (
    SessionEventBus,
    emit_status_event,
    reset_stream_sink,
    set_stream_sink,
)


class FakeStreamingLLM:
    def __init__(self, chunks: list[str] | None = None, stream_error: Exception | None = None):
        self.chunks = chunks or []
        self.stream_error = stream_error
        self.chat_calls = 0

    async def chat(self, **_kwargs) -> str:
        self.chat_calls += 1
        return "fallback question"

    async def chat_stream(self, **_kwargs):
        if self.stream_error:
            raise self.stream_error
        for chunk in self.chunks:
            yield chunk


def _state() -> dict:
    return {
        "question_plan": [
            QuestionItem(
                id=1,
                content="请介绍一下你的项目。",
                skill_tags=["Python", "LangGraph"],
                difficulty="medium",
                reference_points=["项目架构"],
            )
        ],
        "current_question_index": 0,
        "conversation_history": [],
        "max_follow_ups": 2,
    }


@pytest.mark.asyncio
async def test_ask_question_streams_chunks_and_returns_complete_state(monkeypatch):
    llm = FakeStreamingLLM(["第一段", "第二段"])
    events: list[dict] = []

    async def sink(event: dict) -> None:
        events.append(event)

    monkeypatch.setattr(interviewer, "get_llm_client", lambda: llm)
    token = set_stream_sink(sink)
    try:
        result = await interviewer.ask_question(_state())
    finally:
        reset_stream_sink(token)

    assert [event["event"] for event in events] == ["start", "delta", "delta", "end"]
    assert "".join(event["chunk"] for event in events if event["event"] == "delta") == "第一段第二段"
    assert events[-1]["content"] == "第一段第二段"
    assert result["conversation_history"][0].content == "第一段第二段"
    assert llm.chat_calls == 0


@pytest.mark.asyncio
async def test_streaming_falls_back_to_chat_before_first_chunk(monkeypatch):
    llm = FakeStreamingLLM(stream_error=RuntimeError("provider streaming unavailable"))
    events: list[dict] = []

    async def sink(event: dict) -> None:
        events.append(event)

    monkeypatch.setattr(interviewer, "get_llm_client", lambda: llm)
    token = set_stream_sink(sink)
    try:
        result = await interviewer.ask_question(_state())
    finally:
        reset_stream_sink(token)

    assert result["conversation_history"][0].content == "fallback question"
    assert [event["event"] for event in events] == ["start", "delta", "end"]
    assert events[1]["chunk"] == "fallback question"
    assert events[-1]["done"] is True
    assert llm.chat_calls == 1


@pytest.mark.asyncio
async def test_streaming_falls_back_to_chat_when_provider_returns_no_text(monkeypatch):
    llm = FakeStreamingLLM([])
    events: list[dict] = []

    async def sink(event: dict) -> None:
        events.append(event)

    monkeypatch.setattr(interviewer, "get_llm_client", lambda: llm)
    token = set_stream_sink(sink)
    try:
        result = await interviewer.ask_question(_state())
    finally:
        reset_stream_sink(token)

    assert result["conversation_history"][0].content == "fallback question"
    assert [event["event"] for event in events] == ["start", "delta", "end"]
    assert events[1]["chunk"] == "fallback question"
    assert events[-1]["content"] == "fallback question"
    assert llm.chat_calls == 1


@pytest.mark.asyncio
async def test_status_event_uses_the_same_transport_sink():
    events: list[dict] = []

    async def sink(event: dict) -> None:
        events.append(event)

    token = set_stream_sink(sink)
    try:
        await emit_status_event("planning_questions", "正在规划问题。")
    finally:
        reset_stream_sink(token)

    assert events == [
        {
            "type": "status",
            "stage": "planning_questions",
            "message": "正在规划问题。",
        }
    ]


@pytest.mark.asyncio
async def test_session_event_bus_fans_out_bounded_events():
    bus = SessionEventBus(max_queue_size=2)
    first = bus.subscribe()
    second = bus.subscribe()

    await bus.publish({"type": "status", "stage": "one", "message": "1"})
    await bus.publish({"type": "status", "stage": "two", "message": "2"})
    await bus.publish({"type": "status", "stage": "three", "message": "3"})

    assert (await first.get())["sequence"] == 2
    assert (await second.get())["event"]["stage"] == "two"

    bus.unsubscribe(second)
    await bus.publish({"type": "status", "stage": "four", "message": "4"})
    assert (await first.get())["sequence"] == 3


@pytest.mark.asyncio
async def test_background_answer_task_deduplicates_same_answer_id(monkeypatch):
    manager = SessionManager()
    session = manager.create_session(
        session_id="background-answer-session",
        jd_text="Python FastAPI LangGraph developer position",
    )
    data = manager._sessions[session.session_id]
    data.graph_started = True

    gate = __import__("asyncio").Event()
    calls = 0

    async def fake_submit(_session_id, _answer, *, stream_sink=None):
        nonlocal calls
        calls += 1
        await gate.wait()
        if stream_sink:
            await stream_sink({"type": "status", "stage": "done", "message": "done"})
        return {"interview_complete": False, "conversation_history": []}

    monkeypatch.setattr(manager, "submit_answer", fake_submit)
    first = manager.ensure_answer_task(session.session_id, "answer", "answer-1")
    second = manager.ensure_answer_task(session.session_id, "answer", "answer-1")
    assert first is second

    with pytest.raises(RuntimeError, match="Another answer"):
        manager.ensure_answer_task(session.session_id, "different", "answer-2")

    gate.set()
    result = await first
    assert result["interview_complete"] is False
    assert calls == 1

    cached = await manager.ensure_answer_task(session.session_id, "answer", "answer-1")
    assert cached["interview_complete"] is False
    assert calls == 1


@pytest.mark.asyncio
async def test_background_start_task_is_shared_and_publishes_events(monkeypatch):
    manager = SessionManager()
    session = manager.create_session(
        session_id="background-start-session",
        jd_text="Python FastAPI LangGraph developer position",
    )
    queue = manager.subscribe_events(session.session_id)
    calls = 0

    async def fake_start(_session_id, *, stream_sink=None):
        nonlocal calls
        calls += 1
        await stream_sink(
            {
                "event": "start",
                "stream_id": "stream-1",
                "kind": "question",
            }
        )
        await asyncio.sleep(0)
        await stream_sink(
            {
                "event": "end",
                "stream_id": "stream-1",
                "kind": "question",
                "content": "question",
                "done": True,
            }
        )
        return {"interview_complete": False}

    monkeypatch.setattr(manager, "start_interview_graph", fake_start)
    first = manager.ensure_start_task(session.session_id)
    second = manager.ensure_start_task(session.session_id)

    assert first is second
    result = await first
    first_event = await queue.get()
    second_event = await queue.get()

    assert result["interview_complete"] is False
    assert calls == 1
    assert first_event["event"]["event"] == "start"
    assert second_event["event"]["event"] == "end"
