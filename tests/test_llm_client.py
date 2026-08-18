from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.llm.client import LLMClient


class _AsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        self._iterator = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks

    async def create(self, **_kwargs):
        return _AsyncStream(self._chunks)


class _FakeClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(chunks),
        )


@pytest.mark.asyncio
async def test_chat_stream_ignores_empty_provider_chunks():
    chunks = [
        SimpleNamespace(choices=[]),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None),
                ),
            ],
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="第一段"),
                ),
            ],
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="第二段"),
                ),
            ],
        ),
    ]
    client = LLMClient.__new__(LLMClient)
    client._client = _FakeClient(chunks)
    client.model = "test-model"
    client.temperature = 0.7

    result = [part async for part in client.chat_stream(messages=[])]

    assert result == ["第一段", "第二段"]
