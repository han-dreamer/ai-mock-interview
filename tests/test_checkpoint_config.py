import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.services import checkpoint


@pytest.mark.asyncio
async def test_memory_checkpointer_is_default(monkeypatch):
    checkpoint.reset_checkpointer_for_tests()
    monkeypatch.setattr(settings, "checkpointer_backend", "memory")

    saver = await checkpoint.init_checkpointer()

    assert isinstance(saver, MemorySaver)
    assert checkpoint.get_checkpointer() is saver
    await checkpoint.close_checkpointer()


@pytest.mark.asyncio
async def test_unknown_checkpointer_backend_fails_clearly(monkeypatch):
    checkpoint.reset_checkpointer_for_tests()
    monkeypatch.setattr(settings, "checkpointer_backend", "sqlite")

    with pytest.raises(ValueError, match="Unsupported CHECKPOINTER_BACKEND"):
        await checkpoint.init_checkpointer()

    checkpoint.reset_checkpointer_for_tests()
