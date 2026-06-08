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


def test_memory_backend_factories_select_postgres(monkeypatch):
    from app.memory.store import get_memory_store
    from app.memory.vector_store import get_memory_vector_store

    monkeypatch.setattr(settings, "memory_store_backend", "postgres")
    monkeypatch.setattr(settings, "memory_vector_backend", "pgvector")
    monkeypatch.setattr("app.memory.vector_store._memory_vector_store", None)

    store = get_memory_store()
    vector_store = get_memory_vector_store()

    assert store.__class__.__name__ == "PostgresMemoryStore"
    assert vector_store.__class__.__name__ == "PgVectorMemoryVectorStore"
