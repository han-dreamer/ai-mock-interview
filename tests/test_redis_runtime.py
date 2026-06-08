import pytest
from fastapi.testclient import TestClient

from app.api import interview_rest
from app.cache import locks, rate_limiter
from app.cache.rate_limiter import RateLimitResult
from app.main import app
from app.models.user import UserInDB
from app.security import get_current_user


def _fake_user() -> UserInDB:
    return UserInDB(
        id="user-1",
        username="user-1@example.test",
        password_hash="hash",
        display_name="Test User",
        role="user",
        is_active=True,
    )


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.ttls = {}
        self.lock_acquired = True

    async def incr(self, key):
        self.values[key] = int(self.values.get(key, 0)) + 1
        return self.values[key]

    async def expire(self, key, seconds):
        self.ttls[key] = seconds
        return True

    async def ttl(self, key):
        return self.ttls.get(key, 60)

    async def set(self, key, value, nx=False, px=None, ex=None):
        if nx and key in self.values:
            return False
        if not self.lock_acquired:
            return False
        self.values[key] = value
        if ex is not None:
            self.ttls[key] = ex
        if px is not None:
            self.ttls[key] = px
        return True

    async def get(self, key):
        return self.values.get(key)

    async def delete(self, key):
        self.values.pop(key, None)
        return 1


@pytest.mark.asyncio
async def test_rate_limiter_blocks_after_limit(monkeypatch):
    fake = FakeRedis()

    async def fake_get_redis():
        return fake

    monkeypatch.setattr(rate_limiter, "get_redis", fake_get_redis)

    first = await rate_limiter.check_rate_limit("interview:start", "user:a", 1, 60)
    second = await rate_limiter.check_rate_limit("interview:start", "user:a", 1, 60)

    assert first.allowed is True
    assert second.allowed is False
    assert second.retry_after_seconds == 61


@pytest.mark.asyncio
async def test_redis_lock_rejects_concurrent_answer(monkeypatch):
    fake = FakeRedis()
    fake.lock_acquired = False

    async def fake_get_redis():
        return fake

    monkeypatch.setattr(locks, "get_redis", fake_get_redis)

    with pytest.raises(RuntimeError, match="already processing"):
        async with locks.session_answer_lock("session-1"):
            raise AssertionError("lock should not enter the critical section")


def test_rest_start_rate_limit_returns_429(monkeypatch):
    async def blocked(*_args, **_kwargs):
        return RateLimitResult(
            allowed=False,
            limit=1,
            remaining=0,
            retry_after_seconds=30,
            key="rl:test",
        )

    monkeypatch.setattr(interview_rest, "check_rate_limit", blocked)

    async def override_current_user():
        return _fake_user()

    app.dependency_overrides[get_current_user] = override_current_user
    client = TestClient(app)

    try:
        response = client.post(
            "/api/interview/start",
            json={"jd_text": "Python FastAPI LangGraph AI application developer position"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"
