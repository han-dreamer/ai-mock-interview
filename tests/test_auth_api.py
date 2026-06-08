from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.api import auth
from app.main import app
from app.models.user import UserInDB


def _user(username="alice", password_hash="hash") -> UserInDB:
    return UserInDB(
        id="user-1",
        username=username,
        password_hash=password_hash,
        display_name="Alice",
        role="user",
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )


def test_register_returns_token_and_user(monkeypatch):
    created = {}

    async def fake_get_user_by_username(_username):
        return None

    async def fake_create_user(**kwargs):
        created.update(kwargs)
        return _user(username=kwargs["username"], password_hash=kwargs["password_hash"])

    monkeypatch.setattr(auth, "get_user_by_username", fake_get_user_by_username)
    monkeypatch.setattr(auth, "create_user", fake_create_user)

    client = TestClient(app)
    response = client.post(
        "/api/auth/register",
        json={"username": "Alice", "password": "secret123", "display_name": "Alice"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert body["user"]["username"] == "alice"
    assert created["username"] == "alice"
    assert created["password_hash"] != "secret123"


def test_register_rejects_duplicate_username(monkeypatch):
    async def fake_get_user_by_username(_username):
        return _user()

    monkeypatch.setattr(auth, "get_user_by_username", fake_get_user_by_username)

    client = TestClient(app)
    response = client.post(
        "/api/auth/register",
        json={"username": "alice", "password": "secret123"},
    )

    assert response.status_code == 409


def test_login_and_me(monkeypatch):
    password_hash = auth.hash_password("secret123")

    async def fake_get_user_by_username(_username):
        return _user(password_hash=password_hash)

    async def fake_update_last_login(_user_id, _when):
        return None

    async def fake_get_user_by_id(_user_id):
        return _user(password_hash=password_hash)

    monkeypatch.setattr(auth, "get_user_by_username", fake_get_user_by_username)
    monkeypatch.setattr(auth, "update_last_login", fake_update_last_login)
    monkeypatch.setattr("app.security.get_user_by_id", fake_get_user_by_id)

    client = TestClient(app)
    logged_in = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "secret123"},
    )

    assert logged_in.status_code == 200
    token = logged_in.json()["access_token"]

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert me.status_code == 200
    assert me.json()["id"] == "user-1"
    assert me.json()["username"] == "alice"


def test_login_rejects_bad_password(monkeypatch):
    password_hash = auth.hash_password("secret123")

    async def fake_get_user_by_username(_username):
        return _user(password_hash=password_hash)

    monkeypatch.setattr(auth, "get_user_by_username", fake_get_user_by_username)

    client = TestClient(app)
    response = client.post(
        "/api/auth/login",
        json={"username": "alice", "password": "wrong-password"},
    )

    assert response.status_code == 401


def test_me_requires_bearer_token():
    client = TestClient(app)
    response = client.get("/api/auth/me")

    assert response.status_code == 401
