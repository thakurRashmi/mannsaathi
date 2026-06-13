"""Smoke tests so we know the scaffold actually boots."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "service": "mannsaathi-backend"}


def test_chat_echoes_user_message() -> None:
    res = client.post("/api/chat", json={"message": "hello"})
    assert res.status_code == 200
    body = res.json()
    assert "hello" in body["reply"]
    assert body["is_crisis"] is False
