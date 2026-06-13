"""Smoke tests so we know the scaffold actually boots."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["service"] == "mannsaathi-backend"
    # llm_provider field is informational; just confirm it's present.
    assert "llm_provider" in body


def test_chat_request_validates_input() -> None:
    """Empty messages should be rejected by Pydantic validation."""
    res = client.post("/api/chat", json={"message": ""})
    assert res.status_code == 422  # validation error
