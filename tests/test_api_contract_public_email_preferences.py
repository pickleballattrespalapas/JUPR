from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def _tables():
    return {
        "player_profile_update_subscriptions": [
            {
                "id": "sub-1",
                "club_id": "club",
                "player_id": 10,
                "email": "player@example.com",
                "unsubscribe_token": "tok-1",
                "request_status": "active",
                "preferences_json": {"send_only_if_changed": True},
                "verified_at": "2026-01-01T00:00:00Z",
                "unsubscribed_at": None,
            }
        ]
    }


def _install_env(monkeypatch, supabase):
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)


def test_public_email_preferences_lookup_masks_email(monkeypatch):
    tables = _tables()
    _install_env(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).get("/email-preferences?token=tok-1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["found"] is True
    assert payload["subscription"]["email_masked"] == "p****r@example.com"
    assert payload["subscription"]["request_status"] == "active"


def test_public_email_preferences_unsubscribe_by_token(monkeypatch):
    tables = _tables()
    _install_env(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).post(
        "/email-preferences/unsubscribe",
        json={"token": "tok-1", "scope": "global"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["scope"] == "global"
    assert payload["subscription"]["request_status"] == "unsubscribed"
    assert tables["player_profile_update_subscriptions"][0]["request_status"] == "unsubscribed"


def test_public_email_preferences_unsubscribe_requires_identifier(monkeypatch):
    _install_env(monkeypatch, FakeSupabase(_tables()))

    response = TestClient(app).post("/email-preferences/unsubscribe", json={"scope": "player_updates"})

    assert response.status_code == 400
    assert "unsubscribe token" in response.json()["detail"]
