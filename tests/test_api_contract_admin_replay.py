from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_replay_service import FakeSupabase, fake_storage

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def test_admin_replay_status_disabled_contract(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    response = TestClient(app).get("/admin/clubs/club/replay-history")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["apply_endpoint"] is None
    assert payload["options"] == ["ALL (Full System Reset)"]


def test_admin_replay_post_disabled_before_auth(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", raising=False)
    called = {"auth": False}

    def fake_auth(*_args, **_kwargs):
        called["auth"] = True
        raise AssertionError("auth should not run while replay is disabled")

    monkeypatch.setattr("services.api.admin_replay_routes.authenticate_bearer", fake_auth)
    response = TestClient(app).post("/admin/clubs/club/replay-history", json={"target_reset": "Open", "confirmation_text": "REPLAY"})

    assert response.status_code == 403
    assert called == {"auth": False}


def test_admin_replay_status_enabled_contract(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(fake_storage()))

    response = TestClient(app).get("/admin/clubs/club/replay-history")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["apply_endpoint"] == "/admin/clubs/{club_id}/replay-history"
    assert "Open" in payload["options"]


def test_admin_replay_post_contract(monkeypatch):
    storage = fake_storage()
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(storage))
    monkeypatch.setattr(
        "services.api.admin_replay_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_replay_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="super_admin"),
    )

    def fake_replay_history(**kwargs):
        return {
            "target_reset": kwargs["target_reset"],
            "players_updated": kwargs["target_reset"].startswith("ALL"),
            "skipped_incomplete": 0,
            "matches_rewritten": 3,
            "matches_snapshots_updated_rows": 3,
            "league_ratings_rows": 4,
            "matches_scanned_total": 5,
        }

    monkeypatch.setattr("jupr_app.services.admin_replay_service.replay_history", fake_replay_history)

    response = TestClient(app).post(
        "/admin/clubs/club/replay-history",
        headers={"Authorization": "Bearer local"},
        json={"target_reset": "Open", "confirmation_text": "REPLAY", "source": "test"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["result"]["matches_rewritten"] == 3
    assert storage["admin_activity_log"][0]["action_type"] == "replay_history"
