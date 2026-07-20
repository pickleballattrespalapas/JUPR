from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app
from jupr_app.domain.notifications.player_profile_update_repo import StaleCommunicationsStateError


def _install_env(monkeypatch, supabase):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "local-service")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_player_updates_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_player_updates_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_admin_player_updates_send_range_contract(monkeypatch):
    supabase = FakeSupabase({"admin_activity_log": []})
    _install_env(monkeypatch, supabase)
    captured: dict[str, object] = {}

    def fake_run(supabase_arg, **kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "mode": "player_update_range_send",
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "generation_result": {"saved": 2, "queued": 2},
            "send_result": {"attempted": 2, "sent": 2, "skipped": 0, "errors": 0, "email_mode": "dry_run"},
            "warnings": [],
        }

    monkeypatch.setattr("services.api.admin_player_updates_routes.run_admin_player_update_range", fake_run)

    response = TestClient(app).post(
        "/admin/clubs/club/player-updates/send-range",
        headers={"Authorization": "Bearer local"},
        json={
            "start_date": "2026-04-01",
            "end_date": "2026-04-07",
            "only_players_with_matches": True,
            "send_now": True,
            "confirmation_text": "SEND PLAYER UPDATES",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "player_update_range_send"
    assert payload["generation_result"]["saved"] == 2
    assert payload["send_result"]["sent"] == 2
    assert captured["actor_email"] == "admin@example.com"
    assert captured["actor_role"] == "club_owner"


def test_public_status_is_sanitized(monkeypatch) -> None:
    supabase = FakeSupabase({"player_profile_update_subscriptions": []})
    _install_env(monkeypatch, supabase)
    response = TestClient(app).get("/admin/clubs/club/player-updates/status")
    assert response.status_code == 200
    payload = response.json()
    assert "smtp_status" not in payload
    assert "active_subscription_count" not in payload
    assert "service_role_configured" not in payload
    assert isinstance(payload["smtp_configured"], bool)


def test_admin_player_updates_send_range_requires_confirmation(monkeypatch):
    supabase = FakeSupabase({"admin_activity_log": []})
    _install_env(monkeypatch, supabase)

    def fake_run(_supabase_arg, **kwargs):
        if str(kwargs.get("confirmation_text") or "").strip().upper() != "SEND PLAYER UPDATES":
            raise ValueError("Type SEND PLAYER UPDATES to send player update emails.")
        return {"ok": True}

    monkeypatch.setattr("services.api.admin_player_updates_routes.run_admin_player_update_range", fake_run)

    response = TestClient(app).post(
        "/admin/clubs/club/player-updates/send-range",
        headers={"Authorization": "Bearer local"},
        json={"start_date": "2026-04-01", "end_date": "2026-04-07", "confirmation_text": "SEND"},
    )

    assert response.status_code == 400
    assert "SEND PLAYER UPDATES" in response.json()["detail"]


def test_admin_player_updates_outbox_stale_write_is_conflict(monkeypatch):
    supabase = FakeSupabase({"admin_activity_log": []})
    _install_env(monkeypatch, supabase)

    def fake_delete(*_args, **_kwargs):
        raise StaleCommunicationsStateError("Outbox row changed. Reload the queue.")

    monkeypatch.setattr("services.api.admin_player_updates_routes.delete_outbox_rows", fake_delete)
    response = TestClient(app).post(
        "/admin/clubs/club/player-updates/outbox/delete",
        headers={"Authorization": "Bearer local"},
        json={
            "items": [{"id": "outbox-1", "expected_row_version": 2}],
            "confirmation_text": "DELETE QUEUED UPDATES",
        },
    )

    assert response.status_code == 409
    assert "Reload" in response.json()["detail"]


def test_admin_player_updates_workspace_requires_server_service_role(monkeypatch):
    supabase = FakeSupabase({"admin_activity_log": []})
    _install_env(monkeypatch, supabase)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    response = TestClient(app).get(
        "/admin/clubs/club/player-updates/workspace?start_date=2026-04-01&end_date=2026-04-07",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 503
    assert "must never be configured in the browser" in response.json()["detail"]
