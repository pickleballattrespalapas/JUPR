from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_player_editor_service import FakeSupabase, fake_storage
from jupr_app.services.admin_guarded_write_service import GuardedWriteRecoveryRequired

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def _install_auth(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.api.admin_player_editor_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="owner@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_player_editor_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_player_editor_status_disabled_contract(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    response = TestClient(app).get("/admin/clubs/club/players/editor/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["players_endpoint"] is None


def test_player_editor_list_disabled_before_auth(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", raising=False)
    called = {"auth": False}

    def fake_auth(*_args, **_kwargs):
        called["auth"] = True
        raise AssertionError("auth should not run while player editor flag is disabled")

    monkeypatch.setattr("services.api.admin_player_editor_routes.authenticate_bearer", fake_auth)

    response = TestClient(app).get("/admin/clubs/club/players/editor/players")

    assert response.status_code == 403
    assert called == {"auth": False}


def test_player_editor_status_enabled_contract(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-role")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(fake_storage()))

    response = TestClient(app).get("/admin/clubs/club/players/editor/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["players_endpoint"] == "/admin/clubs/{club_id}/players/editor/players"
    assert payload["transactional_merge_ready"] is True
    assert payload["player_merge_endpoint"] == "/admin/clubs/{club_id}/players/editor/merge"
    assert payload["player_count"] == 2


def test_player_editor_list_and_detail_contract(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(fake_storage()))
    _install_auth(monkeypatch)

    client = TestClient(app)
    list_response = client.get("/admin/clubs/club/players/editor/players", headers={"Authorization": "Bearer local"})
    detail_response = client.get("/admin/clubs/club/players/editor/players/1", headers={"Authorization": "Bearer local"})

    assert list_response.status_code == 200
    assert list_response.json()["count"] == 2
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["player"]["name"] == "Alex"
    assert detail["league_ratings"][0]["league_name"] == "Open"


def test_player_editor_create_and_patch_contract(monkeypatch):
    storage = fake_storage()
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(storage))
    _install_auth(monkeypatch)

    client = TestClient(app)
    create_response = client.post(
        "/admin/clubs/club/players/editor/players",
        headers={"Authorization": "Bearer local"},
        json={"name": "Casey", "starting_jupr": 3.25, "idempotency_key": "player-create-casey", "source": "test"},
    )
    assert create_response.status_code == 200
    created = create_response.json()["player"]
    assert created["name"] == "Casey"

    patch_response = client.patch(
        f"/admin/clubs/club/players/editor/players/{created['id']}",
        headers={"Authorization": "Bearer local"},
        json={"name": "Casey R", "rating_jupr": 3.6, "starting_jupr": 3.25, "active": True, "expected_state_fingerprint": created["state_fingerprint"], "idempotency_key": "player-update-casey", "source": "test"},
    )
    assert patch_response.status_code == 200
    patched = patch_response.json()["player"]
    assert patched["name"] == "Casey R"
    assert patched["rating"] == 1440.0
    action_types = [row["action_type"] for row in storage["admin_activity_log"]]
    assert "create_player_editor_player" in action_types
    assert "update_player_editor_player" in action_types


def test_player_editor_league_rating_patch_contract(monkeypatch):
    storage = fake_storage()
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(storage))
    _install_auth(monkeypatch)

    client = TestClient(app)
    detail = client.get(
        "/admin/clubs/club/players/editor/players/1",
        headers={"Authorization": "Bearer local"},
    ).json()
    response = client.patch(
        "/admin/clubs/club/players/editor/players/1/league-ratings/10",
        headers={"Authorization": "Bearer local"},
        json={
            "rating_jupr": 3.8,
            "starting_jupr": 3.5,
            "is_active": False,
            "expected_state_fingerprint": detail["league_ratings"][0]["state_fingerprint"],
            "idempotency_key": "league-rating-update",
            "confirmation_text": "SAVE LEAGUE RATING",
            "source": "test",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "player_editor_league_rating_update"
    assert payload["league_rating"]["rating"] == 1520.0
    assert payload["league_ratings"][0]["rating"] == 1520.0
    assert any(row["action_type"] == "update_player_editor_league_rating" for row in storage["admin_activity_log"])


def test_player_editor_ambiguous_create_returns_structured_recovery(monkeypatch):
    storage = fake_storage()
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(storage))
    monkeypatch.setattr(
        "services.api.admin_player_editor_routes.create_admin_player_editor_player",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            GuardedWriteRecoveryRequired("player-create-timeout", "Player outcome is uncertain.")
        ),
    )
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/players/editor/players",
        headers={"Authorization": "Bearer local"},
        json={"name": "Casey", "starting_jupr": 3.25, "idempotency_key": "player-create-timeout"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "RECOVERY_REQUIRED",
        "kind": "uncertain",
        "message": "Player outcome is uncertain.",
        "operation_key": "player-create-timeout",
        "recovery_required": True,
    }


def test_player_editor_ambiguous_patches_return_structured_recovery(monkeypatch):
    storage = fake_storage()
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(storage))
    monkeypatch.setattr(
        "services.api.admin_player_editor_routes.update_admin_player_editor_player",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            GuardedWriteRecoveryRequired("player-update-timeout", "Player edit outcome is uncertain.")
        ),
    )
    monkeypatch.setattr(
        "services.api.admin_player_editor_routes.update_admin_player_editor_league_rating",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            GuardedWriteRecoveryRequired("rating-update-timeout", "Rating edit outcome is uncertain.")
        ),
    )
    _install_auth(monkeypatch)
    client = TestClient(app)

    player_response = client.patch(
        "/admin/clubs/club/players/editor/players/1",
        headers={"Authorization": "Bearer local"},
        json={"name": "Alex R", "expected_state_fingerprint": "0" * 64, "idempotency_key": "player-update-timeout"},
    )
    rating_response = client.patch(
        "/admin/clubs/club/players/editor/players/1/league-ratings/10",
        headers={"Authorization": "Bearer local"},
        json={"rating_jupr": 3.8, "expected_state_fingerprint": "0" * 64, "idempotency_key": "rating-update-timeout", "confirmation_text": "SAVE LEAGUE RATING"},
    )

    assert player_response.status_code == 409
    assert player_response.json()["detail"]["code"] == "RECOVERY_REQUIRED"
    assert player_response.json()["detail"]["operation_key"] == "player-update-timeout"
    assert rating_response.status_code == 409
    assert rating_response.json()["detail"]["code"] == "RECOVERY_REQUIRED"
    assert rating_response.json()["detail"]["operation_key"] == "rating-update-timeout"
