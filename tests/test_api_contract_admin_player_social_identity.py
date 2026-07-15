from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def social_identity_tables() -> dict[str, list[dict]]:
    return {
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex Smith", "rating": 1400, "starting_rating": 1400, "wins": 0, "losses": 0, "matches_played": 0, "active": True},
            {"club_id": "club", "id": 2, "name": "Blair Jones", "rating": 1200, "starting_rating": 1200, "wins": 0, "losses": 0, "matches_played": 0, "active": True},
        ],
        "league_ratings": [],
        "matches": [],
        "club_people": [
            {"club_id": "club", "id": "person-1", "display_name": "Alex Smith", "normalized_name": "alex smith", "linked_player_id": None, "first_seen_on": "2026-01-01", "last_seen_on": "2026-01-08"},
            {"club_id": "club", "id": "person-2", "display_name": "Social Blair", "normalized_name": "social blair", "linked_player_id": 2, "first_seen_on": "2026-01-01", "last_seen_on": "2026-01-08"},
        ],
        "admin_activity_log": [],
    }


def _install_env(monkeypatch, supabase):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_player_editor_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_player_editor_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_admin_player_social_identity_list_contract(monkeypatch):
    supabase = FakeSupabase(social_identity_tables())
    _install_env(monkeypatch, supabase)

    response = TestClient(app).get(
        "/admin/clubs/club/players/editor/social-identities",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["summary"]["people"] == 2
    assert payload["summary"]["linked"] == 1
    assert payload["people"][1]["linked_player_name"] == "Blair Jones"


def test_admin_player_social_identity_update_contract(monkeypatch):
    tables = social_identity_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/players/editor/social-identities/person-1",
        headers={"Authorization": "Bearer local"},
        json={"linked_player_id": 1, "confirmation_text": "LINK SOCIAL"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "player_social_identity_update"
    assert tables["club_people"][0]["linked_player_id"] == 1
    assert tables["admin_activity_log"][0]["action_type"] == "update_player_social_identity_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_player_social_identity_auto_link_contract(monkeypatch):
    tables = social_identity_tables()
    tables["club_people"][1]["linked_player_id"] = None
    tables["club_people"][1]["display_name"] = "Blair Jones"
    tables["club_people"][1]["normalized_name"] = "blair jones"
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/players/editor/social-identities/auto-link",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "LINK SOCIAL"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["linked_count"] == 2
    assert {row["linked_player_id"] for row in tables["club_people"]} == {1, 2}
    assert tables["admin_activity_log"][0]["action_type"] == "auto_link_player_social_identities_admin"


def test_admin_player_social_identity_update_requires_confirmation(monkeypatch):
    supabase = FakeSupabase(social_identity_tables())
    _install_env(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/players/editor/social-identities/person-1",
        headers={"Authorization": "Bearer local"},
        json={"linked_player_id": 1, "confirmation_text": "LINK"},
    )

    assert response.status_code == 400
    assert "LINK SOCIAL" in response.json()["detail"]
