from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def tournament_tables():
    return {
        "tournaments": [
            {
                "club_id": "club",
                "id": "tour_1",
                "name": "Spring Classic",
                "status": "PUBLISHED",
                "start_date": "2026-04-10",
                "end_date": "2026-04-12",
                "created_at": "2026-03-01T00:00:00Z",
                "updated_at": "2026-03-02T00:00:00Z",
            },
            {
                "club_id": "club",
                "id": "tour_archived",
                "name": "Old Classic",
                "status": "ARCHIVED",
                "created_at": "2025-03-01T00:00:00Z",
            },
        ],
        "tournament_registration_settings": [
            {
                "id": "regset_1",
                "tournament_id": "tour_1",
                "registration_slug": "spring-classic",
                "registration_status": "open",
                "waitlist_enabled": True,
                "partner_board_enabled": True,
            }
        ],
        "tournament_registration_days": [
            {"id": "day_1", "tournament_id": "tour_1", "label": "Friday", "date": "2026-04-10", "enabled": True, "sort_order": 1}
        ],
        "tournament_event_options": [
            {
                "id": "event_1",
                "tournament_id": "tour_1",
                "registration_day_id": "day_1",
                "event_family_label": "Gender Doubles",
                "division_name": "3.5",
                "event_format_default": "round_robin",
                "scoring_default": "one_game_to_11",
                "status": "open",
                "enabled": True,
                "sort_order": 1,
            }
        ],
        "tournament_registrations": [
            {
                "id": "registration_1",
                "tournament_id": "tour_1",
                "display_name": "Alex Example",
                "email": "alex@example.com",
                "phone": "555-0100",
                "registration_status": "confirmed",
                "payment_status": "paid",
                "wants_partner_board_contact": True,
                "created_at": "2026-03-03T00:00:00Z",
            }
        ],
        "tournament_registration_selections": [
            {"id": "selection_1", "tournament_id": "tour_1", "registration_id": "registration_1", "event_option_id": "event_1", "partner_mode": "NEEDS_PARTNER"}
        ],
        "admin_activity_log": [],
    }


def test_admin_tournament_status_disabled_contract(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)

    response = TestClient(app).get("/admin/clubs/club/tournaments/admin/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["status"] == "guarded_off"
    assert payload["tournaments_endpoint"] is None


def test_admin_tournament_list_contract(monkeypatch):
    supabase = FakeSupabase(tournament_tables())
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )

    response = TestClient(app).get(
        "/admin/clubs/club/tournaments/admin/tournaments",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_admin_list"
    assert payload["count"] == 1
    assert payload["tournaments"][0]["id"] == "tour_1"
    assert payload["tournaments"][0]["registration_count"] == 1
    assert payload["tournaments"][0]["selection_count"] == 1


def test_admin_tournament_detail_contract(monkeypatch):
    supabase = FakeSupabase(tournament_tables())
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )

    response = TestClient(app).get(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_admin_detail"
    assert payload["tournament"]["registration_status"] == "open"
    assert payload["summary"]["registrations"] == 1
    assert payload["summary"]["selections"] == 1
    assert payload["summary"]["by_registration_status"] == {"confirmed": 1}
    assert payload["summary"]["by_payment_status"] == {"paid": 1}
    assert payload["registrations"][0]["display_name"] == "Alex Example"
    assert payload["event_options"][0]["division_name"] == "3.5"
