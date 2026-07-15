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
            },
            {
                "id": "event_2",
                "tournament_id": "tour_1",
                "registration_day_id": "day_1",
                "event_family_label": "Gender Doubles",
                "division_name": "4.0",
                "event_format_default": "round_robin",
                "scoring_default": "one_game_to_11",
                "status": "open",
                "enabled": True,
                "sort_order": 2,
            },
        ],
        "tournament_registrations": [
            {
                "id": "registration_1",
                "tournament_id": "tour_1",
                "display_name": "Alex Example",
                "email": "alex@example.com",
                "phone": "555-0100",
                "status": "confirmed",
                "payment_status": "paid",
                "notes": "Original note",
                "wants_partner_board_contact": True,
                "created_at": "2026-03-03T00:00:00Z",
            }
        ],
        "tournament_registration_selections": [
            {
                "id": "selection_1",
                "tournament_id": "tour_1",
                "registration_id": "registration_1",
                "registration_day_id": "day_1",
                "event_option_id": "event_1",
                "partner_mode": "NEEDS_PARTNER",
                "show_on_partner_board": True,
            }
        ],
        "tournament_event_draws": [
            {"id": "draw_1", "tournament_id": "tour_1", "registration_day_id": "day_1", "event_option_id": "event_1", "name": "3.5 Draw", "status": "active", "team_count": 2}
        ],
        "tournament_teams": [
            {"id": "team_1", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 1, "player1_id": 1, "player2_id": 2, "source": "REGISTRATION"},
            {"id": "team_2", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 2, "player1_id": 3, "player2_id": 4, "source": "REGISTRATION"},
        ],
        "tournament_games": [
            {"id": "game_1", "tournament_id": "tour_1", "draw_id": "draw_1", "stage": "ROUND_ROBIN", "rr_round_number": 1, "rr_slot_number": 1, "team1_id": "team_1", "team2_id": "team_2", "score_team1": 11, "score_team2": 7, "winner_team_id": "team_1", "status": "complete"}
        ],
        "tournament_podium": [
            {"id": "podium_1", "tournament_id": "tour_1", "draw_id": "draw_1", "placement": 1, "team_id": "team_1", "award_label": "Gold"}
        ],
        "admin_activity_log": [],
    }


def _install_auth(monkeypatch):
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


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
    _install_auth(monkeypatch)

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
    _install_auth(monkeypatch)

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
    assert payload["registrations"][0]["notes"] == "Original note"
    assert payload["event_options"][0]["division_name"] == "3.5"
    assert payload["selections"][0]["event_label"] == "Gender Doubles / 3.5"


def test_admin_tournament_ops_snapshot_contract(monkeypatch):
    supabase = FakeSupabase(tournament_tables())
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).get(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/ops",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_ops_snapshot"
    assert payload["summary"] == {"draws": 1, "teams": 2, "games": 1, "podium": 1, "completed_games": 1}
    assert payload["draws"][0]["id"] == "draw_1"
    assert payload["teams"][0]["team_number"] == 1
    assert payload["games"][0]["winner_team_id"] == "team_1"
    assert payload["podium"][0]["award_label"] == "Gold"


def test_admin_tournament_registration_update_contract(monkeypatch):
    tables = tournament_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/registration_1",
        headers={"Authorization": "Bearer local"},
        json={
            "registration_status": "waitlist",
            "payment_status": "refunded",
            "notes": "Refunded after withdrawal.",
            "confirmation_text": "SAVE REGISTRATION",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_registration_update"
    assert payload["registration"]["registration_status"] == "waitlist"
    assert payload["registration"]["payment_status"] == "refunded"
    assert payload["registration"]["notes"] == "Refunded after withdrawal."
    assert tables["tournament_registrations"][0]["status"] == "waitlist"
    assert tables["tournament_registrations"][0]["payment_status"] == "refunded"
    assert tables["admin_activity_log"][0]["action_type"] == "update_tournament_registration_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_tournament_selection_update_contract(monkeypatch):
    tables = tournament_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/selections/selection_1",
        headers={"Authorization": "Bearer local"},
        json={
            "event_option_id": "event_2",
            "partner_mode": "HAS_PARTNER",
            "partner_name": "Blair Partner",
            "partner_email": "blair@example.com",
            "partner_phone": "555-0101",
            "partner_note": "Confirmed partner.",
            "confirmation_text": "SAVE SELECTION",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_selection_update"
    assert payload["selection"]["event_option_id"] == "event_2"
    assert payload["selection"]["event_label"] == "Gender Doubles / 4.0"
    assert payload["selection"]["partner_mode"] == "HAS_PARTNER"
    assert payload["selection"]["partner_name"] == "Blair Partner"
    assert tables["tournament_registration_selections"][0]["event_option_id"] == "event_2"
    assert tables["tournament_registration_selections"][0]["registration_day_id"] == "day_1"
    assert tables["admin_activity_log"][0]["action_type"] == "update_tournament_registration_selection_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_tournament_bulk_registration_update_contract(monkeypatch):
    tables = tournament_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/bulk",
        headers={"Authorization": "Bearer local"},
        json={
            "registration_ids": ["registration_1"],
            "registration_status": "cancelled",
            "payment_status": "refunded",
            "append_note": "Bulk cancellation.",
            "confirmation_text": "BULK UPDATE REGISTRATIONS",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_registration_bulk_update"
    assert payload["updated_count"] == 1
    assert payload["registration_ids"] == ["registration_1"]
    assert payload["registrations"][0]["registration_status"] == "cancelled"
    assert payload["registrations"][0]["payment_status"] == "refunded"
    assert tables["tournament_registrations"][0]["status"] == "cancelled"
    assert tables["tournament_registrations"][0]["payment_status"] == "refunded"
    assert tables["tournament_registrations"][0]["notes"] == "Original note\nBulk cancellation."
    assert tables["admin_activity_log"][0]["action_type"] == "bulk_update_tournament_registrations_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True
