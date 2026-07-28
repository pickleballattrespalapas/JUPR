from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def tournament_registration_import_tables(*, include_games: bool = False, missing_player: bool = False):
    tables = {
        "tournaments": [
            {"club_id": "club", "id": "tour_1", "name": "Spring Classic", "status": "PUBLISHED"}
        ],
        "tournament_event_draws": [
            {
                "id": "draw_1",
                "tournament_id": "tour_1",
                "name": "3.5 Draw",
                "registration_day_id": "day_1",
                "event_option_id": "event_1",
            }
        ],
        "tournament_event_options": [
            {
                "id": "event_1",
                "tournament_id": "tour_1",
                "eligibility_mode": "STANDARD",
            }
        ],
        "tournament_registrations": [
            {
                "id": "reg_1",
                "tournament_id": "tour_1",
                "display_name": "Alex Example",
                "email": "alex@example.com",
                "status": "confirmed",
                "player_id": None if missing_player else 1,
            },
            {
                "id": "reg_2",
                "tournament_id": "tour_1",
                "display_name": "Blair Partner",
                "email": "blair@example.com",
                "status": "confirmed",
                "player_id": 2,
            },
            {
                "id": "reg_wait",
                "tournament_id": "tour_1",
                "display_name": "Wait List",
                "email": "wait@example.com",
                "status": "waitlist",
                "player_id": 3,
            },
        ],
        "tournament_registration_selections": [
            {
                "id": "sel_1",
                "tournament_id": "tour_1",
                "registration_id": "reg_1",
                "registration_day_id": "day_1",
                "event_option_id": "event_1",
                "partner_email": "blair@example.com",
            },
            {
                "id": "sel_wait",
                "tournament_id": "tour_1",
                "registration_id": "reg_wait",
                "registration_day_id": "day_1",
                "event_option_id": "event_1",
            },
            {
                "id": "sel_other_event",
                "tournament_id": "tour_1",
                "registration_id": "reg_2",
                "registration_day_id": "day_1",
                "event_option_id": "event_2",
            },
        ],
        "tournament_teams": [
            {
                "id": "old_team",
                "tournament_id": "tour_1",
                "draw_id": "draw_1",
                "team_number": 1,
                "player1_id": 9,
                "source": "MANUAL",
            }
        ],
        "tournament_games": [],
        "admin_activity_log": [],
    }
    if include_games:
        tables["tournament_games"].append({"id": "game_1", "tournament_id": "tour_1", "draw_id": "draw_1"})
    return tables


def _install_auth(monkeypatch):
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def _client(monkeypatch, tables):
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)
    return TestClient(app)


def test_admin_tournament_registration_team_import_replace_contract(monkeypatch):
    tables = tournament_registration_import_tables()
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams/import-registrations",
        headers={"Authorization": "Bearer local"},
        json={"import_mode": "REPLACE", "confirmation_text": "IMPORT REGISTRATIONS"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_registration_team_import"
    assert payload["import_mode"] == "REPLACE"
    assert payload["updated_count"] == 1
    assert payload["teams"][0]["player1_id"] == 1
    assert payload["teams"][0]["player2_id"] == 2
    assert payload["teams"][0]["source"] == "REGISTRATION"
    assert len(tables["tournament_teams"]) == 1
    assert tables["tournament_teams"][0]["player1_id"] == 1
    assert tables["admin_activity_log"][0]["action_type"] == "import_tournament_registration_teams_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_tournament_registration_team_import_blocks_after_games(monkeypatch):
    tables = tournament_registration_import_tables(include_games=True)
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams/import-registrations",
        headers={"Authorization": "Bearer local"},
        json={"import_mode": "REPLACE", "confirmation_text": "IMPORT REGISTRATIONS"},
    )

    assert response.status_code == 400
    assert "already has games" in response.json()["detail"]


def test_admin_tournament_registration_team_import_requires_confirmation(monkeypatch):
    tables = tournament_registration_import_tables()
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams/import-registrations",
        headers={"Authorization": "Bearer local"},
        json={"import_mode": "REPLACE", "confirmation_text": "IMPORT"},
    )

    assert response.status_code == 400
    assert "IMPORT REGISTRATIONS" in response.json()["detail"]


def test_admin_tournament_registration_team_import_blocks_unlinked_players(monkeypatch):
    tables = tournament_registration_import_tables(missing_player=True)
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams/import-registrations",
        headers={"Authorization": "Bearer local"},
        json={"import_mode": "REPLACE", "confirmation_text": "IMPORT REGISTRATIONS"},
    )

    assert response.status_code == 400
    assert "could not be resolved" in response.json()["detail"]
