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
                "event_type": "GENDER_DOUBLES",
                "partner_required": True,
                "competition_format": "STANDARD",
                "team_roster_size": 2,
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
                "id": "sel_2",
                "tournament_id": "tour_1",
                "registration_id": "reg_2",
                "registration_day_id": "day_1",
                "event_option_id": "event_1",
                "partner_email": "alex@example.com",
            },
        ],
        "tournament_registration_team_links": [
            {
                "id": "link_1",
                "tournament_id": "tour_1",
                "event_option_id": "event_1",
                "registration1_id": "reg_1",
                "registration2_id": "reg_2",
                "selection1_id": "sel_1",
                "selection2_id": "sel_2",
                "player1_id": 1,
                "player2_id": 2,
                "status": "ADMIN_CONFIRMED",
            }
        ],
        "tournament_registration_team_members": [
            {
                "id": "member_1",
                "team_link_id": "link_1",
                "tournament_id": "tour_1",
                "event_option_id": "event_1",
                "selection_id": "sel_1",
                "registration_id": "reg_1",
                "player_id": 1,
                "player_order": 1,
                "status": "ACTIVE",
            },
            {
                "id": "member_2",
                "team_link_id": "link_1",
                "tournament_id": "tour_1",
                "event_option_id": "event_1",
                "selection_id": "sel_2",
                "registration_id": "reg_2",
                "player_id": 2,
                "player_order": 2,
                "status": "ACTIVE",
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


def test_admin_tournament_registration_team_import_uses_one_canonical_team_for_reciprocal_partner_selections(
    monkeypatch,
):
    tables = tournament_registration_import_tables()
    tables["tournament_registrations"].append(
        {
            "id": "reg_needs_partner",
            "tournament_id": "tour_1",
            "display_name": "Casey Looking",
            "email": "casey@example.com",
            "status": "confirmed",
            "player_id": 3,
        }
    )
    tables["tournament_registration_selections"].append(
        {
            "id": "sel_needs_partner",
            "tournament_id": "tour_1",
            "registration_id": "reg_needs_partner",
            "registration_day_id": "day_1",
            "event_option_id": "event_1",
            "partner_mode": "NEEDS_PARTNER",
        }
    )
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams/import-registrations",
        headers={"Authorization": "Bearer local"},
        json={"import_mode": "REPLACE", "confirmation_text": "IMPORT REGISTRATIONS"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["updated_count"] == 1
    assert len(payload["teams"]) == 1
    assert {
        payload["teams"][0]["player1_id"],
        payload["teams"][0]["player2_id"],
    } == {1, 2}
    assert payload["warnings"] == [
        "Excluded 1 confirmed entry still marked NEEDS_PARTNER."
    ]
    assert len(tables["tournament_teams"]) == 1
    assert len(tables["admin_activity_log"]) == 1


def test_admin_tournament_registration_team_import_does_not_promote_mutual_free_text_to_a_team(
    monkeypatch,
):
    tables = tournament_registration_import_tables()
    tables["tournament_registration_team_links"] = []
    tables["tournament_registration_team_members"] = []
    before = [dict(row) for row in tables["tournament_teams"]]
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams/import-registrations",
        headers={"Authorization": "Bearer local"},
        json={"import_mode": "REPLACE", "confirmation_text": "IMPORT REGISTRATIONS"},
    )

    assert response.status_code == 400
    assert "could not be resolved" in response.json()["detail"]
    assert "Alex Example" in response.json()["detail"]
    assert "Blair Partner" in response.json()["detail"]
    assert tables["tournament_teams"] == before
    assert tables["admin_activity_log"] == []


def test_admin_tournament_registration_team_import_blocks_unlinked_partner_details_alongside_valid_team(
    monkeypatch,
):
    tables = tournament_registration_import_tables()
    tables["tournament_registrations"].append(
        {
            "id": "reg_3",
            "tournament_id": "tour_1",
            "display_name": "Casey Unresolved",
            "email": "casey@example.com",
            "status": "confirmed",
            "player_id": 3,
        }
    )
    tables["tournament_registration_selections"].append(
        {
            "id": "sel_3",
            "tournament_id": "tour_1",
            "registration_id": "reg_3",
            "registration_day_id": "day_1",
            "event_option_id": "event_1",
            "partner_mode": "HAS_PARTNER",
            "partner_email": "unlinked@example.com",
        }
    )
    before = [dict(row) for row in tables["tournament_teams"]]
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams/import-registrations",
        headers={"Authorization": "Bearer local"},
        json={"import_mode": "REPLACE", "confirmation_text": "IMPORT REGISTRATIONS"},
    )

    assert response.status_code == 400
    assert "Casey Unresolved" in response.json()["detail"]
    assert tables["tournament_teams"] == before
    assert tables["admin_activity_log"] == []


def test_admin_tournament_registration_team_import_preserves_singles_as_individual_teams(
    monkeypatch,
):
    tables = tournament_registration_import_tables()
    tables["tournament_event_options"][0].update(
        {
            "event_type": "SINGLES",
            "partner_required": False,
            "team_roster_size": 1,
        }
    )
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams/import-registrations",
        headers={"Authorization": "Bearer local"},
        json={"import_mode": "REPLACE", "confirmation_text": "IMPORT REGISTRATIONS"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["updated_count"] == 2
    assert {team["player1_id"] for team in payload["teams"]} == {1, 2}
    assert all(team["player2_id"] is None for team in payload["teams"])


def test_admin_tournament_registration_team_import_blocks_overlapping_confirmed_links_without_writing(
    monkeypatch,
):
    tables = tournament_registration_import_tables()
    tables["tournament_registrations"].append(
        {
            "id": "reg_3",
            "tournament_id": "tour_1",
            "display_name": "Casey Conflict",
            "email": "casey@example.com",
            "status": "confirmed",
            "player_id": 3,
        }
    )
    tables["tournament_registration_selections"].append(
        {
            "id": "sel_3",
            "tournament_id": "tour_1",
            "registration_id": "reg_3",
            "registration_day_id": "day_1",
            "event_option_id": "event_1",
            "partner_email": "alex@example.com",
        }
    )
    tables["tournament_registration_team_links"].append(
        {
            "id": "link_2",
            "tournament_id": "tour_1",
            "event_option_id": "event_1",
            "registration1_id": "reg_1",
            "registration2_id": "reg_3",
            "selection1_id": "sel_1",
            "selection2_id": "sel_3",
            "player1_id": 1,
            "player2_id": 3,
            "status": "ADMIN_CONFIRMED",
        }
    )
    tables["tournament_registration_team_members"].extend(
        [
            {
                "id": "member_3",
                "team_link_id": "link_2",
                "tournament_id": "tour_1",
                "event_option_id": "event_1",
                "selection_id": "sel_1",
                "registration_id": "reg_1",
                "player_id": 1,
                "player_order": 1,
                "status": "ACTIVE",
            },
            {
                "id": "member_4",
                "team_link_id": "link_2",
                "tournament_id": "tour_1",
                "event_option_id": "event_1",
                "selection_id": "sel_3",
                "registration_id": "reg_3",
                "player_id": 3,
                "player_order": 2,
                "status": "ACTIVE",
            },
        ]
    )
    before = [dict(row) for row in tables["tournament_teams"]]
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams/import-registrations",
        headers={"Authorization": "Bearer local"},
        json={"import_mode": "REPLACE", "confirmation_text": "IMPORT REGISTRATIONS"},
    )

    assert response.status_code == 400
    assert "Duplicate player IDs" in response.json()["detail"]
    assert "1" in response.json()["detail"]
    assert tables["tournament_teams"] == before
    assert tables["admin_activity_log"] == []


def test_admin_tournament_registration_team_import_blocks_mismatched_canonical_member_evidence(
    monkeypatch,
):
    tables = tournament_registration_import_tables()
    tables["tournament_registration_team_members"][1]["registration_id"] = "reg_1"
    before = [dict(row) for row in tables["tournament_teams"]]
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams/import-registrations",
        headers={"Authorization": "Bearer local"},
        json={"import_mode": "REPLACE", "confirmation_text": "IMPORT REGISTRATIONS"},
    )

    assert response.status_code == 400
    assert "link_1" in response.json()["detail"]
    assert tables["tournament_teams"] == before
    assert tables["admin_activity_log"] == []


def test_admin_tournament_registration_team_import_append_blocks_players_already_in_draw(
    monkeypatch,
):
    tables = tournament_registration_import_tables()
    tables["tournament_teams"][0]["player1_id"] = 1
    before = [dict(row) for row in tables["tournament_teams"]]
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams/import-registrations",
        headers={"Authorization": "Bearer local"},
        json={"import_mode": "APPEND", "confirmation_text": "IMPORT REGISTRATIONS"},
    )

    assert response.status_code == 400
    assert "already exist in the current draw" in response.json()["detail"]
    assert "1" in response.json()["detail"]
    assert tables["tournament_teams"] == before
    assert tables["admin_activity_log"] == []


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
