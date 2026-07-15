from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def bulk_team_tables(*, include_games: bool = False):
    tables = {
        "tournaments": [
            {"club_id": "club", "id": "tour_1", "name": "Spring Classic", "status": "PUBLISHED"}
        ],
        "tournament_event_draws": [
            {"id": "draw_1", "tournament_id": "tour_1", "name": "3.5 Draw", "registration_day_id": "day_1", "event_option_id": "event_1"}
        ],
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex Example", "active": True},
            {"club_id": "club", "id": 2, "name": "Blair Partner", "active": True},
            {"club_id": "club", "id": 3, "name": "Casey Third", "active": True},
            {"club_id": "club", "id": 4, "name": "Devon Fourth", "active": True},
        ],
        "tournament_teams": [
            {"id": "old_team", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 1, "player1_id": 9, "source": "MANUAL"}
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


def test_admin_tournament_bulk_team_import_replace_contract(monkeypatch):
    tables = bulk_team_tables()
    client = _client(monkeypatch, tables)
    raw_text = "Player 1,Player 2,Seed,Notes\nAlex Example,Blair Partner,1,Top seed\nCasey Third,Devon Fourth,2,Second seed\n"

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams/import-bulk",
        headers={"Authorization": "Bearer local"},
        json={"raw_text": raw_text, "import_mode": "REPLACE", "confirmation_text": "IMPORT TEAMS"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_bulk_team_import"
    assert payload["import_mode"] == "REPLACE"
    assert payload["updated_count"] == 2
    assert [row["team_number"] for row in payload["teams"]] == [1, 2]
    assert payload["teams"][0]["player1_id"] == 1
    assert payload["teams"][0]["player2_id"] == 2
    assert payload["teams"][0]["source"] == "BULK_UPLOAD"
    assert len(tables["tournament_teams"]) == 2
    assert tables["admin_activity_log"][0]["action_type"] == "import_tournament_bulk_teams_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_tournament_bulk_team_import_blocks_after_games(monkeypatch):
    tables = bulk_team_tables(include_games=True)
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams/import-bulk",
        headers={"Authorization": "Bearer local"},
        json={"raw_text": "Player 1\nAlex Example\n", "import_mode": "REPLACE", "confirmation_text": "IMPORT TEAMS"},
    )

    assert response.status_code == 400
    assert "already has games" in response.json()["detail"]


def test_admin_tournament_bulk_team_import_requires_confirmation(monkeypatch):
    tables = bulk_team_tables()
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams/import-bulk",
        headers={"Authorization": "Bearer local"},
        json={"raw_text": "Player 1\nAlex Example\n", "import_mode": "REPLACE", "confirmation_text": "IMPORT"},
    )

    assert response.status_code == 400
    assert "IMPORT TEAMS" in response.json()["detail"]


def test_admin_tournament_bulk_team_import_blocks_unresolved_names(monkeypatch):
    tables = bulk_team_tables()
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams/import-bulk",
        headers={"Authorization": "Bearer local"},
        json={"raw_text": "Player 1\nUnknown Player\n", "import_mode": "REPLACE", "confirmation_text": "IMPORT TEAMS"},
    )

    assert response.status_code == 400
    assert "Unresolved player" in response.json()["detail"]
