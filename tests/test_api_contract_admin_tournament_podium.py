from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def tournament_podium_tables(*, include_playoffs: bool = True):
    games = [
        {
            "id": "rr_1",
            "tournament_id": "tour_1",
            "draw_id": "draw_1",
            "stage": "ROUND_ROBIN",
            "team_a_id": "team_1",
            "team_b_id": "team_2",
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "team_1",
            "loser_team_id": "team_2",
            "finalized_at": "2026-03-01T00:00:00Z",
        },
        {
            "id": "rr_2",
            "tournament_id": "tour_1",
            "draw_id": "draw_1",
            "stage": "ROUND_ROBIN",
            "team_a_id": "team_3",
            "team_b_id": "team_4",
            "score_a": 11,
            "score_b": 4,
            "winner_team_id": "team_3",
            "loser_team_id": "team_4",
            "finalized_at": "2026-03-01T00:00:00Z",
        },
    ]
    if include_playoffs:
        games.extend(
            [
                {
                    "id": "final_1",
                    "tournament_id": "tour_1",
                    "draw_id": "draw_1",
                    "stage": "PLAYOFF",
                    "playoff_round": "Final",
                    "team_a_id": "team_1",
                    "team_b_id": "team_3",
                    "score_a": 8,
                    "score_b": 11,
                    "winner_team_id": "team_3",
                    "loser_team_id": "team_1",
                    "finalized_at": "2026-03-02T00:00:00Z",
                },
                {
                    "id": "bronze_1",
                    "tournament_id": "tour_1",
                    "draw_id": "draw_1",
                    "stage": "PLAYOFF",
                    "playoff_round": "Bronze",
                    "team_a_id": "team_2",
                    "team_b_id": "team_4",
                    "score_a": 11,
                    "score_b": 5,
                    "winner_team_id": "team_2",
                    "loser_team_id": "team_4",
                    "finalized_at": "2026-03-02T00:00:00Z",
                },
            ]
        )
    return {
        "tournaments": [
            {"club_id": "club", "id": "tour_1", "name": "Spring Classic", "status": "PUBLISHED"}
        ],
        "tournament_event_draws": [
            {"id": "draw_1", "tournament_id": "tour_1", "name": "3.5 Draw", "status": "draft"},
            {"id": "draw_2", "tournament_id": "tour_1", "name": "4.0 Draw", "status": "draft"},
        ],
        "tournament_teams": [
            {"id": "team_1", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 1, "player1_id": 1, "player2_id": 2},
            {"id": "team_2", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 2, "player1_id": 3, "player2_id": 4},
            {"id": "team_3", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 3, "player1_id": 5, "player2_id": 6},
            {"id": "team_4", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 4, "player1_id": 7, "player2_id": 8},
        ],
        "tournament_games": games,
        "tournament_podium": [
            {"id": "old_row", "tournament_id": "tour_1", "draw_id": "draw_2", "placement": 1, "team_id": "other_team", "source": "PLAYOFF"}
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


def _client(monkeypatch, tables):
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)
    return TestClient(app)


def test_admin_tournament_draw_podium_from_playoffs_contract(monkeypatch):
    tables = tournament_podium_tables(include_playoffs=True)
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/podium",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "GENERATE PODIUM"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_draw_podium_generate"
    assert payload["podium_source"] == "PLAYOFF"
    assert [(row["placement"], row["team_id"], row["draw_id"]) for row in payload["podium"]] == [
        (1, "team_3", "draw_1"),
        (2, "team_1", "draw_1"),
        (3, "team_2", "draw_1"),
    ]
    assert [row for row in tables["tournament_podium"] if row.get("draw_id") == "draw_2"]
    assert tables["admin_activity_log"][0]["action_type"] == "generate_tournament_draw_podium_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_tournament_draw_podium_from_round_robin_without_playoffs(monkeypatch):
    tables = tournament_podium_tables(include_playoffs=False)
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/podium",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "GENERATE PODIUM"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["podium_source"] == "ROUND_ROBIN"
    assert len(payload["podium"]) == 3
    assert all(row["draw_id"] == "draw_1" for row in payload["podium"])


def test_admin_tournament_draw_podium_requires_confirmation(monkeypatch):
    tables = tournament_podium_tables(include_playoffs=True)
    client = _client(monkeypatch, tables)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/podium",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "GENERATE"},
    )

    assert response.status_code == 400
    assert "GENERATE PODIUM" in response.json()["detail"]
