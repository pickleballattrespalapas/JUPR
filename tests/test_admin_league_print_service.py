from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from jupr_app.services.admin_league_print_service import (
    build_admin_league_printout,
    build_admin_top_players_printable,
)
from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase


def _match(match_id: int, *, date: str, week: int = 1, with_snapshots: bool = True) -> dict:
    row = {
        "club_id": "club",
        "id": match_id,
        "date": date,
        "league": "Open",
        "match_type": "League",
        "week_tag": f"Week {week}",
        "t1_p1": 1,
        "t1_p2": 2,
        "t2_p1": 3,
        "t2_p2": 4,
        "score_t1": 11,
        "score_t2": 8,
        "is_active": True,
    }
    if with_snapshots:
        row.update(
            {
                "t1_p1_r": 1600,
                "t1_p1_r_end": 1612,
                "t1_p2_r": 1500,
                "t1_p2_r_end": 1512,
                "t2_p1_r": 1400,
                "t2_p1_r_end": 1390,
                "t2_p2_r": 1300,
                "t2_p2_r_end": 1290,
            }
        )
    return row


def print_tables() -> dict[str, list[dict]]:
    return {
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex", "rating": 1640, "active": True},
            {"club_id": "club", "id": 2, "name": "Blair", "rating": 1520, "active": True},
            {"club_id": "club", "id": 3, "name": "Casey", "rating": 1390, "active": True},
            {"club_id": "club", "id": 4, "name": "Devon", "rating": 1290, "active": True},
            {"club_id": "club", "id": 5, "name": "Inactive", "rating": 2000, "active": False},
            {"club_id": "club", "id": 6, "name": "Legacy inactive", "rating": 2100, "is_active": False},
        ],
        "leagues_metadata": [
            {
                "club_id": "club",
                "league_name": "Open",
                "status": "active",
                "is_active": True,
                "started_at": "2026-01-01T00:00:00Z",
                "k_factor": 32,
                "min_games": 1,
                "schedule_config": {
                    "start_date": "2026-02-01",
                    "weekday": 6,
                    "weeks": 2,
                    "time_start": "18:00",
                    "time_end": "20:00",
                },
                "court_board_defaults": {"players_per_court": "4"},
                "rules_config": {},
                "awards_config": {"default_min_games": 1, "default_depth": 1},
                "event_tags": {},
            }
        ],
        "league_ratings": [
            {
                "club_id": "club", "id": 1, "player_id": 1, "league_name": "Open",
                "rating": 1640, "starting_rating": 1600, "wins": 8, "losses": 2,
                "matches_played": 10, "is_active": True,
            },
            {
                "club_id": "club", "id": 2, "player_id": 2, "league_name": "Open",
                "rating": 1520, "starting_rating": 1500, "wins": 7, "losses": 3,
                "matches_played": 10, "is_active": True,
            },
            {
                "club_id": "club", "id": 3, "player_id": 3, "league_name": "Open",
                "rating": 1390, "starting_rating": 1400, "wins": 3, "losses": 7,
                "matches_played": 10, "is_active": True,
            },
            {
                "club_id": "club", "id": 4, "player_id": 4, "league_name": "Open",
                "rating": 1290, "starting_rating": 1300, "wins": 2, "losses": 8,
                "matches_played": 10, "is_active": True,
            },
        ],
        "matches": [
            _match(1, date="2026-02-07T18:00:00Z", week=1),
            _match(2, date="2026-02-14T18:00:00Z", week=2, with_snapshots=False),
        ],
    }


def test_top_players_export_uses_previous_calendar_month_and_eligibility(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    tables = print_tables()
    tables["matches"] = [
        _match(index, date="2026-02-07T18:00:00Z") for index in range(1, 11)
    ] + [
        _match(100 + index, date="2026-03-07T18:00:00Z") for index in range(20)
    ]
    for index, match in enumerate(tables["matches"][:10]):
        # Both activity-column variants must remain excluded from the ranking.
        match["t1_p1"] = 5 if index % 2 == 0 else 6

    payload = build_admin_top_players_printable(
        FakeSupabase(tables),
        club_id="club",
        now_utc=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )

    assert payload["period"]["label"] == "February 2026"
    assert payload["minimum_games"] == 10
    assert [row["player_name"] for row in payload["rankings"]] == ["Blair", "Casey", "Devon"]
    assert all(row["games"] == 10 for row in payload["rankings"])
    assert payload["rankings"][0]["record"] == "10-0"
    assert "Inactive" not in {row["player_name"] for row in payload["rankings"]}
    assert "Legacy inactive" not in {row["player_name"] for row in payload["rankings"]}


def test_league_printout_has_true_weekly_leaders_and_top_performers(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")

    payload = build_admin_league_printout(
        FakeSupabase(print_tables()),
        club_id="club",
        league_name="Open",
        week_num=1,
    )

    assert payload["available_weeks"] == [1, 2]
    assert payload["selected_week"] == 1
    assert payload["weekly_rating_leaders"][0]["player_name"] == "Alex"
    assert payload["weekly_rating_leaders"][0]["rating_delta_jupr"] == 0.03
    assert payload["weekly_win_leaders"][0]["wins"] == 1
    assert payload["season_top_performer_count"] == 4
    assert {row["category_key"] for row in payload["season_top_performers"]} == {
        "highest_rating",
        "most_improved",
        "best_win_pct",
        "most_wins",
    }
    assert payload["detail"]["capabilities"]["roster_mutable"] is True
    assert payload["rating_source"] == "stored_snapshots"
    assert payload["has_printable_data"] is True
    assert payload["printable_sections"] == {
        "schedule": True,
        "weekly_leaders": True,
        "season_leaders": True,
        "standings": True,
        "roster": True,
    }


def test_empty_team_draft_is_not_printable_and_does_not_claim_player_standings(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    tables = print_tables()
    tables["leagues_metadata"][0].update(
        {
            "league_type": "Team",
            "status": "draft",
            "is_active": False,
            "schedule_config": {},
        }
    )
    tables["league_ratings"] = []
    tables["matches"] = []

    payload = build_admin_league_printout(
        FakeSupabase(tables),
        club_id="club",
        league_name="Open",
    )

    assert payload["has_printable_data"] is False
    assert payload["printable_sections"] == {
        "schedule": False,
        "weekly_leaders": False,
        "season_leaders": False,
        "standings": False,
        "roster": False,
    }
    assert payload["warnings"] == [
        "No printable league-night data is available yet; add a schedule, "
        "league roster, or scored results before printing."
    ]


def test_team_player_ratings_only_enable_roster_not_team_standings(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    tables = print_tables()
    tables["leagues_metadata"][0].update(
        {
            "league_type": "Team",
            "status": "draft",
            "is_active": False,
            "schedule_config": {},
        }
    )
    tables["matches"] = []

    payload = build_admin_league_printout(
        FakeSupabase(tables),
        club_id="club",
        league_name="Open",
    )

    assert payload["has_printable_data"] is True
    assert payload["printable_sections"]["standings"] is False
    assert payload["printable_sections"]["roster"] is True


def test_league_printout_replays_missing_selected_week_snapshots(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")

    payload = build_admin_league_printout(
        FakeSupabase(print_tables()),
        club_id="club",
        league_name="Open",
        week_num=2,
    )

    assert payload["rating_source"] == "stored_snapshots_with_python_replay"
    assert payload["warnings"] == [
        "Replayed 1 match(es) in Python because complete stored rating snapshots were unavailable."
    ]


def test_league_printout_rejects_unscored_week(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")

    try:
        build_admin_league_printout(
            FakeSupabase(print_tables()),
            club_id="club",
            league_name="Open",
            week_num=99,
        )
    except ValueError as exc:
        assert "not a scored week" in str(exc)
    else:
        raise AssertionError("Expected an invalid week to be rejected")


require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def _install_api(monkeypatch, tables: dict[str, list[dict]]) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(tables))
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="owner@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_authenticated_print_api_contracts(monkeypatch) -> None:
    tables = print_tables()
    tables["matches"] = [_match(index, date="2026-02-07T18:00:00Z") for index in range(1, 11)]
    _install_api(monkeypatch, tables)
    client = TestClient(app)

    league_response = client.get(
        "/admin/clubs/club/league-manager/leagues/Open/printout?week_num=1",
        headers={"Authorization": "Bearer local"},
    )
    top_response = client.get(
        "/admin/clubs/club/league-manager/top-players-printable?limit=50",
        headers={"Authorization": "Bearer local"},
    )

    assert league_response.status_code == 200
    assert league_response.json()["mode"] == "league_manager_printout"
    assert top_response.status_code == 200
    assert top_response.json()["mode"] == "league_top_players_printable"
    assert "email" not in str(top_response.json())
