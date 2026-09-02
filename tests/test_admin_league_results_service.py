from __future__ import annotations

import pytest

from jupr_app.services.admin_league_results_service import build_admin_league_results
from tests.test_public_league_results_service import FakeSupabase


def archived_results_supabase() -> FakeSupabase:
    return FakeSupabase(
        {
            "leagues_metadata": [
                {
                    "id": "league-archived-1",
                    "club_id": "club",
                    "league_name": "Winter 2025",
                    "league_type": "Individual",
                    "is_active": False,
                    "status": "archived",
                    "min_games": 1,
                    "k_factor": 24,
                    "schedule_config": {"weeks": 1},
                }
            ],
            "players": [
                {
                    "id": player_id,
                    "club_id": "club",
                    "name": name,
                    "rating": 1600 - (player_id * 50),
                    "active": False,
                    "inactive_at": "2026-01-01T00:00:00Z",
                }
                for player_id, name in enumerate(
                    ["Alex", "Blair", "Casey", "Devon"],
                    start=1,
                )
            ],
            "league_ratings": [
                {
                    "club_id": "club",
                    "player_id": 1,
                    "league_name": "Winter 2025",
                    "rating": 1640,
                    "starting_rating": 1600,
                    "wins": 1,
                    "losses": 0,
                    "matches_played": 1,
                    "is_active": False,
                }
            ],
            "matches": [
                {
                    "id": 10,
                    "club_id": "club",
                    "date": "2025-12-01T00:00:00Z",
                    "league": "Winter 2025",
                    "match_type": "Live Match",
                    "week_tag": "Week 1",
                    "t1_p1": 1,
                    "t1_p2": 2,
                    "t2_p1": 3,
                    "t2_p2": 4,
                    "score_t1": 11,
                    "score_t2": 7,
                    "t1_p1_r": 1600,
                    "t1_p1_r_end": 1610,
                    "t1_p2_r": 1500,
                    "t1_p2_r_end": 1510,
                    "t2_p1_r": 1400,
                    "t2_p1_r_end": 1390,
                    "t2_p2_r": 1300,
                    "t2_p2_r_end": 1290,
                    "deleted_at": None,
                }
            ],
        }
    )


def test_admin_results_resolve_exact_archived_league_and_inactive_players() -> None:
    payload = build_admin_league_results(
        archived_results_supabase(),
        club_id="club",
        league_name="Winter 2025",
    )

    assert payload["mode"] == "league_manager_results"
    assert payload["league_id"] == "league-archived-1"
    assert payload["selected_league"] == "Winter 2025"
    assert payload["league_status"] == "archived"
    assert payload["publicly_visible"] is False
    assert payload["standings"][0]["player_name"] == "Alex"
    assert payload["standings"][0]["rating_jupr"] == 4.1
    assert {row["player_name"] for row in payload["weekly_results"]} == {
        "Alex",
        "Blair",
        "Casey",
        "Devon",
    }


def test_admin_results_require_one_exact_metadata_row() -> None:
    with pytest.raises(ValueError, match="not found"):
        build_admin_league_results(
            archived_results_supabase(),
            club_id="club",
            league_name="Winter 2024",
        )


def test_active_admin_results_only_include_current_league_members() -> None:
    tables = {
        "leagues_metadata": [
            {
                "id": "active-1",
                "club_id": "club",
                "league_name": "Summer",
                "league_type": "Individual",
                "match_format": "doubles",
                "is_active": True,
                "status": "active",
                "min_games": 1,
                "k_factor": 24,
                "schedule_config": {"weeks": 1},
            }
        ],
        "players": [
            {"id": player_id, "club_id": "club", "name": name, "rating": 1500, "active": True}
            for player_id, name in ((1, "Current A"), (2, "Current B"), (3, "Former A"), (4, "Former B"))
        ],
        "league_ratings": [
            {"club_id": "club", "player_id": 1, "league_name": "Summer", "rating": 1510, "starting_rating": 1500, "wins": 1, "losses": 0, "matches_played": 1, "is_active": True},
            {"club_id": "club", "player_id": 2, "league_name": "Summer", "rating": 1490, "starting_rating": 1500, "wins": 0, "losses": 1, "matches_played": 1, "is_active": True},
            {"club_id": "club", "player_id": 3, "league_name": "Summer", "rating": 1600, "starting_rating": 1500, "wins": 4, "losses": 0, "matches_played": 4, "is_active": False, "inactive_at": "2026-08-01T00:00:00Z"},
            {"club_id": "club", "player_id": 4, "league_name": "Summer", "rating": 1400, "starting_rating": 1500, "wins": 0, "losses": 4, "matches_played": 4, "is_active": False, "inactive_at": "2026-08-01T00:00:00Z"},
        ],
        "matches": [
            {
                "id": 20,
                "club_id": "club",
                "date": "2026-08-01T00:00:00Z",
                "league": "Summer",
                "match_type": "Live Match",
                "week_tag": "Week 1",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 7,
                "deleted_at": None,
            }
        ],
    }

    payload = build_admin_league_results(
        FakeSupabase(tables), club_id="club", league_name="Summer"
    )

    assert {row["player_name"] for row in payload["standings"]} == {
        "Current A",
        "Current B",
    }
    assert {row["player_name"] for row in payload["weekly_results"]} == {
        "Current A",
        "Current B",
    }
