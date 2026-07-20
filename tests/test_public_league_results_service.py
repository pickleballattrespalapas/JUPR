from __future__ import annotations

from types import SimpleNamespace

from jupr_app.services.public_league_results_service import (
    build_public_league_results,
    get_public_league_results_overview,
)


class FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)
        self._filters: dict[str, object] = {}
        self._limit: int | None = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self._filters[key] = value
        return self

    def limit(self, value):
        self._limit = int(value)
        return self

    def execute(self):
        rows = list(self._rows)
        for key, expected in self._filters.items():
            rows = [row for row in rows if row.get(key) == expected]
        if self._limit is not None:
            rows = rows[: self._limit]
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, tables):
        self._tables = tables

    def table(self, name):
        return FakeQuery(self._tables.get(name, []))


def fake_supabase() -> FakeSupabase:
    return FakeSupabase(
        {
            "players": [
                {"id": 1, "club_id": "club", "name": "Alex", "rating": 1600, "active": True, "email": "private@example.com"},
                {"id": 2, "club_id": "club", "name": "Blair", "rating": 1500, "active": True},
                {"id": 3, "club_id": "club", "name": "Casey", "rating": 1400, "active": True},
                {"id": 4, "club_id": "club", "name": "Devon", "rating": 1300, "active": True},
            ],
            "leagues_metadata": [
                {"club_id": "club", "league_name": "Open", "is_active": True, "status": "active", "min_games": 4, "k_factor": 24},
                {"club_id": "club", "league_name": "Archived", "is_active": False, "status": "archived", "min_games": 4, "k_factor": 24},
            ],
            "league_ratings": [
                {"club_id": "club", "player_id": 1, "league_name": "Open", "rating": 1640, "starting_rating": 1600, "wins": 3, "losses": 1, "matches_played": 4, "is_active": True, "admin_notes": "private"},
                {"club_id": "club", "player_id": 2, "league_name": "Open", "rating": 1500, "starting_rating": 1500, "wins": 2, "losses": 2, "matches_played": 4, "is_active": True},
                {"club_id": "club", "player_id": 3, "league_name": "Open", "rating": 1400, "starting_rating": 1450, "wins": 1, "losses": 3, "matches_played": 4, "is_active": True},
            ],
            "matches": [
                {
                    "id": 10,
                    "club_id": "club",
                    "date": "2026-01-01T00:00:00Z",
                    "league": "Open",
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
                    "admin_flag": "secret",
                },
                {
                    "id": 11,
                    "club_id": "club",
                    "date": "2026-01-08T00:00:00Z",
                    "league": "Open",
                    "match_type": "Live Match",
                    "week_tag": "Week 2",
                    "t1_p1": 1,
                    "t1_p2": 3,
                    "t2_p1": 2,
                    "t2_p2": 4,
                    "score_t1": 9,
                    "score_t2": 11,
                    "t1_p1_r": 1610,
                    "t1_p1_r_end": 1605,
                    "t1_p2_r": 1390,
                    "t1_p2_r_end": 1385,
                    "t2_p1_r": 1510,
                    "t2_p1_r_end": 1515,
                    "t2_p2_r": 1290,
                    "t2_p2_r_end": 1295,
                },
                {"id": 12, "club_id": "club", "league": "Open", "match_type": "PopUp", "week_tag": "Week 2", "score_t1": 11, "score_t2": 1},
            ],
        }
    )


def test_public_league_results_overview_excludes_inactive_leagues() -> None:
    overview = get_public_league_results_overview(fake_supabase(), club_id="club")

    assert overview["leagues"] == [
        {
            "name": "Open",
            "min_games": 4,
            "k_factor": 24,
            "start_week": None,
            "end_week": None,
            "num_weeks": None,
        }
    ]


def test_public_league_results_builds_standings_weekly_and_highlights() -> None:
    payload = build_public_league_results(fake_supabase(), club_id="club", league_name="Open")

    assert payload["selected_league"] == "Open"
    assert payload["standings"][0]["player_name"] == "Alex"
    assert payload["standings"][0]["rank"] == 1
    assert payload["standings"][0]["rating_jupr"] == 4.1
    assert payload["standings"][0]["rating_delta_jupr"] == 0.1
    assert "admin_notes" not in payload["standings"][0]

    assert payload["weeks"] == [
        {"week_num": 1, "week_label": "Week 1", "has_results": True},
        {"week_num": 2, "week_label": "Week 2", "has_results": True},
    ]
    assert payload["selected_week"] == 2
    week_two = [row for row in payload["weekly_results"] if row["week_num"] == 2]
    assert {row["player_name"] for row in week_two} == {"Alex", "Blair", "Casey", "Devon"}
    blair_week_two = next(row for row in week_two if row["player_name"] == "Blair")
    assert blair_week_two["rank"] == 2
    assert blair_week_two["rank_delta"] == 0
    assert blair_week_two["rating_delta_jupr"] == 0.0125
    assert payload["weekly_highlights"]["scope"] == "week"
    assert payload["weekly_highlights"]["week_num"] == 2
    assert payload["weekly_highlights"]["min_games"] == 4
    assert payload["weekly_highlights"]["best_win_pct"] == []
    assert payload["season_highlights"]["scope"] == "season"
    assert payload["season_highlights"]["min_games"] == 4
    assert payload["season_highlights"]["best_win_pct"][0]["player_name"] == "Alex"
    assert len(payload["players"]) == 4
    assert payload["selected_player_id"] == 1
    assert payload["player_summary"]["player_name"] == "Alex"
    assert payload["recent_matches"][0]["match_id"] == 11
    assert payload["recent_matches"][0]["result"] == "L"
    assert "admin_flag" not in payload["weekly_results"][0]


def test_public_league_results_honors_week_player_and_qualification_deep_links() -> None:
    payload = build_public_league_results(
        fake_supabase(),
        club_id="club",
        league_name="Open",
        week_num=1,
        player_id=2,
        weekly_min_games=1,
    )

    assert payload["selected_week"] == 1
    assert payload["selected_player_id"] == 2
    assert payload["player_summary"]["player_name"] == "Blair"
    assert payload["weekly_highlights"]["min_games"] == 1
    assert payload["weekly_highlights"]["best_win_pct"][0]["player_name"] in {"Alex", "Blair"}
    assert payload["recent_matches"][0]["partner"]["player_name"] == "Devon"
    assert [row["player_name"] for row in payload["recent_matches"][0]["opponents"]] == ["Alex", "Casey"]


def test_public_league_results_returns_empty_payload_when_no_leagues() -> None:
    payload = build_public_league_results(FakeSupabase({}), club_id="club")

    assert payload["selected_league"] is None
    assert payload["standings"] == []
    assert payload["weekly_results"] == []
    assert payload["recent_matches"] == []
