from __future__ import annotations

from types import SimpleNamespace

from jupr_app.services.public_league_results_service import (
    _league_matches,
    _parse_week_num,
    build_public_league_results,
    get_public_league_results_overview,
)


def test_week_number_requires_an_explicit_week_label() -> None:
    assert _parse_week_num("Week 2") == 2
    assert _parse_week_num("League week #7 round 1") == 7
    assert _parse_week_num("E2E 4dce01a8-32797001253-1") is None
    assert _parse_week_num("2026-08-24") is None


class FakeQuery:
    def __init__(self, rows, *, strict_select: bool = False):
        self._rows = list(rows)
        self._strict_select = bool(strict_select)
        self._filters: dict[str, object] = {}
        self._limit: int | None = None
        self._selected_columns: list[str] | None = None

    def select(self, columns="*", *_args, **_kwargs):
        if columns != "*":
            self._selected_columns = [
                column.strip() for column in str(columns).split(",") if column.strip()
            ]
        return self

    def eq(self, key, value):
        self._filters[key] = value
        return self

    def is_(self, key, value):
        assert value is None
        self._filters[key] = None
        return self

    def limit(self, value):
        self._limit = int(value)
        return self

    def execute(self):
        rows = list(self._rows)
        if self._strict_select and self._selected_columns is not None:
            schema_columns = {column for row in rows for column in row}
            unknown = [
                column
                for column in self._selected_columns
                if column not in schema_columns
            ]
            if unknown:
                raise RuntimeError(f"Unknown selected columns: {', '.join(unknown)}")
        for key, expected in self._filters.items():
            rows = [row for row in rows if row.get(key) == expected]
        if self._limit is not None:
            rows = rows[: self._limit]
        if self._selected_columns is not None:
            rows = [
                {
                    column: row.get(column)
                    for column in self._selected_columns
                    if column in row
                }
                for row in rows
            ]
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, tables, *, strict_select_tables=()):
        self._tables = tables
        self._strict_select_tables = set(strict_select_tables)
        self.table_calls: list[str] = []

    def table(self, name):
        self.table_calls.append(str(name))
        return FakeQuery(
            self._tables.get(name, []),
            strict_select=name in self._strict_select_tables,
        )


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
                {
                    "club_id": "club",
                    "league_name": "Open",
                    "is_active": True,
                    "status": "active",
                    "min_games": 4,
                    "k_factor": 24,
                    "schedule_config": {},
                },
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
                    "deleted_at": None,
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
                    "deleted_at": None,
                },
                {
                    "id": 12,
                    "club_id": "club",
                    "league": "Open",
                    "match_type": "PopUp",
                    "week_tag": "Week 2",
                    "score_t1": 11,
                    "score_t2": 1,
                    "deleted_at": None,
                },
                {
                    "id": 13,
                    "club_id": "club",
                    "date": "2026-01-15T00:00:00Z",
                    "league": "Open",
                    "match_type": "Live Match",
                    "week_tag": "Week 3",
                    "t1_p1": 3,
                    "t1_p2": 4,
                    "t2_p1": 1,
                    "t2_p2": 2,
                    "score_t1": 11,
                    "score_t2": 7,
                    "deleted_at": "2026-01-16T00:00:00Z",
                },
                {
                    "id": 14,
                    "club_id": "club",
                    "league": "Deleted Only",
                    "match_type": "Live Match",
                    "week_tag": "Week 1",
                    "score_t1": 11,
                    "score_t2": 7,
                    "deleted_at": "2026-01-16T00:00:00Z",
                },
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
                "league_type": "Individual",
                "match_format": "doubles",
                "start_week": None,
            "end_week": None,
            "num_weeks": None,
        }
    ]
    assert overview["past_leagues"] == []


def test_public_league_results_maps_configured_schedule_weeks() -> None:
    overview = get_public_league_results_overview(
        FakeSupabase(
            {
                "leagues_metadata": [
                    {
                        "club_id": "club",
                        "league_name": "Scheduled",
                        "is_active": True,
                        "status": "active",
                        "min_games": 2,
                        "k_factor": 24,
                        "schedule_config": {"weeks": 6},
                    }
                ]
            }
        ),
        club_id="club",
    )

    assert overview["leagues"] == [
            {
                "name": "Scheduled",
                "min_games": 2,
                "k_factor": 24,
                "league_type": "Individual",
                "match_format": "doubles",
                "start_week": None,
            "end_week": None,
            "num_weeks": 6,
        }
    ]
    assert overview["past_leagues"] == []


def test_public_league_results_filters_matches_before_the_fetch_limit() -> None:
    other_rows = [
        {
            "id": index,
            "club_id": "club",
            "league": "Other",
            "match_type": "Live Match",
            "score_t1": 11,
            "score_t2": 7,
            "deleted_at": None,
        }
        for index in range(2500)
    ]
    deleted_target_rows = [
        {
            "id": 2500 + index,
            "club_id": "club",
            "league": "Target",
            "match_type": "Live Match",
            "score_t1": 11,
            "score_t2": 7,
            "deleted_at": "2026-01-01T00:00:00Z",
        }
        for index in range(2500)
    ]
    target = {
        "id": 9001,
        "club_id": "club",
        "league": "Target",
        "match_type": "Live Match",
        "score_t1": 11,
        "score_t2": 7,
        "deleted_at": None,
    }

    rows = _league_matches(
        FakeSupabase({"matches": [*other_rows, *deleted_target_rows, target]}),
        club_id="club",
        league_name="Target",
    )

    assert [row["id"] for row in rows] == [9001]


def test_public_match_select_matches_the_deployed_staging_schema() -> None:
    supabase = fake_supabase()
    supabase._strict_select_tables.add("matches")

    rows = _league_matches(
        supabase,
        club_id="club",
        league_name="Open",
    )

    assert [row["id"] for row in rows] == [10, 11]


def test_public_league_results_builds_standings_weekly_and_highlights() -> None:
    supabase = fake_supabase()
    payload = build_public_league_results(
        supabase, club_id="club", league_name="Open"
    )

    assert payload["selected_league"] == "Open"
    assert payload["standings"][0]["player_name"] == "Alex"
    assert payload["standings"][0]["rank"] == 1
    assert payload["standings"][0]["rating_jupr"] == 4.1
    assert payload["standings"][0]["rating_delta_jupr"] == 0.1
    assert "admin_notes" not in payload["standings"][0]
    assert "Devon" not in {row["player_name"] for row in payload["standings"]}
    alex_season = next(
        row for row in payload["cumulative"] if row["player_name"] == "Alex"
    )
    assert (alex_season["games"], alex_season["wins"], alex_season["losses"]) == (
        4,
        3,
        1,
    )

    assert payload["weeks"] == [
        {"week_num": 1, "week_label": "Week 1", "has_results": True},
        {"week_num": 2, "week_label": "Week 2", "has_results": True},
    ]
    assert payload["selected_week"] == 2
    week_two = [row for row in payload["weekly_results"] if row["week_num"] == 2]
    assert {row["player_name"] for row in week_two} == {"Alex", "Blair", "Casey"}
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
    assert len(payload["players"]) == 3
    assert payload["selected_player_id"] == 1
    assert payload["player_summary"]["player_name"] == "Alex"
    assert (
        payload["player_summary"]["games"],
        payload["player_summary"]["wins"],
        payload["player_summary"]["losses"],
    ) == (4, 3, 1)
    assert payload["recent_matches"][0]["match_id"] == 11
    assert payload["recent_matches"][0]["result"] == "L"
    assert "admin_flag" not in payload["weekly_results"][0]
    assert supabase.table_calls.count("leagues_metadata") == 1
    assert supabase.table_calls.count("players") == 1
    assert supabase.table_calls.count("league_ratings") == 1
    assert supabase.table_calls.count("matches") == 1
    assert "team_league_teams" not in supabase.table_calls
    assert "team_league_fixtures" not in supabase.table_calls


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
    assert (
        payload["player_summary"]["games"],
        payload["player_summary"]["wins"],
        payload["player_summary"]["losses"],
    ) == (4, 2, 2)
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


def test_public_league_results_never_substitutes_an_explicit_missing_league() -> None:
    payload = build_public_league_results(
        fake_supabase(),
        club_id="club",
        league_name="Spring League",
    )

    assert payload["selected_league"] is None
    assert payload["standings"] == []
    assert payload["weekly_results"] == []


def test_public_league_results_does_not_resolve_exact_archived_deep_link() -> None:
    payload = build_public_league_results(
        fake_supabase(),
        club_id="club",
        league_name="Archived",
    )

    assert payload["selected_league"] is None
    assert payload["league"] is None
    assert payload["standings"] == []
    assert payload["weekly_results"] == []
    assert [row["name"] for row in payload["leagues"]] == ["Open"]


def test_public_exact_links_reject_every_historical_lifecycle_status() -> None:
    for status in ("draft", "paused", "archived"):
        league_name = f"Historical {status}"
        payload = build_public_league_results(
            FakeSupabase(
                {
                    "leagues_metadata": [
                        {
                            "club_id": "club",
                            "league_name": league_name,
                            "is_active": status == "paused",
                            "status": status,
                        }
                    ],
                    "league_ratings": [
                        {
                            "club_id": "club",
                            "league_name": league_name,
                            "is_active": status == "paused",
                        }
                    ],
                    "matches": [
                        {
                            "club_id": "club",
                            "league": league_name,
                            "match_type": "Live Match",
                            "score_t1": 11,
                            "score_t2": 7,
                            "deleted_at": None,
                        }
                    ],
                }
            ),
            club_id="club",
            league_name=league_name,
        )

        assert payload["selected_league"] is None
        assert payload["standings"] == []


def test_ended_league_is_available_only_in_the_past_collection() -> None:
    league_name = "Finished League"
    payload = build_public_league_results(
        FakeSupabase(
            {
                "leagues_metadata": [
                    {
                        "club_id": "club",
                        "league_name": league_name,
                        "is_active": False,
                        "status": "ended",
                    }
                ],
                "league_ratings": [
                    {
                        "club_id": "club",
                        "league_name": league_name,
                        "player_id": 9,
                        "rating": 1520,
                        "starting_rating": 1480,
                        "wins": 4,
                        "losses": 2,
                        "matches_played": 6,
                        "is_active": False,
                    }
                ],
                "matches": [],
                "players": [
                    {
                        "id": 9,
                        "club_id": "club",
                        "name": "Former Player",
                        "rating": 1520,
                        "active": False,
                        "inactive_at": "2026-08-01",
                    }
                ],
            }
        ),
        club_id="club",
        league_name=league_name,
    )

    assert payload["selected_league"] == league_name
    assert payload["leagues"] == []
    assert [row["name"] for row in payload["past_leagues"]] == [league_name]
    assert [row["player_name"] for row in payload["standings"]] == ["Former Player"]


def test_inactive_metadata_blocks_match_and_rating_fallback_from_public_overview() -> None:
    supabase = FakeSupabase(
        {
            "leagues_metadata": [
                {
                    "club_id": "club",
                    "league_name": "Archived",
                    "is_active": False,
                    "status": "archived",
                }
            ],
            "league_ratings": [
                {
                    "club_id": "club",
                    "league_name": "Archived",
                    "is_active": True,
                }
            ],
            "matches": [
                {
                    "club_id": "club",
                    "league": "Archived",
                    "match_type": "Live Match",
                    "score_t1": 11,
                    "score_t2": 7,
                    "deleted_at": None,
                }
            ],
        }
    )

    assert get_public_league_results_overview(supabase, club_id="club") == {
        "leagues": [],
        "past_leagues": [],
    }
