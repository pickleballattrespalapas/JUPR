from __future__ import annotations

from types import SimpleNamespace

import pytest

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
        self._start = 0
        self._order = None
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

    def order(self, key):
        self._order = key
        return self

    def range(self, start, end):
        self._start = start
        self._limit = end - start + 1
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
        if self._order:
            rows.sort(key=lambda row: row.get(self._order, ""))
        if self._limit is not None:
            rows = rows[self._start : self._start + self._limit]
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
    assert (
        payload["player_summary"]["games"],
        payload["player_summary"]["wins"],
        payload["player_summary"]["losses"],
    ) == (4, 3, 1)
    assert payload["recent_matches"][0]["match_id"] == 11
    assert payload["recent_matches"][0]["result"] == "L"
    assert "admin_flag" not in payload["weekly_results"][0]
    assert supabase.table_calls.count("leagues_metadata") == 2
    assert supabase.table_calls.count("players") == 2
    assert supabase.table_calls.count("league_ratings") == 2
    assert supabase.table_calls.count("matches") == 2
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


def test_weekly_report_opens_latest_played_week_before_future_schedule() -> None:
    db = fake_supabase()
    for row in db._tables['leagues_metadata']:
        if row.get('league_name') == 'Open':
            row['schedule_config'] = {'weeks': 12}
    data = build_public_league_results(db, club_id='club', league_name='Open')
    assert data['selected_week'] == 2
    assert data['weekly_highlights']['week_num'] == 2
    # A direct selection of an unplayed scheduled week still works.
    selected = build_public_league_results(db, club_id='club', league_name='Open', week_num=12)
    assert selected['selected_week'] == 12


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


@pytest.mark.parametrize('cap', [137, 1000])
def test_complete_league_history_survives_server_row_caps(monkeypatch, cap):
    db = fake_supabase()
    template = db._tables['matches'][0]
    db._tables['matches'] = [dict(template, id=i, week_tag=f'Week {1 + i // 83}') for i in range(1408)]
    db._tables['matches'] += [dict(template, id=9000, club_id='other'), dict(template, id=9001, deleted_at='2026-01-01')]
    original = FakeQuery.execute
    monkeypatch.setattr(FakeQuery, 'execute', lambda self: SimpleNamespace(data=original(self).data[:cap]))
    matches = _league_matches(db, club_id='club', league_name='Open')
    assert len(matches) == 1408
    assert len({row['id'] for row in matches}) == 1408
    result = build_public_league_results(db, club_id='club', league_name='Open')
    assert [week['week_num'] for week in result['weeks']] == list(range(1, 18))


def test_failed_later_history_page_does_not_publish_partial_results(monkeypatch):
    from jupr_app.data.paged_reads import DataReadUnavailable
    db = fake_supabase()
    original = FakeQuery.execute
    def execute(query):
        if query._start:
            raise RuntimeError('database unavailable')
        return original(query)
    monkeypatch.setattr(FakeQuery, 'execute', execute)
    with pytest.raises(DataReadUnavailable):
        build_public_league_results(db, club_id='club', league_name='Open')


def test_weekly_stat_rankings_follow_selected_week_and_card_rules():
    db = fake_supabase()
    payload = build_public_league_results(db, club_id="club", league_name="Open", week_num=2, weekly_min_games=1)
    rankings = payload["weekly_stat_rankings"]
    assert payload["selected_week"] == 2
    assert "court" not in rankings
    for key in ("biggest_climbers", "best_win_pct", "most_active"):
        assert [row["player_id"] for row in rankings[key]["entries"][:3]] == [
            row["player_id"] for row in payload["weekly_highlights"][key]
        ]
    weekly = [row for row in payload["weekly_results"] if row["week_num"] == 2]
    assert rankings["record_games"]["total_qualified"] == len(weekly)
    assert [row["metric_value"] for row in rankings["record_games"]["entries"]] == sorted(
        [row["games"] for row in weekly], reverse=True
    )
    assert "private@example.com" not in str(rankings)


def test_season_stat_rankings_match_season_preview_and_ignore_selected_week():
    week_one = build_public_league_results(fake_supabase(), club_id="club", league_name="Open", week_num=1, weekly_min_games=1)
    week_two = build_public_league_results(fake_supabase(), club_id="club", league_name="Open", week_num=2, weekly_min_games=20)
    rankings = week_one["season_stat_rankings"]
    assert rankings == week_two["season_stat_rankings"]
    assert set(rankings) == {"biggest_climbers", "best_win_pct", "most_active"}
    for key, ranking in rankings.items():
        assert [row["player_id"] for row in ranking["entries"][:3]] == [
            row["player_id"] for row in week_one["season_highlights"][key]
        ]
        assert "this week" not in ranking["qualification"]
    assert rankings["best_win_pct"]["qualification"] == "At least 4 games this league season."


def test_completed_week_keeps_all_fifty_two_participants_without_reactivating_them(monkeypatch):
    """Two 50-player rounds with substitutions must not shrink to 34 active people."""
    import jupr_app.services.public_league_results_service as service

    db = fake_supabase()
    db._tables["players"] = [
        {"id": pid, "club_id": "club", "name": f"Participant {pid}", "rating": 1500,
         "active": pid > 18, "inactive_at": "2026-10-01" if pid <= 18 else None}
        for pid in range(1, 53)
    ]
    db._tables["league_ratings"] = [
        {"club_id": "club", "player_id": pid, "league_name": "Open", "rating": 1500,
         "starting_rating": 1500, "wins": 4, "losses": 4, "matches_played": 8,
         "is_active": pid > 18, "inactive_at": "2026-10-01" if pid <= 18 else None}
        for pid in range(1, 53)
    ]
    template = db._tables["matches"][0]
    matches = []
    for round_index in range(2):
        participants = list(range(1, 51)) if round_index == 0 else [*range(1, 49), 51, 52]
        for index in range(50):
            slots = dict(zip(("t1_p1", "t1_p2", "t2_p1", "t2_p2"),
                             [participants[(index + shift) % 50] for shift in range(4)]))
            matches.append(dict(template, id=1000 + round_index * 50 + index,
                                week_tag="Week 4", **slots))
    db._tables["matches"] = matches
    award_inputs = {}
    monkeypatch.setattr(service, "_public_award_progress", lambda *_args, **kwargs:
                        award_inputs.update(kwargs) or {"awards": [], "award_count": 0})

    payload = build_public_league_results(db, club_id="club", league_name="Open",
                                         week_num=4, player_id=1)

    weekly = payload["weekly_results"]
    assert len(weekly) == 52
    assert {row["player_id"] for row in weekly} == set(range(1, 53))
    assert sum(row["games"] for row in weekly) == 400
    assert {row["player_id"]: row["games"] for row in weekly if row["player_id"] >= 49} == {
        49: 4, 50: 4, 51: 4, 52: 4,
    }
    assert payload["weekly_stat_rankings"]["record_games"]["total_qualified"] == 52
    assert len(payload["players"]) == 52
    assert payload["selected_player_id"] == 1
    assert payload["player_summary"]["games"] == 8
    assert payload["player_summary"]["rank"] is None
    assert len(payload["player_weekly"]) == 1
    # Correcting event history must not silently change current eligibility or awards.
    assert {row["player_id"] for row in payload["standings"]} == set(range(19, 53))
    assert {row["player_id"] for row in award_inputs["league_rows"]} == set(range(19, 53))
    assert {row["id"] for row in award_inputs["player_rows"]} == set(range(19, 53))
    assert db._tables["players"][0]["active"] is False
    assert db.table_calls.count("players") == 2  # one complete paged read, no second eligibility fetch


@pytest.mark.parametrize("inactive_source", ["profile", "league", "both"])
def test_later_inactivity_does_not_rewrite_a_played_week(inactive_source):
    db = fake_supabase()
    before = build_public_league_results(db, club_id="club", league_name="Open",
                                        week_num=1, weekly_min_games=1)
    if inactive_source in {"profile", "both"}:
        db._tables["players"][0].update(active=False, inactive_at="2026-11-08")
    if inactive_source in {"league", "both"}:
        db._tables["league_ratings"][0].update(is_active=False, inactive_at="2026-11-08")

    after = build_public_league_results(db, club_id="club", league_name="Open",
                                       week_num=1, player_id=1, weekly_min_games=1)

    assert after["weekly_results"] == before["weekly_results"]
    assert after["weekly_highlights"] == before["weekly_highlights"]
    assert after["weekly_stat_rankings"] == before["weekly_stat_rankings"]
    assert after["selected_player_id"] == 1
    assert after["player_weekly"]
    assert 1 not in {row["player_id"] for row in after["standings"]}


def test_weekly_history_keeps_identity_and_tenant_boundaries():
    db = fake_supabase()
    db._tables["players"][3].update(name="Former identity (merged into 1)", active=False)
    db._tables["players"].append({"id": 99, "club_id": "other", "name": "Other club", "active": False})
    template = db._tables["matches"][0]
    db._tables["matches"].append(dict(template, id=99, club_id="other", t1_p1=99))

    payload = build_public_league_results(db, club_id="club", league_name="Open")

    assert {row["player_id"] for row in payload["weekly_results"]} == {1, 2, 3}
    assert {row["player_id"] for row in payload["players"]} == {1, 2, 3}
    assert "Other club" not in str(payload)
    assert "Former identity" not in str(payload)
