from __future__ import annotations

from jupr_app.services.leaderboard_service import LeaderboardDataUnavailable, build_public_leaderboard


class _Response:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, store, table_name):
        self.store = store
        self.table_name = table_name
        self.filters = {}

    def select(self, _columns):
        return self

    def eq(self, key, value):
        self.filters[key] = value
        return self

    def range(self, start, end):
        self.page_bounds = (start, end)
        return self

    def order(self, *_args, **_kwargs):
        return self

    def execute(self):
        rows = [dict(row) for row in self.store.get(self.table_name, [])]
        for key, value in self.filters.items():
            rows = [row for row in rows if str(row.get(key)) == str(value)]
        if hasattr(self, "page_bounds"):
            rows = rows[self.page_bounds[0]:self.page_bounds[1] + 1]
        return _Response(rows)


class _Supabase:
    def __init__(self, store):
        self.store = store

    def table(self, table_name):
        return _Query(self.store, table_name)


def _fixture():
    return _Supabase(
        {
            "players": [
                {
                    "id": 1,
                    "club_id": "club-1",
                    "name": "Avery Ace",
                    "rating": 1600,
                    "starting_rating": 1500,
                    "wins": 9,
                    "losses": 1,
                    "matches_played": 10,
                    "active": True,
                    "email": "private@example.com",
                },
                {
                    "id": 2,
                    "club_id": "club-1",
                    "name": "Blake Baseline",
                    "rating": 1500,
                    "starting_rating": 1520,
                    "wins": 5,
                    "losses": 5,
                    "matches_played": 10,
                    "active": True,
                },
                {
                    "id": 3,
                    "club_id": "club-1",
                    "name": "Casey Court",
                    "rating": 1400,
                    "starting_rating": 1400,
                    "wins": 1,
                    "losses": 2,
                    "matches_played": 3,
                    "active": False,
                    "phone": "+1-private",
                },
                {"id": 99, "club_id": "other-club", "name": "Wrong Club", "rating": 2200, "active": True},
            ],
            "league_ratings": [
                {"club_id": "club-1", "league_name": "Pro", "player_id": 1, "rating": 1640, "starting_rating": 1560, "wins": 7, "losses": 1, "matches_played": 8, "is_active": True},
                {"club_id": "club-1", "league_name": "Pro", "player_id": 2, "rating": 1520, "starting_rating": 1520, "wins": 2, "losses": 2, "matches_played": 4, "is_active": True},
                {"club_id": "club-1", "league_name": "Pro", "player_id": 3, "rating": 1410, "starting_rating": 1400, "wins": 1, "losses": 2, "matches_played": 3, "is_active": False},
                {"club_id": "club-1", "league_name": "Past Season", "player_id": 1, "rating": 1620, "starting_rating": 1500, "wins": 6, "losses": 2, "matches_played": 8, "is_active": False},
                {"club_id": "club-1", "league_name": "Draft Test", "player_id": 1, "rating": 1800, "is_active": True},
                {"club_id": "club-1", "league_name": "Paused Test", "player_id": 1, "rating": 1800, "is_active": True},
                {"club_id": "club-1", "league_name": "Archived", "player_id": 1, "rating": 1700, "is_active": False},
            ],
            "leagues_metadata": [
                {"club_id": "club-1", "league_name": "Pro", "is_active": True, "status": "active", "min_games": 6},
                {"club_id": "club-1", "league_name": "Past Season", "is_active": False, "status": "ended", "min_games": 4},
                {"club_id": "club-1", "league_name": "Draft Test", "is_active": True, "status": "draft", "min_games": 0},
                {"club_id": "club-1", "league_name": "Paused Test", "is_active": False, "status": "paused", "min_games": 0},
                {"club_id": "club-1", "league_name": "Archived", "is_active": False, "status": "archived", "min_games": 2},
            ],
            "badges": [
                {"badge_id": "champ", "name": "Champion", "prestige": 100, "category": "League", "icon_key": "league_champion", "rarity": "Rare", "admin_notes": "private"},
                {"badge_id": "climber", "name": "Climber", "prestige": 50, "category": "Rating"},
                {"badge_id": "streak", "name": "Streak", "prestige": 20, "category": "Play"},
                {"badge_id": "extra", "name": "Extra", "prestige": 1, "category": "Play"},
            ],
            "player_badges": [
                {"club_id": "club-1", "player_id": 1, "badge_id": "extra", "earned_at": "2026-01-01", "secret": "private"},
                {"club_id": "club-1", "player_id": 1, "badge_id": "champ", "earned_at": "2026-04-01"},
                {"club_id": "club-1", "player_id": 1, "badge_id": "climber", "earned_at": "2026-03-01"},
                {"club_id": "club-1", "player_id": 1, "badge_id": "streak", "earned_at": "2026-02-01"},
                {"club_id": "other-club", "player_id": 1, "badge_id": "extra", "earned_at": "2026-05-01"},
            ],
        }
    )


def _assert_private_fields_absent(value):
    denied = {"email", "phone", "secret", "admin_notes", "subscription_token"}
    if isinstance(value, dict):
        assert denied.isdisjoint(value)
        for child in value.values():
            _assert_private_fields_absent(child)
    elif isinstance(value, list):
        for child in value:
            _assert_private_fields_absent(child)


def test_overall_is_a_real_active_default_projection_with_search_and_paging():
    payload = build_public_leaderboard(
        _fixture(),
        club_id="club-1",
        league_name="OVERALL",
        search="a",
        sort="name",
        limit=1,
        offset=0,
    )

    assert [scope["name"] for scope in payload["scopes"]] == ["OVERALL", "Pro"]
    assert payload["selected_scope"] == "OVERALL"
    assert payload["filters"] == {
        "league_view": "active",
        "status": "active",
        "search": "a",
        "sort": "name",
    }
    assert payload["summary"] == {
        "ranked_players": 3,
        "active_players": 2,
        "inactive_players": 1,
        "leaderboard_scopes": 2,
        "filtered_players": 2,
    }
    assert len(payload["leaderboard"]) == 1
    assert payload["leaderboard"][0]["player_name"] == "Avery Ace"
    assert payload["leaderboard"][0]["rating_jupr"] == 4.0
    assert payload["leaderboard"][0]["rating_gain_jupr"] == 0.25
    assert payload["leaderboard"][0]["qualified"] is None
    assert payload["pagination"] == {"total": 2, "offset": 0, "limit": 1, "has_more": True}
    assert all(row["is_active"] is True for row in payload["leaderboard"])
    _assert_private_fields_absent(payload)


def test_league_projection_preserves_rank_gap_qualification_badges_and_snapshot():
    payload = build_public_leaderboard(
        _fixture(),
        club_id="club-1",
        league_name="pro",
        status="all",
        player_id="3",
        limit=50,
    )

    assert payload["selected_scope"] == "Pro"
    assert payload["scope"]["min_games"] == 6
    first, second, third = payload["leaderboard"]
    assert [row["rank"] for row in payload["leaderboard"]] == [1, 2, 3]
    assert first["rating_jupr"] == 4.1
    assert round(first["rating_gain_jupr"], 6) == 0.2
    assert first["gap_jupr"] is None
    assert round(second["gap_jupr"], 6) == 0.3
    assert first["qualified"] is True
    assert second["qualified"] is False
    assert third["is_active"] is False
    assert [badge["badge_id"] for badge in first["badges"]] == ["champ", "climber", "streak"]
    assert first["badge_count"] == 4
    assert payload["snapshot"]["player_id"] == 3
    assert payload["highlights"]["highest_rating"][0]["player_id"] == 1
    assert {row["player_id"] for row in payload["highlights"]["highest_rating"]} == {1}
    _assert_private_fields_absent(payload)


def test_past_view_contains_only_ended_leagues_and_never_archived_or_draft() -> None:
    payload = build_public_leaderboard(
        _fixture(),
        club_id="club-1",
        league_view="past",
        status="all",
    )

    assert [scope["name"] for scope in payload["scopes"]] == ["Past Season"]
    assert payload["selected_scope"] == "Past Season"
    assert payload["filters"]["league_view"] == "past"
    assert {row["league_name"] for row in payload["leaderboard"]} == {"Past Season"}


def test_inactive_filter_and_invalid_scope_are_deterministic():
    inactive = build_public_leaderboard(_fixture(), club_id="club-1", league_name="Pro", status="inactive")
    assert [row["player_id"] for row in inactive["leaderboard"]] == [3]
    assert inactive["leaderboard"][0]["rank"] == 1

    fallback = build_public_leaderboard(_fixture(), club_id="club-1", league_name="does-not-exist")
    assert fallback["selected_scope"] == "OVERALL"
    assert all(row["league_name"] == "OVERALL" for row in fallback["leaderboard"])


def test_core_server_projection_failure_is_not_misreported_as_empty():
    class _BrokenSupabase:
        def table(self, _table_name):
            raise RuntimeError("permission denied")

    try:
        build_public_leaderboard(_BrokenSupabase(), club_id="club-1")
    except LeaderboardDataUnavailable as exc:
        assert "players" in str(exc)
    else:
        raise AssertionError("Expected a server projection failure")
