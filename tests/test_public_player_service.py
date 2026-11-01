from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest

from jupr_app.data.paged_reads import DataReadUnavailable

from jupr_app.services.public_player_service import (
    build_public_player_directory,
    get_public_match_detail,
    get_public_matches,
    get_public_player_profile,
    get_public_players,
)


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, table_name: str, rows_by_table: dict[str, list[dict]]):
        self.table_name = table_name
        self.rows_by_table = rows_by_table
        self.filters: dict[str, object] = {}
        self.in_filters: dict[str, set[str]] = {}
        self.or_filters: list[tuple[str, str]] = []
        self.null_filters: list[str] = []
        self.order_by: list[tuple[str, bool]] = []
        self.row_limit: int | None = None

    def select(self, *_args, **_kwargs):
        self.selected_columns = str(_args[0]).split(",") if _args else ["*"]
        return self

    def eq(self, key, value):
        self.filters[str(key)] = value
        return self

    def in_(self, key, values):
        self.in_filters[str(key)] = {str(value) for value in values}
        return self

    def or_(self, expression):
        self.or_filters = [(key, value) for key, operator, value in (part.split(".", 2) for part in expression.split(",")) if operator == "eq"]
        return self

    def is_(self, key, value):
        assert value == "null"
        self.null_filters.append(key)
        return self

    def order(self, key, desc=False):
        self.order_by.append((key, desc))
        return self

    def limit(self, value):
        self.row_limit = int(value)
        return self

    def range(self, start, end):
        self.page_bounds = (start, end)
        return self

    def execute(self):
        rows = list(self.rows_by_table.get(self.table_name, []))
        for key, value in self.filters.items():
            rows = [row for row in rows if str(row.get(key)) == str(value)]
        for key, values in self.in_filters.items():
            rows = [row for row in rows if str(row.get(key)) in values]
        if self.or_filters:
            rows = [row for row in rows if any(str(row.get(key)) == value for key, value in self.or_filters)]
        for key in self.null_filters:
            rows = [row for row in rows if row.get(key) is None]
        for key, descending in reversed(self.order_by):
            rows.sort(key=lambda row: (row.get(key) is None, row.get(key)), reverse=descending)
        if self.row_limit is not None:
            rows = rows[: self.row_limit]
        if hasattr(self, "page_bounds"):
            rows = rows[self.page_bounds[0]:self.page_bounds[1] + 1]
        if hasattr(self, "server_cap"):
            rows = rows[:self.server_cap]
        if self.table_name == "player_badges" and self.selected_columns != ["*"]:
            rows = [{key: row.get(key) for key in self.selected_columns} for row in rows]
        return FakeResponse(rows)


class FakeSupabase:
    def __init__(self):
        self.rows_by_table = {
            "players": [
                {"id": 1, "club_id": "club-1", "name": "Alex", "rating": 1600, "starting_rating": 1520, "wins": 3, "losses": 1, "matches_played": 4, "singles_rating": 1560, "singles_wins": 1, "singles_losses": 0, "singles_matches_played": 1, "active": True, "email": "private@example.com", "legal_name": "Private Legal Name"},
                {"id": 2, "club_id": "club-1", "name": "Blair", "rating": 1500, "wins": 2, "losses": 2, "matches_played": 4, "active": True},
                {"id": 3, "club_id": "club-1", "name": "Casey", "rating": 1490, "wins": 1, "losses": 2, "matches_played": 3, "active": True},
                {"id": 4, "club_id": "club-1", "name": "Devon", "rating": 1480, "wins": 1, "losses": 2, "matches_played": 3, "active": True},
                {"id": 5, "club_id": "club-1", "name": "Inactive Izzy", "rating": 1450, "wins": 0, "losses": 1, "matches_played": 1, "active": False, "inactive_at": "2026-06-01"},
            ],
            "league_ratings": [
                {"id": 11, "club_id": "club-1", "player_id": 1, "league_name": "Open", "rating": 1600, "wins": 3, "losses": 1, "matches_played": 4, "is_active": True},
                {"id": 12, "club_id": "club-1", "player_id": 1, "league_name": "Archived Test", "rating": 1800, "wins": 10, "losses": 0, "matches_played": 10, "is_active": True}
            ],
            "leagues_metadata": [
                {"club_id": "club-1", "league_name": "Open", "status": "active", "is_active": True},
                {"club_id": "club-1", "league_name": "Archived Test", "status": "archived", "is_active": False}
            ],
            "matches": [
                {
                    "id": 99,
                    "club_id": "club-1",
                    "date": "2026-07-02",
                    "league": "Open",
                    "t1_p1": 1,
                    "t1_p2": 2,
                    "t2_p1": 3,
                    "t2_p2": 4,
                    "score_t1": 11,
                    "score_t2": 8,
                    "elo_delta": 4.5,
                    "t1_p1_r": 1595,
                    "t1_p1_r_end": 1600,
                    "t1_p2_r": 1495,
                    "t1_p2_r_end": 1500,
                    "t2_p1_r": 1495,
                    "t2_p1_r_end": 1490,
                    "t2_p2_r": 1485,
                    "t2_p2_r_end": 1480,
                    "notes": "private admin note",
                },
                {
                    "id": 100,
                    "club_id": "club-1",
                    "date": "2026-07-03",
                    "league": "Singles Night",
                    "match_format": "singles",
                    "rating_scope": "singles",
                    "t1_p1": 1,
                    "t1_p2": None,
                    "t2_p1": 3,
                    "t2_p2": None,
                    "score_t1": 11,
                    "score_t2": 9,
                    "elo_delta": 4,
                    "t1_p1_r": 1556,
                    "t1_p1_r_end": 1560,
                    "t2_p1_r": 1494,
                    "t2_p1_r_end": 1490,
                },
            ],
            "badges": [
                {"badge_id": "first_win", "name": "First Win", "category": "Milestone", "prestige": 10, "rarity": "Common", "is_active": True, "admin_notes": "private"},
                {"badge_id": "tournament_champion", "name": "Tournament Champion", "category": "Trophy", "prestige": 100, "rarity": "Legendary", "is_active": True},
            ],
            "player_badges": [
                {"club_id": "club-1", "player_id": 1, "badge_id": "first_win", "earned_at": "2026-07-02T00:00:00Z", "context_type": "overall", "context_id": "private-context", "secret": "private"},
                {"club_id": "club-1", "player_id": 1, "badge_id": "tournament_champion", "earned_at": "2026-07-04T00:00:00Z", "context_type": "tournament", "context_id": "summer:podium:1", "value_num": 1, "value_json": {"placement": 1, "tournament_name": "Summer Cup", "contact_email": "private@example.com"}},
            ],
            "player_profile_update_subscriptions": [
                {"club_id": "club-1", "player_id": 1, "request_status": "active", "verified_at": "2026-07-01T00:00:00Z", "email": "private@example.com", "unsubscribe_token": "private-token"}
            ],
            "club_people": [
                {"id": "person-1", "club_id": "club-1", "display_name": "Private Social Alias", "normalized_name": "private social alias", "linked_player_id": 1}
            ],
            "live_event_participants": [
                {"id": "participant-1", "event_id": "event-1", "club_person_id": "person-1", "linked_player_id": 1, "display_name_snapshot": "Private Social Alias"},
                {"id": "participant-2", "event_id": "event-1", "club_person_id": "person-2", "linked_player_id": 2},
                {"id": "participant-3", "event_id": "event-1", "club_person_id": "person-3", "linked_player_id": 3},
                {"id": "participant-4", "event_id": "event-1", "club_person_id": "person-4", "linked_player_id": 4},
            ],
            "live_events": [
                {"id": "event-1", "club_id": "club-1", "name": "Friday Social", "event_type": "round_robin", "event_date": "2026-07-05", "status": "saved", "result_mode": "social_unrated", "summary_json": {"event_tags": {"skill_levels": ["3.5"]}, "private_contacts": ["private@example.com"]}, "submitted_by": "private@example.com"}
            ],
            "live_event_matches": [
                {"event_id": "event-1", "played_on": "2026-07-05", "t1_p1_participant_id": "participant-1", "t1_p2_participant_id": "participant-2", "t2_p1_participant_id": "participant-3", "t2_p2_participant_id": "participant-4", "score_t1": 11, "score_t2": 7}
            ],
        }

        for player in self.rows_by_table["players"]:
            player["last_game_at"] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

    def table(self, table_name):
        return FakeQuery(str(table_name), self.rows_by_table)


def test_public_players_are_sanitized_and_searchable():
    rows = get_public_players(FakeSupabase(), club_id="club-1", search="alex")

    assert len(rows) == 1
    assert rows[0]["name"] == "Alex"
    assert rows[0]["rating"] == 1600.0
    assert rows[0]["rating_jupr"] == 4.0
    assert "email" not in rows[0]
    assert "legal_name" not in rows[0]
    assert "club_id" not in rows[0]


def test_player_cabinet_keeps_distinct_ids_for_repeated_partner_achievements():
    supabase = FakeSupabase()
    supabase.rows_by_table["badges"].append({
        "badge_id": "matches_together_10", "name": "10 Matches Together", "is_active": True,
    })
    for award_id, partner, revoked_at in [(101, "Blair", None), (102, "Casey", None), (103, "Devon", "2026-07-06")]:
        supabase.rows_by_table["player_badges"].append({
            "id": award_id, "club_id": "club-1", "player_id": 1,
            "badge_id": "matches_together_10", "context_type": "overall",
            "context_id": f"pair:1:{award_id}", "earned_at": "2026-07-05T00:00:00Z",
            "revoked_at": revoked_at,
            "value_json": {"rule_version": "program-badges-v1", "tape_excerpt": f"10 matches with {partner}.", "private_note": "do not publish"},
        })

    profile = get_public_player_profile(supabase, club_id="club-1", player_id=1)
    badge = next(b for b in profile["awards"]["badges"] if b["badge_id"] == "matches_together_10")
    assert badge["count"] == 2
    assert badge["achievements"] == [
        {"id": "101", "earned_at": "2026-07-05T00:00:00Z", "detail": "10 matches with Blair."},
        {"id": "102", "earned_at": "2026-07-05T00:00:00Z", "detail": "10 matches with Casey."},
    ]


def test_trophy_case_uses_league_award_rank_and_keeps_progression_in_cabinet():
    supabase = FakeSupabase()
    base = {"club_id": "club-1", "player_id": 1, "earned_at": "2026-07-05T00:00:00Z", "context_type": "league"}
    awards = [
        {"badge_id": "top_performer_most_wins", "value_num": 66,
         "context_id": "Summer:top_performer:most_wins:1",
         "value_json": {"rank": 1, "metric_value": 66, "league_id": "Summer"}},
        {"badge_id": "top_performer_most_improved", "value_num": 0.47,
         "context_id": "Summer:top_performer:most_improved:2",
         "value_json": '{"rank": 2, "metric_value": 0.47, "league_id": "Summer"}'},
        {"badge_id": "top_performer_best_win_pct", "value_num": 87,
         "context_id": "Summer:top_performer:best_win_pct:3"},
        {"badge_id": "mountain_climber", "value_num": 27},
        {"badge_id": "mountain_climber", "value_num": 1},
        {"badge_id": "mountain_climber", "value_num": 47, "value_json": {"rank": 1}},
    ]
    supabase.rows_by_table["player_badges"].extend({**base, **award} for award in awards)
    profile = get_public_player_profile(supabase, club_id="club-1", player_id=1)
    result = profile["awards"]
    assert result["trophy_count"] == 4
    assert {row["badge_id"]: row["placement"] for row in result["trophies"]} == {
        "tournament_champion": 1, "top_performer_most_wins": 1,
        "top_performer_most_improved": 2, "top_performer_best_win_pct": 3,
    }
    assert next(row for row in result["trophies"] if row["badge_id"] == "top_performer_most_wins")["context_label"] == "Summer"
    assert {row["badge_id"]: row["count"] for row in result["badges"]} == {"first_win": 1, "mountain_climber": 3}
    assert result["badge_award_count"] == 4
    assert len(supabase.rows_by_table["player_badges"]) == 8  # Display filtering preserves the earned records.


@pytest.mark.parametrize("badge_id,context_type,evidence,expected", [
    ("tournament_champion", "tournament", {"value_json": {"placement": 1}}, 1),
    ("tournament_runner_up", "tournament", {"context_id": "cup:draw:open:podium:2", "value_num": 66}, 2),
    ("tournament_third_place", "tournament", {"value_num": 3.0}, 3),
    ("league_champion", "league", {}, 1),
    ("league_runner_up", "league_end_award", {"value_json": {"podium_place": "2"}}, 2),
    ("top_performer_most_wins", "league", {"value_json": {"rank": "3.0"}, "value_num": 66}, 3),
    ("top_performer_most_wins", "league", {"value_num": 1, "value_json": {"category_key": "most_wins"}}, None),
    ("top_performer_most_wins", "league", {"value_json": {"rank": 4}, "value_num": 1}, None),
    ("top_performer_most_wins", "tournament", {"value_json": {"rank": 1}}, None),
    ("mountain_climber", "league_end", {"value_json": {"placement": 1}}, None),
    ("first_win", "tournament", {"context_id": "cup:podium:1"}, None),
    ("tournament_champion", "overall", {"value_json": {"placement": 1}}, None),
    ("tournament_champion", "tournament", {"value_json": {"placement": 27}}, None),
    ("tournament_champion", "tournament", {"value_num": 66}, None),
    ("tournament_champion", "tournament", {"value_json": {"placement": 0}, "context_id": "cup:podium:1"}, None),
    ("tournament_champion", "tournament", {"value_json": {"placement": 1.5}}, None),
    ("tournament_champion", "tournament", {"value_json": {"placement": True}}, None),
    ("tournament_champion", "tournament", {"value_json": {"placement": "bad"}}, None),
    ("tournament_champion", "tournament", {"value_json": {"placement": 1}, "revoked_at": "2026-07-06"}, None),
])
def test_public_trophy_case_requires_a_top_three_award(badge_id, context_type, evidence, expected):
    supabase = FakeSupabase()
    supabase.rows_by_table["player_badges"] = [{
        "club_id": "club-1", "player_id": 1, "badge_id": badge_id,
        "context_type": context_type, "earned_at": "2026-07-05", **evidence,
    }]
    awards = get_public_player_profile(supabase, club_id="club-1", player_id=1)["awards"]
    assert [row["placement"] for row in awards["trophies"]] == ([] if expected is None else [expected])
    assert awards["trophy_count"] == len(awards["trophies"])
    assert awards["badge_award_count"] == int(expected is None and not evidence.get("revoked_at"))


def test_public_directory_includes_inactive_profiles_even_for_legacy_status_filters():
    for status in (None, "all", "active", "inactive"):
        kwargs = {"status": status} if status else {}
        directory = build_public_player_directory(FakeSupabase(), club_id="club-1", limit=2, **kwargs)
        assert directory["filters"] == {"search": "", "status": "all", "sort": "rating"}
        assert directory["summary"] == {"public_players": 5, "active_players": 4, "inactive_players": 1, "filtered_players": 5}
        assert len(directory["players"]) == 2
        assert directory["pagination"]["has_more"] is True

        inactive = build_public_player_directory(FakeSupabase(), club_id="club-1", search="izzy", **kwargs)
        assert [row["name"] for row in inactive["players"]] == ["Inactive Izzy"]
        assert inactive["players"][0]["is_active"] is False


def test_player_directory_excludes_merged_identities_without_changing_activity():
    import copy
    supabase = FakeSupabase()
    supabase.rows_by_table["players"].append({
        "id": 99, "club_id": "club-1", "name": "Old Name (MERGED into Alex #1)", "active": False,
    })
    before = copy.deepcopy(supabase.rows_by_table)
    directory = build_public_player_directory(supabase, club_id="club-1", limit=100)
    assert len(directory["players"]) == 5
    assert 99 not in [row["id"] for row in directory["players"]]
    assert supabase.rows_by_table == before
    assert get_public_player_profile(supabase, club_id="club-1", player_id=99) is None


def test_public_player_profile_includes_leagues_and_recent_matches():
    profile = get_public_player_profile(FakeSupabase(), club_id="club-1", player_id=1)

    assert profile is not None
    assert profile["player"]["name"] == "Alex"
    assert profile["league_ratings"][0]["league_name"] == "Open"
    assert [row["league_name"] for row in profile["league_ratings"]] == ["Open"]
    assert profile["recent_matches"][0]["team_1"][0]["name"] == "Alex"
    assert profile["recent_matches"][0]["score_t1"] == 11
    assert profile["recent_matches"][0]["match_format_label"] == "Singles"


def test_rating_history_fallback_converts_raw_elo_delta_to_jupr_units():
    supabase = FakeSupabase()
    supabase.rows_by_table["matches"][0]["t1_p1_r_end"] = None

    profile = get_public_player_profile(supabase, club_id="club-1", player_id=1)

    assert profile is not None
    point = next(row for row in profile["rating_history"] if row["match_id"] == 99)
    assert point["rating_before_jupr"] == 3.9875
    assert point["rating_after_jupr"] == 3.99875
    assert point["rating_delta_jupr"] == 0.01125


def test_pending_verified_update_request_is_not_publicly_reopened():
    supabase = FakeSupabase()
    supabase.rows_by_table["player_profile_update_subscriptions"][0]["request_status"] = "pending_admin_review"

    profile = get_public_player_profile(supabase, club_id="club-1", player_id=1)

    assert profile is not None
    assert profile["verified_updates"] == {"status": "pending", "can_request": False}


def _assert_private_fields_absent(value):
    denied = {
        "email",
        "email_normalized",
        "phone",
        "legal_name",
        "normalized_name",
        "display_name_snapshot",
        "participant_key",
        "club_person_id",
        "participant_id",
        "subscription_id",
        "unsubscribe_token",
        "request_note",
        "admin_note",
        "verified_by",
        "preferences_json",
        "submitted_by",
        "submitted_by_name",
        "source_event_uid",
        "raw_event_json",
        "private_contacts",
        "secret",
        "admin_notes",
        "notes",
        "club_id",
        "context_id",
    }
    if isinstance(value, dict):
        assert denied.isdisjoint(value)
        for child in value.values():
            _assert_private_fields_absent(child)
    elif isinstance(value, list):
        for child in value:
            _assert_private_fields_absent(child)


def test_public_profile_closes_rating_award_relationship_social_and_privacy_parity():
    profile = get_public_player_profile(FakeSupabase(), club_id="club-1", player_id=1, history_limit=100)

    assert profile is not None
    assert profile["identity"] == {
        "display_name": "Alex",
        "public_name_policy": "public_display_name",
        "verification_status": "enabled",
    }
    assert profile["verified_updates"] == {"status": "enabled", "can_request": False}
    assert [row["label"] for row in profile["rating_breakdowns"]] == ["Doubles / overall", "Singles"]
    assert {point["match_format_label"] for point in profile["rating_history"]} == {"Doubles", "Singles"}
    assert profile["rating_summary"]["starting_rating_jupr"] == 3.8
    assert profile["rating_summary"]["highest_rating_jupr"] == 4.0
    assert profile["rating_summary"]["lowest_rating_jupr"] == 3.8
    assert profile["awards"]["badge_count"] == 1
    assert profile["awards"]["badge_award_count"] == 1
    assert profile["awards"]["trophy_count"] == 1
    assert [row["name"] for row in profile["awards"]["badges"]] == ["First Win"]
    assert profile["awards"]["trophies"][0]["context_label"] == "Summer Cup"
    assert profile["relationships"]["best_partner"]["player_name"] == "Blair"
    assert profile["relationships"]["rival"]["player_name"] in {"Casey", "Devon"}
    assert profile["social"]["identity"] == {"linked": True, "label": "Club Social identity linked"}
    assert profile["social"]["summary"]["wins"] == 1
    assert profile["social"]["skill_breakdown"][0]["label"] == "3.5"
    assert profile["social"]["skill_breakdown"][0]["score_diff"] == 4
    assert profile["match_history"][0]["match_format_label"] == "Singles"
    _assert_private_fields_absent(profile)
    serialized = repr(profile).casefold()
    for private_value in ("private@example.com", "private legal name", "private social alias", "private-token", "private-context"):
        assert private_value not in serialized


def test_public_matches_include_linkable_public_players():
    matches = get_public_matches(FakeSupabase(), club_id="club-1")

    assert len(matches) == 2
    doubles = next(match for match in matches if match["match_format"] == "doubles")
    assert doubles["team_1"][0] == {"id": 1, "name": "Alex"}
    assert doubles["team_2"][1] == {"id": 4, "name": "Devon"}
    assert "t1_p1_r" not in doubles


def test_public_match_detail_includes_rating_snapshot_without_raw_columns():
    detail = get_public_match_detail(FakeSupabase(), club_id="club-1", match_id=99)

    assert detail is not None
    assert detail["id"] == 99
    assert detail["rating_snapshot"]["team_1"][0]["start_rating"] == 1595.0
    assert detail["rating_snapshot"]["team_1"][0]["end_rating"] == 1600.0
    assert "t1_p1_r" not in detail


def test_soft_deleted_matches_are_suppressed_from_all_public_match_reads():
    supabase = FakeSupabase()
    supabase.rows_by_table["matches"][0]["deleted_at"] = "2026-07-10T00:00:00Z"

    assert get_public_match_detail(supabase, club_id="club-1", match_id=99) is None
    assert all(
        match["id"] != 99
        for match in get_public_matches(supabase, club_id="club-1")
    )


def _history_database(match_count=41):
    db = FakeSupabase()
    template = db.rows_by_table["matches"][0]
    old_matches = []
    positions = ("t1_p1", "t1_p2", "t2_p1", "t2_p2")
    for index in range(match_count):
        row = {**template, "id": index + 1, "date": "2026-02-04T12:00:00+00:00"}
        position = positions[index % 4]
        row["t1_p1"], row[position] = row[position], row["t1_p1"]
        old_matches.append(row)
    # More than the old club-wide cutoff, all belonging to other players.
    newer_matches = [
        {**template, "id": 10000 + index, "date": "2026-10-01", "t1_p1": 5}
        for index in range(700)
    ]
    db.rows_by_table["matches"] = newer_matches + old_matches + [
        {**template, "id": 20000, "club_id": "club-2"},
        {**template, "id": 20001, "deleted_at": "2026-07-10"},
    ]
    db.rows_by_table["players"][0].update(active=False, last_game_at="2026-02-04")
    return db


def test_returning_player_retains_all_old_matches_in_every_position_and_derived_stats():
    db = _history_database()
    profile = get_public_player_profile(db, club_id="club-1", player_id="1")

    assert profile["history"]["total_matches"] == 41
    assert [row["id"] for row in profile["match_history"]] == list(range(41, 0, -1))
    assert len(profile["recent_matches"]) == 12
    assert len(profile["rating_history"]) == 41
    assert sum(row["matches"] for row in profile["rating_breakdowns"]) == 41
    assert sum(row["matches"] for row in profile["relationships"]["partners"]) == 41
    assert sum(row["matches"] for row in profile["relationships"]["rivals"]) == 82
    assert profile["player"]["is_active"] is False
    assert [row["id"] for row in get_public_matches(db, club_id="club-1", player_id=1, limit=12, offset=12)] == list(range(29, 17, -1))
    _assert_private_fields_absent(profile)


def test_long_player_history_pages_past_server_caps_without_losing_totals_or_stats():
    db = _history_database(match_count=1205)
    original_table = db.table

    def capped_table(table_name):
        query = original_table(table_name)
        if table_name == "matches":
            query.server_cap = 73
        return query

    db.table = capped_table
    pages = [get_public_player_profile(db, club_id="club-1", player_id=1, history_limit=500, history_offset=offset) for offset in (0, 500, 1000)]
    assert [len(page["match_history"]) for page in pages] == [500, 500, 205]
    assert [page["history"]["has_more"] for page in pages] == [True, True, False]
    assert [page["history"]["history_offset"] for page in pages] == [0, 500, 1000]
    assert all(page["history"]["total_matches"] == 1205 for page in pages)
    assert all(len(page["rating_history"]) == 1205 for page in pages)
    assert pages[0]["relationships"] == pages[2]["relationships"]
    assert [row["id"] for page in pages for row in page["match_history"]] == list(range(1205, 0, -1))


def test_league_filter_precedes_recent_and_full_history_pagination():
    db = _history_database(match_count=80)
    for row in db.rows_by_table["matches"]:
        if row["id"] <= 80:
            row["league"] = "Previous season" if row["id"] <= 25 else "Open"
    profile = get_public_player_profile(db, club_id="club-1", player_id=1, history_limit=12, history_offset=12, history_league="Previous season")

    assert profile["history"]["total_matches"] == 80
    assert profile["history"]["filtered_matches"] == 25
    assert profile["history"]["leagues"] == ["Open", "Previous season"]
    assert [row["id"] for row in profile["recent_matches"]] == list(range(25, 13, -1))
    assert [row["id"] for row in profile["match_history"]] == list(range(13, 1, -1))
    assert len(profile["rating_history"]) == 80
    final = get_public_player_profile(db, club_id="club-1", player_id=1, history_limit=12, history_offset=999, history_league="Previous season")
    assert final["history"]["history_offset"] == 24
    assert [row["id"] for row in final["match_history"]] == [1]
    empty = get_public_player_profile(db, club_id="club-1", player_id=1, history_offset=50, history_league="Missing league")
    assert empty["history"]["filtered_matches"] == 0
    assert empty["history"]["history_offset"] == 0
    assert empty["match_history"] == []


def test_failed_history_read_is_an_error_instead_of_an_empty_or_partial_record():
    db = _history_database(match_count=501)
    original_table = db.table

    def failing_table(table_name):
        query = original_table(table_name)
        execute = query.execute

        def failing_execute():
            if table_name == "matches" and getattr(query, "page_bounds", (0, 0))[0] > 0:
                raise RuntimeError("History page unavailable")
            return execute()

        query.execute = failing_execute
        return query

    db.table = failing_table
    with pytest.raises(DataReadUnavailable):
        get_public_player_profile(db, club_id="club-1", player_id=1)
    with pytest.raises(DataReadUnavailable):
        get_public_matches(db, club_id="club-1", player_id=1)


def test_invalid_player_id_cannot_expand_the_history_query():
    db = _history_database()
    assert get_public_matches(db, club_id="club-1", player_id="1,t1_p1.gt.0") == []


def test_directory_partial_name_suggestions_are_accent_insensitive_and_club_scoped():
    db = FakeSupabase()
    db.rows_by_table['players'] += [
        {'id': 61, 'club_id': 'club-1', 'name': 'Valé Verdugo', 'rating': 1700, 'active': True},
        {'id': 62, 'club_id': 'club-1', 'name': 'Valé Verdugo', 'rating': 1600, 'active': True},
        {'id': 63, 'club_id': 'club-2', 'name': 'Valé Verdugo', 'rating': 1800, 'active': True},
    ]
    result = build_public_player_directory(db, club_id='club-1', search='VER VALE')
    assert {row['id'] for row in result['players']} == {61, 62}
    assert all('email' not in row for row in result['players'])


def test_directory_search_reads_past_supabase_row_cap():
    db = FakeSupabase()
    db.rows_by_table['players'] = [
        {'id': index, 'club_id': 'club-1', 'name': f'Player {index}', 'active': True}
        for index in range(1201)
    ] + [{'id': 1202, 'club_id': 'club-1', 'name': 'Valé Verdugo', 'active': True}]
    result = build_public_player_directory(db, club_id='club-1', search='verdu')
    assert [row['id'] for row in result['players']] == [1202]
    assert result['summary']['public_players'] == 1202
