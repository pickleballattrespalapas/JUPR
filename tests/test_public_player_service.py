from __future__ import annotations

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
        self.row_limit: int | None = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters[str(key)] = value
        return self

    def in_(self, key, values):
        self.in_filters[str(key)] = {str(value) for value in values}
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, value):
        self.row_limit = int(value)
        return self

    def execute(self):
        rows = list(self.rows_by_table.get(self.table_name, []))
        for key, value in self.filters.items():
            rows = [row for row in rows if str(row.get(key)) == str(value)]
        for key, values in self.in_filters.items():
            rows = [row for row in rows if str(row.get(key)) in values]
        if self.row_limit is not None:
            rows = rows[: self.row_limit]
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
                {"id": 11, "club_id": "club-1", "player_id": 1, "league_name": "Open", "rating": 1600, "wins": 3, "losses": 1, "matches_played": 4, "is_active": True}
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


def test_public_directory_defaults_active_and_supports_inactive_search_and_paging():
    active = build_public_player_directory(FakeSupabase(), club_id="club-1", limit=2)

    assert active["filters"] == {"search": "", "status": "active", "sort": "rating"}
    assert active["summary"] == {"public_players": 5, "active_players": 4, "inactive_players": 1, "filtered_players": 4}
    assert len(active["players"]) == 2
    assert active["pagination"]["has_more"] is True
    assert all(row["is_active"] for row in active["players"])

    inactive = build_public_player_directory(FakeSupabase(), club_id="club-1", search="izzy", status="inactive")
    assert [row["name"] for row in inactive["players"]] == ["Inactive Izzy"]


def test_public_player_profile_includes_leagues_and_recent_matches():
    profile = get_public_player_profile(FakeSupabase(), club_id="club-1", player_id=1)

    assert profile is not None
    assert profile["player"]["name"] == "Alex"
    assert profile["league_ratings"][0]["league_name"] == "Open"
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
