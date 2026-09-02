from __future__ import annotations

from types import SimpleNamespace

from jupr_app.services.admin_league_manager_service import (
    LEAGUE_MANAGER_EXTENDED_SELECT,
    LEAGUE_MANAGER_MINIMAL_SELECT,
    _league_row_payload,
    build_admin_league_manager_status,
    build_league_schedule_ics,
    get_admin_league_manager_detail,
    league_schedule_ics_filename,
    list_admin_league_manager_leagues,
)


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters: list[tuple[str, object]] = []
        self.order_key: str | None = None
        self.order_desc = False
        self.limit_value: int | None = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def order(self, key, desc=False):
        self.order_key = key
        self.order_desc = bool(desc)
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def execute(self):
        rows = list(self.storage.setdefault(self.table_name, []))
        for key, expected in self.filters:
            rows = [row for row in rows if str(row.get(key)) == str(expected)]
        if self.order_key:
            rows = sorted(rows, key=lambda row: str(row.get(self.order_key) or ""), reverse=self.order_desc)
        if self.limit_value is not None:
            rows = rows[: self.limit_value]
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, storage):
        self.storage = storage

    def table(self, name):
        return FakeQuery(self.storage, name)


def fake_storage():
    return {
        "leagues_metadata": [
            {
                "club_id": "club",
                "id": 101,
                "league_name": "Open",
                "is_active": True,
                "status": "active",
                "k_factor": 32,
                "min_games": 4,
                "schedule_config": {"start_date": "2026-07-01", "weekday": 2, "weeks": 3, "time_start": "18:00", "time_end": "20:00"},
                "court_board_defaults": {"total_courts": 4},
                "rules_config": {"move_up_down": True},
                "awards_config": {"default_depth": 1},
            },
            {"club_id": "club", "id": 102, "league_name": "Advanced", "is_active": False, "status": "ended", "k_factor": 24},
        ],
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex", "rating": 1520, "active": True, "last_game_at": "2026-07-08T18:00:00Z"},
            {"club_id": "club", "id": 2, "name": "Blair", "rating": 1320, "active": True},
            {"club_id": "club", "id": 3, "name": "Casey", "active": True},
        ],
        "league_ratings": [
            {"club_id": "club", "id": 10, "player_id": 1, "league_name": "Open", "rating": 1500, "starting_rating": 1400, "wins": 4, "losses": 1, "matches_played": 5, "is_active": True},
            {"club_id": "club", "id": 11, "player_id": 2, "league_name": "Open", "rating": 1300, "starting_rating": 1320, "wins": 2, "losses": 3, "matches_played": 5, "is_active": True},
        ],
    }


def test_league_manager_status_disabled_is_db_free(monkeypatch) -> None:
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", raising=False)

    payload = build_admin_league_manager_status(None, club_id="club")

    assert payload["enabled"] is False
    assert payload["leagues_endpoint"] is None
    assert payload["league_duplicate_endpoint"] is None


def test_league_manager_status_enabled_counts_leagues(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")

    payload = build_admin_league_manager_status(FakeSupabase(fake_storage()), club_id="club")

    assert payload["enabled"] is True
    assert payload["status"] == "ready_for_league_manager_roster_and_live_pilot"
    assert payload["league_create_endpoint"] == "/admin/clubs/{club_id}/league-manager/leagues"
    assert payload["league_duplicate_endpoint"] == "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/duplicate"
    assert payload["league_lifecycle_endpoint"] == "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/lifecycle"
    assert payload["league_schedule_preview_endpoint"] == "/admin/clubs/{club_id}/league-manager/leagues/{league_name}/schedule/preview"
    assert payload["league_count"] == 2
    assert payload["active_count"] == 1


def test_league_manager_list_and_detail(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    supabase = FakeSupabase(fake_storage())

    listing = list_admin_league_manager_leagues(supabase, club_id="club")
    detail = get_admin_league_manager_detail(supabase, club_id="club", league_name="Open")

    assert listing["count"] == 2
    assert LEAGUE_MANAGER_EXTENDED_SELECT.split(",")[0] == "id"
    assert LEAGUE_MANAGER_MINIMAL_SELECT.split(",")[0] == "id"
    assert listing["leagues"][1]["league_name"] == "Open"
    assert listing["leagues"][1]["league_id"] == "101"
    assert detail["league"]["league_id"] == "101"
    assert detail["league"]["status"] == "active"
    assert len(detail["schedule_preview"]) == 3
    assert detail["schedule_ics"].count("BEGIN:VEVENT") == 3
    assert "DTSTART;TZID=UTC:20260701T180000" in detail["schedule_ics"]
    assert detail["schedule_ics_filename"] == "open-schedule.ics"
    assert detail["standings"][0]["player_name"] == "Alex"
    assert detail["standings"][0]["rank"] == 1
    assert detail["standings"][0]["rating_jupr"] == 3.75
    assert detail["roster_count"] == 3
    assert detail["league_roster_count"] == 2
    assert detail["roster"][0]["player_name"] == "Alex"
    assert detail["roster"][0]["overall_rating_jupr"] == 3.8
    assert detail["roster"][2]["player_name"] == "Casey"
    assert detail["roster"][2]["in_league"] is False
    assert detail["validation"]["valid"] is True
    assert detail["capabilities"]["settings_mode"] == "description_only"
    assert detail["capabilities"]["roster_mutable"] is True
    assert detail["capabilities"]["lifecycle_actions"] == ["pause", "end"]


def test_paused_league_roster_is_review_only(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    storage = fake_storage()
    storage["leagues_metadata"][0].update(status="paused", is_active=False)

    detail = get_admin_league_manager_detail(
        FakeSupabase(storage), club_id="club", league_name="Open"
    )

    assert detail["league"]["status"] == "paused"
    assert detail["capabilities"]["roster_mutable"] is False


def test_active_manager_detail_hides_inactive_members_but_archive_keeps_them(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    storage = fake_storage()
    storage["league_ratings"].append(
        {
            "club_id": "club",
            "id": 12,
            "player_id": 3,
            "league_name": "Open",
            "rating": 1700,
            "starting_rating": 1400,
            "wins": 8,
            "losses": 1,
            "matches_played": 9,
            "is_active": False,
            "inactive_at": "2026-08-01T00:00:00Z",
        }
    )
    supabase = FakeSupabase(storage)

    active = get_admin_league_manager_detail(
        supabase, club_id="club", league_name="Open"
    )
    assert "Casey" not in {row["player_name"] for row in active["standings"]}

    storage["leagues_metadata"][0].update(status="archived", is_active=False)
    archived = get_admin_league_manager_detail(
        supabase, club_id="club", league_name="Open"
    )
    assert "Casey" in {row["player_name"] for row in archived["standings"]}


def test_legacy_inactive_status_is_always_a_draft() -> None:
    for is_active in (True, False, None):
        assert _league_row_payload({"status": " INACTIVE ", "is_active": is_active})["status"] == "draft"


def test_legacy_false_active_flag_without_a_status_remains_ended() -> None:
    assert _league_row_payload({"is_active": False})["status"] == "ended"


def test_league_schedule_ics_matches_preview_blackouts_and_escapes_text() -> None:
    content = build_league_schedule_ics(
        {
            "start_date": "2026-07-01",
            "weekday": 2,
            "weeks": 3,
            "time_start": "18:15",
            "time_end": "20:45",
            "timezone": "America/Chicago",
            "blackout_dates": ["2026-07-08"],
        },
        league_name="Open, Summer; Night\nLeague",
    )

    assert content.count("BEGIN:VEVENT") == 2
    assert "20260708" not in content
    assert "DTSTAMP:" in content
    assert "DTSTART;TZID=America/Chicago:20260701T181500" in content
    assert "DTEND;TZID=America/Chicago:20260715T204500" in content
    assert "SUMMARY:Open\\, Summer\\; Night\\nLeague" in content
    assert "\r\n" in content
    assert league_schedule_ics_filename("Open / Summer") == "open-summer-schedule.ics"


def test_league_schedule_ics_is_empty_without_a_preview() -> None:
    assert build_league_schedule_ics({}, league_name="Open") == ""


def test_league_schedule_ics_rejects_an_invalid_timezone_token() -> None:
    content = build_league_schedule_ics(
        {"start_date": "2026-07-01", "weekday": 2, "weeks": 1, "timezone": "America/Chicago\r\nBAD:TOKEN"},
        league_name="Open",
    )

    assert "DTSTART;TZID=UTC:20260701T180000" in content
    assert "BAD" not in content
