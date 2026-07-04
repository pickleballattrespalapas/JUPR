from __future__ import annotations

from types import SimpleNamespace

from jupr_app.services.admin_league_manager_service import (
    build_admin_league_manager_status,
    get_admin_league_manager_detail,
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
            {"club_id": "club", "league_name": "Advanced", "is_active": False, "status": "ended", "k_factor": 24},
        ],
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex"},
            {"club_id": "club", "id": 2, "name": "Blair"},
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


def test_league_manager_status_enabled_counts_leagues(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")

    payload = build_admin_league_manager_status(FakeSupabase(fake_storage()), club_id="club")

    assert payload["enabled"] is True
    assert payload["status"] == "ready_for_league_manager_read_foundation"
    assert payload["league_count"] == 2
    assert payload["active_count"] == 1


def test_league_manager_list_and_detail(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    supabase = FakeSupabase(fake_storage())

    listing = list_admin_league_manager_leagues(supabase, club_id="club")
    detail = get_admin_league_manager_detail(supabase, club_id="club", league_name="Open")

    assert listing["count"] == 2
    assert listing["leagues"][1]["league_name"] == "Open"
    assert detail["league"]["status"] == "active"
    assert len(detail["schedule_preview"]) == 3
    assert detail["standings"][0]["player_name"] == "Alex"
    assert detail["standings"][0]["rank"] == 1
    assert detail["standings"][0]["rating_jupr"] == 3.75
