from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.podium_awards import award_league_podium_badges
from jupr_app.ui.pages.players import (
    _decorate_trophies_with_leagues,
    build_inactive_league_options,
    filter_player_league_trophies,
    get_player_trophy_case,
)


class FakeTable:
    def __init__(self, storage, name):
        self.storage = storage
        self.name = name
        self.filters = []

    def select(self, _cols):
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def upsert(self, rows, on_conflict=None):
        existing = self.storage.setdefault(self.name, [])
        if not on_conflict:
            existing.extend(rows)
            return self
        keys = [c.strip() for c in str(on_conflict).split(",") if c.strip()]
        existing_keys = {tuple(row.get(k) for k in keys) for row in existing}
        for row in rows:
            key = tuple(row.get(k) for k in keys)
            if key in existing_keys:
                continue
            existing.append(row)
            existing_keys.add(key)
        return self

    def execute(self):
        data = list(self.storage.get(self.name, []))
        for op, column, value in self.filters:
            if op == "eq":
                data = [row for row in data if str(row.get(column)) == str(value)]
        return SimpleNamespace(data=data)


class FakeSupabase:
    def __init__(self, storage=None):
        self.storage = storage or {}

    def table(self, name):
        return FakeTable(self.storage, name)


def test_build_inactive_league_options_prefers_completed():
    df_meta = pd.DataFrame(
        [
            {
                "league_name": "Spring 2024 Ladder",
                "is_active": False,
                "status": "completed",
                "end_date": "2024-05-01",
                "season_label": "Spring 2024",
            },
            {
                "league_name": "Summer 2024 Ladder",
                "is_active": False,
                "status": "completed",
                "end_date": "2024-08-01",
                "season_label": "Summer 2024",
            },
            {
                "league_name": "Fall 2024 Ladder",
                "is_active": True,
                "status": "active",
                "end_date": "2099-11-01",
                "season_label": "Fall 2024",
            },
        ]
    )
    options = build_inactive_league_options(df_meta, pd.DataFrame(), pd.DataFrame())
    assert options["league_name"].tolist() == ["Summer 2024 Ladder", "Spring 2024 Ladder"]


def test_filter_player_league_trophies_scoped_to_player_and_league():
    df = pd.DataFrame(
        [
            {
                "player_id": 1,
                "badge_id": "league_champion",
                "context_type": "league",
                "context_id": "Spring 2024 Ladder:podium:1",
                "value_json": {"league_id": "Spring 2024 Ladder", "rank": 1},
            },
            {
                "player_id": 2,
                "badge_id": "league_champion",
                "context_type": "league",
                "context_id": "Spring 2024 Ladder:podium:1",
                "value_json": {"league_id": "Spring 2024 Ladder", "rank": 1},
            },
            {
                "player_id": 1,
                "badge_id": "league_runner_up",
                "context_type": "league",
                "context_id": "Fall 2024 Ladder:podium:2",
                "value_json": {"league_id": "Fall 2024 Ladder", "rank": 2},
            },
        ]
    )

    filtered = filter_player_league_trophies(df, 1, "Spring 2024 Ladder")

    assert filtered["player_id"].tolist() == [1]
    assert filtered["badge_id"].tolist() == ["league_champion"]


def test_award_league_podium_badges_is_idempotent():
    df_leagues = pd.DataFrame(
        [
            {"league_name": "Spring 2024 Ladder", "player_id": 1, "rating": 1500, "matches_played": 10},
            {"league_name": "Spring 2024 Ladder", "player_id": 2, "rating": 1400, "matches_played": 9},
            {"league_name": "Spring 2024 Ladder", "player_id": 3, "rating": 1300, "matches_played": 8},
        ]
    )
    storage = {"player_badges": []}
    supabase = FakeSupabase(storage)
    ctx = SimpleNamespace(
        df_leagues=df_leagues,
        club_id="club",
        supabase=supabase,
        public_mode=False,
    )

    award_league_podium_badges(ctx, "Spring 2024 Ladder")
    award_league_podium_badges(ctx, "Spring 2024 Ladder")

    inserted = storage["player_badges"]
    assert len(inserted) == 3
    badge_map = {(row["player_id"], row["badge_id"]) for row in inserted}
    assert badge_map == {
        (1, "league_champion"),
        (2, "league_runner_up"),
        (3, "league_third_place"),
    }


def test_get_player_trophy_case_scopes_and_orders():
    df = pd.DataFrame(
        [
            {
                "player_id": 1,
                "badge_id": "league_champion",
                "prestige": 90,
                "earned_at": "2024-08-10T10:00:00Z",
                "value_json": {"league_id": "Summer 2024 Ladder"},
            },
            {
                "player_id": 1,
                "badge_id": "league_runner_up",
                "prestige": 80,
                "earned_at": "2024-07-10T10:00:00Z",
                "value_json": {"league_id": "Spring 2024 Ladder"},
            },
            {
                "player_id": 1,
                "badge_id": "league_runner_up",
                "prestige": 80,
                "earned_at": "2024-08-11T10:00:00Z",
                "value_json": {"league_id": "Summer 2024 Ladder"},
            },
            {
                "player_id": 2,
                "badge_id": "league_champion",
                "prestige": 99,
                "earned_at": "2024-09-10T10:00:00Z",
                "value_json": {"league_id": "Summer 2024 Ladder"},
            },
        ]
    )
    completed = {"Spring 2024 Ladder", "Summer 2024 Ladder"}
    trophy_case = get_player_trophy_case(df, 1, completed, limit=8)

    assert trophy_case["player_id"].tolist() == [1, 1, 1]
    assert trophy_case["badge_id"].tolist() == [
        "league_champion",
        "league_runner_up",
        "league_runner_up",
    ]
    assert trophy_case["earned_at"].tolist() == [
        "2024-08-10T10:00:00Z",
        "2024-08-11T10:00:00Z",
        "2024-07-10T10:00:00Z",
    ]


def test_get_player_trophy_case_filters_other_players():
    df = pd.DataFrame(
        [
            {
                "player_id": 1,
                "badge_id": "top_performer_most_wins",
                "prestige": 120,
                "earned_at": "2024-08-10T10:00:00Z",
                "value_json": {"league_id": "Summer 2024 Ladder"},
            },
            {
                "player_id": 2,
                "badge_id": "top_performer_most_wins",
                "prestige": 120,
                "earned_at": "2024-08-11T10:00:00Z",
                "value_json": {"league_id": "Summer 2024 Ladder"},
            },
        ]
    )
    completed = {"Summer 2024 Ladder"}

    trophy_case = get_player_trophy_case(df, 1, completed, limit=8)

    assert trophy_case["player_id"].tolist() == [1]


def test_decorate_trophies_handles_missing_prestige_column():
    df = pd.DataFrame(
        [
            {
                "player_id": 1,
                "badge_id": "league_champion",
                "value_json": {"league_id": "Summer 2024 Ladder"},
            },
            {
                "player_id": 2,
                "badge_id": "league_runner_up",
                "value_json": {"league_id": "Spring 2024 Ladder"},
            },
        ]
    )

    decorated = _decorate_trophies_with_leagues(df, {})

    assert decorated["prestige_num"].tolist() == [0, 0]
