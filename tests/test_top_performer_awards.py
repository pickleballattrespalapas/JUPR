from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.awards import compute_top_performer_awards
from jupr_app.domain.gamification.top_performer_awards import ensure_league_top_performer_awards


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

    def in_(self, column, values):
        self.filters.append(("in", column, set(values)))
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
            elif op == "in":
                data = [row for row in data if str(row.get(column)) in {str(v) for v in value}]
        return SimpleNamespace(data=data)


class FakeSupabase:
    def __init__(self, storage=None):
        self.storage = storage or {}

    def table(self, name):
        return FakeTable(self.storage, name)


def test_compute_top_performer_awards_returns_expected_winners():
    qualified = pd.DataFrame(
        [
            {
                "_pid": 1,
                "name": "Ace",
                "rating": 1600,
                "JUPR": 4.0,
                "rating_gain": 200,
                "wins": 10,
                "matches_played": 12,
                "Win %": 83.3,
            },
            {
                "_pid": 2,
                "name": "Blaze",
                "rating": 1700,
                "JUPR": 4.25,
                "rating_gain": 100,
                "wins": 9,
                "matches_played": 12,
                "Win %": 75.0,
            },
            {
                "_pid": 3,
                "name": "Clutch",
                "rating": 1500,
                "JUPR": 3.75,
                "rating_gain": 300,
                "wins": 11,
                "matches_played": 12,
                "Win %": 91.7,
            },
        ]
    )

    awards = compute_top_performer_awards(qualified, min_games=6, winners_per_category=1)
    winners = {award["category_key"]: award["player_id"] for award in awards}

    assert winners == {
        "highest_rating": 2,
        "most_improved": 3,
        "best_win_pct": 3,
        "most_wins": 3,
    }


def test_ensure_league_top_performer_awards_is_idempotent():
    df_leagues = pd.DataFrame(
        [
            {
                "league_name": "Spring 2024 Ladder",
                "player_id": 1,
                "rating": 1600,
                "starting_rating": 1400,
                "wins": 9,
                "losses": 3,
                "matches_played": 12,
            },
            {
                "league_name": "Spring 2024 Ladder",
                "player_id": 2,
                "rating": 1700,
                "starting_rating": 1500,
                "wins": 10,
                "losses": 2,
                "matches_played": 12,
            },
            {
                "league_name": "Spring 2024 Ladder",
                "player_id": 3,
                "rating": 1500,
                "starting_rating": 1200,
                "wins": 11,
                "losses": 1,
                "matches_played": 12,
            },
        ]
    )
    df_meta = pd.DataFrame(
        [
            {
                "league_name": "Spring 2024 Ladder",
                "min_games": 6,
            }
        ]
    )
    storage = {"player_badges": []}
    supabase = FakeSupabase(storage)
    ctx = SimpleNamespace(
        df_leagues=df_leagues,
        df_meta=df_meta,
        club_id="club",
        supabase=supabase,
        public_mode=False,
        id_to_name={1: "Ace", 2: "Blaze", 3: "Clutch"},
    )

    ensure_league_top_performer_awards(ctx, "Spring 2024 Ladder")
    ensure_league_top_performer_awards(ctx, "Spring 2024 Ladder")

    inserted = storage["player_badges"]
    assert len(inserted) == 4
    badge_ids = {row["badge_id"] for row in inserted}
    assert badge_ids == {
        "top_performer_highest_rating",
        "top_performer_most_improved",
        "top_performer_best_win_pct",
        "top_performer_most_wins",
    }
