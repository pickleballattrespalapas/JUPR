from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.ensure_badges import ensure_badges
from jupr_app.domain.gamification.participation import compute_lifetime_games


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

    def insert(self, rows):
        self.storage.setdefault(self.name, []).extend(rows)
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
                data = [row for row in data if row.get(column) in value]
        return SimpleNamespace(data=data)


class FakeSupabase:
    def __init__(self, storage=None):
        self.storage = storage or {}

    def table(self, name):
        return FakeTable(self.storage, name)


def test_compute_lifetime_games_counts_team_rows():
    df_matches = pd.DataFrame(
        [
            {"t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 8},
            {"t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 9},
            {"t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": None, "score_t1": 11, "score_t2": 9},
            {"t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 0, "score_t2": 0},
        ]
    )
    ctx = SimpleNamespace(df_matches=df_matches, club_id="club")
    counts = compute_lifetime_games(ctx)
    assert counts == {1: 2, 2: 2, 3: 2, 4: 2}


def test_compute_lifetime_games_counts_player_rows():
    df_matches = pd.DataFrame(
        [
            {"player_id": 10, "score_t1": 11, "score_t2": 6},
            {"player_id": 10, "score_t1": 11, "score_t2": 9},
            {"player_id": 11, "score_t1": 0, "score_t2": 0},
        ]
    )
    ctx = SimpleNamespace(df_matches=df_matches, club_id="club")
    counts = compute_lifetime_games(ctx)
    assert counts == {10: 2}


def test_participation_badges_awarded_via_engine_idempotent():
    rows = [{"player_id": 1, "score_t1": 11, "score_t2": 9, "club_id": "club"} for _ in range(210)]
    rows.append({"player_id": 2, "score_t1": 11, "score_t2": 9, "club_id": "club"})
    df_matches = pd.DataFrame(rows)

    storage = {"player_badges": []}
    supabase = FakeSupabase(storage)
    ctx = SimpleNamespace(
        df_matches=df_matches,
        club_id="club",
        supabase=supabase,
        public_mode=False,
    )

    ensure_badges(ctx)
    ensure_badges(ctx)

    inserted = storage["player_badges"]
    badge_map = {(row["player_id"], row["badge_id"]) for row in inserted}
    assert badge_map >= {
        (1, "participant"),
        (1, "dedicated_participant_50"),
        (1, "lifetime_participant_200"),
        (2, "participant"),
    }
