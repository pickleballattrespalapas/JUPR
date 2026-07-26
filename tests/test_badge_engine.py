from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.badge_engine import compute_candidates_for_club
from jupr_app.domain.gamification.badge_registry import active_badge_ids, registry
from jupr_app.domain.gamification.badge_types import BadgeCandidate
from jupr_app.domain.gamification.badges_repo import upsert_player_badges
from jupr_app.domain.gamification.copy_pack import load_copy_pack
from jupr_app.domain.gamification.ensure_badges import ensure_badges


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
        keys = [c.strip() for c in str(on_conflict or "").split(",") if c.strip()]
        existing_keys = {tuple(row.get(k) for k in keys) for row in existing} if keys else set()
        for row in rows:
            key = tuple(row.get(k) for k in keys) if keys else None
            if key is not None and key in existing_keys:
                continue
            existing.append(row)
            if key is not None:
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
        self.storage = storage if storage is not None else {}

    def table(self, name):
        return FakeTable(self.storage, name)


def test_registry_includes_copy_pack_badges():
    pack = load_copy_pack()
    copy_ids = set((pack.get("badges") or {}).keys())
    reg_ids = set(registry().keys())
    active_copy_ids = copy_ids & active_badge_ids()
    assert active_copy_ids <= reg_ids


def test_engine_returns_candidates():
    df_matches = pd.DataFrame(
        [
            {
                "id": "m1",
                "club_id": "club",
                "league": "Open",
                "date": "2024-01-05T10:00:00Z",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 7,
            }
        ]
    )
    df_players_all = pd.DataFrame(
        [
            {"id": 1, "rating": 1200},
            {"id": 2, "rating": 1200},
            {"id": 3, "rating": 1200},
            {"id": 4, "rating": 1200},
        ]
    )
    ctx = SimpleNamespace(
        df_matches=df_matches,
        df_players_all=df_players_all,
        df_leagues=pd.DataFrame(),
        club_id="club",
    )
    candidates = list(compute_candidates_for_club("club", ctx=ctx))
    assert candidates
    assert all(isinstance(c, BadgeCandidate) for c in candidates)


def test_badge_upsert_adds_tape_excerpt_and_is_idempotent():
    storage = {}
    supabase = FakeSupabase(storage)
    candidate = BadgeCandidate(
        badge_id="participant",
        player_id=1,
        club_id="club",
        context_type="overall",
        context_id="overall",
        match_id=None,
        value_json={"games": 1},
    )
    created = upsert_player_badges(supabase, "club", [candidate])
    assert len(created) == 1
    rows = storage["player_badges"]
    assert rows[0]["value_json"]["tape_excerpt"]
    assert rows[0]["value_json"]["tape_title"]

    created_again = upsert_player_badges(supabase, "club", [candidate])
    assert created_again == []
    assert len(storage["player_badges"]) == 1


def test_ensure_badges_is_idempotent():
    df_matches = pd.DataFrame(
        [
            {
                "id": "m1",
                "club_id": "club",
                "league": "Open",
                "date": "2024-01-05T10:00:00Z",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 7,
            }
        ]
    )
    df_players_all = pd.DataFrame(
        [
            {"id": 1, "rating": 1200},
            {"id": 2, "rating": 1200},
            {"id": 3, "rating": 1200},
            {"id": 4, "rating": 1200},
        ]
    )
    storage = {}
    supabase = FakeSupabase(storage)
    ctx = SimpleNamespace(
        df_matches=df_matches,
        df_players_all=df_players_all,
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
        id_to_name={},
        club_id="club",
        supabase=supabase,
        public_mode=False,
    )
    ensure_badges(ctx)
    ensure_badges(ctx)
    assert len(storage.get("player_badges", [])) == len(
        {
            tuple(
                row.get(k)
                for k in ["club_id", "player_id", "badge_id", "context_type", "context_id"]
            )
            for row in storage["player_badges"]
        }
    )


def test_registry_covers_active_badges():
    reg_ids = set(registry().keys())
    missing = active_badge_ids() - reg_ids
    assert not missing
