from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.recompute import run_badge_recompute


class FakeTable:
    def __init__(self, storage, name):
        self.storage = storage
        self.name = name
        self.filters = []
        self.update_payload = None
        self.insert_payload = None

    def select(self, _cols):
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column, values):
        self.filters.append(("in", column, set(values)))
        return self

    def insert(self, payload):
        self.insert_payload = payload
        return self

    def update(self, payload):
        self.update_payload = payload
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

        if self.insert_payload is not None:
            payloads = self.insert_payload
            if isinstance(payloads, dict):
                payloads = [payloads]
            stored = self.storage.setdefault(self.name, [])
            inserted = []
            for payload in payloads:
                row = dict(payload)
                row.setdefault("id", f"{self.name}_{len(stored) + 1}")
                stored.append(row)
                inserted.append(row)
            return SimpleNamespace(data=inserted)

        if self.update_payload is not None:
            for row in data:
                row.update(self.update_payload)
            return SimpleNamespace(data=data)

        return SimpleNamespace(data=data)


class FakeSupabase:
    def __init__(self, storage=None):
        self.storage = storage if storage is not None else {}

    def table(self, name):
        return FakeTable(self.storage, name)


def _build_ctx():
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
    return SimpleNamespace(
        supabase=None,
        club_id="club",
        df_matches=df_matches,
        df_players_all=pd.DataFrame(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
        df_badges=pd.DataFrame([{"badge_id": "participant", "state": "live"}]),
        df_player_badges=pd.DataFrame(),
        name_to_id={},
        id_to_name={},
        public_mode=False,
        admin_logged_in=True,
    )


def test_recompute_dry_run_writes_nothing():
    storage = {}
    supabase = FakeSupabase(storage)
    ctx = _build_ctx()
    summary = run_badge_recompute(
        supabase,
        club_id="club",
        mode="dry-run",
        ctx=ctx,
        badge_id="participant",
        allow_strict_global=True,
    )
    assert summary["new_awards_count"] >= 0
    assert storage.get("player_badges", []) == []
    assert storage.get("badge_eval_runs")


def test_recompute_append_only_idempotent():
    storage = {}
    supabase = FakeSupabase(storage)
    ctx = _build_ctx()
    run_badge_recompute(
        supabase,
        club_id="club",
        mode="append-only",
        ctx=ctx,
        badge_id="participant",
        allow_strict_global=True,
    )
    first_count = len(storage.get("player_badges", []))
    run_badge_recompute(
        supabase,
        club_id="club",
        mode="append-only",
        ctx=ctx,
        badge_id="participant",
        allow_strict_global=True,
    )
    second_count = len(storage.get("player_badges", []))
    assert first_count == second_count
    assert all(row.get("awarded_by") == "recompute" for row in storage.get("player_badges", []))


def test_recompute_strict_revokes_missing():
    storage = {
        "player_badges": [
            {
                "id": "pb1",
                "club_id": "club",
                "player_id": 1,
                "badge_id": "participant",
                "context_id": "overall",
            }
        ]
    }
    supabase = FakeSupabase(storage)
    ctx = SimpleNamespace(
        supabase=None,
        club_id="club",
        df_matches=pd.DataFrame(),
        df_players_all=pd.DataFrame(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
        df_badges=pd.DataFrame([{"badge_id": "participant", "state": "live"}]),
        df_player_badges=pd.DataFrame(),
        name_to_id={},
        id_to_name={},
        public_mode=False,
        admin_logged_in=True,
    )
    run_badge_recompute(
        supabase,
        club_id="club",
        mode="strict",
        ctx=ctx,
        badge_id="participant",
        allow_strict_global=True,
        created_by="admin",
        revoke_reason="strict cleanup",
    )
    row = storage["player_badges"][0]
    assert row.get("revoked_at")
    assert row.get("revoked_by") == "admin"
    assert row.get("revoke_reason") == "strict cleanup"
