from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from jupr_app.domain.gamification.recompute import run_badge_recompute


class FakeTable:
    def __init__(self, storage, name, mutations):
        self.storage = storage
        self.name = name
        self.mutations = mutations
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
        self.mutations.append(("insert", self.name))
        self.insert_payload = payload
        return self

    def update(self, payload):
        self.mutations.append(("update", self.name))
        self.update_payload = payload
        return self

    def upsert(self, rows, on_conflict=None):
        self.mutations.append(("upsert", self.name))
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
        self.mutations = []

    def table(self, name):
        return FakeTable(self.storage, name, self.mutations)


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
    assert summary["read_only"] is True
    assert storage.get("player_badges", []) == []
    assert storage.get("badge_eval_runs", []) == []
    assert supabase.mutations == []


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


def test_recompute_strict_global_blocked_without_scope():
    supabase = FakeSupabase({})
    with pytest.raises(ValueError, match="Strict mode requires"):
        run_badge_recompute(
            supabase,
            club_id="club",
            mode="strict",
            ctx=_build_ctx(),
            allow_strict_global=False,
        )


def test_recompute_append_only_does_not_revoke_stale_rows():
    storage = {
        "player_badges": [
            {
                "id": "legacy1",
                "club_id": "club",
                "player_id": 99,
                "badge_id": "participant",
                "context_id": "overall",
                "revoked_at": None,
            }
        ]
    }
    supabase = FakeSupabase(storage)
    run_badge_recompute(
        supabase,
        club_id="club",
        mode="append-only",
        ctx=_build_ctx(),
        badge_id="participant",
        allow_strict_global=True,
    )
    stale = next(row for row in storage["player_badges"] if row["id"] == "legacy1")
    assert stale.get("revoked_at") is None


def test_recompute_strict_scoped_to_badge_id_only():
    storage = {
        "player_badges": [
            {
                "id": "legacy-high",
                "club_id": "club",
                "player_id": 7,
                "badge_id": "high_roller",
                "context_id": "legacy",
                "revoked_at": None,
            },
            {
                "id": "keep-other",
                "club_id": "club",
                "player_id": 7,
                "badge_id": "participant",
                "context_id": "overall",
                "revoked_at": None,
            },
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
        df_badges=pd.DataFrame([
            {"badge_id": "high_roller", "state": "live"},
            {"badge_id": "participant", "state": "live"},
        ]),
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
        badge_id="high_roller",
        allow_strict_global=False,
    )
    legacy = next(row for row in storage["player_badges"] if row["id"] == "legacy-high")
    untouched = next(row for row in storage["player_badges"] if row["id"] == "keep-other")
    assert legacy.get("revoked_at") is not None
    assert untouched.get("revoked_at") is None


def test_recompute_append_only_uses_hybrid_safe_legacy_rows():
    storage = {}
    supabase = FakeSupabase(storage)
    ctx = SimpleNamespace(
        supabase=None,
        club_id="club",
        df_matches=pd.DataFrame(
            [
                {
                    "id": "legacy-1",
                    "club_id": "club",
                    "date": "2024-02-01T10:00:00Z",
                    "league_name": "Open",
                    "team1_player1": 1,
                    "team2_player1": 2,
                    "team1_score": 11,
                    "team2_score": 7,
                }
            ]
        ),
        # Participation badges intentionally use persisted standings totals;
        # match-derived badges below exercise the hybrid legacy fact source.
        df_players_all=pd.DataFrame(
            [
                {"id": 1, "wins": 1, "losses": 0, "matches_played": 1},
                {"id": 2, "wins": 0, "losses": 1, "matches_played": 1},
            ]
        ),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
        df_badges=pd.DataFrame(
            [
                {"badge_id": "participant", "state": "live"},
                {"badge_id": "first_win", "state": "live"},
            ]
        ),
        df_player_badges=pd.DataFrame(),
        name_to_id={},
        id_to_name={},
        public_mode=False,
        admin_logged_in=True,
    )
    run_badge_recompute(
        supabase,
        club_id="club",
        mode="append-only",
        ctx=ctx,
        allow_strict_global=True,
    )
    badge_pairs = {(row["player_id"], row["badge_id"]) for row in storage.get("player_badges", [])}
    assert (1, "participant") in badge_pairs
    assert (1, "first_win") in badge_pairs
