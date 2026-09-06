from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.badge_engine import compute_candidates_for_club
from jupr_app.domain.gamification.recompute import run_badge_recompute


class FakeTable:
    def __init__(self, storage, name):
        self.storage = storage
        self.name = name
        self.filters = []
        self.update_payload = None

    def select(self, _cols):
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column, values):
        self.filters.append(("in", column, set(values)))
        return self

    def insert(self, payload):
        payloads = payload if isinstance(payload, list) else [payload]
        self.storage.setdefault(self.name, []).extend(payloads)
        return self

    def upsert(self, rows, on_conflict=None, ignore_duplicates=False):
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

    def update(self, payload):
        self.update_payload = payload
        return self

    def range(self, start, end):
        self.page_bounds = (start, end)
        return self

    def order(self, *_args, **_kwargs):
        return self

    def execute(self):
        data = list(self.storage.get(self.name, []))
        for op, column, value in self.filters:
            if op == "eq":
                data = [row for row in data if str(row.get(column)) == str(value)]
            elif op == "in":
                data = [row for row in data if row.get(column) in value]
        if self.update_payload is not None:
            for row in data:
                row.update(self.update_payload)
        if hasattr(self, "page_bounds"):
            data = data[self.page_bounds[0]:self.page_bounds[1] + 1]
        return SimpleNamespace(data=data)


class FakeSupabase:
    def __init__(self, storage=None):
        self.storage = storage if storage is not None else {}

    def table(self, name):
        return FakeTable(self.storage, name)


def _ctx(df_matches: pd.DataFrame, df_players_all: pd.DataFrame, badge_ids: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        supabase=FakeSupabase({}),
        club_id="club",
        df_matches=df_matches,
        df_players_all=df_players_all,
        df_players_active=df_players_all,
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
        df_badges=pd.DataFrame([{"badge_id": bid, "state": "live"} for bid in badge_ids]),
        df_player_badges=pd.DataFrame(),
        name_to_id={},
        id_to_name={},
        public_mode=False,
        admin_logged_in=True,
    )


def test_high_roller_uses_standings_when_facts_are_lower():
    matches = pd.DataFrame([{"id": f"m{i}", "club_id": "club", "league": "Open", "date": "2024-01-01", "t1_p1": 1, "t2_p1": 2, "score_t1": 11, "score_t2": 9} for i in range(42)])
    players = pd.DataFrame(
        [
            {"id": 1, "wins": 112, "losses": 20, "matches_played": 132},
            {"id": 2, "wins": 20, "losses": 112, "matches_played": 132},
        ]
    )
    ctx = _ctx(matches, players, ["high_roller"])

    candidates = list(compute_candidates_for_club("club", ctx=ctx))
    high_roller = [c for c in candidates if c.badge_id == "high_roller" and c.player_id == 1]
    assert len(high_roller) == 1
    assert high_roller[0].value_json["wins"] == 112


def test_participant_badges_use_standings_totals():
    matches = pd.DataFrame()
    players = pd.DataFrame(
        [
            {"id": 1, "wins": 5, "losses": 0, "matches_played": 5},
            {"id": 2, "wins": 0, "losses": 0, "matches_played": 0},
        ]
    )
    ctx = _ctx(matches, players, ["participant"])
    candidates = list(compute_candidates_for_club("club", ctx=ctx))

    assert {(c.player_id, c.badge_id) for c in candidates} == {(1, "participant")}


def test_match_dependent_badges_still_use_match_facts():
    matches = pd.DataFrame()
    players = pd.DataFrame([{"id": 1, "wins": 20, "losses": 0, "matches_played": 20}])
    ctx = _ctx(matches, players, ["first_win"])
    candidates = list(compute_candidates_for_club("club", ctx=ctx))

    assert not any(c.badge_id == "first_win" for c in candidates)


def test_canonical_only_badges_remain_strict():
    matches = pd.DataFrame(
        [
            {
                "id": "legacy-1",
                "club_id": "club",
                "date": "2024-01-01",
                "league_name": "Open",
                "team1_player1": 1,
                "team2_player1": 2,
                "team1_score": 11,
                "team2_score": 9,
            }
        ]
    )
    players = pd.DataFrame([{"id": 1, "wins": 1, "losses": 0, "matches_played": 1}])
    ctx = _ctx(matches, players, ["david_vs_goliath"])

    candidates = list(compute_candidates_for_club("club", ctx=ctx))
    assert not any(c.badge_id == "david_vs_goliath" for c in candidates)


def test_strict_recompute_revokes_high_roller_with_new_source_policy():
    storage = {
        "player_badges": [
            {
                "id": "legacy-high",
                "club_id": "club",
                "player_id": 1,
                "badge_id": "high_roller",
                "context_id": "legacy",
                "revoked_at": None,
            }
        ]
    }
    supabase = FakeSupabase(storage)
    ctx = _ctx(
        pd.DataFrame(),
        pd.DataFrame([{"id": 1, "wins": 42, "losses": 30, "matches_played": 72}]),
        ["high_roller"],
    )

    run_badge_recompute(
        supabase,
        club_id="club",
        mode="strict",
        ctx=ctx,
        badge_id="high_roller",
        allow_strict_global=False,
        revoke_reason="strict cleanup",
    )

    legacy = storage["player_badges"][0]
    assert legacy.get("revoked_at") is not None


def test_recompute_append_only_remains_idempotent_for_high_roller():
    storage = {}
    supabase = FakeSupabase(storage)
    ctx = _ctx(
        pd.DataFrame(),
        pd.DataFrame([{"id": 1, "wins": 112, "losses": 8, "matches_played": 120}]),
        ["high_roller"],
    )

    run_badge_recompute(
        supabase,
        club_id="club",
        mode="append-only",
        ctx=ctx,
        badge_id="high_roller",
        allow_strict_global=True,
    )
    first = len(storage.get("player_badges", []))
    run_badge_recompute(
        supabase,
        club_id="club",
        mode="append-only",
        ctx=ctx,
        badge_id="high_roller",
        allow_strict_global=True,
    )
    second = len(storage.get("player_badges", []))

    assert first == second
