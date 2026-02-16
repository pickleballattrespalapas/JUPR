from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue


class FakeTable:
    def __init__(self, storage: dict, name: str):
        self.storage = storage
        self.name = name
        self.filters: list[tuple[str, str, object]] = []
        self.sort_key = None
        self.sort_desc = False
        self.limit_count = None
        self.update_payload = None

    def select(self, _cols):
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column, values):
        self.filters.append(("in", column, set(values)))
        return self

    def order(self, column, desc=False):
        self.sort_key = column
        self.sort_desc = desc
        return self

    def limit(self, count):
        self.limit_count = count
        return self

    def update(self, payload):
        self.update_payload = payload
        return self

    def insert(self, payload):
        rows = payload if isinstance(payload, list) else [payload]
        existing = self.storage.setdefault(self.name, [])
        for row in rows:
            row = dict(row)
            row.setdefault("id", f"{self.name}_{len(existing) + 1}")
            existing.append(row)
        self.storage.setdefault("ops", []).append((self.name, "insert"))
        return self

    def upsert(self, payload, on_conflict=None):
        keys = [c.strip() for c in str(on_conflict or "").split(",") if c.strip()]
        row_list = payload if isinstance(payload, list) else [payload]
        existing = self.storage.setdefault(self.name, [])
        for row in row_list:
            row = dict(row)
            match = None
            if keys:
                for current in existing:
                    if all(str(current.get(k)) == str(row.get(k)) for k in keys):
                        match = current
                        break
            if match is None:
                row.setdefault("id", f"{self.name}_{len(existing) + 1}")
                existing.append(row)
            else:
                match.update(row)
        self.storage.setdefault("ops", []).append((self.name, "upsert"))
        return self

    def execute(self):
        rows = list(self.storage.get(self.name, []))
        for op, col, val in self.filters:
            if op == "eq":
                rows = [r for r in rows if str(r.get(col)) == str(val)]
            elif op == "in":
                rows = [r for r in rows if r.get(col) in val]
        if self.sort_key:
            rows = sorted(rows, key=lambda row: row.get(self.sort_key), reverse=self.sort_desc)
        if self.limit_count is not None:
            rows = rows[: int(self.limit_count)]
        if self.update_payload is not None:
            for row in rows:
                row.update(self.update_payload)
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, storage=None):
        self.storage = storage if storage is not None else {}

    def table(self, name):
        return FakeTable(self.storage, name)


def test_badge_v3_pipeline_updates_facts_before_enqueue_and_award(monkeypatch):
    storage = {
        "matches": [],
        "players": [
            {"club_id": "club", "id": 1, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
            {"club_id": "club", "id": 2, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
            {"club_id": "club", "id": 3, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
            {"club_id": "club", "id": 4, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
        ],
        "league_ratings": [],
        "player_badge_facts": [],
        "badge_eval_queue": [],
        "badges": [{"club_id": "club", "badge_id": "grinder", "status": "published", "award_count": 0, "is_locked": False}],
        "badge_rule_conditions": [{"badge_id": "grinder", "fact_key": "total_matches", "operator": ">=", "value_numeric": 1}],
        "player_badges": [],
    }
    supabase = FakeSupabase(storage)

    def _fake_submit_match(*, club_id, context_type, context_id, match_payload, idempotency_key, run_context_hooks):
        row = dict(match_payload)
        row["club_id"] = club_id
        row["context_type"] = context_type
        row["context_id"] = context_id
        row["id"] = idempotency_key or f"m{len(storage['matches']) + 1}"
        storage["matches"].append(row)
        storage.setdefault("ops", []).append(("matches", "insert"))
        return {"id": row["id"]}

    monkeypatch.setattr("jupr_app.domain.match_processing.submit_match", _fake_submit_match)

    match_list = [{
        "t1_p1": 1,
        "t1_p2": 2,
        "t2_p1": 3,
        "t2_p2": 4,
        "s1": 11,
        "s2": 7,
        "league": "Open",
        "date": "2024-01-01T00:00:00Z",
        "match_type": "League",
    }]

    process_matches(
        match_list,
        supabase=supabase,
        club_id="club",
        name_to_id={},
        df_players_all=pd.DataFrame(storage["players"]),
        df_leagues=pd.DataFrame(columns=["club_id", "player_id", "league_name", "rating"]),
        df_meta=pd.DataFrame(columns=["league_name", "k_factor"]),
    )

    ops = storage.get("ops", [])
    assert ("badge_eval_queue", "upsert") in ops
    assert not storage.get("player_badge_facts", [])

    ctx = SimpleNamespace(
        supabase=supabase,
        club_id="club",
        df_matches=pd.DataFrame(storage["matches"]),
        df_players_all=pd.DataFrame(storage["players"]),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
        df_badges=pd.DataFrame([{"badge_id": "grinder", "eval_triggers": ["match_recorded"]}]),
        df_player_badges=pd.DataFrame(),
        name_to_id={},
        id_to_name={},
        public_mode=False,
        admin_logged_in=True,
    )

    result = process_badge_eval_queue(supabase, max_jobs=1, time_budget_seconds=2, ctx=ctx)
    assert result == {"processed": 1, "errored": 0}
    assert len(storage.get("player_badges", [])) == 4

    fact_rows = storage.get("player_badge_facts", [])
    for pid in [1, 2, 3, 4]:
        assert any(
            row["player_id"] == pid and row["fact_key"] == "total_matches" and float(row["fact_value_num"]) == 1.0
            for row in fact_rows
        )
