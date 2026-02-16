from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from postgrest.exceptions import APIError

from jupr_app.domain.gamification.badge_queue import BADGE_QUEUE_TABLE, enqueue_badge_eval
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue


class FakeTable:
    def __init__(self, storage, name):
        self.storage = storage
        self.name = name
        self.filters = []
        self.sort_key = None
        self.sort_desc = False
        self.limit_count = None
        self.update_payload = None
        self._insert_result = None

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

    def insert(self, payload, **kwargs):
        if self.storage.get("raise_missing_table"):
            raise APIError({"code": "PGRST205", "message": "missing table"})
        rows = payload if isinstance(payload, list) else [payload]
        stored = self.storage.setdefault(self.name, [])
        conflict_cols = [c.strip() for c in str(kwargs.get("on_conflict") or "").split(",") if c.strip()]
        ignore_duplicates = bool(kwargs.get("ignore_duplicates"))
        inserted = []
        for row in rows:
            row = dict(row)
            match = None
            if conflict_cols:
                for current in stored:
                    if all(str(current.get(c)) == str(row.get(c)) for c in conflict_cols):
                        match = current
                        break
            if match is not None and ignore_duplicates:
                continue
            if match is None:
                row.setdefault("id", f"{self.name}_{len(stored) + 1}")
                stored.append(row)
                inserted.append(row)
            else:
                match.update(row)
                inserted.append(match)
        self._insert_result = inserted
        return self

    def upsert(self, rows, on_conflict=None):
        if self.storage.get("raise_missing_table"):
            raise APIError({"code": "PGRST205", "message": "missing table"})
        existing = self.storage.setdefault(self.name, [])
        keys = [c.strip() for c in str(on_conflict or "").split(",") if c.strip()]
        existing_keys = {tuple(row.get(k) for k in keys) for row in existing} if keys else set()
        row_list = rows if isinstance(rows, list) else [rows]
        for row in row_list:
            row = dict(row)
            key = tuple(row.get(k) for k in keys) if keys else None
            if key is not None and key in existing_keys:
                continue
            row.setdefault("id", f"{self.name}_{len(existing) + 1}")
            existing.append(row)
            if key is not None:
                existing_keys.add(key)
        return self

    def execute(self):
        if self.storage.get("raise_missing_table"):
            raise APIError({"code": "PGRST205", "message": "missing table"})
        if self._insert_result is not None:
            data = list(self._insert_result)
            self._insert_result = None
            return SimpleNamespace(data=data)
        data = list(self.storage.get(self.name, []))
        for op, column, value in self.filters:
            if op == "eq":
                data = [row for row in data if str(row.get(column)) == str(value)]
            elif op == "in":
                data = [row for row in data if row.get(column) in value]
        if self.sort_key:
            data = sorted(data, key=lambda row: row.get(self.sort_key), reverse=self.sort_desc)
        if self.limit_count is not None:
            data = data[: int(self.limit_count)]
        if self.update_payload is not None:
            for row in data:
                row.update(self.update_payload)
        return SimpleNamespace(data=data)


class FakeSupabase:
    def __init__(self, storage=None):
        self.storage = storage if storage is not None else {}

    def table(self, name):
        return FakeTable(self.storage, name)


def _build_ctx(supabase=None):
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
    df_badges = pd.DataFrame(
        [
            {"badge_id": "grinder", "state": "live", "eval_triggers": ["match_recorded"]},
        ]
    )
    return SimpleNamespace(
        supabase=supabase,
        club_id="club",
        df_matches=df_matches,
        df_players_all=pd.DataFrame(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
        df_badges=df_badges,
        df_player_badges=pd.DataFrame(),
        name_to_id={},
        id_to_name={},
        public_mode=False,
        admin_logged_in=True,
    )


def test_worker_processes_queue_and_awards_badge():
    storage = {
        "badges": [{"club_id": "club", "badge_id": "grinder", "status": "published", "award_count": 0, "is_locked": False}],
        "badge_rule_conditions": [{"badge_id": "grinder", "fact_key": "total_matches", "operator": ">=", "value_numeric": 1}],
        "player_badge_facts": [],
        "player_badges": [],
    }
    supabase = FakeSupabase(storage)
    enqueue_badge_eval(
        supabase,
        club_id="club",
        event_type="match_recorded",
        player_ids=[1],
        match_id="m1",
    )
    ctx = _build_ctx(supabase)
    result = process_badge_eval_queue(supabase, max_jobs=1, time_budget_seconds=2, ctx=ctx)
    assert result["processed"] == 1
    assert storage.get("player_badges")


def test_worker_dedupes_duplicate_events():
    storage = {
        "badges": [{"club_id": "club", "badge_id": "grinder", "status": "published", "award_count": 0, "is_locked": False}],
        "badge_rule_conditions": [{"badge_id": "grinder", "fact_key": "total_matches", "operator": ">=", "value_numeric": 1}],
        "player_badge_facts": [],
        "player_badges": [],
    }
    supabase = FakeSupabase(storage)
    enqueue_badge_eval(
        supabase,
        club_id="club",
        event_type="match_recorded",
        player_ids=[1],
        match_id="m1",
    )
    enqueue_badge_eval(
        supabase,
        club_id="club",
        event_type="match_recorded",
        player_ids=[1],
        match_id="m1",
    )
    ctx = _build_ctx(supabase)
    process_badge_eval_queue(supabase, max_jobs=2, time_budget_seconds=2, ctx=ctx)
    assert len(storage.get("player_badges", [])) == 1


def test_worker_error_marks_queue(monkeypatch):
    storage = {
        "badges": [{"club_id": "club", "badge_id": "grinder", "status": "published", "award_count": 0, "is_locked": False}],
        "badge_rule_conditions": [{"badge_id": "grinder", "fact_key": "total_matches", "operator": ">=", "value_numeric": 1}],
        "player_badge_facts": [],
        "player_badges": [],
    }
    supabase = FakeSupabase(storage)
    enqueue_badge_eval(
        supabase,
        club_id="club",
        event_type="match_recorded",
        player_ids=[1],
        match_id="m1",
    )
    ctx = _build_ctx(supabase)

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "jupr_app.domain.gamification.v3_engine.evaluate_badges_v3_for_player",
        boom,
    )
    process_badge_eval_queue(supabase, max_jobs=1, time_budget_seconds=2, ctx=ctx)
    rows = storage.get("badge_eval_queue", [])
    assert rows[0]["status"] == "error"
    assert rows[0]["attempts"] == 1


def test_enqueue_badge_eval_missing_table_is_ignored():
    storage = {"raise_missing_table": True}
    supabase = FakeSupabase(storage)
    enqueue_badge_eval(
        supabase,
        club_id="club",
        event_type="match_recorded",
        player_ids=[1],
        match_id="m1",
    )
    assert storage.get(BADGE_QUEUE_TABLE) is None


def test_worker_passes_match_id_to_fact_engine(monkeypatch):
    storage = {
        "badges": [{"club_id": "club", "badge_id": "grinder", "status": "published", "award_count": 0, "is_locked": False}],
        "badge_rule_conditions": [{"badge_id": "grinder", "fact_key": "total_matches", "operator": ">=", "value_numeric": 1}],
        "player_badge_facts": [],
        "player_badges": [],
    }
    supabase = FakeSupabase(storage)
    enqueue_badge_eval(
        supabase,
        club_id="club",
        event_type="match_recorded",
        player_ids=[1],
        match_id="m42",
    )

    captured = {}

    def fake_update(*_args, **kwargs):
        captured["match_id"] = kwargs.get("match_id")

    monkeypatch.setattr(
        "jupr_app.domain.gamification.badge_worker.update_match_facts_for_players",
        fake_update,
    )
    process_badge_eval_queue(supabase, max_jobs=1, time_budget_seconds=2, ctx=_build_ctx(supabase))

    assert captured["match_id"] == "m42"
