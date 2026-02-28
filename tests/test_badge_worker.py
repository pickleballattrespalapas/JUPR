from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from postgrest.exceptions import APIError

from jupr_app.domain.gamification.badge_queue import BADGE_QUEUE_TABLE, enqueue_badge_eval
from jupr_app.domain.gamification.badge_worker import (
    process_badge_eval_queue,
    process_badge_eval_queue_until_empty,
)


class FakeTable:
    def __init__(self, storage, name):
        self.storage = storage
        self.name = name
        self.filters = []
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
        if self.storage.get("raise_missing_table"):
            raise APIError({"code": "PGRST205", "message": "missing table"})
        rows = payload if isinstance(payload, list) else [payload]
        stored = self.storage.setdefault(self.name, [])
        for row in rows:
            row = dict(row)
            row.setdefault("id", f"{self.name}_{len(stored) + 1}")
            stored.append(row)
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
    df_badges = pd.DataFrame(
        [
            {"badge_id": "participant", "state": "live", "eval_triggers": ["match_recorded"]},
        ]
    )
    return SimpleNamespace(
        supabase=None,
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
    storage = {}
    supabase = FakeSupabase(storage)
    enqueue_badge_eval(
        supabase,
        club_id="club",
        event_type="match_recorded",
        player_ids=[1],
        match_id="m1",
    )
    ctx = _build_ctx()
    result = process_badge_eval_queue(supabase, max_jobs=1, time_budget_seconds=2, ctx=ctx)
    assert result["processed"] == 1
    assert storage.get("player_badges")


def test_worker_dedupes_duplicate_events():
    storage = {}
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
    ctx = _build_ctx()
    process_badge_eval_queue(supabase, max_jobs=2, time_budget_seconds=2, ctx=ctx)
    assert len(storage.get("player_badges", [])) == 1


def test_worker_error_marks_queue(monkeypatch):
    storage = {}
    supabase = FakeSupabase(storage)
    enqueue_badge_eval(
        supabase,
        club_id="club",
        event_type="match_recorded",
        player_ids=[1],
        match_id="m1",
    )
    ctx = _build_ctx()

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "jupr_app.domain.gamification.badge_worker.compute_candidates_for_player",
        boom,
    )
    process_badge_eval_queue(supabase, max_jobs=1, time_budget_seconds=2, ctx=ctx)
    rows = storage.get("badge_eval_queue", [])
    assert rows[0]["status"] == "error"
    assert rows[0]["attempts"] == 1
    assert "RuntimeError" in str(rows[0].get("last_error") or "")


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


def test_worker_drain_until_empty_stops_on_empty(monkeypatch):
    results = iter([
        {"processed": 10, "errored": 0},
        {"processed": 10, "errored": 0},
        {"processed": 10, "errored": 0},
        {"processed": 0, "errored": 0},
    ])

    monkeypatch.setattr(
        "jupr_app.domain.gamification.badge_worker.process_badge_eval_queue",
        lambda *_args, **_kwargs: next(results),
    )

    result = process_badge_eval_queue_until_empty(
        supabase=object(),
        club_id="club",
        batch_max_jobs=10,
        max_wall_clock_seconds=10.0,
    )

    assert result["total_processed"] == 30
    assert result["total_errored"] == 0
    assert result["loops"] == 4
    assert result["stopped_reason"] == "empty"


def test_worker_drain_until_empty_trips_error_circuit_breaker(monkeypatch):
    results = iter([
        {"processed": 0, "errored": 1},
        {"processed": 0, "errored": 1},
        {"processed": 0, "errored": 1},
    ])

    monkeypatch.setattr(
        "jupr_app.domain.gamification.badge_worker.process_badge_eval_queue",
        lambda *_args, **_kwargs: next(results),
    )

    result = process_badge_eval_queue_until_empty(
        supabase=object(),
        club_id="club",
        batch_max_jobs=10,
        max_wall_clock_seconds=10.0,
        max_errors=10,
    )

    assert result["total_processed"] == 0
    assert result["total_errored"] == 3
    assert result["loops"] == 3
    assert result["stopped_reason"] == "error_circuit_breaker"


def test_worker_respects_time_budget_deadline(monkeypatch):
    clock = {"now": 0.0}
    jobs = [
        {"id": "j1", "club_id": "club", "event_type": "match_recorded", "player_ids": [1], "context_id": "overall"},
        {"id": "j2", "club_id": "club", "event_type": "match_recorded", "player_ids": [1], "context_id": "overall"},
        {"id": "j3", "club_id": "club", "event_type": "match_recorded", "player_ids": [1], "context_id": "overall"},
    ]
    acked: list[str] = []

    def fake_monotonic():
        return clock["now"]

    def fake_dequeue(_supabase):
        return jobs.pop(0) if jobs else None

    def fake_ack(_supabase, job_id, status, error=None):
        if status == "done":
            acked.append(str(job_id))

    monkeypatch.setattr("jupr_app.domain.gamification.badge_worker.time.monotonic", fake_monotonic)
    monkeypatch.setattr("jupr_app.domain.gamification.badge_worker.dequeue_badge_eval", fake_dequeue)
    monkeypatch.setattr("jupr_app.domain.gamification.badge_worker.ack_badge_eval", fake_ack)
    monkeypatch.setattr("jupr_app.domain.gamification.badge_worker._resolve_context", lambda *_a, **_k: _build_ctx())
    monkeypatch.setattr("jupr_app.domain.gamification.badge_worker._update_incremental_facts", lambda *_a, **_k: None)

    def fake_compute(*_args, **_kwargs):
        clock["now"] += 0.7
        return []

    monkeypatch.setattr("jupr_app.domain.gamification.badge_worker.compute_candidates_for_player", fake_compute)

    result = process_badge_eval_queue(object(), max_jobs=10, time_budget_seconds=1.0, ctx=_build_ctx())

    assert result["processed"] == 2
    assert acked == ["j1", "j2"]


def test_worker_drain_stops_on_max_wall_clock(monkeypatch):
    clock = {"now": 0.0}
    budgets: list[float] = []

    def fake_monotonic():
        return clock["now"]

    def fake_batch(*_args, **kwargs):
        budget = float(kwargs["time_budget_seconds"])
        budgets.append(budget)
        clock["now"] += budget + 0.25
        return {"processed": 1, "errored": 0}

    monkeypatch.setattr("jupr_app.domain.gamification.badge_worker.time.monotonic", fake_monotonic)
    monkeypatch.setattr("jupr_app.domain.gamification.badge_worker.process_badge_eval_queue", fake_batch)

    result = process_badge_eval_queue_until_empty(
        supabase=object(),
        club_id="club",
        batch_max_jobs=10,
        per_batch_time_budget_seconds=0.9,
        max_wall_clock_seconds=1.0,
    )

    assert result["stopped_reason"] == "max_wall_clock"
    assert result["loops"] == 1
    assert budgets == [0.9]
