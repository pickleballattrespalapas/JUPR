from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

from postgrest.exceptions import APIError

from jupr_app.domain.gamification.badge_queue import (
    BADGE_QUEUE_CLAIM_RPC,
    BADGE_QUEUE_TABLE,
    dequeue_badge_eval,
    enqueue_badge_eval,
)
from jupr_app.domain.gamification.badge_worker import (
    _resolve_context,
    _update_incremental_facts,
    load_live_badge_data,
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

    def is_(self, column, value):
        self.filters.append(("is", column, value))
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
            elif op == "is" and value is None:
                data = [row for row in data if row.get(column) is None]
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
        self.claim_lock = threading.Lock()
        self.rpc_calls = []

    def table(self, name):
        return FakeTable(self.storage, name)

    def rpc(self, name, params):
        self.rpc_calls.append((name, dict(params)))
        return FakeBadgeQueueClaim(self, name, params)


class FakeBadgeQueueClaim:
    def __init__(self, client, name, params):
        self.client = client
        self.name = name
        self.params = dict(params)

    def execute(self):
        if self.name != BADGE_QUEUE_CLAIM_RPC:
            raise AssertionError(f"unexpected RPC: {self.name}")
        if self.client.storage.get("raise_missing_claim_rpc"):
            raise APIError({"code": "PGRST202", "message": "function not found"})
        club_id = str(self.params.get("p_club_id") or "")
        with self.client.claim_lock:
            pending = [
                row
                for row in self.client.storage.get(BADGE_QUEUE_TABLE, [])
                if str(row.get("club_id") or "") == club_id and row.get("status") == "pending"
            ]
            pending.sort(key=lambda row: (str(row.get("created_at") or ""), str(row.get("id") or "")))
            if not pending:
                return SimpleNamespace(data=[])
            job = pending[0]
            job["status"] = "processing"
            job["attempts"] = int(job.get("attempts") or 0) + 1
            return SimpleNamespace(data=[dict(job)])


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
        df_players_all=pd.DataFrame(
            [
                {
                    "id": 1,
                    "wins": 1,
                    "losses": 0,
                    "matches_played": 1,
                }
            ]
        ),
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
    result = process_badge_eval_queue(supabase, "club", max_jobs=1, time_budget_seconds=2, ctx=ctx)
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
    process_badge_eval_queue(supabase, "club", max_jobs=2, time_budget_seconds=2, ctx=ctx)
    assert len(storage.get("player_badges", [])) == 1


def test_worker_direct_load_excludes_soft_deleted_matches():
    storage = {
        "matches": [
            {"id": "active", "club_id": "club", "deleted_at": None},
            {
                "id": "deleted",
                "club_id": "club",
                "deleted_at": "2026-07-26T00:00:00Z",
            },
        ]
    }
    loaded = load_live_badge_data(FakeSupabase(storage), "club")
    assert loaded[3]["id"].tolist() == ["active"]


def test_worker_supplied_context_excludes_soft_deleted_matches():
    ctx = _build_ctx()
    ctx.df_matches = pd.concat(
        [
            ctx.df_matches.assign(deleted_at=None),
            ctx.df_matches.assign(id="deleted", deleted_at="2026-07-26T00:00:00Z"),
        ],
        ignore_index=True,
    )

    resolved = _resolve_context(ctx, object(), "club", 5000)

    assert resolved is not ctx
    assert resolved.df_matches["id"].tolist() == ["m1"]
    assert ctx.df_matches["id"].tolist() == ["m1", "deleted"]


def test_match_updated_does_not_increment_matches_seen():
    storage = {}
    supabase = FakeSupabase(storage)
    job = {"club_id": "club", "event_type": "match_updated"}

    _update_incremental_facts(supabase, job, [1], "overall")

    assert storage.get("player_badge_facts") is None


def test_match_recorded_still_increments_matches_seen():
    storage = {}
    supabase = FakeSupabase(storage)
    job = {"club_id": "club", "event_type": "match_recorded"}

    _update_incremental_facts(supabase, job, [1], "overall")

    assert storage["player_badge_facts"][0]["fact_value_num"] == 1


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
    process_badge_eval_queue(supabase, "club", max_jobs=1, time_budget_seconds=2, ctx=ctx)
    rows = storage.get("badge_eval_queue", [])
    assert rows[0]["status"] == "error"
    assert rows[0]["attempts"] == 1
    assert "RuntimeError" in str(rows[0].get("last_error") or "")


def test_enqueue_badge_eval_missing_table_is_ignored():
    storage = {"raise_missing_table": True}
    supabase = FakeSupabase(storage)
    status = enqueue_badge_eval(
        supabase,
        club_id="club",
        event_type="match_recorded",
        player_ids=[1],
        match_id="m1",
    )
    assert status["queued"] is False
    assert status["reason"] == "missing_table"
    assert storage.get(BADGE_QUEUE_TABLE) is None


def test_queue_deduplication_and_processing_are_club_scoped():
    storage = {}
    supabase = FakeSupabase(storage)
    for club_id in ("club", "other"):
        enqueue_badge_eval(
            supabase,
            club_id=club_id,
            event_type="match_recorded",
            player_ids=[1],
            match_id="shared-match-id",
        )

    # Same-club retries still deduplicate, while another club may use the same
    # source match identifier independently.
    enqueue_badge_eval(
        supabase,
        club_id="club",
        event_type="match_recorded",
        player_ids=[1],
        match_id="shared-match-id",
    )
    assert len(storage[BADGE_QUEUE_TABLE]) == 2

    result = process_badge_eval_queue(
        supabase,
        "club",
        max_jobs=1,
        time_budget_seconds=2,
        ctx=_build_ctx(),
    )

    assert result == {"processed": 1, "errored": 0}
    rows_by_club = {row["club_id"]: row for row in storage[BADGE_QUEUE_TABLE]}
    assert rows_by_club["club"]["status"] == "done"
    assert rows_by_club["other"]["status"] == "pending"
    assert supabase.rpc_calls[0] == (BADGE_QUEUE_CLAIM_RPC, {"p_club_id": "club"})


def test_atomic_dequeue_claims_one_pending_job_at_most_once():
    storage = {
        BADGE_QUEUE_TABLE: [
            {
                "id": "job-1",
                "created_at": "2026-07-18T00:00:00Z",
                "club_id": "club",
                "event_type": "match_recorded",
                "status": "pending",
                "attempts": 0,
            }
        ]
    }
    supabase = FakeSupabase(storage)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: dequeue_badge_eval(supabase, club_id="club"), range(2)))

    claimed = [row for row in results if row is not None]
    assert [row["id"] for row in claimed] == ["job-1"]
    assert storage[BADGE_QUEUE_TABLE][0]["status"] == "processing"
    assert storage[BADGE_QUEUE_TABLE][0]["attempts"] == 1


def test_dequeue_fails_closed_when_atomic_claim_migration_is_missing():
    supabase = FakeSupabase({"raise_missing_claim_rpc": True})

    with pytest.raises(RuntimeError, match="20260718141016_badge_eval_queue_atomic_club_claim"):
        dequeue_badge_eval(supabase, club_id="club")


def test_worker_max_jobs_caps_errors_as_well_as_successes(monkeypatch):
    storage = {}
    supabase = FakeSupabase(storage)
    for match_id in ("m1", "m2"):
        enqueue_badge_eval(
            supabase,
            club_id="club",
            event_type="match_recorded",
            player_ids=[1],
            match_id=match_id,
        )

    monkeypatch.setattr(
        "jupr_app.domain.gamification.badge_worker._process_job_with_retry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = process_badge_eval_queue(
        supabase,
        "club",
        max_jobs=1,
        time_budget_seconds=2,
        ctx=_build_ctx(),
    )

    assert result == {"processed": 0, "errored": 1}
    assert [row["status"] for row in storage[BADGE_QUEUE_TABLE]].count("pending") == 1


def test_worker_drain_until_empty_stops_on_empty(monkeypatch):
    results = iter([
        {"processed": 10, "errored": 0},
        {"processed": 10, "errored": 0},
        {"processed": 10, "errored": 0},
        {"processed": 0, "errored": 0},
    ])

    seen_club_ids = []

    def fake_batch(_supabase, club_id, **_kwargs):
        seen_club_ids.append(club_id)
        return next(results)

    monkeypatch.setattr(
        "jupr_app.domain.gamification.badge_worker.process_badge_eval_queue",
        fake_batch,
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
    assert seen_club_ids == ["club"] * 4


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

    def fake_dequeue(_supabase, *, club_id):
        assert club_id == "club"
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

    result = process_badge_eval_queue(
        object(),
        "club",
        max_jobs=10,
        time_budget_seconds=1.0,
        ctx=_build_ctx(),
    )

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
