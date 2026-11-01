from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from uuid import UUID

import pytest

from jupr_app.domain.notifications import player_profile_update_repo as repo
from jupr_app.domain.notifications import player_update_sender as sender
from jupr_app.workers import player_update_email_scheduler as scheduler
from jupr_app.workers import player_update_email_worker as worker


class Query:
    def __init__(self, db, table):
        self.db, self.name = db, table
        self.filters, self.sort = [], []
        self.window, self.cap, self.payload = None, None, None
        self.insert_payload = None

    def select(self, *_): return self
    def eq(self, key, value):
        self.filters.append(("eq", key, value))
        return self
    def gt(self, key, value):
        self.filters.append(("gt", key, value))
        return self
    def in_(self, key, values):
        self.filters.append(("in", key, values))
        return self
    def order(self, key, desc=False):
        self.sort.append((key, desc))
        return self
    def range(self, start, end):
        self.window = (start, end + 1)
        return self
    def limit(self, count):
        self.cap = count
        return self
    def insert(self, payload):
        self.insert_payload = payload
        return self
    def update(self, payload):
        self.payload = payload
        return self

    def execute(self):
        self.db.queries.append((self.name, deepcopy(self.filters)))
        rows = self.db.tables.setdefault(self.name, [])
        if self.insert_payload is not None:
            row = deepcopy(self.insert_payload)
            row.setdefault("id", str(UUID(int=self.db.next_id)))
            self.db.next_id += 1
            row.setdefault("created_at", row["id"])
            row.setdefault("row_version", 1)
            rows.append(row)
            return SimpleNamespace(data=[deepcopy(row)])
        selected = list(rows)
        for op, key, expected in self.filters:
            def value(row):
                if "->>" in key:
                    column, field = key.split("->>")
                    return (row.get(column) or {}).get(field)
                return row.get(key)
            if op == "eq": selected = [r for r in selected if value(r) == expected]
            elif op == "gt": selected = [r for r in selected if value(r) > expected]
            else: selected = [r for r in selected if value(r) in expected]
        for key, desc in reversed(self.sort):
            selected.sort(key=lambda r: r.get(key) or "", reverse=desc)
        if self.window is not None: selected = selected[slice(*self.window)]
        if self.cap is not None: selected = selected[:self.cap]
        if self.payload is not None:
            for row in selected:
                row.update(deepcopy(self.payload))
                row["row_version"] += 1
        return SimpleNamespace(data=deepcopy(selected))


class Database:
    def __init__(self):
        self.tables, self.queries, self.next_id = {}, [], 1
        self.closed = 0
        self.options = SimpleNamespace(httpx_client=SimpleNamespace(close=self.close))
    def close(self): self.closed += 1
    def table(self, name): return Query(self, name)


def enable(monkeypatch):
    for key, value in {
        "JUPR_ENV": "production", "FLY_APP_NAME": "juprleagues-api",
        "SUPABASE_URL": "https://dnoockbwfenunhcibwfn.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "test-key", "JUPR_PRODUCTION_WRITE_POLICY": "enabled",
        "JUPR_STAGING_WRITE_WAVE": "none", "JUPR_EMAIL_MODE": "live",
        "JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS": "1",
        "JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL": "1", "JUPR_REQUIRE_WORKER_RUN_LOG": "1",
    }.items(): monkeypatch.setenv(key, value)
    monkeypatch.setattr(scheduler, "get_smtp_config_status", lambda: {"ok": True})


def queue(db, *, club="club", player=1, matches=None):
    subs = db.tables.setdefault("player_profile_update_subscriptions", [])
    if not any(s["club_id"] == club and s["player_id"] == player for s in subs):
        subs.append({"id": f"{club}-{player}", "club_id": club, "player_id": player,
            "email": "player@example.test", "request_status": "active", "row_version": 1,
            "unsubscribe_token": "test-unsubscribe", "preferences_json": {}})
    result = repo.queue_player_updates_for_affected_subscribers(db, club_id=club,
        affected_player_ids=[player], match_dates=["2026-11-01"], match_ids=matches or [101, 102])
    assert result["queued"] == 1
    return db.tables["player_profile_update_outbox"][-1]


@pytest.fixture
def delivery(monkeypatch):
    enable(monkeypatch)
    db, sent, computed = Database(), [], []
    monkeypatch.setattr(scheduler, "make_supabase", lambda *_: db)
    monkeypatch.setattr(worker, "make_supabase", lambda *_: db)
    monkeypatch.setattr(sender, "render_player_digest_chart_png", lambda _: None)
    def compute(_ctx, **kwargs):
        computed.append(kwargs)
        return {"player_name": "Player", "summary": {"matches_played": 2, "overall_delta": .02},
                "matches_played_rows": [{"id": mid} for mid in kwargs["match_ids"]]}
    monkeypatch.setattr(sender, "compute_player_weekly_digest", compute)
    monkeypatch.setattr(sender, "send_email_with_inline_chart", lambda **kw: sent.append(kw) or "<test-message>")
    return db, sent, computed


def test_published_session_is_delivered_once_through_real_queue_claim_and_worker(delivery):
    db, sent, computed = delivery
    first = queue(db)
    second = queue(db, club="other", player=2, matches=[201])
    # A publisher retry uses the same identity and cannot create another email.
    retry = repo.queue_player_updates_for_affected_subscribers(db, club_id="club",
        affected_player_ids=[1], match_dates=["2026-11-01"], match_ids=["0102", "101", 102])
    assert retry["already_queued"] == 1 and retry["queued"] == 0
    assert len(db.tables["player_profile_update_outbox"]) == 2
    result = scheduler.deliver_pending_updates()
    assert result["sent"] == 2 and result["errors"] == 0
    assert {tuple(c["match_ids"]) for c in computed} == {("101", "102"), ("201",)}
    assert len(sent) == 2 and all("test-unsubscribe" in x["unsubscribe_url"] for x in sent)
    assert first["send_status"] == second["send_status"] == "sent"
    assert first["attempt_count"] == second["attempt_count"] == 1
    assert scheduler.deliver_pending_updates()["sent"] == 0
    assert len(sent) == 2
    assert all(r["status"] == "success" for r in db.tables["worker_run_log"])
    assert db.closed >= 4


def test_only_automatic_pending_sessions_are_selected(delivery):
    db, sent, _ = delivery
    eligible = queue(db)
    for index, change in enumerate([
        {"queue_operation_key": "manual-preview"},
        {"digest_snapshot_json": {}},
        {"send_status": "sending"},
        {"send_status": "error"},
        {"digest_snapshot_json": {"update_scope": "session", "match_ids": [True]}},
    ], 10):
        db.tables["player_profile_update_outbox"].append({**deepcopy(eligible), "id": str(UUID(int=index)), **change})
    result = scheduler.deliver_pending_updates()
    assert result["sent"] == len(sent) == 1
    assert result["ignored"] == 2
    assert [r["send_status"] for r in db.tables["player_profile_update_outbox"]] == [
        "sent", "pending", "pending", "sending", "error", "pending"]


def test_selected_old_row_is_not_hidden_by_newer_manual_rows(delivery):
    db, sent, _ = delivery
    selected = queue(db)
    for index in range(50):
        db.tables["player_profile_update_outbox"].append({**deepcopy(selected),
            "id": str(UUID(int=index + 10)), "created_at": f"newer-{index}", "queue_operation_key": "manual"})
    result = scheduler.deliver_pending_updates()
    assert result["sent"] == len(sent) == 1
    assert selected["send_status"] == "sent"


def test_unsubscribe_after_publication_is_respected(delivery):
    db, sent, _ = delivery
    row = queue(db)
    db.tables["player_profile_update_subscriptions"][0]["request_status"] = "unsubscribed"
    result = scheduler.deliver_pending_updates()
    assert result["skipped"] == 1 and not sent
    assert row["send_status"] == "skipped"


def test_uncertain_smtp_result_is_never_automatically_resent(delivery, monkeypatch):
    db, sent, _ = delivery
    row = queue(db)
    def uncertain(**kwargs):
        sent.append(kwargs)
        raise TimeoutError("delivery outcome unknown")
    monkeypatch.setattr(sender, "send_email_with_inline_chart", uncertain)
    result = scheduler.deliver_pending_updates()
    assert result["uncertain"] == 1 and row["send_status"] == "sending"
    assert db.tables["worker_run_log"][-1]["status"] == "failed"
    scheduler.deliver_pending_updates()
    assert len(sent) == 1


def test_waiting_first_batch_does_not_starve_later_ready_sessions(delivery, monkeypatch):
    db, _, _ = delivery
    for player in range(1, 53): queue(db, player=player)
    selected = []
    def waiting(_club, **kwargs):
        selected.extend(i["id"] for i in kwargs["outbox_items"])
        return {"waiting_for_awards": len(kwargs["outbox_items"])}
    monkeypatch.setattr(scheduler, "run_player_update_email_worker", waiting)
    cursor = None
    for _ in range(3):
        result = scheduler.deliver_pending_updates(scan_after=cursor)
        cursor = result["next_cursor"]
    assert len(selected) == len(set(selected)) == 52
    assert cursor is None


def test_manual_session_previews_cannot_fill_the_scan_forever(delivery, monkeypatch):
    db, sent, _ = delivery
    row = queue(db)
    rows = db.tables["player_profile_update_outbox"]
    for i in range(1, 251):
        rows.append({**deepcopy(row), "id": str(UUID(int=i)), "queue_operation_key": "manual"})
    row["id"] = str(UUID(int=300))
    first = scheduler.deliver_pending_updates()
    assert first["sent"] == 0 and first["next_cursor"]
    second = scheduler.deliver_pending_updates(scan_after=first["next_cursor"])
    assert second["sent"] == len(sent) == 1 and second["next_cursor"] is None


@pytest.mark.parametrize("key,value", [
    ("JUPR_ENV", "staging"), ("FLY_APP_NAME", "juprleagues-api-staging"),
    ("SUPABASE_URL", "https://sijpxjxvdtrehmqvirfi.supabase.co"),
    ("JUPR_PRODUCTION_WRITE_POLICY", "read_only"), ("JUPR_EMAIL_MODE", "dry_run"),
    ("JUPR_STAGING_WRITE_WAVE", "open"), ("SUPABASE_SERVICE_ROLE_KEY", ""),
    ("JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS", "0"),
    ("JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL", "0"), ("JUPR_REQUIRE_WORKER_RUN_LOG", "0"),
])
def test_disabled_or_nonproduction_scheduler_never_accesses_database(monkeypatch, key, value):
    enable(monkeypatch)
    assert scheduler.scheduler_enabled()
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(scheduler, "make_supabase", lambda *_: pytest.fail("no DB access allowed"))
    assert not scheduler.scheduler_enabled()
    assert scheduler.deliver_pending_updates()["examined"] == 0


def test_scheduler_records_heartbeat_without_exposing_cursor(monkeypatch):
    scheduler._delivery_status.clear()
    def poll(**kwargs):
        assert kwargs["scan_after"] is None
        assert not kwargs["should_stop"]()
        return {"next_cursor": "private-outbox-id", "examined": 1, "sent": 1, "errors": 0}
    async def stop(_): raise asyncio.CancelledError
    monkeypatch.setattr(scheduler, "deliver_pending_updates", poll)
    monkeypatch.setattr(scheduler.asyncio, "sleep", stop)
    with pytest.raises(asyncio.CancelledError): asyncio.run(scheduler.run_scheduler())
    status = scheduler.delivery_status()
    assert status["last_completed_at"] and status["last_sent_at"]
    assert "private-outbox-id" not in str(status)


def test_automatic_worker_advances_awards_then_waits_for_next_poll(delivery, monkeypatch):
    db, _, _ = delivery
    calls = []
    monkeypatch.setattr(worker, "send_pending_player_update_emails", lambda *_a, **_kw: {"waiting_for_awards": 2})
    monkeypatch.setattr("jupr_app.domain.gamification.badge_worker.process_badge_eval_queue",
        lambda *a, **kw: calls.append((a, kw)) or {"processed": 1, "errored": 0})
    result = worker.run_player_update_email_worker("club", outbox_items=[], advance_waiting_awards=True)
    assert result["sent"] == 0 and result["waiting_for_awards"] == 2
    assert result["badge_jobs_processed"] == 1
    assert calls == [((db, "club"), {"max_jobs": 1, "time_budget_seconds": 2})]


def test_missing_required_run_marker_prevents_delivery(delivery, monkeypatch):
    db, sent, _ = delivery
    original = Query.execute
    def execute(self):
        if self.name == "worker_run_log" and self.insert_payload is not None:
            return SimpleNamespace(data=[{}])
        return original(self)
    monkeypatch.setattr(Query, "execute", execute)
    queue(db)
    result = scheduler.deliver_pending_updates()
    assert result["errors"] == 1 and not sent
    assert db.closed == 2


def test_shutdown_finishes_claimed_email_and_leaves_next_row_pending(delivery, monkeypatch):
    from threading import Event
    db, sent, _ = delivery
    first = queue(db)
    second = queue(db, player=2)
    stop_event = Event()
    compute = sender.compute_player_weekly_digest
    def stop_during_render(*args, **kwargs):
        stop_event.set()
        return compute(*args, **kwargs)
    monkeypatch.setattr(sender, "compute_player_weekly_digest", stop_during_render)
    result = scheduler.deliver_pending_updates(should_stop=stop_event.is_set)
    assert result["sent"] == result["attempted"] == len(sent) == 1
    assert {first["send_status"], second["send_status"]} == {"sent", "pending"}
    assert sum(row.get("attempt_count", 0) for row in (first, second)) == 1
    monkeypatch.setattr(sender, "compute_player_weekly_digest", compute)
    assert scheduler.deliver_pending_updates()["sent"] == 1
    assert first["attempt_count"] == second["attempt_count"] == 1


def test_api_shutdown_waits_for_inflight_smtp_and_worker_receipt(delivery, monkeypatch):
    from threading import Event
    from services.api import main
    db, _, _ = delivery
    first = queue(db)
    second = queue(db, player=2)
    entered, release = Event(), Event()
    def smtp(**kwargs):
        entered.set()
        assert release.wait(5)
        return "<drained-message>"
    monkeypatch.setattr(sender, "send_email_with_inline_chart", smtp)
    monkeypatch.setattr(main, "_log_runtime_guardrails", lambda: None)
    monkeypatch.setattr(main, "scheduler_enabled", lambda: True)
    async def exercise():
        await main.startup_checks()
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            shutdown = asyncio.create_task(main.stop_player_email_worker())
            await asyncio.sleep(0)
            assert main.app.state.player_email_stop.is_set()
            assert not shutdown.done()
        finally:
            release.set()
        await asyncio.wait_for(shutdown, 3)
        assert not main.app.state.player_email_task.cancelled()
    asyncio.run(exercise())
    assert {first["send_status"], second["send_status"]} == {"sent", "pending"}
    assert sum(row.get("attempt_count", 0) for row in (first, second)) == 1
    assert db.tables["worker_run_log"][0]["status"] == "success"


def test_already_stopped_scheduler_does_not_access_queue(monkeypatch):
    from threading import Event
    stop = Event()
    stop.set()
    monkeypatch.setattr(scheduler, "deliver_pending_updates", lambda **kw: pytest.fail("already stopped"))
    asyncio.run(scheduler.run_scheduler(stop_event=stop))


def test_shutdown_during_award_read_does_not_claim_next_email(delivery, monkeypatch):
    from threading import Event
    db, sent, _ = delivery
    row = queue(db)
    stop = Event()
    def awards_ready(*args):
        stop.set()
        return True
    monkeypatch.setattr(sender, "_session_awards_ready", awards_ready)
    result = scheduler.deliver_pending_updates(should_stop=stop.is_set)
    assert result["attempted"] == 0 and not sent
    assert row["send_status"] == "pending" and not row.get("attempt_count")
