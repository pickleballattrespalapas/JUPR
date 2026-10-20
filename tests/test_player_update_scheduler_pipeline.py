"""Exercise the real queue, poll, worker, and CAS sender without external I/O."""
from __future__ import annotations

from copy import deepcopy
from datetime import date
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest

from jupr_app.domain.notifications import player_update_sender as sender
from jupr_app.domain.notifications.player_profile_update_repo import (
    create_outbox_row,
    queue_player_updates_for_affected_subscribers,
    session_player_update_operation_key,
)
from jupr_app.workers import player_update_email_scheduler as scheduler
from jupr_app.workers import player_update_email_worker as worker
from tests.test_player_update_queueing import _Query
from tests.test_production_email_activation import _enable


class _Database:
    """Return detached rows and model the database's row-version trigger."""

    def __init__(self):
        self.tables = {
            "player_profile_update_subscriptions": [{
                "id": "sub-7", "club_id": "club", "player_id": 7,
                "email": "player@example.test", "request_status": "active",
                "unsubscribe_token": "test-token", "verified_at": "2026-10-10",
            }],
            "player_profile_update_outbox": [],
            "badge_eval_queue": [],
            "worker_run_log": [],
        }
        self.queries = []
        self.before_execute = None

    def table(self, name):
        assert name in self.tables, f"Unexpected table (including date-window cache): {name}"
        return _Query(self, name)

    def execute(self, query):
        self.queries.append(query)
        if self.before_execute:
            self.before_execute(query)
        rows = self.tables[query.table]
        if query._op == "insert":
            row = deepcopy(query._payload)
            row.setdefault("id", f"{query.table}-{len(rows) + 1}")
            row.setdefault("created_at", f"2026-10-10T12:00:{len(rows):02d}Z")
            row.setdefault("row_version", 1)
            row.setdefault("attempt_count", 0)
            rows.append(row)
            return SimpleNamespace(data=[deepcopy(row)])

        def field(row, name):
            if "->>" in name:
                column, key = name.split("->>")
                value = row.get(column)
                return value.get(key) if isinstance(value, dict) else None
            return row.get(name)

        selected = list(rows)
        for operation, column, value in query._filters:
            if operation == "eq":
                selected = [row for row in selected if str(field(row, column)) == str(value)]
            elif operation == "in":
                selected = [row for row in selected if str(field(row, column)) in {str(v) for v in value}]
            else:
                raise AssertionError(operation)
        if query._order:
            column, descending = query._order
            selected.sort(key=lambda row: str(row.get(column) or ""), reverse=descending)
        if query._range:
            first, last = query._range
            selected = selected[first:last + 1]
        if query._limit is not None:
            selected = selected[:query._limit]
        if query._op == "update":
            for row in selected:
                row.update(deepcopy(query._payload))
                row["row_version"] += 1
        else:
            assert query._op == "select"
        return SimpleNamespace(data=deepcopy(selected))


@pytest.fixture
def pipeline(monkeypatch):
    # These flags exercise the real scheduler guard; all DB and SMTP factories
    # below are replaced before any production function is called.
    _enable(monkeypatch)
    db = _Database()
    calls = SimpleNamespace(computes=[], smtp=[])
    monkeypatch.setattr(scheduler, "make_supabase", lambda *_args: db)
    monkeypatch.setattr(worker, "make_supabase", lambda *_args: db)

    def compute(_ctx, **kwargs):
        calls.computes.append(deepcopy(kwargs))
        return {"summary": {"matches_played": len(kwargs["match_ids"])},
                "matches_played_rows": [{"id": mid} for mid in kwargs["match_ids"]]}

    monkeypatch.setattr(sender, "compute_player_weekly_digest", compute)
    monkeypatch.setattr(sender, "render_player_digest_chart_png", lambda _digest: None)
    monkeypatch.setattr(sender, "build_player_update_email_subject", lambda _digest: "Session update")
    monkeypatch.setattr(sender, "build_player_update_email_html", lambda *_args: "Session update")
    monkeypatch.setattr(sender, "build_player_update_email_text",
                        lambda digest: ",".join(row["id"] for row in digest["matches_played_rows"]))

    def smtp(**kwargs):
        # The actual sender must hold a durable claim before contacting SMTP.
        claimed = [row for row in db.tables["player_profile_update_outbox"]
                   if row.get("delivery_attempt_id") == kwargs["message_id"]]
        assert len(claimed) == 1 and claimed[0]["send_status"] == "sending"
        assert claimed[0]["row_version"] == 2
        calls.smtp.append(deepcopy(kwargs))
        return "mock-provider-message"

    monkeypatch.setattr(sender, "send_email_with_inline_chart", smtp)
    return SimpleNamespace(db=db, calls=calls)


def _queue(case, *, match_ids=None, badge_refs=None):
    return queue_player_updates_for_affected_subscribers(
        case.db, club_id="club", affected_player_ids=[7],
        match_dates=["2026-10-10"], match_ids=match_ids or [102, "0101", 102],
        session_evidence={"player_updates": [{"id": 7, "rating_before": 3.0, "rating_after": 3.04}]},
        badge_refs=badge_refs,
    )


def test_operation_key_preserves_existing_session_keys():
    expected = str(uuid5(NAMESPACE_URL, 'jupr:session-player-update:club:["101","102"]'))
    assert session_player_update_operation_key(club_id="club", match_ids=[102, "0101", 102]) == expected


def test_scheduler_json_filter_uses_the_client_without_network(monkeypatch):
    import httpx
    from supabase import ClientOptions, create_client

    _enable(monkeypatch)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=[])

    transport = httpx.Client(transport=httpx.MockTransport(respond))
    client = create_client("https://example.test", "test-key", options=ClientOptions(httpx_client=transport))
    monkeypatch.setattr(scheduler, "make_supabase", lambda *_args: client)
    assert scheduler.deliver_pending_updates() == {"clubs": 0, "sent": 0, "errors": 0}
    assert len(requests) == 1 and requests[0].url.host == "example.test"
    assert requests[0].url.params["digest_snapshot_json->>update_scope"] == "eq.session"
    assert requests[0].url.params["send_status"] == "eq.pending"
    assert transport.is_closed


def test_actual_producer_reaches_sender_once_with_exact_saved_games(pipeline):
    case = pipeline
    assert _queue(case)["queued"] == 1
    original = deepcopy(case.db.tables["player_profile_update_outbox"][0])
    assert _queue(case)["already_queued"] == 1

    # A second session on the same day remains a distinct update.
    assert _queue(case, match_ids=[103])["queued"] == 1
    # Admin previews and old rows must stay pending, regardless of their dates.
    for key, snapshot in [("manual-preview", {"summary": {"matches_played": 99}}),
                          ("legacy", {})]:
        create_outbox_row(case.db, subscription_id="sub-7", club_id="club", player_id=7,
            week_start=date(2026, 10, 10), week_end=date(2026, 10, 10),
            email="player@example.test", operation_key=key, digest_snapshot=snapshot)

    assert scheduler.deliver_pending_updates() == {"clubs": 1, "sent": 2, "errors": 0}
    assert {tuple(call["match_ids"]) for call in case.calls.computes} == {("101", "102"), ("103",)}
    assert all(call["session_evidence"] == original["digest_snapshot_json"]["session_evidence"]
               for call in case.calls.computes)
    assert {call["text_body"] for call in case.calls.smtp} == {"101,102", "103"}
    assert all(call["to_email"] == "player@example.test" for call in case.calls.smtp)
    rows = case.db.tables["player_profile_update_outbox"]
    assert [row["send_status"] for row in rows] == ["sent", "sent", "pending", "pending"]
    assert rows[0]["digest_snapshot_json"] == original["digest_snapshot_json"]
    assert rows[0]["row_version"] == 3 and rows[0]["attempt_count"] == 1
    assert scheduler.deliver_pending_updates() == {"clubs": 0, "sent": 0, "errors": 0}
    assert len(case.calls.smtp) == 2
    assert case.db.tables["worker_run_log"][0]["summary_json"]["sent"] == 2


@pytest.mark.parametrize("snapshot,key", [
    ({"update_scope": "session", "match_ids": [101]}, "manual-preview"),
    ({"update_scope": "session", "match_ids": [101]}, "wrong-club-key"),
    ({"update_scope": "session", "match_ids": []}, "invalid"),
    ({"update_scope": "session", "match_ids": [True]}, "invalid"),
    ({"update_scope": "session"}, "invalid"),
    ({}, "legacy"),
])
def test_invalid_or_manually_queued_scopes_are_never_claimed(pipeline, snapshot, key):
    _queue(pipeline)
    row = pipeline.db.tables["player_profile_update_outbox"][0]
    row["digest_snapshot_json"] = snapshot
    row["queue_operation_key"] = (
        session_player_update_operation_key(club_id="other-club", match_ids=[101])
        if key == "wrong-club-key" else key
    )
    assert scheduler.deliver_pending_updates() == {"clubs": 0, "sent": 0, "errors": 0}
    assert row["send_status"] == "pending" and row["attempt_count"] == 0
    assert pipeline.calls.smtp == pipeline.calls.computes == []


def test_saved_session_waits_for_its_badges_before_claiming(pipeline):
    _queue(pipeline, badge_refs=["101", "102"])
    jobs = pipeline.db.tables["badge_eval_queue"]
    jobs.extend({"id": ref, "club_id": "club", "event_type": "match_recorded",
                 "match_id": ref, "status": "pending", "attempts": 0, "row_version": 1}
                for ref in ["101", "102"])
    assert scheduler.deliver_pending_updates() == {"clubs": 1, "sent": 0, "errors": 0}
    row = pipeline.db.tables["player_profile_update_outbox"][0]
    assert row["send_status"] == "pending" and row["attempt_count"] == 0
    assert pipeline.calls.smtp == pipeline.calls.computes == []
    assert pipeline.db.tables["worker_run_log"][0]["summary_json"]["waiting_for_awards"] == 1
    for job in jobs:
        job["status"] = "done"
    assert scheduler.deliver_pending_updates() == {"clubs": 1, "sent": 1, "errors": 0}
    assert len(pipeline.calls.smtp) == 1


def test_competing_claim_between_read_and_update_prevents_delivery(pipeline):
    _queue(pipeline)

    def competing_worker(query):
        if query.table == "player_profile_update_outbox" and query._op == "update":
            pipeline.db.before_execute = None
            row = pipeline.db.tables["player_profile_update_outbox"][0]
            row.update(send_status="sending", row_version=2, attempt_count=1)

    pipeline.db.before_execute = competing_worker
    assert scheduler.deliver_pending_updates() == {"clubs": 1, "sent": 0, "errors": 0}
    assert pipeline.calls.smtp == pipeline.calls.computes == []
    assert pipeline.db.tables["worker_run_log"][0]["summary_json"]["stale"] == 1
    assert scheduler.deliver_pending_updates() == {"clubs": 0, "sent": 0, "errors": 0}


def test_changed_version_after_poll_requires_a_fresh_selection(pipeline):
    _queue(pipeline)

    def edited_after_poll(query):
        if query.table == "worker_run_log" and query._op == "insert":
            pipeline.db.before_execute = None
            pipeline.db.tables["player_profile_update_outbox"][0]["row_version"] += 1

    pipeline.db.before_execute = edited_after_poll
    assert scheduler.deliver_pending_updates() == {"clubs": 1, "sent": 0, "errors": 0}
    row = pipeline.db.tables["player_profile_update_outbox"][0]
    assert row["send_status"] == "pending" and row["attempt_count"] == 0
    assert pipeline.calls.smtp == pipeline.calls.computes == []
    assert pipeline.db.tables["worker_run_log"][0]["summary_json"]["stale"] == 1


def test_dry_run_blocks_scheduler_and_worker_never_calls_smtp(pipeline, monkeypatch):
    _queue(pipeline)
    row = pipeline.db.tables["player_profile_update_outbox"][0]
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
    pipeline.db.queries.clear()
    assert scheduler.deliver_pending_updates() == {"clubs": 0, "sent": 0, "errors": 0}
    assert pipeline.db.queries == []
    # The explicit dry-run worker still exercises the same saved session and CAS.
    result = worker.run_player_update_email_worker("club", outbox_items=[{
        "id": row["id"], "expected_row_version": row["row_version"],
    }])
    assert result["sent"] == 1 and result["email_mode"] == "dry_run"
    assert row["provider_message_id"] == "dry_run"
    assert pipeline.calls.smtp == []
