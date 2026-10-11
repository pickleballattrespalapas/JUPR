from __future__ import annotations

from copy import deepcopy
from datetime import date
from types import SimpleNamespace

import pytest

from jupr_app.domain.notifications import player_update_sender as sender


class _BadgeQuery:
    def __init__(self, db):
        self.db = db
        self.filters = []
        self.window = None
        self.payload = None

    def select(self, columns):
        self.columns = columns
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column, values):
        self.filters.append(("in", column, list(values)))
        return self

    def update(self, payload):
        self.payload = deepcopy(payload)
        return self

    def order(self, column):
        assert column == "id"
        return self

    def range(self, start, end):
        self.window = (start, end)
        return self

    def execute(self):
        self.db.queries.append(self)
        if self.db.read_error:
            raise RuntimeError("Badge queue read unavailable")
        if self.payload is not None and self.db.before_update is not None:
            self.db.before_update(self.db)
        rows = list(self.db.rows)
        for op, column, value in self.filters:
            if op == "eq":
                rows = [row for row in rows if row.get(column) == value]
            else:
                rows = [row for row in rows if row.get(column) in value]
        rows.sort(key=lambda row: row["id"])
        if self.window is not None:
            rows = rows[self.window[0] : self.window[1] + 1]
        if self.payload is not None:
            for row in rows:
                row.update(deepcopy(self.payload))
        return SimpleNamespace(data=deepcopy(rows))


class _BadgeDatabase:
    def __init__(self, rows=(), *, read_error=False):
        self.rows = list(rows)
        self.queries = []
        self.read_error = read_error
        self.before_update = None

    def table(self, name):
        # The sender's best-effort subscription watermark is unrelated to
        # readiness. Its failure must not change successful outbox delivery.
        if name == "player_profile_update_subscriptions":
            raise RuntimeError("No subscription writes in this sender fixture")
        assert name == "badge_eval_queue"
        return _BadgeQuery(self)


def _job(ref, status="done", *, club="club-1", event="match_recorded", job_id=None, attempts=1):
    return {
        "id": job_id or f"job-{ref}",
        "club_id": club,
        "event_type": event,
        "match_id": str(ref),
        "status": status,
        "attempts": attempts,
    }


def _session_snapshot():
    return {
        "update_scope": "session",
        "match_ids": ["101", "102"],
        "session_evidence": {
            "player_updates": [{"id": 7, "rating_before": 3.0, "rating_after": 3.04}]
        },
        "badge_refs": ["101", "102"],
    }


@pytest.fixture
def sender_case(monkeypatch):
    monkeypatch.setattr(sender, "get_email_mode", lambda: "dry_run")
    outbox = {
        "id": "outbox-1",
        "club_id": "club-1",
        "subscription_id": "subscription-1",
        "player_id": 7,
        "email": "verified@example.test",
        "week_start": "2026-10-10",
        "week_end": "2026-10-10",
        "send_status": "pending",
        "row_version": 1,
        "digest_snapshot_json": _session_snapshot(),
    }
    digest = {
        "update_scope": "session",
        "player_name": "Ada",
        "summary": {"matches_played": 2, "overall_delta": 0.04},
        "matches_played_rows": [{"id": "101"}, {"id": "102"}],
        "badges_earned": [{"badge_id": "session-badge"}],
    }
    calls = SimpleNamespace(claims=[], statuses=[], computes=[], renders=[], smtp=[])

    def claim(_db, **kwargs):
        calls.claims.append(kwargs)
        outbox.update(send_status="sending", row_version=2)
        return {**outbox, "delivery_attempt_id": "attempt-1"}

    def update(_db, outbox_id, **kwargs):
        assert outbox_id == outbox["id"]
        calls.statuses.append(kwargs)
        outbox["send_status"] = kwargs["send_status"]

    def compute(ctx, **kwargs):
        calls.computes.append(kwargs)
        return deepcopy(digest)

    def render(payload, *_args, **_kwargs):
        calls.renders.append(deepcopy(payload))
        return "Rendered session"

    def forbidden(*_args, **_kwargs):
        pytest.fail("Session send must not read or write the date-window digest cache")

    def smtp(**kwargs):
        calls.smtp.append(kwargs)
        pytest.fail("dry_run must never call SMTP")

    monkeypatch.setattr(sender, "list_outbox_rows", lambda *_args, **_kwargs: [outbox])
    monkeypatch.setattr(sender, "claim_outbox_row_for_send", claim)
    monkeypatch.setattr(sender, "update_outbox_status", update)
    monkeypatch.setattr(sender, "compute_player_weekly_digest", compute)
    monkeypatch.setattr(sender, "_safe_digest_for_week", forbidden)
    monkeypatch.setattr(sender, "save_digest", forbidden)
    monkeypatch.setattr(sender, "_safe_subscription", lambda *_args: {
        "id": "subscription-1", "request_status": "active",
        "preferences_json": {"send_only_if_changed": False},
    })
    monkeypatch.setattr(sender, "ensure_unsubscribe_token", lambda *_args: "test-unsubscribe-token")
    monkeypatch.setattr(sender, "render_player_digest_chart_png", render)
    monkeypatch.setattr(sender, "build_player_update_email_subject", render)
    monkeypatch.setattr(sender, "build_player_update_email_html", render)
    monkeypatch.setattr(sender, "build_player_update_email_text", render)
    monkeypatch.setattr(sender, "send_email_with_inline_chart", smtp)
    return SimpleNamespace(outbox=outbox, digest=digest, calls=calls)


def _send(case, db):
    return sender.send_pending_player_update_emails(
        SimpleNamespace(supabase=db, club_id="club-1"),
        public_base_url="https://example.test",
    )


def test_session_send_uses_exact_games_and_evidence_without_weekly_cache(sender_case):
    case = sender_case
    db = _BadgeDatabase([_job("101"), _job("102"), _job("unrelated", "pending")])
    evidence = deepcopy(case.outbox["digest_snapshot_json"]["session_evidence"])

    result = _send(case, db)

    assert result["sent"] == 1
    assert result["errors"] == result["waiting_for_awards"] == 0
    assert case.calls.computes == [{
        "player_id": 7, "start_date": date(2026, 10, 10), "end_date": date(2026, 10, 10),
        "match_ids": ["101", "102"], "session_evidence": evidence,
    }]
    assert len(case.calls.claims) == 1
    assert case.calls.statuses[0]["send_status"] == "sent"
    assert case.calls.statuses[0]["provider_message_id"] == "dry_run"
    assert case.calls.smtp == []
    assert len(case.calls.renders) == 4
    for rendered in case.calls.renders:
        assert {key: value for key, value in rendered.items() if key != "links"} == case.digest
        assert rendered["links"]["unsubscribe"].endswith("?token=test-unsubscribe-token")
    assert db.queries
    for query in db.queries:
        assert ("eq", "club_id", "club-1") in query.filters
        assert ("eq", "event_type", "match_recorded") in query.filters
        assert ("in", "match_id", ["101", "102"]) in query.filters


@pytest.mark.parametrize("status", ["pending", "processing", "error", "missing", "read_error"])
def test_session_waits_for_every_exact_badge_job_before_claim(sender_case, status):
    case = sender_case
    rows = [
        _job("101"),
        _job("102", club="another-club", job_id="other-club"),
        _job("102", event="season_completed", job_id="other-event"),
        _job("unrelated"),
    ]
    if status not in {"missing", "read_error"}:
        rows.append(_job("102", status))
    db = _BadgeDatabase(rows, read_error=status == "read_error")

    result = _send(case, db)

    assert result["waiting_for_awards"] == 1
    assert result["sent"] == result["errors"] == 0
    assert case.outbox["send_status"] == "pending"
    assert case.outbox["row_version"] == 1
    assert case.calls.claims == case.calls.statuses == case.calls.computes == []
    assert case.calls.renders == case.calls.smtp == []


@pytest.mark.parametrize("attempts", [1, 2])
def test_late_badge_failure_requeues_only_exact_session_job_before_email_claim(sender_case, attempts):
    case = sender_case
    case.outbox["digest_snapshot_json"]["badge_refs"] = ["101", "102", "103"]
    failed_job = {
        **_job("102", "error", attempts=attempts),
        "last_error": "Badge worker timed out after approval",
        "processed_at": "2026-10-10T19:00:00Z",
    }
    preserved = [
        _job("101"),
        _job("103", "processing"),
        _job("102", "error", club="another-club", job_id="foreign-club"),
        _job("102", "error", event="season_completed", job_id="foreign-event"),
        _job("outside-session", "error"),
    ]
    db = _BadgeDatabase([failed_job, *deepcopy(preserved)])

    result = _send(case, db)

    assert result["waiting_for_awards"] == 1
    assert result["sent"] == result["errors"] == 0
    assert case.outbox["send_status"] == "pending"
    assert case.calls.claims == case.calls.statuses == case.calls.computes == []
    assert case.calls.renders == case.calls.smtp == []
    assert failed_job == {
        **_job("102", "pending", attempts=attempts), "last_error": None, "processed_at": None,
    }
    assert db.rows[1:] == preserved
    updates = [query for query in db.queries if query.payload is not None]
    assert len(updates) == 1
    assert updates[0].payload == {"status": "pending", "last_error": None, "processed_at": None}
    assert set(updates[0].filters) == {
        ("eq", "club_id", "club-1"), ("eq", "id", "job-102"),
        ("eq", "event_type", "match_recorded"), ("eq", "match_id", "102"),
        ("eq", "status", "error"), ("eq", "attempts", attempts),
    }


def test_badge_recovery_cas_does_not_reset_job_changed_since_read(sender_case):
    case = sender_case
    db = _BadgeDatabase([_job("101"), _job("102", "error", attempts=1)])

    def concurrently_started(database):
        database.rows[1].update(status="processing", attempts=2)

    db.before_update = concurrently_started

    result = _send(case, db)

    assert result["waiting_for_awards"] == 1
    assert db.rows[1]["status"] == "processing"
    assert db.rows[1]["attempts"] == 2
    assert case.outbox["send_status"] == "pending"
    assert case.calls.claims == case.calls.computes == case.calls.renders == case.calls.smtp == []


@pytest.mark.parametrize("attempts", [3, 4])
def test_exhausted_badge_recovery_claims_outbox_as_actionable_error_without_sending(sender_case, attempts):
    case = sender_case
    rows = [_job("101"), _job("102", "error", attempts=attempts)]
    db = _BadgeDatabase(deepcopy(rows))

    result = _send(case, db)

    assert result["errors"] == 1
    assert result["sent"] == result["waiting_for_awards"] == 0
    assert case.outbox["send_status"] == "error"
    assert len(case.calls.claims) == len(case.calls.statuses) == 1
    recorded_error = case.calls.statuses[0]
    assert recorded_error["send_status"] == "error"
    assert recorded_error["expected_status"] == "sending"
    assert recorded_error["expected_row_version"] == 2
    assert "badge" in recorded_error["error_text"].lower()
    assert db.rows == rows
    assert not any(query.payload is not None for query in db.queries)
    assert case.calls.computes == case.calls.renders == case.calls.smtp == []


def test_session_uses_saved_badge_refs_even_when_they_are_batch_references(sender_case):
    case = sender_case
    case.outbox["digest_snapshot_json"]["badge_refs"] = ["batch-approval-1"]
    db = _BadgeDatabase([_job("batch-approval-1"), _job("101", "pending")])

    result = _send(case, db)

    assert result["sent"] == 1
    assert case.calls.computes[0]["match_ids"] == ["101", "102"]
    assert ("in", "match_id", ["batch-approval-1"]) in db.queries[0].filters


@pytest.mark.parametrize("match_ids", [None, [], "101", [True], [{}], [""]])
def test_invalid_session_game_scope_fails_closed(sender_case, match_ids):
    case = sender_case
    case.outbox["digest_snapshot_json"]["match_ids"] = match_ids

    result = _send(case, _BadgeDatabase([_job("101"), _job("102")]))

    assert result["sent"] == 0
    assert result["errors"] == 1
    assert case.calls.computes == case.calls.renders == case.calls.smtp == []


@pytest.mark.parametrize("snapshot", [None, {}])
def test_old_automatic_row_without_snapshot_fails_closed(sender_case, snapshot):
    case = sender_case
    case.outbox["digest_snapshot_json"] = snapshot

    result = _send(case, _BadgeDatabase())

    assert result["sent"] == 0
    assert result["errors"] == 1
    assert case.calls.computes == case.calls.renders == case.calls.smtp == []


def test_empty_computed_session_digest_does_not_fall_back_to_date_window(sender_case):
    case = sender_case
    case.digest.clear()

    result = _send(case, _BadgeDatabase([_job("101"), _job("102")]))

    assert result["sent"] == 0
    assert result["errors"] == 1
    assert len(case.calls.computes) == 1
    assert case.calls.renders == case.calls.smtp == []


def test_manual_full_snapshot_is_rendered_without_session_computation_or_badge_gate(sender_case):
    case = sender_case
    manual_digest = {
        "player_name": "Ada", "summary": {"matches_played": 6},
        "matches_played_rows": [{"id": "historical-match"}],
    }
    case.outbox["digest_snapshot_json"] = deepcopy(manual_digest)
    db = _BadgeDatabase(read_error=True)

    result = _send(case, db)

    assert result["sent"] == 1
    assert result["waiting_for_awards"] == 0
    assert db.queries == case.calls.computes == case.calls.smtp == []
    assert len(case.calls.renders) == 4
    assert all({k: v for k, v in rendered.items() if k != "links"} == manual_digest
               for rendered in case.calls.renders)
