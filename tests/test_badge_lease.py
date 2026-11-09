from types import SimpleNamespace
import asyncio

import pytest

from jupr_app.domain.gamification.badge_lease import BadgeJobLease, BadgeLeaseLost, isolated_worker_context
from jupr_app.domain.gamification.badge_queue import ack_badge_eval, dequeue_badge_eval
from jupr_app.domain.gamification import badge_worker
from jupr_app.workers.isolated_badge_worker import validate_worker_target


class LeaseDB:
    def __init__(self, current=True):
        self.current = current
        self.calls = []

    def rpc(self, name, args):
        self.calls.append((name, args))
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=self.current))


def job():
    return {"id": "job", "club_id": "club", "claim_token": "lease"}


def test_lease_loss_prevents_completion_and_does_not_blindly_retry():
    db = LeaseDB()
    lease = BadgeJobLease(db, job())
    lease.renew()
    db.current = False
    with pytest.raises(BadgeLeaseLost):
        lease.renew()
    db.current = True
    with pytest.raises(BadgeLeaseLost):
        lease.renew()
    assert len(db.calls) == 2  # No renewal once ownership became uncertain.


def test_fenced_ack_rejects_stale_owner():
    db = LeaseDB(False)
    with pytest.raises(BadgeLeaseLost):
        ack_badge_eval(db, job_id="job", status="done", claim_token="stale")
    assert db.calls[0][0] == "finish_badge_eval_queue_job"
    assert db.calls[0][1]["p_claim_token"] == "stale"


def test_lease_enabled_worker_fails_closed_on_legacy_claim(monkeypatch):
    monkeypatch.setenv("JUPR_BADGE_QUEUE_LEASES", "1")
    with pytest.raises(BadgeLeaseLost):
        with BadgeJobLease(LeaseDB(), {"id": "job", "club_id": "club"}):
            pytest.fail("Unfenced job executed")


def test_worker_does_not_ack_or_process_unowned_job(monkeypatch):
    monkeypatch.setenv("JUPR_BADGE_QUEUE_LEASES", "1")
    monkeypatch.setattr(badge_worker, "dequeue_badge_eval", lambda *a, **k: job())
    monkeypatch.setattr(badge_worker, "_process_job_with_retry", lambda *a, **k: pytest.fail("Job executed after lease lost"))
    monkeypatch.setattr(badge_worker, "ack_badge_eval", lambda *a, **k: pytest.fail("Unowned job acknowledged"))
    assert badge_worker.process_badge_eval_queue(LeaseDB(False), "club", max_jobs=1) == {"processed": 0, "errored": 1}


def test_heartbeat_thread_stops_on_completion(monkeypatch):
    monkeypatch.setenv("JUPR_BADGE_QUEUE_LEASES", "1")
    db, data = LeaseDB(), job()
    with BadgeJobLease(db, data) as lease:
        assert data["_assert_owned"]
        assert lease._thread.is_alive()
    assert not lease._thread.is_alive()
    assert "_assert_owned" not in data


def test_external_worker_requires_explicit_activation_and_exact_target(monkeypatch):
    monkeypatch.setenv("JUPR_BADGE_WORKER_CLUB_ID", "tres_palapas")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("FLY_APP_NAME", "juprleagues-api-staging")
    monkeypatch.setenv("JUPR_BADGE_WORKER_MODE", "external")
    monkeypatch.setenv("JUPR_BADGE_QUEUE_LEASES", "1")
    monkeypatch.setenv("SUPABASE_URL", "https://sijpxjxvdtrehmqvirfi.supabase.co")
    validate_worker_target("tres_palapas")
    monkeypatch.setenv("FLY_APP_NAME", "juprleagues-api")
    with pytest.raises(ValueError):
        validate_worker_target("tres_palapas")


@pytest.mark.parametrize("origin", ["https://dnoockbwfenunhcibwfn.supabase.co", "https://example.supabase.co",
                                   "https://sijpxjxvdtrehmqvirfi.supabase.co/other", "http://sijpxjxvdtrehmqvirfi.supabase.co"])
def test_staging_worker_rejects_wrong_database_before_connecting(monkeypatch, origin):
    monkeypatch.setenv("JUPR_BADGE_WORKER_CLUB_ID", "tres_palapas")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("FLY_APP_NAME", "juprleagues-api-staging")
    monkeypatch.setenv("JUPR_BADGE_WORKER_MODE", "external")
    monkeypatch.setenv("JUPR_BADGE_QUEUE_LEASES", "1")
    monkeypatch.setenv("SUPABASE_URL", origin)
    with pytest.raises(ValueError, match="database origin"):
        validate_worker_target("tres_palapas")


def test_lease_claim_is_explicit_and_club_checked(monkeypatch):
    monkeypatch.setenv("JUPR_BADGE_QUEUE_LEASES", "1")
    db = LeaseDB([job()])
    assert dequeue_badge_eval(db, club_id="club")["claim_token"] == "lease"
    assert db.calls[0][0] == "claim_badge_eval_queue_job_leased"
    assert db.calls[0][1]["p_lease_seconds"] == 120
    with pytest.raises(RuntimeError):
        dequeue_badge_eval(db, club_id="other")


def test_api_inline_processing_defers_in_external_mode(monkeypatch):
    monkeypatch.setenv("JUPR_BADGE_WORKER_MODE", "external")
    monkeypatch.setattr(badge_worker, "dequeue_badge_eval", lambda *a, **k: pytest.fail("API claimed a job"))
    assert badge_worker.process_badge_eval_queue(None, "club") == {"processed": 0, "errored": 0}
    assert badge_worker.process_badge_eval_queue_until_empty(None, "club")["stopped_reason"] == "external_worker_required"
    monkeypatch.setattr(badge_worker, "dequeue_badge_eval", lambda *a, **k: None)
    with isolated_worker_context():
        assert badge_worker.process_badge_eval_queue(None, "club") == {"processed": 0, "errored": 0}
    monkeypatch.setattr(badge_worker, "dequeue_badge_eval", lambda *a, **k: pytest.fail("Worker context leaked"))
    badge_worker.process_badge_eval_queue(None, "club")


def test_external_mode_does_not_start_embedded_program_worker(monkeypatch):
    from services.api.program_badge_worker import install_program_badge_worker
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("FLY_APP_NAME", "juprleagues-api-staging")
    monkeypatch.setenv("JUPR_BADGE_WORKER_MODE", "external")
    callbacks = {}
    def on_event(name):
        def register(fn):
            callbacks[name] = fn
            return fn
        return register
    app = SimpleNamespace(on_event=on_event, state=SimpleNamespace())
    install_program_badge_worker(app, get_supabase_client=lambda: pytest.fail("Embedded worker accessed database"))
    asyncio.run(callbacks["startup"]())
    assert not hasattr(app.state, "program_badge_worker")


def test_two_external_api_contexts_cannot_poll_program_badges(monkeypatch):
    from jupr_app.domain.gamification import program_badge_service
    monkeypatch.setenv("JUPR_BADGE_WORKER_MODE", "external")
    monkeypatch.setattr(program_badge_service, "program_badge_writes_enabled", lambda: True)
    for _ in range(2):
        assert program_badge_service.process_pending_program_badges(None) == {"processed": 0, "errors": 0}
    db = SimpleNamespace(rpc=lambda *a, **k: SimpleNamespace(execute=lambda: SimpleNamespace(data=[])))
    with isolated_worker_context():
        assert program_badge_service.process_pending_program_badges(db) == {"processed": 0, "errors": 0}
