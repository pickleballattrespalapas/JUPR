import json
import asyncio
import logging
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from jupr_app.domain.gamification import badge_lease, badge_worker, program_badge_service
from jupr_app.workers import isolated_badge_worker as worker


@pytest.fixture
def enabled(monkeypatch):
    for name, value in {
        "JUPR_ENV": "staging", "FLY_APP_NAME": "juprleagues-api-staging",
        "SUPABASE_URL": "https://sijpxjxvdtrehmqvirfi.supabase.co",
        "JUPR_BADGE_WORKER_MODE": "external", "JUPR_BADGE_QUEUE_LEASES": "1",
        "JUPR_BADGE_WORKER_CLUB_ID": "tres_palapas",
        "JUPR_STAGING_WRITE_WAVE": "open",
    }.items():
        monkeypatch.setenv(name, value)


class StateDB:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def table(self, name):
        self.calls.append(("table", name))
        return self

    def select(self, fields):
        self.calls.append(("select", fields))
        return self

    def eq(self, field, value):
        self.calls.append(("eq", field, value))
        return self

    def lte(self, field, value):
        self.calls.append(("lte", field, value))
        return self

    def limit(self, value):
        self.calls.append(("limit", value))
        return self

    def execute(self):
        return SimpleNamespace(data=self.rows)


def test_scoped_program_query_limits_after_exact_club_and_retry_filters(enabled, monkeypatch):
    db = StateDB([{"club_id": "tres_palapas", "revision": 2, "applied_revision": 1}])
    processed = []
    monkeypatch.setattr(program_badge_service, "reconcile_program_badges", lambda db, club: processed.append(club))
    with badge_lease.isolated_worker_context():
        result = program_badge_service.process_pending_program_badges(db, limit=1, club_id="tres_palapas")
    assert result == {"processed": 1, "errors": 0}
    assert processed == ["tres_palapas"]
    assert db.calls[:3] == [
        ("table", "badge_program_state"),
        ("select", "club_id,revision,applied_revision,retry_after"),
        ("eq", "club_id", "tres_palapas"),
    ]
    assert db.calls[3][0:2] == ("lte", "retry_after")
    assert db.calls[3][2].endswith("+00:00")
    assert db.calls[4] == ("limit", 1)


@pytest.mark.parametrize("rows", [[], [{"club_id": "tres_palapas", "revision": 2, "applied_revision": 2}]])
def test_scoped_program_skips_absent_or_current_revision(enabled, monkeypatch, rows):
    monkeypatch.setattr(program_badge_service, "reconcile_program_badges", lambda *args: pytest.fail("Unneeded calculation"))
    with badge_lease.isolated_worker_context():
        assert program_badge_service.process_pending_program_badges(StateDB(rows), club_id="tres_palapas") == {"processed": 0, "errors": 0}


@pytest.mark.parametrize("rows", [
    [{"club_id": "other", "revision": 2, "applied_revision": 1}],
    [{"club_id": "tres_palapas", "revision": 2}],
    [{"club_id": "tres_palapas", "revision": True, "applied_revision": 0}],
])
def test_scoped_program_refuses_wrong_club_or_incomplete_state(enabled, monkeypatch, rows):
    monkeypatch.setattr(program_badge_service, "reconcile_program_badges", lambda *args: pytest.fail("Unverified calculation"))
    with badge_lease.isolated_worker_context(), pytest.raises(RuntimeError, match="read completely"):
        program_badge_service.process_pending_program_badges(StateDB(rows), club_id="tres_palapas")


def test_shutdown_during_program_phase_does_not_claim_queue_job(enabled, monkeypatch):
    stop = threading.Event()
    def program(db, *, limit, club_id):
        assert limit == 1 and club_id == "tres_palapas"
        stop.set()
        return {"processed": 1, "errors": 0}
    monkeypatch.setattr(worker, "process_pending_program_badges", program)
    monkeypatch.setattr(worker, "process_badge_eval_queue_until_empty", lambda *args, **kw: pytest.fail("New job claimed after shutdown"))
    result = worker.run_cycle(None, "tres_palapas", stop=stop)
    assert result["program"]["processed"] == 1
    assert result["queue"]["stopped_reason"] == "shutdown"
    monkeypatch.setattr(worker, "process_pending_program_badges", lambda *args, **kw: pytest.fail("New program phase after shutdown"))
    assert worker.run_cycle(None, "tres_palapas", stop=stop)["program"]["processed"] == 0


@pytest.mark.parametrize("wave", ["none", "communications", "league-live-submit", ""])
def test_staging_write_stop_prevents_all_worker_database_processing(enabled, monkeypatch, wave):
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", wave)
    monkeypatch.setattr(worker, "process_pending_program_badges", lambda *a, **k: pytest.fail("Program processing while writes disabled"))
    monkeypatch.setattr(worker, "process_badge_eval_queue_until_empty", lambda *a, **k: pytest.fail("Queue processing while writes disabled"))
    assert worker.run_cycle(object(), "tres_palapas") == {
        "program": {"processed": 0, "errors": 0},
        "queue": {"total_processed": 0, "total_errored": 0, "stopped_reason": "writes_disabled"},
    }


@pytest.mark.parametrize("wave", ["open", "badge-diagnostics"])
def test_authorized_staging_wave_allows_both_worker_processors(enabled, monkeypatch, wave):
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", wave)
    calls = []
    monkeypatch.setattr(worker, "process_pending_program_badges", lambda *a, **k: calls.append("program") or {"processed": 0, "errors": 0})
    monkeypatch.setattr(worker, "process_badge_eval_queue_until_empty", lambda *a, **k: calls.append("queue") or {"total_processed": 0, "total_errored": 0})
    worker.run_cycle(object(), "tres_palapas")
    assert calls == ["program", "queue"]


def test_shutdown_keeps_inflight_queue_heartbeat_until_fenced_completion(enabled, monkeypatch):
    stop, heartbeat = threading.Event(), threading.Event()
    calls = []
    class DB:
        def rpc(self, name, args):
            calls.append(name)
            def execute():
                if name == "heartbeat_badge_eval_queue_job" and threading.current_thread().name == "badge-job-heartbeat":
                    heartbeat.set()
                return SimpleNamespace(data=True)
            return SimpleNamespace(execute=execute)
    monkeypatch.setattr(badge_lease, "HEARTBEAT_SECONDS", 0.005)
    monkeypatch.setattr(worker, "process_pending_program_badges", lambda *args, **kw: {"processed": 0, "errors": 0})
    monkeypatch.setattr(badge_worker, "dequeue_badge_eval", lambda *args, **kw: {
        "id": "job", "club_id": "tres_palapas", "claim_token": "lease",
    })
    def complete_inflight(*args, **kwargs):
        stop.set()
        assert heartbeat.wait(1), "Current job lost heartbeat during graceful shutdown"
    monkeypatch.setattr(badge_worker, "_process_job_with_retry", complete_inflight)
    result = worker.run_cycle(DB(), "tres_palapas", stop=stop)
    assert result["queue"]["total_processed"] == 1
    assert calls[-1] == "finish_badge_eval_queue_job"
    assert calls.count("finish_badge_eval_queue_job") == 1
    assert not any(t.name == "badge-job-heartbeat" for t in threading.enumerate())


@pytest.mark.parametrize("program_errors,queue_errors,expected", [(0, 0, 0), (1, 0, 1), (0, 1, 1)])
def test_once_exit_code_includes_reported_phase_errors(enabled, monkeypatch, program_errors, queue_errors, expected):
    monkeypatch.setattr(worker.logging.config, "dictConfig", lambda *args: None)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(worker, "_resolve_supabase_config", lambda: ("url", "key", "source"))
    monkeypatch.setattr(worker, "make_supabase", lambda *args: object())
    monkeypatch.setattr(worker, "run_cycle", lambda *args, **kw: {
        "program": {"processed": 0, "errors": program_errors},
        "queue": {"total_processed": 0, "total_errored": queue_errors},
    })
    assert worker.main(["--club-id", "tres_palapas", "--once"]) == expected


def test_once_exception_is_failed_without_logging_exception_text(enabled, monkeypatch, caplog):
    monkeypatch.setattr(worker.logging.config, "dictConfig", lambda *args: None)
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(worker, "_resolve_supabase_config", lambda: ("url", "key", "source"))
    monkeypatch.setattr(worker, "make_supabase", lambda *args: object())
    def fail(*args, **kwargs):
        raise ValueError("sensitive-provider-text")
    monkeypatch.setattr(worker, "run_cycle", fail)
    with caplog.at_level(logging.INFO, logger="jupr.badge.worker"):
        assert worker.main(["--club-id", "tres_palapas", "--once"]) == 1
    assert "sensitive-provider-text" not in caplog.text
    assert "badge.worker.stopped" in caplog.text


def test_actual_log_config_emits_info_ownership_fields_and_excludes_payloads():
    script = '''
import json, logging, logging.config
from pathlib import Path
logging.config.dictConfig(json.loads(Path("services/api/logging.json").read_text()))
logging.getLogger("jupr.badge.worker").info("badge.job.heartbeat", extra={
    "job_id": "job", "worker_id": "machine:pid", "club_id": "tres_palapas",
    "elapsed_ms": 25, "claim_token": "private-token", "payload": {"name": "Private Name"},
})
'''
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    assert payload["level"] == "INFO"
    assert payload["event"] == "badge.job.heartbeat"
    assert payload["job_id"] == "job" and payload["worker_id"] == "machine:pid"
    assert payload["club_id"] == "tres_palapas" and payload["elapsed_ms"] == 25
    assert "claim_token" not in payload and "payload" not in payload
    assert "private-token" not in result.stdout and "Private Name" not in result.stdout


@pytest.mark.parametrize("scope", ["", "other", "tres_palapas,other"])
def test_isolated_entrypoint_requires_exact_explicit_pilot_scope(enabled, monkeypatch, scope):
    monkeypatch.setenv("JUPR_BADGE_WORKER_CLUB_ID", scope)
    with pytest.raises(ValueError, match="scope"):
        worker.validate_worker_target("tres_palapas")


def test_scoped_api_saves_defer_pilot_only_and_preserve_failed_enqueue_evidence(enabled):
    assert badge_lease.processing_deferred("tres_palapas")
    assert not badge_lease.processing_deferred("other")
    assert not badge_lease.processing_deferred()  # Global poller must serve other clubs.
    assert badge_lease.deferred_badge_summary({"queued": True}, club_id="other") is None
    assert badge_lease.deferred_badge_summary({"queued": False}, club_id="other") is None
    pilot = badge_lease.deferred_badge_summary({"queued": True}, club_id="tres_palapas", match_ref="saved")
    assert pilot["deferred"] and pilot["notification_badge_refs"] == ["saved"]
    failed = badge_lease.deferred_badge_summary({"queued": False}, club_id="tres_palapas", match_ref="saved")
    assert failed["mode"] == "queue_error" and failed["recovery_match_refs"] == ["saved"]


def test_two_api_contexts_defer_pilot_but_other_club_retains_legacy_claims(enabled, monkeypatch):
    calls, processed = [], []
    class DB:
        def rpc(self, name, args):
            calls.append((name, args))
            assert name == "claim_badge_eval_queue_job"
            assert args == {"p_club_id": "other"}
            data = [{"id": "other-job", "club_id": "other"}]
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=data))

        def table(self, name):
            assert name == "badge_eval_queue"
            return self

        def update(self, payload):
            assert payload["status"] == "done"
            calls.append(("legacy_ack", payload))
            return self

        def eq(self, field, value):
            assert (field, value) == ("id", "other-job")
            return self

        def execute(self):
            return SimpleNamespace(data=[])

    monkeypatch.setattr(badge_worker, "_process_job_with_retry", lambda db, job, **kw: processed.append(job["club_id"]))
    for _ in range(2):
        assert badge_worker.process_badge_eval_queue(None, "tres_palapas", max_jobs=1) == {"processed": 0, "errored": 0}
        assert badge_worker.process_badge_eval_queue_until_empty(None, "tres_palapas")["stopped_reason"] == "external_worker_required"
        assert badge_worker.process_badge_eval_queue(DB(), "other", max_jobs=1) == {"processed": 1, "errored": 0}
    assert processed == ["other", "other"]
    assert sum(name == "legacy_ack" for name, _ in calls) == 2
    assert not any(name == "claim_badge_eval_queue_job_leased" for name, _ in calls)


def test_pilot_first_pending_program_does_not_starve_other_club_at_limit_one(enabled, monkeypatch):
    calls, processed = [], []
    def rpc(name, args):
        calls.append((name, args))
        assert name == "pending_program_badge_clubs_v1" and args == {"p_limit": 2}
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=[{"club_id": "tres_palapas"}, {"club_id": "other"}]))
    monkeypatch.setattr(program_badge_service, "reconcile_program_badges", lambda db, club: processed.append(club))
    for _ in range(2):
        assert program_badge_service.process_pending_program_badges(SimpleNamespace(rpc=rpc), limit=1) == {"processed": 1, "errors": 0}
        assert program_badge_service.process_pending_program_badges(None, club_id="tres_palapas") == {"processed": 0, "errors": 0}
    assert processed == ["other", "other"]


def test_scoped_mode_keeps_embedded_program_loop_for_other_clubs(enabled, monkeypatch):
    from services.api.program_badge_worker import install_program_badge_worker
    callbacks, created = {}, []
    def on_event(name):
        def register(fn):
            callbacks[name] = fn
            return fn
        return register
    def create_task(coro):
        created.append(True)
        coro.close()
        return "background-task"
    monkeypatch.setattr(asyncio, "create_task", create_task)
    app = SimpleNamespace(on_event=on_event, state=SimpleNamespace())
    install_program_badge_worker(app, get_supabase_client=lambda: None)
    asyncio.run(callbacks["startup"]())
    assert created == [True]
    assert app.state.program_badge_worker == "background-task"


def test_program_failure_keeps_backoff_without_provider_text_in_logs(enabled, monkeypatch, caplog):
    calls = []
    def rpc(name, args):
        calls.append((name, args))
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=[{"club_id": "other"}]))
    def fail(*args, **kwargs):
        raise RuntimeError("sensitive-provider-text")
    monkeypatch.setattr(program_badge_service, "reconcile_program_badges", fail)
    with caplog.at_level(logging.ERROR, logger="jupr.badge.worker"):
        assert program_badge_service.process_pending_program_badges(SimpleNamespace(rpc=rpc)) == {"processed": 0, "errors": 1}
    assert calls[-1] == ("program_badge_failure_v1", {"p_club_id": "other"})
    assert "sensitive-provider-text" not in caplog.text
    record = next(record for record in caplog.records if record.message == "badge.program.failed")
    assert record.exception_type == "RuntimeError" and record.club_id == "other"
    assert record.exc_info is None
