import pytest

from jupr_app.workers import badge_queue_worker as bqw


class _Resp:
    def __init__(self, data=None):
        self.data = data


class _Query:
    def __init__(self, table, state):
        self.table = table
        self.state = state
        self.payload = None

    def insert(self, payload):
        self.payload = payload
        return self

    def update(self, payload):
        self.payload = payload
        return self

    def eq(self, *_args):
        return self

    def execute(self):
        if self.state.get("missing"):
            raise Exception("relation worker_run_log does not exist")
        if self.table == "worker_run_log" and self.payload and self.payload.get("status") == "started":
            self.state.setdefault("events", []).append(("start", self.payload))
            return _Resp([{"id": "run-1"}])
        if self.table == "worker_run_log" and self.payload:
            self.state.setdefault("events", []).append((self.payload.get("status"), self.payload))
        return _Resp([])


class _Supabase:
    def __init__(self, state):
        self.state = state

    def table(self, name):
        return _Query(name, self.state)


def test_worker_success_logs_summary(monkeypatch):
    state = {}
    monkeypatch.setattr(bqw, "make_supabase", lambda *_: _Supabase(state))
    monkeypatch.setattr(bqw, "process_badge_eval_queue_until_empty", lambda *_, **__: {"total_errored": 0, "processed": 3})
    monkeypatch.setenv("SUPABASE_URL", "u")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "k")
    out = bqw.run_badge_queue_worker("club-1")
    assert out["ok"] is True
    assert any(evt[0] == "success" for evt in state["events"])


def test_worker_failure_logs_error(monkeypatch):
    state = {}
    monkeypatch.setattr(bqw, "make_supabase", lambda *_: _Supabase(state))
    monkeypatch.setattr(bqw, "process_badge_eval_queue_until_empty", lambda *_, **__: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setenv("SUPABASE_URL", "u")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "k")
    with pytest.raises(RuntimeError):
        bqw.run_badge_queue_worker("club-1")
    assert any(evt[0] == "failed" for evt in state["events"])


def test_missing_table_degrades_when_not_strict(monkeypatch):
    state = {"missing": True}
    monkeypatch.setattr(bqw, "make_supabase", lambda *_: _Supabase(state))
    monkeypatch.setattr(bqw, "process_badge_eval_queue_until_empty", lambda *_, **__: {"total_errored": 0})
    monkeypatch.setenv("SUPABASE_URL", "u")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "k")
    out = bqw.run_badge_queue_worker("club-1")
    assert out["ok"] is True


def test_missing_table_fails_when_strict(monkeypatch):
    state = {"missing": True}
    monkeypatch.setattr(bqw, "make_supabase", lambda *_: _Supabase(state))
    monkeypatch.setattr(bqw, "process_badge_eval_queue_until_empty", lambda *_, **__: {"total_errored": 0})
    monkeypatch.setenv("SUPABASE_URL", "u")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "k")
    monkeypatch.setenv("JUPR_REQUIRE_WORKER_RUN_LOG", "1")
    with pytest.raises(Exception):
        bqw.run_badge_queue_worker("club-1")
