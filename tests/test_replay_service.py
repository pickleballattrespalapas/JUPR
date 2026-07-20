from types import SimpleNamespace

import pytest

from jupr_app.services import replay_service


class FakeTable:
    def __init__(self, name: str, state: dict):
        self.name = name
        self.state = state
        self.pending = None

    def insert(self, row):
        self.pending = ("insert", row)
        return self

    def select(self, *_args, **_kwargs):
        self.pending = ("select", None)
        return self

    def limit(self, _value):
        return self

    def update(self, row):
        self.pending = ("update", row)
        return self

    def eq(self, key, value):
        self.state["eq_calls"].append((self.name, key, value))
        self.state.setdefault("filters", []).append((key, value))
        return self

    def execute(self):
        kind, payload = self.pending
        filters = list(self.state.pop("filters", []))
        if self.name == "replay_jobs" and kind == "select":
            rows = list(self.state["jobs"])
            for key, value in filters:
                rows = [row for row in rows if str(row.get(key)) == str(value)]
            return SimpleNamespace(data=rows)
        if self.name == "replay_jobs" and kind == "insert":
            self.state["insert_rows"].append(payload)
            row = {"id": self.state["job_id"], **payload}
            self.state["jobs"].append(row)
            return SimpleNamespace(data=[row])
        if self.name == "replay_jobs" and kind == "update":
            self.state["updates"].append(payload)
            rows = list(self.state["jobs"])
            for key, value in filters:
                rows = [row for row in rows if str(row.get(key)) == str(value)]
            for row in rows:
                row.update(payload)
            return SimpleNamespace(data=[dict(row) for row in rows])
        raise AssertionError(f"Unexpected execute for {self.name}/{kind}")


class FakeSupabase:
    def __init__(self):
        self.state = {"insert_rows": [], "updates": [], "eq_calls": [], "job_id": "job-123", "jobs": []}

    def table(self, name):
        return FakeTable(name, self.state)


def test_run_replay_with_job_tracking_success(monkeypatch):
    supabase = FakeSupabase()
    replay_result = {"skipped_incomplete": 0, "matches_rewritten": 4, "league_ratings_rows": 10}

    monkeypatch.setattr(replay_service, "replay_history", lambda **_: replay_result)

    out = replay_service.run_replay_with_job_tracking(
        supabase=supabase,
        club_id="club-a",
        df_meta=None,
        target_reset="ALL (Full System Reset)",
        actor_email="admin@example.com",
        actor_role="owner",
        progress_cb=None,
    )

    assert out["job_id"] == "job-123"
    assert out["job_status"] == "succeeded"
    assert out["result"] == replay_result

    assert supabase.state["insert_rows"][0]["status"] == "pending"
    assert supabase.state["updates"][0]["status"] == "running"
    assert supabase.state["updates"][1]["status"] == "succeeded"
    assert supabase.state["updates"][1]["result_json"] == replay_result


def test_run_replay_with_job_tracking_failure_marks_failed(monkeypatch):
    supabase = FakeSupabase()

    def _boom(**_):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(replay_service, "replay_history", _boom)

    with pytest.raises(RuntimeError, match="kaboom"):
        replay_service.run_replay_with_job_tracking(
            supabase=supabase,
            club_id="club-a",
            df_meta=None,
            target_reset="League A",
            progress_cb=None,
        )

    assert supabase.state["updates"][0]["status"] == "running"
    assert supabase.state["updates"][1]["status"] == "failed"
    assert supabase.state["updates"][1]["error_text"] == "kaboom"


def test_existing_pending_job_is_claimed_and_completed(monkeypatch):
    supabase = FakeSupabase()
    supabase.state["jobs"] = [{
        "id": "job-123",
        "club_id": "club-a",
        "target_reset": "Open",
        "status": "pending",
        "idempotency_key": "same-key",
        "result_json": {},
    }]
    calls = []
    monkeypatch.setattr(replay_service, "replay_history", lambda **kwargs: calls.append(kwargs) or {"matches_rewritten": 2})

    result = replay_service.run_replay_with_job_tracking(
        supabase=supabase,
        club_id="club-a",
        df_meta=None,
        target_reset="Open",
        idempotency_key="same-key",
    )

    assert result["job_status"] == "succeeded"
    assert len(calls) == 1


def test_existing_running_job_is_not_replayed(monkeypatch):
    supabase = FakeSupabase()
    supabase.state["jobs"] = [{
        "id": "job-123",
        "club_id": "club-a",
        "target_reset": "Open",
        "status": "running",
        "idempotency_key": "same-key",
        "result_json": {},
    }]
    monkeypatch.setattr(replay_service, "replay_history", lambda **_: (_ for _ in ()).throw(AssertionError("must not replay")))

    result = replay_service.run_replay_with_job_tracking(
        supabase=supabase,
        club_id="club-a",
        df_meta=None,
        target_reset="Open",
        idempotency_key="same-key",
    )

    assert result["job_status"] == "running"
    assert result["idempotent_replay"] is True


def test_is_replay_jobs_table_missing_error_code_42p01():
    exc = Exception({"code": "42P01", "message": "relation replay_jobs does not exist"})
    assert replay_service.is_replay_jobs_table_missing_error(exc)


def test_is_replay_jobs_table_missing_error_code_pgrst205():
    exc = Exception({"code": "PGRST205", "message": "Could not find table replay_jobs in schema cache"})
    assert replay_service.is_replay_jobs_table_missing_error(exc)


def test_is_replay_jobs_table_missing_error_text_only():
    exc = Exception("replay_jobs does not exist")
    assert replay_service.is_replay_jobs_table_missing_error(exc)
