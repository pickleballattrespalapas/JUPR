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

    def update(self, row):
        self.pending = ("update", row)
        return self

    def eq(self, key, value):
        self.state["eq_calls"].append((self.name, key, value))
        return self

    def execute(self):
        kind, payload = self.pending
        if self.name == "replay_jobs" and kind == "insert":
            self.state["insert_rows"].append(payload)
            return SimpleNamespace(data=[{"id": self.state["job_id"], **payload}])
        if self.name == "replay_jobs" and kind == "update":
            self.state["updates"].append(payload)
            return SimpleNamespace(data=[payload])
        raise AssertionError(f"Unexpected execute for {self.name}/{kind}")


class FakeSupabase:
    def __init__(self):
        self.state = {"insert_rows": [], "updates": [], "eq_calls": [], "job_id": "job-123"}

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
