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
        self.state = {
            "insert_rows": [],
            "updates": [],
            "eq_calls": [],
            "job_id": "job-123",
            "jobs": [],
            "rpc_calls": [],
            "finish_allowed": True,
            "heartbeat_allowed": True,
        }

    def table(self, name):
        return FakeTable(name, self.state)

    def rpc(self, name, params):
        owner = self

        class Rpc:
            def execute(self):
                owner.state["rpc_calls"].append((name, dict(params)))
                jobs = owner.state["jobs"]
                job = next(
                    row
                    for row in jobs
                    if str(row.get("id")) == str(params.get("p_job_id"))
                    and str(row.get("club_id")) == str(params.get("p_club_id"))
                )
                if name == "claim_replay_job_atomic":
                    if job.get("status") == "succeeded":
                        return SimpleNamespace(
                            data={
                                "ok": True,
                                "claimed": False,
                                "job_id": job["id"],
                                "status": "succeeded",
                                "result_json": dict(job.get("result_json") or {}),
                                "idempotent_replay": True,
                            }
                        )
                    if job.get("status") == "running":
                        return SimpleNamespace(
                            data={
                                "ok": False,
                                "claimed": False,
                                "code": "REPLAY_JOB_ALREADY_LEASED",
                                "job_id": job["id"],
                                "status": "running",
                                "lease_expires_at": "2099-01-01T00:00:00Z",
                            }
                        )
                    if job.get("status") == "failed" and not params.get(
                        "p_retry_failed"
                    ):
                        return SimpleNamespace(
                            data={
                                "ok": False,
                                "claimed": False,
                                "code": "REPLAY_JOB_FAILED",
                                "job_id": job["id"],
                                "status": "failed",
                                "error_text": job.get("error_text"),
                            }
                        )
                    job.update(
                        {
                            "status": "running",
                            "lease_token": "11111111-1111-1111-1111-111111111111",
                            "attempt_count": int(job.get("attempt_count") or 0) + 1,
                        }
                    )
                    return SimpleNamespace(
                        data={
                            "ok": True,
                            "claimed": True,
                            "job_id": job["id"],
                            "status": "running",
                            "lease_token": job["lease_token"],
                            "attempt_count": job["attempt_count"],
                            "idempotent_replay": False,
                        }
                    )
                if name == "heartbeat_replay_job_atomic":
                    if not owner.state["heartbeat_allowed"]:
                        return SimpleNamespace(
                            data={
                                "ok": False,
                                "renewed": False,
                                "code": "REPLAY_LEASE_LOST",
                            }
                        )
                    return SimpleNamespace(
                        data={
                            "ok": True,
                            "renewed": True,
                            "job_id": job["id"],
                            "status": "running",
                        }
                    )
                if name == "finish_replay_job_atomic":
                    if not owner.state["finish_allowed"]:
                        return SimpleNamespace(
                            data={
                                "ok": False,
                                "finished": False,
                                "code": "REPLAY_LEASE_LOST",
                            }
                        )
                    job.update(
                        {
                            "status": params["p_status"],
                            "result_json": dict(params.get("p_result_json") or {}),
                            "error_text": params.get("p_error_text"),
                            "lease_token": None,
                        }
                    )
                    return SimpleNamespace(
                        data={
                            "ok": True,
                            "finished": True,
                            "job_id": job["id"],
                            "status": job["status"],
                            "result_json": job["result_json"],
                            "error_text": job["error_text"],
                        }
                    )
                raise AssertionError(f"Unexpected RPC {name}")

        return Rpc()


def test_run_replay_with_job_tracking_success(monkeypatch):
    supabase = FakeSupabase()
    replay_calls = []
    replay_result = {
        "skipped_incomplete": 0,
        "matches_rewritten": 4,
        "league_ratings_rows": 10,
        "singles_replay_supported": True,
    }

    monkeypatch.setattr(
        replay_service,
        "replay_history",
        lambda **kwargs: replay_calls.append(kwargs) or replay_result,
    )

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
    assert [name for name, _params in supabase.state["rpc_calls"]] == [
        "claim_replay_job_atomic",
        "finish_replay_job_atomic",
    ]
    finish_params = supabase.state["rpc_calls"][1][1]
    assert finish_params["p_status"] == "succeeded"
    assert finish_params["p_result_json"] == replay_result
    assert finish_params["p_lease_token"]
    assert replay_calls[0]["write_fence"] == {
        "job_id": "job-123",
        "lease_token": "11111111-1111-1111-1111-111111111111",
        "worker_id": replay_calls[0]["write_fence"]["worker_id"],
    }
    assert replay_calls[0]["write_fence"]["worker_id"].startswith(
        "replay-worker:"
    )
    assert callable(replay_calls[0]["before_write_batch"])


def test_lease_loss_before_mutation_batch_is_fatal(monkeypatch):
    supabase = FakeSupabase()
    supabase.state["heartbeat_allowed"] = False

    def _replay(**kwargs):
        kwargs["before_write_batch"]()
        raise AssertionError("lease loss must stop before mutation")

    monkeypatch.setattr(replay_service, "replay_history", _replay)

    with pytest.raises(
        replay_service.ReplayLeaseLostError,
        match="could not be renewed",
    ):
        replay_service.run_replay_with_job_tracking(
            supabase=supabase,
            club_id="club-a",
            df_meta=None,
            target_reset="ALL (Full System Reset)",
        )

    assert [name for name, _params in supabase.state["rpc_calls"]] == [
        "claim_replay_job_atomic",
        "heartbeat_replay_job_atomic",
    ]
    assert supabase.state["jobs"][0]["status"] == "running"


def test_full_replay_without_singles_attestation_marks_job_failed(monkeypatch):
    supabase = FakeSupabase()
    monkeypatch.setattr(
        replay_service,
        "replay_history",
        lambda **_: {"matches_rewritten": 4},
    )

    with pytest.raises(RuntimeError, match="singles recovery"):
        replay_service.run_replay_with_job_tracking(
            supabase=supabase,
            club_id="club-a",
            df_meta=None,
            target_reset="ALL (Full System Reset)",
        )

    finish_params = supabase.state["rpc_calls"][-1][1]
    assert finish_params["p_status"] == "failed"
    assert "singles recovery" in finish_params["p_error_text"]


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

    finish_params = supabase.state["rpc_calls"][-1][1]
    assert finish_params["p_status"] == "failed"
    assert finish_params["p_error_text"] == "kaboom"


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


def test_existing_failed_job_is_reclaimed_only_for_guarded_recovery(
    monkeypatch,
):
    supabase = FakeSupabase()
    supabase.state["jobs"] = [
        {
            "id": "job-123",
            "club_id": "club-a",
            "target_reset": "Open",
            "status": "failed",
            "idempotency_key": "same-key",
            "result_json": {},
            "error_text": "first attempt failed",
        }
    ]
    monkeypatch.setattr(
        replay_service,
        "replay_history",
        lambda **_kwargs: {"matches_rewritten": 2},
    )

    with pytest.raises(RuntimeError, match="first attempt failed"):
        replay_service.run_replay_with_job_tracking(
            supabase=supabase,
            club_id="club-a",
            df_meta=None,
            target_reset="Open",
            idempotency_key="same-key",
        )

    recovered = replay_service.run_replay_with_job_tracking(
        supabase=supabase,
        club_id="club-a",
        df_meta=None,
        target_reset="Open",
        idempotency_key="same-key",
        retry_failed=True,
    )

    assert recovered["job_status"] == "succeeded"
    assert supabase.state["jobs"][0]["status"] == "succeeded"


def test_direct_operation_job_does_not_create_a_second_replay(monkeypatch):
    supabase = FakeSupabase()
    supabase.state["jobs"] = [
        {
            "id": "job-123",
            "club_id": "club-a",
            "target_reset": "Open",
            "status": "pending",
            "idempotency_key": "match-exclusion:operation",
            "result_json": {},
        }
    ]
    monkeypatch.setattr(
        replay_service,
        "replay_history",
        lambda **_kwargs: {"matches_rewritten": 2},
    )

    result = replay_service.run_replay_with_job_tracking(
        supabase=supabase,
        club_id="club-a",
        df_meta=None,
        target_reset="Open",
        replay_job_id="job-123",
    )

    assert result["job_status"] == "succeeded"
    assert supabase.state["insert_rows"] == []


def test_lost_finish_lease_never_blindly_updates_job(monkeypatch):
    supabase = FakeSupabase()
    supabase.state["finish_allowed"] = False
    monkeypatch.setattr(
        replay_service,
        "replay_history",
        lambda **_kwargs: {"matches_rewritten": 2},
    )

    with pytest.raises(replay_service.ReplayLeaseLostError):
        replay_service.run_replay_with_job_tracking(
            supabase=supabase,
            club_id="club-a",
            df_meta=None,
            target_reset="Open",
        )

    assert supabase.state["jobs"][0]["status"] == "running"
    assert supabase.state["updates"] == []


def test_is_replay_jobs_table_missing_error_code_42p01():
    exc = Exception({"code": "42P01", "message": "relation replay_jobs does not exist"})
    assert replay_service.is_replay_jobs_table_missing_error(exc)


def test_is_replay_jobs_table_missing_error_code_pgrst205():
    exc = Exception({"code": "PGRST205", "message": "Could not find table replay_jobs in schema cache"})
    assert replay_service.is_replay_jobs_table_missing_error(exc)


def test_is_replay_jobs_table_missing_error_text_only():
    exc = Exception("replay_jobs does not exist")
    assert replay_service.is_replay_jobs_table_missing_error(exc)
