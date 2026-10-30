"""Planning runs off the API process; admission and failure never imply a save."""
from concurrent.futures import Future
import asyncio
import logging
import threading
import time

import httpx
import pytest
from fastapi import FastAPI

from jupr_app.services.generator_compute import (
    GeneratorComputeUnavailable, ScheduleWorker, compute_unavailable_cause,
)


def preview_args(size=4, rounds=1):
    return dict(generator_kind="round_robin", play_format="doubles", title="Test",
        participant_names=[f"Player {n}" for n in range(size)], total_rounds=rounds,
        court_count=size // 4, standings_sort="wins", scoring_mode="unscored", rating_mode="unrated")


class PendingPool:
    def __init__(self):
        self.jobs = []
    def submit(self, *args):
        future = Future()
        future.set_running_or_notify_cancel()
        self.jobs.append(future)
        return future


def test_timeout_holds_capacity_until_pure_job_finishes(monkeypatch):
    worker = ScheduleWorker(capacity=1, timeout_seconds=.01)
    pool = PendingPool()
    monkeypatch.setattr(worker, "_executor", lambda: pool)
    with pytest.raises(GeneratorComputeUnavailable, match="No changes were saved"):
        worker.run("engine", "create_generator_preview", **preview_args())
    with pytest.raises(GeneratorComputeUnavailable, match="Another schedule"):
        worker.run("engine", "create_generator_preview", **preview_args())
    assert len(pool.jobs) == 1
    pool.jobs[0].set_result(({"late": True}, {}))
    with pytest.raises(GeneratorComputeUnavailable, match="No changes were saved"):
        worker.run("engine", "create_generator_preview", **preview_args())
    assert len(pool.jobs) == 2
    pool.jobs[1].set_result(({}, {}))


def test_admission_failure_releases_slot(monkeypatch):
    worker = ScheduleWorker(capacity=1)
    def unavailable():
        raise GeneratorComputeUnavailable("restarting")
    monkeypatch.setattr(worker, "_executor", unavailable)
    for _ in range(2):
        with pytest.raises(GeneratorComputeUnavailable, match="restarting"):
            worker.run("engine", "create_generator_preview", **preview_args())


def test_wrapped_compute_failure_is_recognized_as_retryable():
    try:
        try:
            raise GeneratorComputeUnavailable("busy")
        except ValueError as exc:
            raise ValueError("domain rejected") from exc
    except ValueError as exc:
        assert str(compute_unavailable_cause(exc)) == "busy"
    assert compute_unavailable_cause(ValueError("invalid score")) is None


def test_real_worker_preserves_domain_errors_and_can_recover():
    worker = ScheduleWorker()
    try:
        with pytest.raises(ValueError):
            worker.run("engine", "create_generator_preview", **{**preview_args(), "participant_names": []})
        result = worker.run("engine", "create_generator_preview", **preview_args())
        assert len(result["participants"]) == 4
        assert len(result["rounds"]) >= 1
    finally:
        worker.close()


def test_simple_api_remains_responsive_during_real_cpu_planning(monkeypatch, caplog):
    worker = ScheduleWorker(timeout_seconds=60)
    submitted = threading.Event()
    pool = worker._executor()
    submit = pool.submit
    def observed_submit(*args, **kwargs):
        result = submit(*args, **kwargs)
        submitted.set()
        return result
    monkeypatch.setattr(pool, "submit", observed_submit)
    app = FastAPI()
    @app.get("/status")
    async def status():
        return {"ok": True}
    @app.post("/plan")
    def plan():
        result = worker.run("engine", "create_generator_preview", **preview_args(19, 21))
        return {"participants": len(result["participants"])}
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            planning = asyncio.create_task(client.post("/plan"))
            assert await asyncio.to_thread(submitted.wait, 10)
            assert not planning.done()
            start = time.monotonic()
            response = await client.get("/status")
            assert response.json() == {"ok": True}
            assert time.monotonic() - start < 2
            assert (await planning).json() == {"participants": 19}
    try:
        with caplog.at_level(logging.INFO, logger="jupr.api.compute"):
            asyncio.run(exercise())
        timings = [r for r in caplog.records if r.getMessage() == "generator.compute.complete"]
        assert timings and timings[-1].cpu_ms > 0
        assert timings[-1].queue_ms >= 0
    finally:
        worker.close()


def test_old_broken_waiter_does_not_remove_replacement_pool(monkeypatch):
    from concurrent.futures.process import BrokenProcessPool
    worker = ScheduleWorker()
    replacement = object()
    class BrokenPool:
        def submit(self, *args):
            # Another caller already discarded this pool and replaced it.
            worker._pool = replacement
            raise BrokenProcessPool("old pool")
        def shutdown(self, **kwargs):
            pass
    old = BrokenPool()
    monkeypatch.setattr(worker, "_executor", lambda: old)
    with pytest.raises(GeneratorComputeUnavailable):
        worker.run("engine", "create_generator_preview", **preview_args())
    assert worker._pool is replacement


def test_failed_public_planning_never_commits_session(monkeypatch):
    from copy import deepcopy
    from tests.test_public_play_generator_service import FakeSupabase, key, requester, token_secret
    from jupr_app.services import public_play_generator_service as service
    db = FakeSupabase()
    created = service.create_public_play_generator_session(db, club_id="club",
        **preview_args(), participant_player_ids={}, preview_fingerprint=None, idempotency_key=key("worker-start"),
        requester_hash=requester(), token_secret=token_secret())
    before = deepcopy(db.db["live_sessions"])
    def unavailable(*args, **kwargs):
        raise GeneratorComputeUnavailable("Schedule planning took too long. No changes were saved. Please try again.")
    monkeypatch.setattr(service, "mutate_generator_roster", unavailable)
    with pytest.raises(service.PublicPlayGeneratorError) as error:
        service.mutate_public_play_generator_roster(db, club_id="club",
            session_key=created["session"]["session_key"], action="remove", participant_id="p-1",
            name=None, player_id=None, substitute_scope="rest", roster_order=None,
            edit_token=created["edit_token"], expected_version=created["session"]["version"],
            idempotency_key=key("worker-remove"), requester_hash=requester())
    assert compute_unavailable_cause(error.value)
    assert db.db["live_sessions"] == before
    assert db.db["public_live_operations"][-1]["status"] == "rejected"


def test_public_and_admin_handlers_return_retryable_503():
    from fastapi import HTTPException
    from services.api.main import _raise_public_live_write_error
    from services.api.admin_play_generator_routes import _handle
    for handler in (_raise_public_live_write_error, _handle):
        with pytest.raises(HTTPException) as error:
            handler(GeneratorComputeUnavailable("Another schedule is being prepared."))
        assert error.value.status_code == 503
        assert error.value.headers["Retry-After"] == "2"


def test_durable_checkin_stays_successful_when_planner_is_busy(monkeypatch):
    from tests.test_hub_arrival_registration import empty_hub
    from jupr_app.services import round_robin_hub_registration_service as registration
    from jupr_app.services import admin_round_robin_hub_service as hubs
    db, hub = empty_hub()
    def busy(*args, **kwargs):
        raise GeneratorComputeUnavailable("Another schedule is being prepared.")
    monkeypatch.setattr(hubs, "command_round_robin_hub", busy)
    response = registration.submit_hub_checkin(db, hub_id=hub["hub"]["id"],
        session_key="rr-a", name="New Player")
    assert response["ok"] is True
    assert len(db.storage[registration.CHECKIN_TABLE]) == 1


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="POSIX Uvicorn SIGTERM lifecycle")
def test_uvicorn_fixture_reaps_worker_and_closes_output_pipes_on_sigterm():
    """Match the browser runner: ASGI shutdown must reap the real child process.

    Checking only API exit status misses orphan planners keeping Node's piped
    stdout alive. communicate() additionally proves every output pipe closes.
    """
    import json
    import os
    from pathlib import Path
    import signal
    import socket
    import subprocess
    import sys
    import urllib.error
    import urllib.request
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    process = subprocess.Popen([sys.executable, "-m", "uvicorn", "hub_registration_api:app",
        "--app-dir", "tests/e2e", "--host", "127.0.0.1", "--port", str(port)], cwd=root,
        env={**os.environ, "HUB_EMPTY_REGISTRATION_FIXTURE": "1", "JUPR_GENERATOR_PROCESS_WORKER": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                with urllib.request.urlopen(origin + "/hub-fixture-result", timeout=1) as response:
                    assert response.status == 200
                break
            except (OSError, urllib.error.URLError):
                if process.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail("Loopback worker lifecycle fixture did not start")
                time.sleep(.1)
        request = urllib.request.Request(origin + "/admin/clubs/club/play-generators/preview",
            data=json.dumps(preview_args()).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer local-reviewer"})
        with urllib.request.urlopen(request, timeout=15) as response:
            assert response.status == 200
        process.terminate()
        output, _ = process.communicate(timeout=8)
        assert "Application shutdown complete" in output
        assert process.returncode in (0, -signal.SIGTERM)
    finally:
        # A regressed orphan can outlive its API parent, so always kill the
        # test's isolated process group, not just the already-exited parent.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate(timeout=5)
