from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import logging
import os
from threading import Barrier

from fastapi import FastAPI
from fastapi.testclient import TestClient

from jupr_app.observability.runtime_memory import RequestMemoryTracker, runtime_memory_snapshot
from services.api.logging_config import RequestJSONFormatter
from services.api.middleware import StructuredRequestLoggingMiddleware


def test_memory_samples_report_current_rss_and_machine_available_ram(tmp_path):
    (tmp_path / "self").mkdir()
    (tmp_path / "self/statm").write_text("99999 256 4 0 0 0 0\n")
    (tmp_path / "meminfo").write_text("MemTotal: 2097152 kB\nMemAvailable: 524288 kB\n")
    assert runtime_memory_snapshot(tmp_path) == {
        "process_rss_mb": round(256 * os.sysconf("SC_PAGE_SIZE") / 1048576, 2),
        "machine_available_mb": 512,
    }
    (tmp_path / "self/statm").write_text("malformed")
    (tmp_path / "meminfo").write_text("MemAvailable: unavailable kB\n")
    assert runtime_memory_snapshot(tmp_path) == {}
    assert runtime_memory_snapshot(tmp_path / "missing") == {}


def test_overlapping_handlers_are_counted_and_sampler_failure_does_not_leak_count():
    def unavailable():
        raise OSError("proc unavailable")

    tracker = RequestMemoryTracker(unavailable)
    all_started = Barrier(8)

    def handler(_):
        started = tracker.start()
        all_started.wait(timeout=5)
        return tracker.finish(started)

    with ThreadPoolExecutor(max_workers=8) as pool:
        observations = list(pool.map(handler, range(8)))
    assert sorted(row["handlers_in_flight_start"] for row in observations) == list(range(1, 9))
    assert sorted(row["handlers_in_flight_end"] for row in observations) == list(range(8))
    assert tracker.finish(tracker.start())["handlers_in_flight_end"] == 0


def test_cancellation_releases_handler_count_and_request_context():
    from starlette.requests import Request

    middleware = StructuredRequestLoggingMiddleware(FastAPI())
    request = Request({"type": "http", "method": "GET", "path": "/cancel", "headers": []})

    async def cancelled(_):
        raise asyncio.CancelledError()

    async def exercise():
        try:
            await middleware.dispatch(request, cancelled)
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Cancellation must be propagated")

    asyncio.run(exercise())
    assert middleware._memory_tracker._in_flight == 0


def test_success_and_failure_keep_memory_logs_safe_and_counts_balanced(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="jupr.api.request")
    monkeypatch.setenv("FLY_MACHINE_ID", "test-machine")
    app = FastAPI()
    app.add_middleware(StructuredRequestLoggingMiddleware)

    @app.get("/players/{player_id}")
    def ok(player_id: str):
        return {"ok": True}

    @app.get("/fail")
    def fail():
        raise ValueError("fixture failure")

    with TestClient(app, raise_server_exceptions=False) as client:
        for path, status, request_id in [
            ("/players/PRIVATE_PLAYER", 200, "memory-ok"),
            ("/fail", 500, "memory-fail"),
            ("/players/PRIVATE_PLAYER", 200, "memory-after-fail"),
        ]:
            response = client.get(path + "?edit_token=PRIVATE_TOKEN", headers={"x-request-id": request_id})
            assert response.status_code == status

    records = [json.loads(RequestJSONFormatter().format(record))
               for record in caplog.records if record.name == "jupr.api.request"]
    assert len(records) == 3
    for record in records:
        assert record["handlers_in_flight_start"] == 1
        assert record["handlers_in_flight_end"] == 0
        assert record["process_rss_mb_start"] > 0
        assert record["process_rss_mb_end"] > 0
        assert record["machine_available_mb_start"] >= 0
        assert record["machine_available_mb_end"] >= 0
        assert record["machine_id"] == "test-machine"
    assert records[0]["path"] == records[2]["path"] == "/players/{player_id}"
    assert records[1]["event"] == "request.failed"
    assert records[1]["exception_type"] == "ValueError"
    assert "PRIVATE_PLAYER" not in json.dumps(records)
    assert "PRIVATE_TOKEN" not in json.dumps(records)
