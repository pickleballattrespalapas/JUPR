"""Exercise real SDK query chains and concurrent ASGI-to-thread propagation."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from postgrest import SyncPostgrestClient
from supabase import create_client

from jupr_app.observability.request_timing import (
    RequestTiming, current_timing, instrument_supabase_client, measured_phase,
)
from services.api.logging_config import RequestJSONFormatter
from services.api.middleware import StructuredRequestLoggingMiddleware
from services.api.middleware import StagingWriteWaveMiddleware


def test_real_sdk_chains_count_success_and_error_without_retaining_query_data(monkeypatch):
    # The fixture uses only MockTransport; don't inherit a workstation proxy.
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    requests = []
    def transport(request):
        requests.append(request)
        if request.url.path.endswith("/rpc/fail"):
            return httpx.Response(500, json={"message": "private failure", "code": "XX000"})
        return httpx.Response(200, json=[{"id": 1}])

    raw = create_client("https://example.invalid", "fixture-key")
    raw._postgrest = SyncPostgrestClient("https://example.invalid/rest/v1", retry_enabled=False,
        http_client=httpx.Client(transport=httpx.MockTransport(transport)))
    assert instrument_supabase_client(raw) is raw  # Background clients unchanged.
    timing = RequestTiming("sdk-test")
    token = current_timing.set(timing)
    try:
        client = instrument_supabase_client(raw)
        assert instrument_supabase_client(client) is client
        assert client.supabase_url == raw.supabase_url
        result = client.table("players").select("id").not_.is_("id", "null").eq("name", "PRIVATE_PLAYER").range(0, 4).execute()
        assert result.data == [{"id": 1}]
        assert client.from_("players").select("id").execute().data == result.data
        assert client.rpc("success", {"secret": "PRIVATE_PAYLOAD"}).execute().data == result.data
        with pytest.raises(Exception):
            client.rpc("fail", {}).execute()
    finally:
        current_timing.reset(token)
        raw._postgrest.aclose()
    snapshot = timing.snapshot()
    assert snapshot["db_calls"] == 4
    assert snapshot["db_errors"] == 1
    assert snapshot["db_elapsed_ms"] >= 0
    assert snapshot["db_thread_cpu_ms"] >= 0
    assert "PRIVATE" not in json.dumps(snapshot)
    assert len(requests) == 4


def test_concurrent_requests_keep_metrics_separate_across_sync_threads(caplog):
    app = FastAPI()
    app.add_middleware(StructuredRequestLoggingMiddleware)
    barrier = threading.Barrier(2, timeout=5)

    class Query:
        def execute(self):
            return SimpleNamespace(data=[])
    class Client:
        def table(self, name):
            return Query()

    @app.get("/players/{player_id}")
    def player(player_id: str, calls: int):
        barrier.wait()
        db = instrument_supabase_client(Client())
        for _ in range(calls):
            db.table("players").execute()
        with measured_phase("leaderboard_calculation"):
            sum(n * n for n in range(1000))
        with measured_phase(player_id):
            pass  # Untrusted/high-cardinality phase labels are ignored.
        return {"ok": True}

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            replies = await asyncio.gather(*[
                client.get(f"/players/PRIVATE_{n}?calls={n}", headers={"x-request-id": f"req-{n}"})
                for n in (1, 3)
            ])
            assert [r.status_code for r in replies] == [200, 200]
        assert current_timing.get() is None

    caplog.set_level(logging.INFO, logger="jupr.api.request")
    asyncio.run(exercise())
    records = {r.request_id: r for r in caplog.records if r.name == "jupr.api.request"}
    assert records["req-1"].db_calls == 1
    assert records["req-3"].db_calls == 3
    for record in records.values():
        assert record.path == "/players/{player_id}"
        assert record.phases["leaderboard_calculation"]["calls"] == 1
        assert "PRIVATE" not in RequestJSONFormatter().format(record)


def test_measurement_failure_preserves_original_outcome(monkeypatch):
    timing = RequestTiming("measurement-failure")
    def broken(*args):
        raise RuntimeError("telemetry unavailable")
    monkeypatch.setattr(timing, "add_db", broken)
    monkeypatch.setattr(timing, "add_phase", broken)
    token = current_timing.set(timing)
    try:
        class Query:
            def execute(self):
                raise ValueError("original error")
        client = instrument_supabase_client(SimpleNamespace(table=lambda _: Query()))
        with pytest.raises(ValueError, match="original error"):
            with measured_phase("badge_calculation"):
                client.table("badges").execute()
        with measured_phase("badge_calculation"):
            result = 42
        assert result == 42
    finally:
        current_timing.reset(token)


def test_formatter_keeps_request_and_worker_metrics_but_not_payload():
    record = logging.makeLogRecord({"msg": "badge.job.complete", "levelname": "INFO", "job_id": "job-1",
        "worker_id": "worker-1", "club_id": "test_club", "phase": "evaluation", "cpu_ms": 1.5,
        "request_id": "request-1", "db_calls": 3, "db_elapsed_ms": 22,
        "request_body": "PRIVATE_PAYLOAD", "authorization": "PRIVATE_TOKEN"})
    payload = json.loads(RequestJSONFormatter().format(record))
    assert payload["job_id"] == "job-1"
    assert payload["db_calls"] == 3
    assert "PRIVATE" not in json.dumps(payload)


def test_broken_log_sink_does_not_change_success_or_exception(monkeypatch):
    app = FastAPI()
    app.add_middleware(StructuredRequestLoggingMiddleware)

    @app.get("/ok")
    def ok():
        return {"ok": True}

    @app.get("/failed")
    def failed():
        raise ValueError("original error")

    def broken(*args, **kwargs):
        raise RuntimeError("log sink unavailable")
    monkeypatch.setattr("services.api.middleware.logger.info", broken)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            assert (await client.get("/ok")).json() == {"ok": True}
            with pytest.raises(ValueError, match="original error"):
                await client.get("/failed")
        assert current_timing.get() is None
    asyncio.run(exercise())


def test_adapter_does_not_invent_unsupported_rpc_capability():
    raw = SimpleNamespace(rpc=None)
    token = current_timing.set(RequestTiming("no-rpc"))
    try:
        assert instrument_supabase_client(raw).rpc is None
    finally:
        current_timing.reset(token)


def test_early_denials_log_registered_template_without_private_values(monkeypatch, caplog):
    app = FastAPI()
    app.add_middleware(StagingWriteWaveMiddleware)
    app.add_middleware(StructuredRequestLoggingMiddleware)

    @app.post("/players/{player_id}")
    def update_player(player_id: str):
        pytest.fail("Denied requests must not execute the endpoint")

    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("JUPR_PRODUCTION_WRITE_POLICY", "read_only")
    caplog.set_level(logging.INFO, logger="jupr.api.request")

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            for path in ("/players/PRIVATE_NAME?token=PRIVATE_SECRET", "/unknown/PRIVATE_CAPABILITY"):
                assert (await client.post(path)).status_code == 403
    asyncio.run(exercise())
    records = [r for r in caplog.records if r.name == "jupr.api.request"]
    assert [r.path for r in records] == ["/players/{player_id}", "<unmatched>"]
    assert all("PRIVATE" not in RequestJSONFormatter().format(r) for r in records)
