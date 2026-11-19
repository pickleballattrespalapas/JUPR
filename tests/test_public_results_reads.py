import asyncio
from threading import Event

import anyio.to_thread
import httpx
import pytest
from fastapi import FastAPI, Header, HTTPException

from services.api.public_alias_middleware import install_public_alias_middleware
from services.api.public_league_results_routes import install_public_league_results_routes
from services.api.public_results_reads import PublicResultsReads
from services.api.public_tournament_results_routes import install_public_tournament_results_routes


async def wait_for(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


def test_overlapping_visitors_share_work_but_next_read_gets_fresh_scores():
    async def scenario():
        reads = PublicResultsReads(max_active=2)
        release = Event()
        calls = []

        def load():
            calls.append(1)
            assert release.wait(3)
            return {"scores": [len(calls)]}

        visitors = [asyncio.create_task(reads.run("same", load)) for _ in range(48)]
        try:
            await wait_for(lambda: reads._waiters == 48 and calls)
            assert len(calls) == 1
            visitors[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await visitors[0]
            assert len(reads._pending) == 1
        finally:
            release.set()
        results = await asyncio.gather(*visitors[1:])
        results[0]["scores"].append("changed by caller")
        assert all(result == {"scores": [1]} for result in results[1:])
        assert await reads.run("same", load) == {"scores": [2]}
        assert not reads._pending
        assert reads._waiters == 0

    asyncio.run(scenario())


def test_active_work_and_waiting_visitors_are_bounded_before_thread_admission():
    async def scenario():
        reads = PublicResultsReads(max_active=2, max_waiters=3)
        release = Event()
        calls = []

        def load():
            calls.append(1)
            assert release.wait(3)
            return "done"

        visitors = [asyncio.create_task(reads.run(key, load)) for key in ("a", "a", "b")]
        try:
            await wait_for(lambda: reads._waiters == 3 and len(calls) == 2)
            with pytest.raises(HTTPException) as full:
                await reads.run("a", load)
            assert full.value.status_code == 503
            assert full.value.headers == {"Retry-After": "5", "Cache-Control": "no-store"}
            visitors[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await visitors[0]
            with pytest.raises(HTTPException):
                await reads.run("different club or filters", load)
            assert len(calls) == 2
        finally:
            release.set()
        assert await asyncio.gather(*visitors[1:]) == ["done", "done"]
        assert await reads.run("different club or filters", lambda: "fresh") == "fresh"

    asyncio.run(scenario())


def test_failure_is_shared_only_while_running_and_retry_can_recover():
    async def scenario():
        reads = PublicResultsReads(max_active=1)
        release = Event()
        calls = []

        def fail():
            calls.append(1)
            assert release.wait(3)
            raise RuntimeError("upstream unavailable")

        visitors = [asyncio.create_task(reads.run("same", fail)) for _ in range(8)]
        try:
            await wait_for(lambda: reads._waiters == 8 and calls)
        finally:
            release.set()
        errors = await asyncio.gather(*visitors, return_exceptions=True)
        assert len(calls) == 1
        assert all(isinstance(error, RuntimeError) for error in errors)
        assert await reads.run("same", lambda: "recovered") == "recovered"

    asyncio.run(scenario())


def test_disconnected_only_visitor_does_not_release_running_worker():
    async def scenario():
        reads = PublicResultsReads(max_active=1)
        started, release = Event(), Event()

        def load():
            started.set()
            assert release.wait(3)
            raise RuntimeError("failed after disconnect")

        visitor = asyncio.create_task(reads.run("same", load))
        try:
            await wait_for(started.is_set)
            visitor.cancel()
            with pytest.raises(asyncio.CancelledError):
                await visitor
            with pytest.raises(HTTPException):
                await reads.run("other", lambda: "must wait")
        finally:
            release.set()
        await wait_for(lambda: not reads._pending)
        assert await reads.run("other", lambda: "recovered") == "recovered"

    asyncio.run(scenario())


def test_public_pressure_leaves_staff_and_tournament_reads_available(monkeypatch):
    """Exercise real public route registration and privacy with a tiny worker pool."""
    monkeypatch.setenv("JUPR_ENABLE_PUBLIC_PLAYER_ALIASES", "true")
    release = Event()
    loads, aliases, writes = [], [], []

    def results(_db, *, club_id, league_name, **kwargs):
        loads.append((club_id, league_name))
        assert release.wait(3)
        return {"name": "Canonical Player", "score": 11, "league": league_name}

    monkeypatch.setattr("services.api.public_league_results_routes.build_public_league_results", results)
    monkeypatch.setattr("services.api.public_tournament_results_routes.build_public_tournament_results",
                        lambda _db, **kwargs: {"score": 11, "tournament_id": kwargs["tournament_id"]})

    def load_aliases(_db, club_id):
        aliases.append(club_id)
        return {"Canonical Player": "Public Alias"}

    monkeypatch.setattr("services.api.public_alias_middleware.load_player_aliases", load_aliases)
    app = FastAPI()
    dependencies = dict(get_club=lambda slug: {"id": slug}, get_supabase_client=lambda: object(),
                        public_club_payload=lambda club, slug: club)
    install_public_league_results_routes(app, **dependencies)
    install_public_tournament_results_routes(app, **dependencies)
    install_public_alias_middleware(app, get_club=dependencies["get_club"],
                                    get_supabase_client=dependencies["get_supabase_client"])

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/admin/score")
    def score(authorization: str | None = Header(default=None)):
        if authorization != "Bearer fixture-staff":
            raise HTTPException(401)
        writes.append(11)
        return {"saved": 11}

    async def scenario():
        limiter = anyio.to_thread.current_default_thread_limiter()
        original_tokens = limiter.total_tokens
        limiter.total_tokens = 5
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://fixture") as client:
            visitors = [asyncio.create_task(client.get(
                f"/clubs/alpha/league-results?league_name=league-{index % 4}")) for index in range(48)]
            try:
                await wait_for(lambda: len(loads) == 4 and app.state.public_results_reads["statistics"]._waiters == 48)
                async with asyncio.timeout(2):
                    assert (await client.get("/health")).json() == {"ok": True}
                    assert (await client.post("/admin/score")).status_code == 401
                    saved = await client.post("/admin/score", headers={"Authorization": "Bearer fixture-staff"})
                    assert saved.json() == {"saved": 11}
                    tournament = await client.get("/clubs/alpha/tournament-results?tournament_id=live")
                    assert tournament.json()["score"] == 11
                    for path in ("/clubs/beta/league-results?league_name=league-0",
                                 "/clubs/alpha/league-results?league_name=different"):
                        busy = await client.get(path)
                        assert busy.status_code == 503
                        assert busy.headers["retry-after"] == "5"
                    assert (await client.get("/clubs/alpha/league-results?week=0")).status_code == 422
                assert writes == [11]
            finally:
                release.set()
                responses = await asyncio.gather(*visitors)
                limiter.total_tokens = original_tokens
            assert len(loads) == 4
            assert len(aliases) == 1
            assert all(response.status_code == 200 and response.json()["name"] == "Public Alias"
                       for response in responses)
            assert {response.json()["league"] for response in responses} == {f"league-{i}" for i in range(4)}
            fresh = await client.get("/clubs/alpha/league-results?league_name=league-0")
            assert fresh.status_code == 200
            assert len(loads) == 5

    asyncio.run(scenario())
