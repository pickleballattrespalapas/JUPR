"""The shipped disabled policy prevents work while preserving formal awards."""
from types import SimpleNamespace
import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jupr_app.domain.gamification import badge_policy
from jupr_app.domain.gamification.badge_queue import enqueue_badge_eval, dequeue_badge_eval
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue, process_badge_eval_queue_until_empty
from jupr_app.domain.gamification.program_badge_service import process_pending_program_badges, reconcile_program_badges
from jupr_app.domain.gamification.live_awards import run_live_badge_awards
from jupr_app.domain.gamification.recompute import run_badge_recompute
from jupr_app.domain.matches.side_effects import run_badge_side_effects, ensure_notification_badge_job
from jupr_app.services.leaderboard_service import _badge_map
from jupr_app.workers.badge_queue_worker import run_badge_queue_worker
from services.api.public_badge_codex_routes import install_public_badge_codex_routes
from services.api.admin_badge_diagnostics_routes import install_admin_badge_diagnostics_routes
from services.api.program_badge_worker import install_program_badge_worker


class NoDatabase:
    def __getattr__(self, name):
        raise AssertionError(f"Retired badges accessed database: {name}")


def no_database(*args, **kwargs):
    raise AssertionError("Retired badges requested a database client")


def test_badge_retirement_cannot_be_overridden_by_old_deployment_flags(monkeypatch):
    for name, value in {"JUPR_ENV": "staging", "JUPR_STAGING_WRITE_WAVE": "open",
                        "JUPR_BADGE_WORKER_MODE": "external", "JUPR_BADGE_QUEUE_LEASES": "1",
                        "JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS": "1"}.items():
        monkeypatch.setenv(name, value)
    assert badge_policy.badges_enabled() is False
    db = NoDatabase()
    assert enqueue_badge_eval(db, club_id="tres_palapas", event_type="match_recorded")["reason"] == "badges_retired"
    assert dequeue_badge_eval(db, club_id="tres_palapas") is None
    assert process_badge_eval_queue(db, "tres_palapas") == {"processed": 0, "errored": 0}
    assert process_badge_eval_queue_until_empty(db, "tres_palapas")["stopped_reason"] == "badges_retired"
    assert process_pending_program_badges(db) == {"processed": 0, "errors": 0}
    assert run_badge_queue_worker("tres_palapas")["stopped_reason"] == "badges_retired"
    assert run_live_badge_awards(db, club_id="tres_palapas", player_ids=[1], event_type="match_recorded")["mode"] == "disabled"
    with pytest.raises(PermissionError, match="retired"):
        reconcile_program_badges(db, "tres_palapas")
    with pytest.raises(PermissionError, match="retired"):
        run_badge_recompute(db, club_id="tres_palapas")


def test_match_side_effects_and_visible_leaderboard_do_no_badge_work():
    db = NoDatabase()
    assert ensure_notification_badge_job(supabase=db, club_id="club", match_ids=[1], player_ids=[1]) == ""
    result = run_badge_side_effects(supabase=db, club_id="club", has_badge_eligible_match=True,
                                  affected_players={1}, db_matches=[{"id": 1}], match_payloads=[])
    assert result["mode"] == "disabled"
    assert not result.get("notification_badge_refs")
    assert _badge_map(db, club_id="club", player_ids=("1",)) == {}


def test_badge_public_routes_hidden_and_admin_status_retired_before_client(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS", "1")
    app = FastAPI()
    install_public_badge_codex_routes(app, get_club=no_database, get_supabase_client=no_database, public_club_payload=no_database)
    install_admin_badge_diagnostics_routes(app, get_supabase_client=no_database)
    client = TestClient(app)
    assert client.get("/clubs/tres-palapas/badges").status_code == 404
    assert client.get("/clubs/tres-palapas/badges/first_win/earners").status_code == 404
    status = client.get("/admin/clubs/tres_palapas/badges/status")
    assert status.status_code == 200 and status.json()["status"] == "retired"
    assert status.json()["enabled"] is False
    assert client.get("/admin/clubs/tres_palapas/badges/options").status_code == 403


def test_embedded_worker_does_not_start_when_badges_retired(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("FLY_APP_NAME", "juprleagues-api-staging")
    app = FastAPI()
    install_program_badge_worker(app, get_supabase_client=no_database)
    for handler in app.router.on_startup:
        asyncio.run(handler())
    assert not hasattr(app.state, "program_badge_worker")


def test_only_top_three_formal_trophies_can_pass_league_close(monkeypatch):
    import importlib
    service = importlib.import_module("jupr_app.domain.gamification.ensure_badges")
    from jupr_app.domain.gamification.badge_types import BadgeCandidate
    def candidate(badge, rank):
        return BadgeCandidate(badge_id=badge, player_id=1, club_id="club", context_type="league",
                              context_id=f"league:top_performer:most_wins:{rank}",
                              match_id=None, value_json={"rank": rank}, value_num=66)
    winners = [candidate("top_performer_most_wins", 1), candidate("top_performer_most_wins", 3)]
    all_candidates = [*winners, candidate("top_performer_most_wins", 27), candidate("first_win", 1)]
    monkeypatch.setattr(service, "compute_candidates_for_club", lambda **kwargs: all_candidates)
    saved=[]
    monkeypatch.setattr(service, "upsert_player_badges", lambda db, club, rows: saved.extend(rows) or rows)
    ctx = SimpleNamespace(supabase=NoDatabase(), club_id="club")
    assert service.ensure_badges(ctx) == []
    assert service.ensure_badges(ctx, league_id="league", status="seasonal", award_timing="on_league_close") == winners
    assert saved == winners


def test_weekly_recap_omits_retired_achievements_before_reading():
    from jupr_app.domain.recaps.weekly_report import _achievements
    assert _achievements(SimpleNamespace(supabase=NoDatabase(), club_id="club"), [{"id": 1}]) == []


def test_admin_worker_controls_stay_retired_before_database(monkeypatch):
    from services.api.admin_tools_routes import install_admin_tools_routes
    from jupr_app.services.admin_tools_service import build_admin_worker_status
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    app = FastAPI()
    install_admin_tools_routes(app, get_supabase_client=no_database)
    client = TestClient(app)
    for path in ("badge-queue", "badge-recompute"):
        assert client.post(f"/admin/clubs/club/tools/workers/{path}", json={}).status_code == 403
    assert build_admin_worker_status(NoDatabase(), club_id="club")["status"] == "retired"


def test_production_direct_and_embedded_worker_stay_retired(monkeypatch):
    from services.api.program_badge_worker import check_pending_badges
    from jupr_app.services.production_feature_policy import production_feature_enabled

    for name, value in {"JUPR_ENV": "production", "FLY_APP_NAME": "juprleagues-api",
                        "JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS": "1",
                        "JUPR_PRODUCTION_WRITE_POLICY": "enabled",
                        "JUPR_STAGING_WRITE_WAVE": "none"}.items():
        monkeypatch.setenv(name, value)
    assert production_feature_enabled("badges", "tres_palapas") is False
    check_pending_badges(NoDatabase())
    app = FastAPI()
    install_program_badge_worker(app, get_supabase_client=no_database)
    for handler in app.router.on_startup:
        asyncio.run(handler())
    assert not hasattr(app.state, "program_badge_worker")
