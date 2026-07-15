from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def tournament_award_tables():
    return {
        "tournaments": [
            {"club_id": "club", "id": "tour_1", "name": "Spring Classic", "status": "PUBLISHED"}
        ],
        "tournament_event_draws": [
            {"id": "draw_1", "tournament_id": "tour_1", "name": "3.5 Draw", "event_option_id": "event_1"}
        ],
        "tournament_teams": [
            {"id": "team_1", "tournament_id": "tour_1", "draw_id": "draw_1", "event_option_id": "event_1", "team_number": 1, "player1_id": 1, "player2_id": 2},
            {"id": "team_2", "tournament_id": "tour_1", "draw_id": "draw_1", "event_option_id": "event_1", "team_number": 2, "player1_id": 3, "player2_id": 4},
            {"id": "team_3", "tournament_id": "tour_1", "draw_id": "draw_1", "event_option_id": "event_1", "team_number": 3, "player1_id": 5, "player2_id": 6},
        ],
        "tournament_podium": [
            {"id": "pod_1", "tournament_id": "tour_1", "draw_id": "draw_1", "placement": 1, "team_id": "team_1", "source": "PLAYOFF"},
            {"id": "pod_2", "tournament_id": "tour_1", "draw_id": "draw_1", "placement": 2, "team_id": "team_2", "source": "PLAYOFF"},
            {"id": "pod_3", "tournament_id": "tour_1", "draw_id": "draw_1", "placement": 3, "team_id": "team_3", "source": "PLAYOFF"},
        ],
        "admin_activity_log": [],
    }


def _install_auth(monkeypatch):
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_admin_tournament_draw_podium_awards_contract(monkeypatch):
    tables = tournament_award_tables()
    supabase = FakeSupabase(tables)
    awarded_candidates = []

    def fake_upsert_player_badges(_supabase, club_id, candidates, **_kwargs):
        assert club_id == "club"
        awarded_candidates.extend(list(candidates))
        return list(candidates)

    monkeypatch.setattr("jupr_app.domain.gamification.badges_repo.upsert_player_badges", fake_upsert_player_badges)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/podium/awards",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "AWARD PODIUM"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_draw_podium_award"
    assert payload["candidate_count"] == 6
    assert payload["awarded_count"] == 6
    assert len(awarded_candidates) == 6
    assert {candidate.badge_id for candidate in awarded_candidates} == {
        "tournament_champion",
        "tournament_runner_up",
        "tournament_third_place",
    }
    assert all(":draw:draw_1:podium:" in str(candidate.context_id) for candidate in awarded_candidates)
    assert tables["admin_activity_log"][0]["action_type"] == "award_tournament_draw_podium_badges_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_tournament_draw_podium_awards_requires_podium(monkeypatch):
    tables = tournament_award_tables()
    tables["tournament_podium"] = []
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/podium/awards",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "AWARD PODIUM"},
    )

    assert response.status_code == 400
    assert "Generate a draw-scoped podium" in response.json()["detail"]


def test_admin_tournament_draw_podium_awards_requires_confirmation(monkeypatch):
    tables = tournament_award_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/podium/awards",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "AWARD"},
    )

    assert response.status_code == 400
    assert "AWARD PODIUM" in response.json()["detail"]
