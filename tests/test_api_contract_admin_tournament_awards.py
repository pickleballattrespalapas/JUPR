from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app
from jupr_app.domain.tournament_podium import PODIUM_BADGE_MAP
from jupr_app.services.admin_tournament_ops_service import (
    get_admin_tournament_ops_state_fingerprint,
)
from jupr_app.services.admin_tournament_podium_review_service import (
    PODIUM_REVIEW_ACTION,
    PODIUM_REVIEW_CONTRACT,
    build_admin_tournament_podium_review_fingerprint,
)


def tournament_award_tables():
    return {
        "tournaments": [
            {"club_id": "club", "id": "tour_1", "name": "Spring Classic", "status": "PUBLISHED"}
        ],
        "tournament_event_draws": [
            {"id": "draw_1", "tournament_id": "tour_1", "name": "3.5 Draw", "event_option_id": "event_1", "updated_at": "2026-04-10T17:00:00Z"}
        ],
        "tournament_teams": [
            {"id": "team_1", "tournament_id": "tour_1", "draw_id": "draw_1", "event_option_id": "event_1", "team_number": 1, "player1_id": 1, "player2_id": 2, "updated_at": "2026-04-10T17:00:00Z"},
            {"id": "team_2", "tournament_id": "tour_1", "draw_id": "draw_1", "event_option_id": "event_1", "team_number": 2, "player1_id": 3, "player2_id": 4, "updated_at": "2026-04-10T17:00:00Z"},
            {"id": "team_3", "tournament_id": "tour_1", "draw_id": "draw_1", "event_option_id": "event_1", "team_number": 3, "player1_id": 5, "player2_id": 6, "updated_at": "2026-04-10T17:00:00Z"},
        ],
        "tournament_games": [
            {"id": "game_1", "tournament_id": "tour_1", "draw_id": "draw_1", "stage": "ROUND_ROBIN", "team_a_id": "team_1", "team_b_id": "team_2", "score_a": 11, "score_b": 7, "winner_team_id": "team_1", "loser_team_id": "team_2", "finalized_at": "2026-04-10T17:00:00Z", "updated_at": "2026-04-10T17:00:00Z"},
        ],
        "tournament_podium": [
            {"id": "pod_1", "tournament_id": "tour_1", "draw_id": "draw_1", "placement": 1, "team_id": "team_1", "source": "PLAYOFF", "updated_at": "2026-04-10T17:00:00Z"},
            {"id": "pod_2", "tournament_id": "tour_1", "draw_id": "draw_1", "placement": 2, "team_id": "team_2", "source": "PLAYOFF", "updated_at": "2026-04-10T17:00:00Z"},
            {"id": "pod_3", "tournament_id": "tour_1", "draw_id": "draw_1", "placement": 3, "team_id": "team_3", "source": "PLAYOFF", "updated_at": "2026-04-10T17:00:00Z"},
        ],
        "admin_activity_log": [],
    }


def install_current_podium_review(tables: dict[str, list[dict]]) -> None:
    fingerprint = build_admin_tournament_podium_review_fingerprint(
        draw=tables["tournament_event_draws"][0],
        teams=tables["tournament_teams"],
        games=tables["tournament_games"],
        podium=tables["tournament_podium"],
    )
    tables["admin_activity_log"] = [
        {
            "club_id": "club",
            "actor_email": "reviewer@example.com",
            "action_type": PODIUM_REVIEW_ACTION,
            "entity_type": "tournament_event_draw",
            "entity_id": "draw_1",
            "after_json": {
                "podium_review_evidence": {
                    "contract": PODIUM_REVIEW_CONTRACT,
                    "tournament_id": "tour_1",
                    "draw_id": "draw_1",
                    "review_fingerprint": fingerprint,
                }
            },
        }
    ]


def _versions(rows: list[dict]) -> list[dict[str, str]]:
    return sorted(
        [
            {"id": str(row["id"]), "updated_at": str(row["updated_at"])}
            for row in rows
        ],
        key=lambda row: row["id"],
    )


def _reviewed_version_payload(tables: dict[str, list[dict]]) -> dict[str, object]:
    return {
        "expected_draw_updated_at": tables["tournament_event_draws"][0]["updated_at"],
        "expected_team_versions": _versions(tables["tournament_teams"]),
        "expected_source_game_versions": _versions(tables["tournament_games"]),
        "expected_podium_versions": _versions(tables["tournament_podium"]),
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


def test_admin_tournament_draw_podium_awards_requires_exact_atomic_review_plan(monkeypatch):
    tables = tournament_award_tables()
    install_current_podium_review(tables)
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

    assert response.status_code == 409
    assert "reviewed draw version is required" in response.json()["detail"]
    assert awarded_candidates == []
    assert not any(
        row.get("action_type") == "award_tournament_draw_podium_badges_admin"
        for row in tables["admin_activity_log"]
    )


def test_guarded_awards_route_forwards_exact_plan_to_atomic_service(monkeypatch):
    tables = tournament_award_tables()
    for row in tables["tournament_podium"]:
        row["updated_at"] = "2026-04-10T17:00:00Z"
    install_current_podium_review(tables)
    tables["tournament_admin_operations"] = []
    tables["player_badges"] = []
    supabase = FakeSupabase(tables)
    expected_podium = [
        {
            "placement": int(row["placement"]),
            "team_id": str(row["team_id"]),
            "source": str(row["source"]),
        }
        for row in tables["tournament_podium"]
    ]
    teams_by_id = {row["id"]: row for row in tables["tournament_teams"]}
    expected_awards = sorted(
        [
            {
                "player_id": int(player_id),
                "badge_id": PODIUM_BADGE_MAP[int(podium["placement"])],
                "context_id": f"tour_1:draw:draw_1:podium:{podium['placement']}",
            }
            for podium in tables["tournament_podium"]
            for player_id in (
                teams_by_id[podium["team_id"]]["player1_id"],
                teams_by_id[podium["team_id"]]["player2_id"],
            )
        ],
        key=lambda row: (row["context_id"], row["badge_id"], row["player_id"]),
    )
    expected_state = get_admin_tournament_ops_state_fingerprint(
        supabase,
        club_id="club",
        tournament_id="tour_1",
    )
    calls: list[dict] = []

    def fake_award(_supabase, **kwargs):
        calls.append(dict(kwargs))
        return {
            "ok": True,
            "mode": "tournament_draw_podium_award_preview"
            if kwargs.get("dry_run")
            else "tournament_draw_podium_award",
            "draw_id": "draw_1",
            "candidate_count": 6,
            "awarded_count": 0 if kwargs.get("dry_run") else 6,
            "warnings": [],
        }

    monkeypatch.setattr(
        "services.api.admin_tournament_routes.award_admin_tournament_draw_podium",
        fake_award,
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.get_admin_tournament_ops_state_fingerprint",
        lambda *_args, **_kwargs: expected_state,
    )
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "tournament-operations")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT", "1")
    monkeypatch.setenv(
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS",
        "1",
    )
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/podium/awards",
        headers={"Authorization": "Bearer local"},
        json={
            "expected_state_fingerprint": expected_state,
            "expected_draw_updated_at": tables["tournament_event_draws"][0]["updated_at"],
            "expected_team_versions": _versions(tables["tournament_teams"]),
            "expected_source_game_versions": _versions(tables["tournament_games"]),
            "expected_podium_versions": _versions(tables["tournament_podium"]),
            "expected_podium": expected_podium,
            "expected_awards": expected_awards,
            "confirmation_text": "AWARD PODIUM",
        },
    )

    assert response.status_code == 200, response.text
    assert len(calls) == 2
    for call in calls:
        assert call["atomic"] is True
        assert call["expected_draw_updated_at"] == "2026-04-10T17:00:00Z"
        assert call["expected_team_versions"] == _versions(tables["tournament_teams"])
        assert call["expected_source_game_versions"] == _versions(tables["tournament_games"])
        assert call["expected_podium_versions"] == _versions(tables["tournament_podium"])
        assert call["expected_podium"] == expected_podium
        assert call["expected_awards"] == expected_awards


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
        json={"confirmation_text": "AWARD PODIUM", **_reviewed_version_payload(tables)},
    )

    assert response.status_code == 400
    assert "Generate a draw-scoped podium" in response.json()["detail"]


def test_admin_tournament_draw_podium_awards_requires_current_explicit_review(monkeypatch):
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
        json={"confirmation_text": "AWARD PODIUM", **_reviewed_version_payload(tables)},
    )
    assert response.status_code == 400
    assert "not been explicitly reviewed" in response.json()["detail"]

    install_current_podium_review(tables)
    tables["tournament_podium"][0]["team_id"] = "team_3"
    stale = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/podium/awards",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "AWARD PODIUM", **_reviewed_version_payload(tables)},
    )
    assert stale.status_code == 400
    assert "changed after review" in stale.json()["detail"]


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
