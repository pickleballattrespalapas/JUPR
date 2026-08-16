from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_admin_tournament_podium_review_service import podium_review_tables

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from jupr_app.services.admin_tournament_ops_service import get_admin_tournament_ops_state_fingerprint
from services.api.main import app


def _versions(rows: list[dict]) -> list[dict[str, str]]:
    return sorted(
        [{"id": str(row["id"]), "updated_at": str(row["updated_at"])} for row in rows],
        key=lambda row: row["id"],
    )


def _install(monkeypatch, supabase: FakeSupabase) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="director@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_admin_podium_review_route_records_exact_current_evidence(monkeypatch) -> None:
    tables = podium_review_tables()
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)
    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour-1/draws/draw-1/podium/review",
        headers={"Authorization": "Bearer local"},
        json={
            "expected_state_fingerprint": get_admin_tournament_ops_state_fingerprint(
                supabase,
                club_id="club",
                tournament_id="tour-1",
            ),
            "expected_draw_updated_at": tables["tournament_event_draws"][0]["updated_at"],
            "expected_team_versions": _versions(tables["tournament_teams"]),
            "expected_source_game_versions": _versions(tables["tournament_games"]),
            "confirmation_text": "REVIEW PODIUM",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["mode"] == "tournament_draw_podium_review"
    assert payload["reviewed"] is True
    assert len(payload["review_fingerprint"]) == 64


def test_admin_podium_review_route_rejects_stale_fingerprint_without_audit(monkeypatch) -> None:
    tables = podium_review_tables()
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)
    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour-1/draws/draw-1/podium/review",
        headers={"Authorization": "Bearer local"},
        json={
            "expected_state_fingerprint": "0" * 64,
            "expected_draw_updated_at": tables["tournament_event_draws"][0]["updated_at"],
            "expected_team_versions": _versions(tables["tournament_teams"]),
            "expected_source_game_versions": _versions(tables["tournament_games"]),
            "confirmation_text": "REVIEW PODIUM",
        },
    )

    assert response.status_code == 409
    assert tables["admin_activity_log"] == []

