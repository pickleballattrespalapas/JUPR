from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from jupr_app.services.admin_weekly_recap_service import apply_weekly_recap_edits
from services.api.main import app


def weekly_recap_tables() -> dict[str, list[dict]]:
    return {
        "weekly_recaps": [
            {
                "id": "recap-1",
                "club_id": "club",
                "week_start": "2026-07-06",
                "week_end": "2026-07-12",
                "status": "draft",
                "generated_json": {"numbers": {"matches": 4}, "spotlight": []},
                "edits_json": {},
                "final_json": {"numbers": {"matches": 4}, "spotlight": []},
            }
        ],
        "admin_activity_log": [],
    }


def _install_env(monkeypatch, supabase) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_weekly_recap_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_weekly_recap_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_admin_weekly_recap_status_disabled_contract(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP", raising=False)
    response = TestClient(app).get("/admin/clubs/club/weekly-recap/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["list_endpoint"] is None


def test_admin_weekly_recap_list_contract(monkeypatch):
    supabase = FakeSupabase(weekly_recap_tables())
    _install_env(monkeypatch, supabase)

    response = TestClient(app).get(
        "/admin/clubs/club/weekly-recap/recaps",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["count"] == 1
    assert payload["recaps"][0]["week_start"] == "2026-07-06"


def test_admin_weekly_recap_generate_requires_confirmation(monkeypatch):
    supabase = FakeSupabase(weekly_recap_tables())
    _install_env(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/weekly-recap/generate",
        headers={"Authorization": "Bearer local"},
        json={"week_start": "2026-07-06", "week_end": "2026-07-12", "confirmation_text": "GENERATE"},
    )

    assert response.status_code == 400
    assert "GENERATE RECAP" in response.json()["detail"]


def test_admin_weekly_recap_save_requires_confirmation(monkeypatch):
    supabase = FakeSupabase(weekly_recap_tables())
    _install_env(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/weekly-recap/recaps/2026-07-06",
        headers={"Authorization": "Bearer local"},
        json={"edits_json": {}, "confirmation_text": "SAVE"},
    )

    assert response.status_code == 400
    assert "SAVE RECAP" in response.json()["detail"]


def test_apply_weekly_recap_edits_uses_spotlight_candidates():
    generated = {"spotlight": [{"key": "TOP_PERFORMER_WEEK", "order": 1, "include": True}], "looking_ahead": []}
    edits = {
        "looking_ahead": ["League finals", "Tournament signup"],
        "spotlight_overrides": {
            "TOP_PERFORMER_WEEK": {
                "players": ["cand-1"],
                "description": "Best range performance.",
                "order": 1,
                "include": True,
            }
        },
    }
    candidates = {
        "TOP_PERFORMER_WEEK": [
            {"candidate_id": "cand-1", "label": "Top Performer", "display": "Alex +0.120", "player_ids": [1]}
        ]
    }

    final = apply_weekly_recap_edits(generated, edits, candidates)

    assert final["looking_ahead"] == ["League finals", "Tournament signup"]
    assert final["spotlight"][0]["players"] == ["Alex +0.120"]
    assert final["spotlight"][0]["description"] == "Best range performance."
