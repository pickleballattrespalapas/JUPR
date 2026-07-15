from __future__ import annotations

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_api_contract_admin_tournament import _install_auth

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def draft_tournament_tables():
    return {
        "tournaments": [
            {"club_id": "club", "id": "draft_1", "name": "Empty Draft", "status": "DRAFT", "created_at": "2026-03-01T00:00:00Z"}
        ],
        "tournament_registration_settings": [{"id": "regset_draft", "tournament_id": "draft_1", "registration_slug": "empty-draft"}],
        "tournament_registration_days": [{"id": "day_draft", "tournament_id": "draft_1", "label": "Friday", "sort_order": 1}],
        "tournament_event_options": [{"id": "event_draft", "tournament_id": "draft_1", "registration_day_id": "day_draft", "division_name": "3.5"}],
        "tournament_registrations": [],
        "tournament_registration_selections": [],
        "tournament_event_draws": [],
        "tournament_teams": [],
        "tournament_games": [],
        "tournament_podium": [],
        "admin_activity_log": [],
    }


def test_admin_tournament_delete_empty_draft_contract(monkeypatch):
    tables = draft_tournament_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/draft_1/delete-draft",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "DELETE DRAFT"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_draft_deleted"
    assert payload["tournament_id"] == "draft_1"
    assert tables["tournaments"] == []
    assert tables["tournament_registration_settings"] == []
    assert tables["tournament_event_options"] == []
    assert tables["tournament_registration_days"] == []
    assert tables["admin_activity_log"][0]["action_type"] == "delete_draft_tournament_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True
