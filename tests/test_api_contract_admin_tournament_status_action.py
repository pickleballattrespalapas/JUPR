from __future__ import annotations

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_api_contract_admin_tournament import _install_auth, tournament_tables

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def test_admin_tournament_archive_and_unarchive_contract(monkeypatch):
    tables = tournament_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    client = TestClient(app)
    archive_response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/status-action",
        headers={"Authorization": "Bearer local"},
        json={"action": "archive", "confirmation_text": "ARCHIVE"},
    )

    assert archive_response.status_code == 200
    archived = archive_response.json()
    assert archived["ok"] is True
    assert archived["mode"] == "tournament_status_action"
    assert archived["action"] == "archive"
    assert archived["tournament"]["status"] == "ARCHIVED"
    assert tables["tournaments"][0]["status"] == "ARCHIVED"

    unarchive_response = client.patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/status-action",
        headers={"Authorization": "Bearer local"},
        json={"action": "unarchive", "confirmation_text": "UNARCHIVE"},
    )

    assert unarchive_response.status_code == 200
    unarchived = unarchive_response.json()
    assert unarchived["action"] == "unarchive"
    assert unarchived["tournament"]["status"] == "DRAFT"
    assert tables["tournaments"][0]["status"] == "DRAFT"
    assert [row["action_type"] for row in tables["admin_activity_log"]] == [
        "archive_tournament_admin",
        "unarchive_tournament_admin",
    ]
    assert all(row["flagged_for_review"] is True for row in tables["admin_activity_log"])
