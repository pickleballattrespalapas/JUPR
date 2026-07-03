from __future__ import annotations

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import fake_supabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def test_admin_match_log_disabled_contract(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    response = TestClient(app).get("/admin/clubs/club/match-log")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["matches"] == []
    assert payload["duplicate_groups"] == []


def test_admin_match_log_enabled_contract(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: fake_supabase())

    response = TestClient(app).get("/admin/clubs/club/match-log?filter=League&limit=20")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["status"] == "planning_only"
    assert payload["summary"]["duplicate_groups"] == 1
    assert payload["duplicate_groups"][0]["delete_ids"] == [2]
    assert payload["duplicate_delete_preview"]["delete_count"] == 1
    assert payload["correction_plan"]["apply_endpoint"] is None
    assert "notes" not in payload["matches"][0]
