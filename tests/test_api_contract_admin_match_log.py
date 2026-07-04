from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase, fake_supabase, fake_tables

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def test_admin_match_log_disabled_contract(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", raising=False)
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", raising=False)
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
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", raising=False)
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: fake_supabase())

    response = TestClient(app).get("/admin/clubs/club/match-log?filter=League&limit=20")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["apply_enabled"] is False
    assert payload["status"] == "planning_only"
    assert payload["summary"]["duplicate_groups"] == 1
    assert payload["duplicate_groups"][0]["delete_ids"] == [2]
    assert payload["duplicate_delete_preview"]["delete_count"] == 1
    assert payload["correction_plan"]["apply_endpoint"] is None
    assert "notes" not in payload["matches"][0]


def test_admin_match_log_apply_disabled_before_auth(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", raising=False)
    called = {"auth": False}

    def fake_auth(*_args, **_kwargs):
        called["auth"] = True
        raise AssertionError("auth should not run while apply flag is disabled")

    monkeypatch.setattr("services.api.admin_match_log_routes.authenticate_bearer", fake_auth)
    response = TestClient(app).patch("/admin/clubs/club/match-log/edits", json={"patches": []})

    assert response.status_code == 403
    assert called == {"auth": False}


def test_admin_match_log_apply_edits_contract(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_match_log_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_match_log_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )

    response = TestClient(app).patch(
        "/admin/clubs/club/match-log/edits",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "APPLY", "patches": [{"id": 1, "week_tag": "Week 2"}], "correction_note": "Fix week"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["updated_count"] == 1
    assert tables["matches"][0]["week_tag"] == "Week 2"


def test_admin_match_log_duplicate_cleanup_contract(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_match_log_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_match_log_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )

    response = TestClient(app).post(
        "/admin/clubs/club/match-log/duplicates/cleanup",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "DELETE", "delete_ids": [2]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["deleted_count"] == 1
    assert [row["id"] for row in tables["matches"]] == [1, 3]
