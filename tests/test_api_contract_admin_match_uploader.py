from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_uploader_service import FakeSupabase, fake_load_data, fake_storage
from jupr_app.services.result_types import ServiceResult

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def test_match_uploader_status_disabled_contract(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    response = TestClient(app).get("/admin/clubs/club/match-uploader/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["submit_endpoint"] is None


def test_match_uploader_submit_disabled_before_auth(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", raising=False)
    called = {"auth": False}

    def fake_auth(*_args, **_kwargs):
        called["auth"] = True
        raise AssertionError("auth should not run while uploader flag is disabled")

    monkeypatch.setattr("services.api.admin_match_uploader_routes.authenticate_bearer", fake_auth)

    response = TestClient(app).post("/admin/clubs/club/match-uploader/batch", json={"matches": []})

    assert response.status_code == 403
    assert called == {"auth": False}


def test_match_uploader_status_enabled_contract(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(fake_storage()))

    response = TestClient(app).get("/admin/clubs/club/match-uploader/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["submit_endpoint"] == "/admin/clubs/{club_id}/match-uploader/batch"
    assert "Open" in payload["league_options"]


def test_match_uploader_submit_contract(monkeypatch):
    storage = fake_storage()
    calls = []

    def fake_submit_match_batch(ctx, matches, **kwargs):
        calls.append({"ctx": ctx, "matches": matches, "kwargs": kwargs})
        return ServiceResult.success(data={"inserted": len(matches), "skipped_incomplete": 0, "skipped_empty": 0, "skipped_unrated": 0})

    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(storage))
    monkeypatch.setattr("jupr_app.services.admin_match_uploader_service.load_data", fake_load_data)
    monkeypatch.setattr("jupr_app.services.admin_match_uploader_service.submit_match_batch", fake_submit_match_batch)
    monkeypatch.setattr(
        "services.api.admin_match_uploader_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_match_uploader_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="scorekeeper"),
    )

    response = TestClient(app).post(
        "/admin/clubs/club/match-uploader/batch",
        headers={"Authorization": "Bearer local"},
        json={
            "source": "test",
            "matches": [
                {
                    "date": "2026-03-01",
                    "league": "Open",
                    "week_tag": "Week 1",
                    "match_type": "Live Match",
                    "t1_p1": 1,
                    "t1_p2": 2,
                    "t2_p1": 3,
                    "t2_p2": 4,
                    "score_t1": 11,
                    "score_t2": 7,
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["submitted_count"] == 1
    assert payload["result"]["inserted"] == 1
    assert calls[0]["matches"][0]["score_t1"] == 11
    assert storage["admin_activity_log"][0]["action_type"] == "submit_match_uploader_batch"
