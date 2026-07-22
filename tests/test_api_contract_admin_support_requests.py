from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def _tables() -> dict[str, list[dict]]:
    return {
        "public_support_requests": [
            {
                "id": "req_1",
                "club_id": "club",
                "club_slug": "club",
                "request_type": "data_correction",
                "status": "new",
                "requester_name": "Alex",
                "requester_email": "alex@example.com",
                "player_name": "Alex",
                "player_id": 1,
                "match_id": "42",
                "subject": "Wrong score",
                "description": "The score is backwards.",
                "requested_action": "Review match 42.",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "identity_status": "not_required",
                "fulfillment_status": "not_required",
                "resolution_action": "none",
            },
            {
                "id": "req_2",
                "club_id": "club",
                "club_slug": "club",
                "request_type": "profile_privacy",
                "status": "in_review",
                "requester_name": "Blair",
                "requester_email": "blair@example.com",
                "player_name": "Blair",
                "subject": "Privacy review",
                "description": "Review my profile.",
                "created_at": "2026-01-02T00:00:00Z",
                "updated_at": "2026-01-02T00:00:00Z",
                "identity_status": "pending",
                "fulfillment_status": "pending",
                "resolution_action": "none",
            },
        ],
        "admin_activity_log": [],
    }


def _install(monkeypatch, supabase):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_SUPPORT_REQUESTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_support_requests_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_support_requests_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner", assigned=True, source="admin_role_assignments"),
    )


def test_admin_support_requests_list_contract(monkeypatch):
    supabase = FakeSupabase(_tables())
    _install(monkeypatch, supabase)

    response = TestClient(app).get(
        "/admin/clubs/club/support-requests?status=new",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "admin_support_requests_list"
    assert len(payload["requests"]) == 1
    assert payload["requests"][0]["id"] == "req_1"
    assert payload["summary"]["by_status"]["new"] == 1


def test_admin_support_request_update_contract(monkeypatch):
    tables = _tables()
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/support-requests/req_1",
        headers={"Authorization": "Bearer local"},
        json={"status": "resolved", "admin_note": "Fixed via Match Log.", "confirmation_text": "SAVE REQUEST STATUS"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["request"]["status"] == "resolved"
    assert tables["public_support_requests"][0]["status"] == "resolved"
    assert tables["public_support_requests"][0]["admin_note"] == "Fixed via Match Log."
    assert tables["admin_activity_log"][0]["action_type"] == "update_public_support_request_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_support_request_terminal_status_allows_optional_note(monkeypatch):
    for target_status in ("resolved", "dismissed"):
        tables = _tables()
        supabase = FakeSupabase(tables)
        _install(monkeypatch, supabase)

        response = TestClient(app).patch(
            "/admin/clubs/club/support-requests/req_1",
            headers={"Authorization": "Bearer local"},
            json={
                "status": target_status,
                "admin_note": "  ",
                "expected_updated_at": "2026-01-01T00:00:00Z",
                "confirmation_text": "SAVE REQUEST STATUS",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["request"]["status"] == target_status
        assert payload["request"]["admin_note"] == ""
        assert payload["request"]["reviewed_by"] == "admin@example.com"
        assert payload["request"]["reviewed_at"]
        assert tables["public_support_requests"][0]["admin_note"] is None
        assert tables["admin_activity_log"][0]["action_type"] == "update_public_support_request_admin"
        assert tables["admin_activity_log"][0]["entity_id"] == "req_1"
        assert tables["admin_activity_log"][0]["before_json"]["status"] == "new"
        assert tables["admin_activity_log"][0]["after_json"]["request"]["status"] == target_status


def test_admin_support_request_update_requires_confirmation(monkeypatch):
    supabase = FakeSupabase(_tables())
    _install(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/support-requests/req_1",
        headers={"Authorization": "Bearer local"},
        json={"status": "resolved", "confirmation_text": "SAVE"},
    )

    assert response.status_code == 400
    assert "SAVE REQUEST STATUS" in response.json()["detail"]


def test_privacy_request_requires_verified_fulfillment_before_resolution(monkeypatch):
    supabase = FakeSupabase(_tables())
    _install(monkeypatch, supabase)

    blocked = TestClient(app).patch(
        "/admin/clubs/club/support-requests/req_2",
        headers={"Authorization": "Bearer local"},
        json={
            "status": "resolved",
            "admin_note": "Reviewed.",
            "expected_updated_at": "2026-01-02T00:00:00Z",
            "identity_status": "pending",
            "fulfillment_status": "pending",
            "resolution_action": "none",
            "confirmation_text": "SAVE REQUEST STATUS",
        },
    )

    assert blocked.status_code == 400
    assert "Verify the requester identity" in blocked.json()["detail"]


def test_privacy_request_resolves_with_completed_sop_evidence(monkeypatch):
    tables = _tables()
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/support-requests/req_2",
        headers={"Authorization": "Bearer local"},
        json={
            "status": "resolved",
            "expected_updated_at": "2026-01-02T00:00:00Z",
            "identity_status": "verified",
            "fulfillment_status": "completed",
            "resolution_action": "alias",
            "resolution_evidence": "Verified by organizer; player, leaderboard, match, and roster projections checked.",
            "confirmation_text": "SAVE REQUEST STATUS",
        },
    )

    assert response.status_code == 200
    row = next(row for row in tables["public_support_requests"] if row["id"] == "req_2")
    assert row["identity_status"] == "verified"
    assert row["fulfillment_status"] == "completed"
    assert row["resolution_action"] == "alias"
    assert row["admin_note"] is None


def test_admin_support_request_rejects_stale_update(monkeypatch):
    supabase = FakeSupabase(_tables())
    _install(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/support-requests/req_1",
        headers={"Authorization": "Bearer local"},
        json={
            "status": "in_review",
            "expected_updated_at": "2025-12-31T00:00:00Z",
            "confirmation_text": "SAVE REQUEST STATUS",
        },
    )

    assert response.status_code == 409
    assert "changed after it was loaded" in response.json()["detail"]
