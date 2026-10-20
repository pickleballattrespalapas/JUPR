"""Exercise the real intake/review/notification flow with isolated in-memory IO."""
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_admin_notification_sources import Database as NotificationDatabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from jupr_app.services.admin_notification_sources import collect_admin_notification_sources
from jupr_app.services.public_support_intake_service import (
    SupportIntakeUnavailableError,
    create_public_support_intake_request,
)
from scripts.staging_write_waves import expected_write_flags
from services.api.main import app


REQUEST = {
    "requester_name": "Alex",
    "requester_email": "alex@example.test",
    "subject": "Please review this record",
    "description": "Details provided for club staff.",
    "consent_to_contact": True,
}
AUTH = {"Authorization": "Bearer local"}


@pytest.fixture
def workflow(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "open")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_SUPPORT_REQUESTS", "1")
    monkeypatch.setenv("JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    assignments = [
        {"club_id": "alpha", "email": "staff@example.test", "user_id": "staff", "role": "club_owner"},
    ]
    rows = {
        "clubs": [{"id": slug, "slug": slug, "name": slug.title()} for slug in ("alpha", "beta")],
        "players": [{"id": 1, "club_id": "alpha", "name": "Alex"}, {"id": 2, "club_id": "beta", "name": "Blair"}],
        "admin_role_assignments": assignments,
        "public_support_requests": [],
        "admin_activity_log": [],
    }
    db = FakeSupabase(rows)
    monkeypatch.setattr("services.api.main.create_client", lambda *_args: db)

    def authenticate(header):
        if header != AUTH["Authorization"]:
            raise HTTPException(401, "Missing valid bearer token")
        return SimpleNamespace(email="staff@example.test", user_id="staff")

    monkeypatch.setattr("services.api.admin_support_requests_routes.authenticate_bearer", authenticate)
    # No mail belongs to this workflow. Fail immediately if a new code path
    # accidentally reaches the transport, even with the dry-run setting.
    monkeypatch.setattr("smtplib.SMTP", lambda *_args, **_kwargs: pytest.fail("Support must not send email"))
    monkeypatch.setattr("smtplib.SMTP_SSL", lambda *_args, **_kwargs: pytest.fail("Support must not send email"))
    return TestClient(app), db, assignments


def _support_items(db, assignments, club_id="alpha"):
    notification_db = NotificationDatabase()
    notification_db.rows = db.tables
    feed = collect_admin_notification_sources(notification_db, club_id=club_id, assignments=assignments)
    return [item for item in feed["items"] if item["category"].startswith("support_")]


@pytest.mark.parametrize("request_type", ["general_support", "data_correction", "profile_privacy"])
def test_intake_becomes_scoped_notification_and_can_be_reviewed_and_completed(workflow, request_type):
    client, db, assignments = workflow
    status = client.get("/clubs/alpha/support/intake/status")
    assert status.status_code == 200
    assert status.headers["cache-control"] == "no-store"
    assert status.json()["accepting_requests"] is True
    assert "request_count" not in status.json()

    payload = {**REQUEST, "request_type": request_type, "player_id": 1, "player_name": "Alex"}
    created = client.post("/clubs/alpha/support/intake", json=payload)
    assert created.status_code == 200
    request_id = created.json()["request"]["id"]
    assert client.post("/clubs/alpha/support/intake", json=payload).json()["deduplicated"] is True
    assert len(db.tables["public_support_requests"]) == 1

    items = _support_items(db, assignments)
    assert [(item["category"], item["source_id"]) for item in items] == [("support_new", request_id)]
    assert parse_qs(urlsplit(items[0]["href"]).query) == {"status": ["new"], "request": [request_id]}
    assert _support_items(db, assignments, "beta") == []
    assert REQUEST["requester_email"] not in str(items)
    assert client.get("/admin/clubs/alpha/support-requests").status_code == 401
    assert client.get("/admin/clubs/beta/support-requests", headers=AUTH).status_code == 403

    queue = client.get("/admin/clubs/alpha/support-requests?status=new", headers=AUTH).json()["requests"]
    assert [row["id"] for row in queue] == [request_id]
    route = f"/admin/clubs/alpha/support-requests/{request_id}"
    transition = {"status": "in_review", "confirmation_text": "SAVE REQUEST STATUS", "expected_updated_at": queue[0]["updated_at"]}
    reviewed = client.patch(route, headers=AUTH, json=transition)
    assert reviewed.status_code == 200
    assert _support_items(db, assignments)[0]["category"] == "support_in_review"
    assert client.patch(route, headers=AUTH, json=transition).status_code == 409

    completion = {**transition, "status": "resolved", "expected_updated_at": reviewed.json()["request"]["updated_at"]}
    if request_type == "profile_privacy":
        assert client.patch(route, headers=AUTH, json=completion).status_code == 400
        completion.update(identity_status="verified", fulfillment_status="completed", resolution_action="alias",
                          resolution_evidence="Identity verified and approved public projections reviewed in fixture.")
    completed = client.patch(route, headers=AUTH, json=completion)
    assert completed.status_code == 200
    assert completed.json()["request"]["reviewed_by"] == "staff@example.test"
    assert _support_items(db, assignments) == []
    audit = [row for row in db.tables["admin_activity_log"] if row["action_type"] == "update_public_support_request_admin"]
    assert len(audit) == 2
    assert all(row["club_id"] == "alpha" and row["entity_id"] == request_id for row in audit)
    assert db.tables["players"][0]["name"] == "Alex"  # queue review does not perform the profile change


@pytest.mark.parametrize("wave", ["none", "public-intake-auth", "support-requests", "open"])
def test_readiness_and_intake_enforce_both_sides_of_staging_wave(workflow, monkeypatch, wave):
    client, db, _ = workflow
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", wave)
    for name, enabled in expected_write_flags(wave).items():
        monkeypatch.setenv(name, "1" if enabled else "0")
    ready = client.get("/clubs/alpha/support/intake/status").json()["accepting_requests"]
    assert ready is (wave == "open")
    response = client.post("/clubs/alpha/support/intake", json=REQUEST)
    assert response.status_code == (200 if ready else 503 if wave == "public-intake-auth" else 403)
    assert len(db.tables["public_support_requests"]) == int(ready)


@pytest.mark.parametrize("environment,policy", [("test", ""), ("production", "enabled")])
def test_disabled_review_refuses_intake_even_when_public_writes_are_allowed(workflow, monkeypatch, environment, policy):
    client, db, _ = workflow
    monkeypatch.setenv("JUPR_ENV", environment)
    monkeypatch.setenv("JUPR_PRODUCTION_WRITE_POLICY", policy)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_SUPPORT_REQUESTS", "0")
    assert client.get("/clubs/alpha/support/intake/status").json()["accepting_requests"] is False
    response = client.post("/clubs/alpha/support/intake", json=REQUEST)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "300"
    assert "JUPR_" not in response.text
    with pytest.raises(SupportIntakeUnavailableError):
        create_public_support_intake_request(db, club_id="alpha", club_slug="alpha", payload=REQUEST)
    assert db.tables["public_support_requests"] == []


def test_readiness_respects_production_read_only_without_enabling_any_writes(workflow, monkeypatch):
    client, db, _ = workflow
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("JUPR_PRODUCTION_WRITE_POLICY", "read_only")
    assert client.get("/clubs/alpha/support/intake/status").json()["accepting_requests"] is False
    assert client.get("/admin/clubs/alpha/support-requests/status").json()["ready_for_review"] is False
    assert client.post("/clubs/alpha/support/intake", json=REQUEST).status_code == 403
    assert db.tables["public_support_requests"] == []


def test_missing_review_schema_fails_closed_without_exposing_details(workflow, monkeypatch):
    client, db, _ = workflow
    original = db.table

    def broken_table(name):
        table = original(name)
        if name == "public_support_requests":
            def execute():
                raise RuntimeError("private database detail: missing fulfillment_status")
            table.execute = execute
        return table

    monkeypatch.setattr(db, "table", broken_table)
    for route, key in [("/clubs/alpha/support/intake/status", "accepting_requests"),
                       ("/admin/clubs/alpha/support-requests/status", "ready_for_review")]:
        response = client.get(route)
        assert response.json()[key] is False
        assert "private database" not in response.text
    assert client.post("/clubs/alpha/support/intake", json=REQUEST).status_code == 503
    assert db.tables["public_support_requests"] == []


def test_unknown_club_and_cross_club_player_are_not_routed_to_another_queue(workflow):
    client, db, _ = workflow
    assert client.get("/clubs/missing/support/intake/status").status_code == 404
    assert client.post("/clubs/missing/support/intake", json=REQUEST).status_code == 404
    assert client.post("/clubs/alpha/support/intake", json={**REQUEST, "player_id": 2}).status_code == 400
    assert db.tables["public_support_requests"] == []
    created = client.post("/clubs/beta/support/intake", json={**REQUEST, "player_id": 2})
    assert created.status_code == 200
    assert db.tables["public_support_requests"][0]["club_id"] == "beta"
    # A valid alpha assignment cannot review a beta request using either URL.
    request_id = created.json()["request"]["id"]
    for club, code in [("alpha", 400), ("beta", 403)]:
        response = client.patch(f"/admin/clubs/{club}/support-requests/{request_id}", headers=AUTH,
                                json={"status": "resolved", "confirmation_text": "SAVE REQUEST STATUS"})
        assert response.status_code == code
    assert db.tables["public_support_requests"][0]["status"] == "new"


@pytest.mark.parametrize("role,revoked", [("read_only", False), ("scorekeeper", False), ("club_owner", True)])
def test_review_completion_requires_current_assigned_write_permission(workflow, role, revoked):
    client, db, assignments = workflow
    created = client.post("/clubs/alpha/support/intake", json=REQUEST)
    request_id = created.json()["request"]["id"]
    assignments[0]["role"] = role
    if revoked:
        assignments[0]["revoked_at"] = "2026-01-01T00:00:00Z"
    response = client.patch(f"/admin/clubs/alpha/support-requests/{request_id}", headers=AUTH,
                            json={"status": "resolved", "confirmation_text": "SAVE REQUEST STATUS"})
    assert response.status_code == 403
    assert db.tables["public_support_requests"][0]["status"] == "new"
    assert _support_items(db, assignments) == []
