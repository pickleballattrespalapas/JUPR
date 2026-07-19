from datetime import datetime, timezone

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def _tables() -> dict[str, list[dict]]:
    return {
        "clubs": [{"id": "club", "slug": "club", "name": "Club"}],
        "players": [{"club_id": "club", "id": 1, "name": "Alex"}],
        "public_support_requests": [],
    }


def _install(monkeypatch, supabase):
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)


def test_public_data_correction_intake_creates_staff_review_row(monkeypatch):
    tables = _tables()
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)

    response = TestClient(app).post(
        "/clubs/club/support/intake",
        json={
            "request_type": "data_correction",
            "requester_name": "Alex Requester",
            "requester_email": "Alex@example.com",
            "player_name": "Alex",
            "player_id": 1,
            "match_id": "123",
            "subject": "Wrong score",
            "description": "The match score was entered backwards.",
            "requested_action": "Please verify and correct the score.",
            "consent_to_contact": True,
            "source": "test",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["request"]["request_type"] == "data_correction"
    assert payload["request"]["status"] == "new"
    row = tables["public_support_requests"][0]
    assert row["club_id"] == "club"
    assert row["requester_email"] == "alex@example.com"
    assert row["player_id"] == 1
    assert row["status"] == "new"


def test_public_privacy_intake_requires_consent(monkeypatch):
    supabase = FakeSupabase(_tables())
    _install(monkeypatch, supabase)

    response = TestClient(app).post(
        "/clubs/club/support/intake",
        json={
            "request_type": "profile_privacy",
            "requester_name": "Alex Requester",
            "requester_email": "alex@example.com",
            "player_name": "Alex",
            "subject": "Privacy review",
            "description": "Please review my public profile display.",
            "consent_to_contact": False,
        },
    )

    assert response.status_code == 400
    assert "Consent" in response.json()["detail"]


def test_public_support_intake_honeypot_returns_generic_success(monkeypatch):
    tables = _tables()
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)

    response = TestClient(app).post(
        "/clubs/club/support/intake",
        json={
            "request_type": "data_correction",
            "requester_name": "Bot",
            "requester_email": "bot@example.com",
            "subject": "Spam",
            "description": "Spam",
            "consent_to_contact": True,
            "website": "https://spam.example",
        },
    )

    assert response.status_code == 200
    assert response.json()["accepted"] is True
    assert tables["public_support_requests"] == []


def test_public_general_support_is_durable_and_exact_retry_is_deduplicated(monkeypatch):
    tables = _tables()
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)
    request = {
        "request_type": "general_support",
        "requester_name": "Alex Requester",
        "requester_email": "alex@example.com",
        "subject": "Cannot find my result",
        "description": "Please help me locate yesterday's result.",
        "consent_to_contact": True,
    }

    first = TestClient(app).post("/clubs/club/support/intake", json=request)
    second = TestClient(app).post("/clubs/club/support/intake", json=request)

    assert first.status_code == 200
    assert first.json()["deduplicated"] is False
    assert second.status_code == 200
    assert second.json()["deduplicated"] is True
    assert len(tables["public_support_requests"]) == 1
    assert tables["public_support_requests"][0]["request_fingerprint"]


def test_public_support_rejects_unsafe_evidence_scheme(monkeypatch):
    supabase = FakeSupabase(_tables())
    _install(monkeypatch, supabase)

    response = TestClient(app).post(
        "/clubs/club/support/intake",
        json={
            "request_type": "data_correction",
            "requester_name": "Alex",
            "requester_email": "alex@example.com",
            "subject": "Wrong score",
            "description": "Please review the score.",
            "evidence_url": "javascript:alert(1)",
            "consent_to_contact": True,
        },
    )

    assert response.status_code == 400
    assert "http or https" in response.json()["detail"]


def test_public_support_rejects_player_from_another_club(monkeypatch):
    tables = _tables()
    tables["players"].append({"club_id": "other", "id": 2, "name": "Other Player"})
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)

    response = TestClient(app).post(
        "/clubs/club/support/intake",
        json={
            "request_type": "profile_privacy",
            "requester_name": "Other",
            "requester_email": "other@example.com",
            "player_name": "Other Player",
            "player_id": 2,
            "subject": "Privacy review",
            "description": "Please review this profile.",
            "consent_to_contact": True,
        },
    )

    assert response.status_code == 400
    assert "does not belong" in response.json()["detail"]


def test_public_support_rate_limit_is_database_backed(monkeypatch):
    tables = _tables()
    now = datetime.now(timezone.utc).isoformat()
    tables["public_support_requests"] = [
        {
            "id": f"req_{index}",
            "club_id": "club",
            "requester_email": "alex@example.com",
            "request_type": "general_support",
            "subject": f"Existing {index}",
            "description": "Existing request",
            "request_fingerprint": f"fingerprint-{index}",
            "created_at": now,
        }
        for index in range(2)
    ]
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_PUBLIC_SUPPORT_RATE_LIMIT_PER_HOUR", "2")

    response = TestClient(app).post(
        "/clubs/club/support/intake",
        json={
            "request_type": "general_support",
            "requester_name": "Alex",
            "requester_email": "alex@example.com",
            "subject": "Another request",
            "description": "This is not a duplicate.",
            "consent_to_contact": True,
        },
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "3600"
