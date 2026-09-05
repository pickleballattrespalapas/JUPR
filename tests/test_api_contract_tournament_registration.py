from __future__ import annotations

import pytest

from tests.conftest import require_api_dependency
from tests.test_public_tournament_registration_service import FakeSupabase, fake_storage

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


@pytest.fixture
def client(monkeypatch):
    storage = fake_storage()
    storage["clubs"] = [{"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas", "admin_notes": "private"}]
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(storage))
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setenv("JUPR_REGISTRATION_CONFIRMATION_SECRET", "api-test-confirmation-secret")
    monkeypatch.setenv("JUPR_WEB_BASE_URL", "https://staging.example.test")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "public-intake-auth")
    monkeypatch.setenv("JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES", "1")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
    return TestClient(app)


@pytest.fixture
def integrity_client(monkeypatch):
    storage = fake_storage()
    storage["clubs"] = [{"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}]
    storage["players"] = [
        {"id": 10, "club_id": "club-1", "name": "Verified Alex", "rating": 1600, "active": True, "inactive_at": None},
        {"id": 11, "club_id": "other-club", "name": "Other Club", "rating": 1200, "active": True, "inactive_at": None},
    ]
    storage["tournament_event_options"].append(
        {
            **storage["tournament_event_options"][0],
            "id": "event2",
            "label": "Advanced Doubles",
            "division_name": "Advanced",
            "sort_order": 2,
        }
    )
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(storage))
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setenv("JUPR_REGISTRATION_CONFIRMATION_SECRET", "api-test-confirmation-secret")
    monkeypatch.setenv("JUPR_WEB_BASE_URL", "https://staging.example.test")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "public-intake-auth")
    monkeypatch.setenv("JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES", "1")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
    return TestClient(app), storage


def test_public_tournament_registration_page_contract(client):
    response = client.get("/clubs/tres-palapas/tournament-registration?registration_slug=tres-open")

    assert response.status_code == 200
    payload = response.json()
    assert payload["club"] == {"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}
    assert payload["available"] is True
    assert payload["registration_open"] is True
    assert payload["tournament"]["name"] == "Tres Palapas Open"
    assert payload["events"][0]["selectable"] is True
    assert "admin_notes" not in payload["club"]
    assert "internal_seed_notes" not in payload["events"][0]


def test_public_tournament_registration_profile_resolution_contract(integrity_client):
    api, storage = integrity_client
    storage["players"][0].update(
        {"name": "Verified Alex", "email": "alex@example.com", "dupr_id": "DUPR-10"}
    )
    response = api.post(
        "/clubs/tres-palapas/tournament-registration/profile-resolution",
        json={
            "registration_slug": "tres-open",
            "first_name": "Verified",
            "last_name": "Alex",
            "email": "alex@example.com",
            "age": 34,
            "gender": "Men",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["profile_match_kind"] == "email_exact"
    assert payload["profile_candidates"][0]["id"] == "10"
    assert payload["profile_policy"]["public_submission_links_player"] is False
    assert "email" not in payload["profile_candidates"][0]


def test_public_tournament_registration_profile_resolution_requires_demographics(client):
    response = client.post(
        "/clubs/tres-palapas/tournament-registration/profile-resolution",
        json={
            "registration_slug": "tres-open",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
        },
    )

    assert response.status_code == 422


def test_public_tournament_registration_submit_and_confirmation_contract(client):
    response = client.post(
        "/clubs/tres-palapas/tournament-registration",
        json={
            "registration_slug": "tres-open",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "doubles_skill": 4.0,
            "age": 34,
            "gender": "Men",
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["registration_id"]
    assert payload["selection_count"] == 1
    assert payload["confirmation_token"]
    assert payload["email_delivery"]["status"] == "dry_run"

    confirm = client.get(
        "/clubs/tres-palapas/tournament-registration/confirmation",
        params={"confirmation_token": payload["confirmation_token"]},
    )
    assert confirm.status_code == 200
    confirm_payload = confirm.json()
    assert confirm_payload["registration"]["display_name"] == "Alex Rivera"
    assert confirm_payload["selections"][0]["event_label"] == "Open"
    assert "phone" not in confirm_payload["registration"]
    assert "email" not in confirm_payload["registration"]
    assert "id" not in confirm_payload["registration"]
    assert "selection_id" not in confirm_payload["selections"][0]

    raw_id_lookup = client.get(
        f"/clubs/tres-palapas/tournament-registration/confirmations/{payload['registration_id']}"
    )
    assert raw_id_lookup.status_code == 404

    tampered = client.get(
        "/clubs/tres-palapas/tournament-registration/confirmation",
        params={"confirmation_token": f"{payload['confirmation_token']}x"},
    )
    assert tampered.status_code == 404


def test_public_tournament_registration_duplicate_email_returns_recovery_conflict(client):
    payload = {
        "registration_slug": "tres-open",
        "first_name": "Alex",
        "last_name": "Rivera",
        "email": "alex@example.com",
        "age": 34,
        "gender": "Men",
        "terms_accepted": True,
        "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
    }
    assert client.post("/clubs/tres-palapas/tournament-registration", json=payload).status_code == 200

    duplicate = client.post("/clubs/tres-palapas/tournament-registration", json=payload)

    assert duplicate.status_code == 409
    assert "Request an edit link" in duplicate.json()["detail"]


def test_public_tournament_registration_submit_requires_age_and_gender(client):
    response = client.post(
        "/clubs/tres-palapas/tournament-registration",
        json={
            "registration_slug": "tres-open",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )

    assert response.status_code == 400
    assert "Age is required" in response.json()["detail"]


def test_public_tournament_registration_honeypot_contract(client):
    response = client.post(
        "/clubs/tres-palapas/tournament-registration",
        json={
            "registration_slug": "tres-open",
            "first_name": "Bot",
            "email": "bot@example.com",
            "terms_accepted": True,
            "website": "filled",
            "selections": [{"event_option_id": "event1"}],
        },
    )

    assert response.status_code == 400


def test_public_tournament_registration_integrity_errors_are_api_400(integrity_client):
    api, storage = integrity_client
    untrusted_player_link = api.post(
        "/clubs/tres-palapas/tournament-registration",
        json={
            "registration_slug": "tres-open",
            "first_name": "Mallory",
            "email": "mallory@example.com",
            "player_id": 11,
            "age": 32,
            "gender": "Women",
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )
    assert untrusted_player_link.status_code == 200
    mallory = next(row for row in storage["tournament_registrations"] if row["email"] == "mallory@example.com")
    assert mallory["player_id"] not in {None, 11}

    duplicate_family = api.post(
        "/clubs/tres-palapas/tournament-registration",
        json={
            "registration_slug": "tres-open",
            "first_name": "Casey",
            "email": "casey@example.com",
            "age": 40,
            "gender": "Women",
            "terms_accepted": True,
            "selections": [
                {"event_option_id": "event1", "partner_mode": "NONE"},
                {"event_option_id": "event2", "partner_mode": "NONE"},
            ],
        },
    )
    assert duplicate_family.status_code == 400
    assert "only one division" in duplicate_family.json()["detail"]

    storage["tournament_event_options"][0].update(
        {"event_type": "SINGLES", "gender_restriction": "WOMEN", "skill_label": "3.5"}
    )
    wrong_gender = api.post(
        "/clubs/tres-palapas/tournament-registration",
        json={
            "registration_slug": "tres-open",
            "first_name": "Jordan",
            "email": "jordan@example.com",
            "gender": "Men",
            "age": 36,
            "singles_skill": 3.2,
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )
    assert wrong_gender.status_code == 400
    assert "women's registrations" in wrong_gender.json()["detail"]
