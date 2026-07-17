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


def test_public_tournament_registration_submit_and_confirmation_contract(client):
    response = client.post(
        "/clubs/tres-palapas/tournament-registration",
        json={
            "registration_slug": "tres-open",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "doubles_skill": 4.0,
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["registration_id"]
    assert payload["selection_count"] == 1

    confirm = client.get(
        f"/clubs/tres-palapas/tournament-registration/confirmations/{payload['registration_id']}?registration_slug=tres-open"
    )
    assert confirm.status_code == 200
    confirm_payload = confirm.json()
    assert confirm_payload["registration"]["display_name"] == "Alex Rivera"
    assert confirm_payload["selections"][0]["event_label"] == "Open"
    assert "phone" not in confirm_payload["registration"]


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
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )
    assert untrusted_player_link.status_code == 200
    mallory = next(row for row in storage["tournament_registrations"] if row["email"] == "mallory@example.com")
    assert mallory["player_id"] is None

    duplicate_family = api.post(
        "/clubs/tres-palapas/tournament-registration",
        json={
            "registration_slug": "tres-open",
            "first_name": "Casey",
            "email": "casey@example.com",
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
            "singles_skill": 3.2,
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )
    assert wrong_gender.status_code == 400
    assert "women's registrations" in wrong_gender.json()["detail"]
