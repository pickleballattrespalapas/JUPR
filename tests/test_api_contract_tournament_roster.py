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


def test_public_tournament_roster_empty_contract(client):
    response = client.get("/clubs/tres-palapas/tournament-roster?registration_slug=tres-open")

    assert response.status_code == 200
    payload = response.json()
    assert payload["club"] == {"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}
    assert payload["available"] is True
    assert payload["tournament"]["name"] == "Tres Palapas Open"
    assert payload["summary"]["total_registrations"] == 0
    assert payload["roster"]["registrations_by_event"] == []
    assert "admin_notes" not in payload["club"]
    assert "internal_seed_notes" not in payload["events"][0]


def test_public_tournament_roster_after_submit_contract(client):
    submit = client.post(
        "/clubs/tres-palapas/tournament-registration",
        json={
            "registration_slug": "tres-open",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "phone": "555-0100",
            "doubles_skill": 4.0,
            "age": 34,
            "gender": "Men",
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )
    assert submit.status_code == 200

    response = client.get("/clubs/tres-palapas/tournament-roster?registration_slug=tres-open")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total_registrations"] == 1
    roster_row = payload["roster"]["registrations_by_event"][0]
    members = roster_row["members"]
    assert members[0]["display_name"] == "Alex Rivera"
    assert roster_row["status"] == "Review"
    assert "email" not in members[0]
    assert "phone" not in members[0]
    assert "registration_id" not in members[0]
    assert "selection_id" not in members[0]
    assert "player_id" not in members[0]
    assert "source_registration_ids" not in roster_row
    assert "source_selection_ids" not in roster_row
