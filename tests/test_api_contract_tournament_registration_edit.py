from __future__ import annotations

import pytest

from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token
from tests.conftest import require_api_dependency
from tests.test_public_tournament_registration_service import FakeSupabase, fake_storage

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


@pytest.fixture
def client_and_storage(monkeypatch):
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "test-secret")
    storage = fake_storage()
    storage["clubs"] = [{"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas", "admin_notes": "private"}]
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(storage))
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    return TestClient(app), storage


def _submit_registration(client: TestClient) -> str:
    response = client.post(
        "/clubs/tres-palapas/tournament-registration",
        json={
            "registration_slug": "tres-open",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "phone": "555-0100",
            "doubles_skill": 4.0,
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )
    assert response.status_code == 200
    return str(response.json()["registration_id"])


def _edit_token(registration_id: str) -> str:
    return build_registration_edit_token(
        tournament_id="t1",
        registration_id=registration_id,
        email="alex@example.com",
        secret="test-secret",
    )


def test_public_registration_edit_page_contract(client_and_storage):
    client, _storage = client_and_storage
    registration_id = _submit_registration(client)
    token = _edit_token(registration_id)

    response = client.get(f"/clubs/tres-palapas/tournament-registration/edit?edit_token={token}&registration_slug=tres-open")

    assert response.status_code == 200
    payload = response.json()
    assert payload["club"] == {"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}
    assert payload["edit_token_valid"] is True
    assert payload["registration"]["email"] == "alex@example.com"
    assert payload["registration"]["phone"] == "555-0100"
    assert payload["selections"][0]["event_option_id"] == "event1"
    assert "admin_notes" not in payload["club"]
    assert "internal_seed_notes" not in payload["events"][0]


def test_public_registration_edit_submit_contract(client_and_storage):
    client, storage = client_and_storage
    registration_id = _submit_registration(client)
    token = _edit_token(registration_id)

    response = client.post(
        "/clubs/tres-palapas/tournament-registration/edit",
        json={
            "edit_token": token,
            "tournament_id": "t1",
            "registration_slug": "tres-open",
            "first_name": "Alexis",
            "last_name": "Rivera",
            "display_name": "Alexis R",
            "email": "changed@example.com",
            "phone": "555-9999",
            "doubles_skill": 4.25,
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NEEDS_PARTNER", "show_on_partner_board": True}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["registration_id"] == registration_id
    assert storage["tournament_registrations"][0]["display_name"] == "Alexis R"
    assert storage["tournament_registrations"][0]["email"] == "alex@example.com"
    assert storage["tournament_registration_selections"][0]["partner_mode"] == "NEEDS_PARTNER"


def test_public_registration_edit_invalid_token_contract(client_and_storage):
    client, _storage = client_and_storage

    response = client.get("/clubs/tres-palapas/tournament-registration/edit?edit_token=bad.token&registration_slug=tres-open")

    assert response.status_code == 400
