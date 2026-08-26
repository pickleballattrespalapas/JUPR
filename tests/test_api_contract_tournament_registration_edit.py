from __future__ import annotations

from copy import deepcopy

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
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "test-registration-edit-secret-32bytes")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
    monkeypatch.setenv("JUPR_WEB_BASE_URL", "https://next.example.com")
    storage = fake_storage()
    storage["clubs"] = [{"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas", "admin_notes": "private"}]
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(storage))
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "local-service-role")
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
            "age": 34,
            "gender": "Men",
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
        secret="test-registration-edit-secret-32bytes",
    )


def _edit_versions(storage):
    registration = storage["tournament_registrations"][0]
    return {
        "expected_updated_at": registration["updated_at"],
        "expected_selection_versions": [
            {"id": row["id"], "updated_at": row["updated_at"]}
            for row in storage["tournament_registration_selections"]
            if row["registration_id"] == registration["id"]
        ],
    }


def test_public_registration_edit_link_request_contract(client_and_storage):
    client, _storage = client_and_storage
    _submit_registration(client)

    response = client.post(
        "/clubs/tres-palapas/tournament-registration/edit-link/request",
        json={"registration_slug": "tres-open", "email": "alex@example.com", "idempotency_key": "edit-link-alex-1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["club"] == {"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}
    assert payload["ok"] is True
    assert payload["accepted"] is True
    assert "edit_token" not in str(payload)
    assert "email_status" not in payload
    assert "provider_message_id" not in payload


def test_public_registration_edit_routes_require_server_credential(client_and_storage, monkeypatch):
    client, _storage = client_and_storage
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    response = client.post(
        "/clubs/tres-palapas/tournament-registration/edit-link/request",
        json={"registration_slug": "tres-open", "email": "alex@example.com", "idempotency_key": "edit-link-alex-2"},
    )

    assert response.status_code == 503
    assert "server credential" in response.json()["detail"]


def test_public_registration_edit_link_request_does_not_enumerate_missing_email(client_and_storage):
    client, _storage = client_and_storage
    _submit_registration(client)

    response = client.post(
        "/clubs/tres-palapas/tournament-registration/edit-link/request",
        json={"registration_slug": "tres-open", "email": "missing@example.com", "idempotency_key": "edit-link-missing"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["accepted"] is True
    assert "edit_token" not in str(payload)
    assert "email_status" not in payload


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
    assert payload["registration"]["updated_at"]
    assert payload["selections"][0]["updated_at"]
    assert response.headers["cache-control"] == "no-store, private"
    assert "admin_notes" not in payload["club"]
    assert "internal_seed_notes" not in payload["events"][0]


def test_public_registration_edit_submit_contract(client_and_storage):
    client, storage = client_and_storage
    registration_id = _submit_registration(client)
    token = _edit_token(registration_id)

    response = client.post(
        "/clubs/tres-palapas/tournament-registration/edit",
        json={
            **_edit_versions(storage),
            "edit_token": token,
            "tournament_id": "t1",
            "registration_slug": "tres-open",
            "first_name": "Alexis",
            "last_name": "Rivera",
            "display_name": "Alexis R",
            "email": "changed@example.com",
            "phone": "555-9999",
            "doubles_skill": 4.25,
            "age": 34,
            "gender": "Men",
            "terms_accepted": True,
            "wants_partner_board_contact": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NEEDS_PARTNER", "show_on_partner_board": True}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["registration_id"] == registration_id
    assert payload["confirmation_delivery"] == {"status": "dry_run", "delivered": False}
    assert payload["confirmation_token"]
    assert payload["email_delivery"]["status"] == "dry_run"
    assert "provider_message_id" not in str(payload)
    assert "to_email" not in str(payload)
    assert storage["tournament_registrations"][0]["display_name"] == "Alexis R"
    assert storage["tournament_registrations"][0]["email"] == "alex@example.com"
    assert storage["tournament_registration_selections"][0]["partner_mode"] == "NEEDS_PARTNER"


def test_public_registration_edit_invalid_token_contract(client_and_storage):
    client, _storage = client_and_storage

    response = client.get("/clubs/tres-palapas/tournament-registration/edit?edit_token=bad.token&registration_slug=tres-open")

    assert response.status_code == 400


def test_public_registration_edit_imported_draw_returns_conflict_without_mutation(client_and_storage):
    client, storage = client_and_storage
    registration_id = _submit_registration(client)
    token = _edit_token(registration_id)
    storage["tournament_teams"].append(
        {
            "id": "team-imported",
            "tournament_id": "t1",
            "draw_id": "draw-1",
            "registration_day_id": "day1",
            "event_option_id": "event1",
            "source": "REGISTRATION",
            "player1_id": storage["tournament_registrations"][0]["player_id"],
        }
    )
    before = deepcopy(storage)

    get_response = client.get(
        f"/clubs/tres-palapas/tournament-registration/edit?edit_token={token}&registration_slug=tres-open"
    )
    post_response = client.post(
        "/clubs/tres-palapas/tournament-registration/edit",
        json={
            **_edit_versions(storage),
            "edit_token": token,
            "tournament_id": "t1",
            "registration_slug": "tres-open",
            "first_name": "Changed",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "age": 34,
            "gender": "Men",
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )

    assert get_response.status_code == 409
    assert post_response.status_code == 409
    assert storage == before
