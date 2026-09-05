from __future__ import annotations

import pytest

from tests.conftest import require_api_dependency
from tests.test_public_tournament_registration_service import FakeSupabase, fake_storage

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token
from jupr_app.domain.tournament_public_references import build_public_tournament_reference
from services.api.main import app


SECRET = "partner-flow-contract-secret-1234567890"


@pytest.fixture
def partner_client(monkeypatch):
    storage = fake_storage()
    storage["clubs"] = [{"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}]
    fake_supabase = FakeSupabase(storage)
    # This API-flow fixture exercises the documented table-backed local-fake
    # path. Transactional RPC behavior has separate migration/domain coverage.
    fake_supabase.rpc = None
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: fake_supabase)
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", SECRET)
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
    return TestClient(app), storage


def _register(api: TestClient, storage: dict, *, first_name: str, email: str, needs_partner: bool = False):
    response = api.post(
        "/clubs/tres-palapas/tournament-registration",
        json={
            "registration_slug": "tres-open",
            "first_name": first_name,
            "last_name": "Player",
            "email": email,
            "doubles_skill": 4.0,
            "age": 34,
            "gender": "Mixed",
            "wants_partner_board_contact": needs_partner,
            "terms_accepted": True,
            "selections": [
                {
                    "event_option_id": "event1",
                    "partner_mode": "NEEDS_PARTNER" if needs_partner else "NONE",
                    "show_on_partner_board": needs_partner,
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    registration_id = response.json()["registration_id"]
    selection_id = next(
        row["id"]
        for row in storage["tournament_registration_selections"]
        if row["registration_id"] == registration_id
    )
    token = build_registration_edit_token(
        tournament_id="t1",
        registration_id=registration_id,
        email=email,
        secret=SECRET,
    )
    return registration_id, selection_id, token


def _interest(api: TestClient, *, requester_selection_id: str, target_selection_id: str, token: str):
    response = api.post(
        "/clubs/tres-palapas/tournament-registration/pairing-interest",
        json={
            "tournament_id": "t1",
            "registration_slug": "tres-open",
            "edit_token": token,
            "requester_selection_id": requester_selection_id,
            "board_entry_key": build_public_tournament_reference(
                tournament_id="t1",
                namespace="partner-board-selection",
                source_id=target_selection_id,
            ),
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _transition(api: TestClient, *, request_id: str, action: str, token: str):
    return api.post(
        f"/clubs/tres-palapas/tournament-registration/pairing-requests/{request_id}/{action}",
        json={"tournament_id": "t1", "registration_slug": "tres-open", "edit_token": token},
    )


def test_partner_board_request_review_decline_cancel_accept_and_stale_contract(partner_client):
    api, storage = partner_client
    _alex_registration, alex_selection, alex_token = _register(api, storage, first_name="Alex", email="alex@example.com")
    _jordan_registration, jordan_selection, jordan_token = _register(api, storage, first_name="Jordan", email="jordan@example.com")
    _casey_registration, casey_selection, casey_token = _register(
        api,
        storage,
        first_name="Casey",
        email="casey@example.com",
        needs_partner=True,
    )

    declined_request = _interest(
        api,
        requester_selection_id=alex_selection,
        target_selection_id=casey_selection,
        token=alex_token,
    )
    incoming = api.get(
        "/clubs/tres-palapas/tournament-registration/pairing-requests",
        params={"tournament_id": "t1", "registration_slug": "tres-open", "edit_token": casey_token},
    )
    assert incoming.status_code == 200
    incoming_payload = incoming.json()
    assert incoming_payload["incoming"][0]["available_actions"] == ["accept", "decline"]
    assert "requester_selection_id" not in incoming_payload["incoming"][0]
    assert "target_selection_id" not in incoming_payload["incoming"][0]
    assert "selection_id" not in incoming_payload["incoming"][0]["requester"]
    assert "registration_id" not in incoming_payload["incoming"][0]["requester"]
    assert "casey@example.com" not in str(incoming_payload).lower()
    assert "alex@example.com" not in str(incoming_payload).lower()

    declined = _transition(api, request_id=declined_request["partner_request_id"], action="decline", token=casey_token)
    declined_retry = _transition(api, request_id=declined_request["partner_request_id"], action="decline", token=casey_token)
    stale_accept = _transition(api, request_id=declined_request["partner_request_id"], action="accept", token=casey_token)
    assert declined.status_code == 200
    assert declined.json()["status"] == "DECLINED"
    assert declined_retry.status_code == 200
    assert declined_retry.json()["idempotent"] is True
    assert stale_accept.status_code == 409

    cancelled_request = _interest(
        api,
        requester_selection_id=jordan_selection,
        target_selection_id=casey_selection,
        token=jordan_token,
    )
    cancelled = _transition(api, request_id=cancelled_request["partner_request_id"], action="cancel", token=jordan_token)
    cancelled_retry = _transition(api, request_id=cancelled_request["partner_request_id"], action="cancel", token=jordan_token)
    wrong_actor_cancel = _transition(api, request_id=cancelled_request["partner_request_id"], action="cancel", token=casey_token)
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "CANCELLED"
    assert cancelled_retry.status_code == 200
    assert cancelled_retry.json()["idempotent"] is True
    assert wrong_actor_cancel.status_code == 400

    accepted_request = _interest(
        api,
        requester_selection_id=alex_selection,
        target_selection_id=casey_selection,
        token=alex_token,
    )
    competing_request = _interest(
        api,
        requester_selection_id=jordan_selection,
        target_selection_id=casey_selection,
        token=jordan_token,
    )
    accepted = _transition(api, request_id=accepted_request["partner_request_id"], action="accept", token=casey_token)
    accepted_retry = _transition(api, request_id=accepted_request["partner_request_id"], action="accept", token=casey_token)
    competing_stale = _transition(api, request_id=competing_request["partner_request_id"], action="accept", token=casey_token)

    assert accepted.status_code == 200
    assert accepted.json()["status"] == "ACCEPTED"
    assert accepted.json()["team_link_id"]
    assert accepted.json()["cancelled_request_ids"] == [competing_request["partner_request_id"]]
    assert accepted_retry.status_code == 200
    assert accepted_retry.json()["idempotent"] is True
    assert competing_stale.status_code == 409
    assert len(storage["tournament_registration_team_links"]) == 1
    assert len(storage["tournament_registration_team_members"]) == 2


def test_partner_board_public_projection_hides_nonconsenting_needs_partner_entry(partner_client):
    api, storage = partner_client
    visible_registration, _visible_selection, _visible_token = _register(
        api,
        storage,
        first_name="Visible",
        email="visible@example.com",
        needs_partner=True,
    )
    _private_registration, _private_selection, _private_token = _register(
        api,
        storage,
        first_name="Private",
        email="private@example.com",
        needs_partner=True,
    )
    private_row = next(row for row in storage["tournament_registrations"] if row["email"] == "private@example.com")
    private_row["wants_partner_board_contact"] = False

    response = api.get("/clubs/tres-palapas/tournament-roster", params={"registration_slug": "tres-open"})

    assert response.status_code == 200
    payload = response.json()
    assert [row["player_name"] for row in payload["roster"]["players_needing_partners"]] == ["Visible Player"]
    assert [row["player_name"] for row in payload["roster"]["partner_board_entries"]] == ["Visible Player"]
    visible_entry = payload["roster"]["partner_board_entries"][0]
    assert visible_entry["player_entry_key"] == build_public_tournament_reference(
        tournament_id="t1",
        namespace="partner-board-registration",
        source_id=visible_registration,
    )
    assert visible_registration not in str(visible_entry)
    assert "visible@example.com" not in str(payload).lower()
    assert "private@example.com" not in str(payload).lower()
