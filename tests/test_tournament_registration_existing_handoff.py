from copy import deepcopy
from urllib.parse import parse_qs, urlparse

import pytest

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient
from services.api.main import app
from jupr_app.services import public_tournament_registration_edit_service as edit_service
from jupr_app.services import public_tournament_registration_service as registration_service
from tests.test_public_tournament_registration_edit_service import _confirmed_reciprocal_registration


@pytest.fixture
def partner_handoff(monkeypatch):
    db, storage, _, _ = _confirmed_reciprocal_registration(monkeypatch)
    storage["clubs"] = [{"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}]
    storage["tournament_event_options"].append({
        **storage["tournament_event_options"][0], "id": "singles", "event_type": "SINGLES",
        "label": "Singles Open", "division_name": "Singles Open", "event_family_label": "Singles",
        "skill_label": "Open", "skill_mode": "OPEN", "partner_required": False,
    })
    monkeypatch.setattr("services.api.main.create_client", lambda *_: db)
    for key, value in {
        "SUPABASE_URL": "http://example.local", "SUPABASE_SERVICE_ROLE_KEY": "test-service-role",
        "SUPABASE_ANON_KEY": "local", "JUPR_ENV": "staging", "JUPR_EMAIL_MODE": "dry_run",
        "JUPR_STAGING_WRITE_WAVE": "public-intake-auth", "JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES": "1",
    }.items():
        monkeypatch.setenv(key, value)
    payload = {
        "registration_slug": "tres-open", "first_name": "Sam", "last_name": "Partner",
        "email": "  PARTNER@example.com ", "age": 61, "gender": "Women", "doubles_skill": 3.52,
        "terms_accepted": True, "selections": [{"event_option_id": "singles", "partner_mode": "NONE"}],
    }
    deliveries = []

    def send(**kwargs):
        deliveries.append(kwargs)
        return {"status": "sent"}

    monkeypatch.setattr(edit_service, "send_tournament_registration_edit_email", send)
    return TestClient(app), db, storage, payload, deliveries


PATH = "/clubs/tres-palapas/tournament-registration"
DOMAIN_TABLES = ("tournament_registrations", "tournament_registration_selections", "tournament_registration_team_links", "players")


def test_partner_can_finish_form_then_use_emailed_link_to_add_event(partner_handoff):
    api, _, storage, payload, deliveries = partner_handoff
    before = {key: deepcopy(storage[key]) for key in DOMAIN_TABLES}
    preflight = api.post(PATH + "/profile-resolution", json=payload)
    assert preflight.status_code == 200
    assert preflight.json()["can_start_new"] is True
    assert deliveries == []

    response = api.post(PATH, json=payload)
    assert response.status_code == 200
    outcome = response.json()
    assert outcome["mode"] == "existing_registration"
    assert outcome["edit_link_status"] == "sent"
    assert "We’ve sent an email" in outcome["message"]
    assert set(outcome) == {"club", "ok", "mode", "edit_link_status", "message"}
    assert {key: storage[key] for key in DOMAIN_TABLES} == before
    assert len(deliveries) == 1
    assert deliveries[0]["registered_email"] == "partner@example.com"
    query = parse_qs(urlparse(deliveries[0]["edit_url"]).query)
    token = query["edit_token"][0]
    edit = api.get(PATH + "/edit", params={"edit_token": token, "tournament_id": "t1"})
    assert edit.status_code == 200
    page = edit.json()
    assert page["registration"]["id"] == "reg-partner"
    assert page["selections"][0]["partner_email"] == "alex@example.com"
    saved = api.post(PATH + "/edit", json={
        **page["registration"], "tournament_id": "t1", "edit_token": token, "terms_accepted": True,
        "expected_updated_at": page["registration"]["updated_at"],
        "expected_selection_versions": [{"id": row["id"], "updated_at": row["updated_at"]} for row in page["selections"]],
        "selections": [*page["selections"], {"event_option_id": "singles", "partner_mode": "NONE"}],
    })
    assert saved.status_code == 200, saved.text
    assert len(storage["tournament_registrations"]) == 2
    assert storage["tournament_registration_team_links"] == before["tournament_registration_team_links"]
    assert {row["event_option_id"] for row in storage["tournament_registration_selections"] if row["registration_id"] == "reg-partner"} == {"event1", "singles"}


def test_repeat_submissions_and_manual_request_send_only_one_link(partner_handoff):
    api, _, _, payload, deliveries = partner_handoff
    first = api.post(PATH, json=payload).json()
    second = api.post(PATH, json={**payload, "attempt_id": "different-browser-attempt"}).json()
    assert first == second
    manual = api.post(PATH + "/edit-link/request", json={
        "registration_slug": "tres-open", "email": payload["email"], "idempotency_key": "another-request-key",
    })
    assert manual.status_code == 200
    assert manual.json()["message"].startswith("If that email matches")
    assert len(deliveries) == 1


def test_email_failure_is_honest_preserves_registration_and_does_not_retry_send(partner_handoff, monkeypatch):
    api, _, storage, payload, deliveries = partner_handoff
    before = {key: deepcopy(storage[key]) for key in DOMAIN_TABLES}

    def failed(**kwargs):
        deliveries.append(kwargs)
        raise TimeoutError("unknown delivery outcome")

    monkeypatch.setattr(edit_service, "send_tournament_registration_edit_email", failed)
    for _ in range(2):
        response = api.post(PATH, json=payload)
        assert response.status_code == 200
        assert response.json()["edit_link_status"] == "unavailable"
        assert "couldn’t confirm" in response.json()["message"]
    assert len(deliveries) == 1
    assert {key: storage[key] for key in DOMAIN_TABLES} == before


def test_imported_registration_explains_organizer_help_without_claiming_email(partner_handoff, monkeypatch):
    api, _, _, payload, deliveries = partner_handoff
    monkeypatch.setattr(edit_service, "registration_has_imported_draw_selection", lambda *_args, **_kwargs: True)
    response = api.post(PATH, json=payload)
    assert response.json()["edit_link_status"] == "locked"
    assert "contact the organizer" in response.json()["message"]
    assert deliveries == []


def test_honeypot_never_triggers_duplicate_recovery_email(partner_handoff):
    api, _, _, payload, deliveries = partner_handoff
    response = api.post(PATH, json={**payload, "website": "spam"})
    assert response.status_code == 400
    assert deliveries == []


@pytest.mark.parametrize("commerce_available", [False, True])
def test_atomic_duplicate_conflict_also_recovers(partner_handoff, monkeypatch, commerce_available):
    api, _, _, payload, deliveries = partner_handoff
    # Model a race after the initial lookup, including the commerce create RPC.
    monkeypatch.setattr(registration_service, "get_registration_by_email", lambda *_: None)
    original_page = registration_service.build_public_tournament_registration_page
    monkeypatch.setattr(registration_service, "build_public_tournament_registration_page",
                        lambda *a, **kw: {**original_page(*a, **kw), "commerce": {"available": commerce_available}})
    monkeypatch.setattr(registration_service, "require_tournament_commerce_mutation_runtime", lambda **_: None)
    monkeypatch.setattr(registration_service, "prepare_public_registration_commerce_transaction", lambda *a, **kw: {})

    def duplicate(*_args, **_kwargs):
        raise ValueError("You’re already registered with this email. Request an edit link to make changes.")

    monkeypatch.setattr(registration_service, "save_registration", duplicate)
    response = api.post(PATH, json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["edit_link_status"] == "sent"
    assert len(deliveries) == 1
