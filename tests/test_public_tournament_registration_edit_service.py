from __future__ import annotations

from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token
from jupr_app.services.public_tournament_registration_edit_service import (
    build_public_tournament_registration_edit_page,
    submit_public_tournament_registration_edit,
)
from jupr_app.services.public_tournament_registration_service import submit_public_tournament_registration
from tests.test_public_tournament_registration_service import FakeSupabase, fake_storage


def _registered_supabase(monkeypatch):
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "test-secret")
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    result = submit_public_tournament_registration(
        supabase,
        club_id="club-1",
        payload={
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
    token = build_registration_edit_token(
        tournament_id="t1",
        registration_id=result["registration_id"],
        email="alex@example.com",
        secret="test-secret",
    )
    return supabase, storage, result["registration_id"], token


def test_registration_edit_page_verifies_token_and_hydrates_registration(monkeypatch) -> None:
    supabase, _storage, registration_id, token = _registered_supabase(monkeypatch)

    payload = build_public_tournament_registration_edit_page(
        supabase,
        club_id="club-1",
        edit_token=token,
        registration_slug="tres-open",
    )

    assert payload["edit_token_valid"] is True
    assert payload["registration"]["id"] == registration_id
    assert payload["registration"]["email"] == "alex@example.com"
    assert payload["registration"]["phone"] == "555-0100"
    assert payload["selections"][0]["event_option_id"] == "event1"
    assert "phone" in payload["registration"]
    assert "admin_notes" not in payload["tournament"]
    assert "internal_seed_notes" not in payload["events"][0]


def test_registration_edit_submit_updates_existing_registration_and_locks_email(monkeypatch) -> None:
    supabase, storage, registration_id, token = _registered_supabase(monkeypatch)

    result = submit_public_tournament_registration_edit(
        supabase,
        club_id="club-1",
        edit_token=token,
        payload={
            "tournament_id": "t1",
            "registration_slug": "tres-open",
            "first_name": "Alexis",
            "last_name": "Rivera",
            "display_name": "Alexis R",
            "email": "evil@example.com",
            "phone": "555-9999",
            "doubles_skill": 4.25,
            "terms_accepted": True,
            "selections": [
                {
                    "event_option_id": "event1",
                    "partner_mode": "NEEDS_PARTNER",
                    "show_on_partner_board": True,
                    "partner_note": "Looking for a steady partner",
                }
            ],
        },
    )

    assert result["ok"] is True
    assert result["registration_id"] == registration_id
    registrations = storage["tournament_registrations"]
    assert len(registrations) == 1
    assert registrations[0]["display_name"] == "Alexis R"
    assert registrations[0]["email"] == "alex@example.com"
    assert registrations[0]["phone"] == "555-9999"
    selections = storage["tournament_registration_selections"]
    assert len(selections) == 1
    assert selections[0]["partner_mode"] == "NEEDS_PARTNER"
    assert selections[0]["show_on_partner_board"] is True


def test_registration_edit_rejects_wrong_club(monkeypatch) -> None:
    supabase, _storage, _registration_id, token = _registered_supabase(monkeypatch)

    try:
        build_public_tournament_registration_edit_page(
            supabase,
            club_id="other-club",
            edit_token=token,
            registration_slug="tres-open",
        )
    except ValueError as exc:
        assert "different club" in str(exc)
    else:
        raise AssertionError("Expected wrong-club edit link rejection")
