from __future__ import annotations

from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token
from jupr_app.services import public_tournament_registration_edit_service as edit_service
from jupr_app.services.public_tournament_registration_edit_service import (
    build_public_tournament_registration_edit_page,
    request_public_tournament_registration_edit_link,
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


def test_registration_edit_link_request_sends_email_without_exposing_match(monkeypatch) -> None:
    supabase, _storage, _registration_id, _token = _registered_supabase(monkeypatch)
    monkeypatch.setenv("JUPR_WEB_BASE_URL", "https://next.example.com")
    captured: dict[str, str] = {}

    def fake_send(**kwargs):
        captured.update({key: str(value) for key, value in kwargs.items()})
        return {"status": "dry_run", "provider_message_id": "dry_run", "to_email": kwargs["registered_email"]}

    monkeypatch.setattr(edit_service, "send_tournament_registration_edit_email", fake_send)

    payload = request_public_tournament_registration_edit_link(
        supabase,
        club_id="club-1",
        club_slug="tres-palapas",
        registration_slug="tres-open",
        email="alex@example.com",
    )

    assert payload == {
        "ok": True,
        "mode": "registration_edit_link_request",
        "accepted": True,
        "message": "If a matching registration exists, an edit link will be sent to that email address.",
    }
    assert captured["registered_email"] == "alex@example.com"
    assert captured["tournament_name"] == "Tres Palapas Open"
    assert captured["edit_url"].startswith("https://next.example.com/clubs/tres-palapas/tournament-registration/edit?")
    assert "edit_token=" in captured["edit_url"]
    assert "edit_token" not in str(payload)


def test_registration_edit_link_request_missing_email_does_not_enumerate(monkeypatch) -> None:
    supabase, _storage, _registration_id, _token = _registered_supabase(monkeypatch)
    calls = {"send": 0}

    def fake_send(**_kwargs):
        calls["send"] += 1
        return {}

    monkeypatch.setattr(edit_service, "send_tournament_registration_edit_email", fake_send)

    payload = request_public_tournament_registration_edit_link(
        supabase,
        club_id="club-1",
        club_slug="tres-palapas",
        registration_slug="tres-open",
        email="missing@example.com",
    )

    assert payload["ok"] is True
    assert payload["accepted"] is True
    assert calls["send"] == 0


def test_registration_edit_link_request_honeypot_is_silent(monkeypatch) -> None:
    supabase, _storage, _registration_id, _token = _registered_supabase(monkeypatch)
    calls = {"send": 0}
    monkeypatch.setattr(edit_service, "send_tournament_registration_edit_email", lambda **_kwargs: calls.__setitem__("send", calls["send"] + 1))

    payload = request_public_tournament_registration_edit_link(
        supabase,
        club_id="club-1",
        club_slug="tres-palapas",
        registration_slug="tres-open",
        email="alex@example.com",
        website="bot field",
    )

    assert payload["ok"] is True
    assert calls["send"] == 0
