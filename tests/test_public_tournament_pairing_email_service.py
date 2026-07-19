from __future__ import annotations

import pytest

from jupr_app.domain.tournament_public_references import build_public_tournament_reference
from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token
from jupr_app.services import public_tournament_partner_request_service as pairing_service
from jupr_app.services.public_tournament_partner_request_service import create_public_tournament_partner_request
from jupr_app.services.public_tournament_registration_service import submit_public_tournament_registration
from tests.test_public_tournament_registration_service import FakeSupabase, fake_storage


def _submit(supabase, *, first_name: str, email: str, mode: str = "NONE", board: bool = False) -> tuple[str, str]:
    result = submit_public_tournament_registration(
        supabase,
        club_id="club-1",
        payload={
            "registration_slug": "tres-open",
            "first_name": first_name,
            "last_name": "Player",
            "email": email,
            "doubles_skill": 4.0,
            "wants_partner_board_contact": board,
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": mode, "show_on_partner_board": board}],
        },
    )
    selection = next(row for row in supabase.storage["tournament_registration_selections"] if row["registration_id"] == result["registration_id"])
    return result["registration_id"], selection["id"]


def test_public_pairing_interest_emails_player_and_organizer(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "test-secret-for-partner-flow-1234567890")
    monkeypatch.setenv("JUPR_WEB_BASE_URL", "https://next.example.com")
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    supabase.rpc = None
    requester_registration_id, requester_selection_id = _submit(supabase, first_name="Alex", email="alex@example.com")
    _target_registration_id, target_selection_id = _submit(supabase, first_name="Casey", email="casey@example.com", mode="NEEDS_PARTNER", board=True)
    token = build_registration_edit_token(
        tournament_id="t1",
        registration_id=requester_registration_id,
        email="alex@example.com",
        secret="test-secret-for-partner-flow-1234567890",
    )
    captured = {}

    def fake_send_pairing_interest_emails(**kwargs):
        captured.update(kwargs)
        return {
            "player": {"status": "dry_run", "provider_message_id": "dry_run", "to_email": kwargs["target_email"]},
            "organizer": {"status": "dry_run", "provider_message_id": "dry_run", "to_email": "joe@juprleagues.com"},
        }

    monkeypatch.setattr(pairing_service, "send_pairing_interest_emails", fake_send_pairing_interest_emails)

    result = create_public_tournament_partner_request(
        supabase,
        club_id="club-1",
        club_slug="tres-palapas",
        tournament_id="t1",
        edit_token=token,
        requester_selection_id=requester_selection_id,
        target_public_entry_key=build_public_tournament_reference(
            tournament_id="t1",
            namespace="partner-board-selection",
            source_id=target_selection_id,
        ),
    )

    assert result["ok"] is True
    assert result["status"] == "PENDING"
    assert result["notification_status"] == {"player": "dry_run", "organizer": "dry_run"}
    assert captured["requester_name"] == "Alex Player"
    assert captured["target_name"] == "Casey Player"
    assert captured["target_email"] == "casey@example.com"
    assert captured["board_url"].startswith("https://")
    assert len(storage["tournament_registration_partner_requests"]) == 1


def test_public_pairing_interest_honeypot_does_not_email(monkeypatch) -> None:
    storage = fake_storage()
    calls = {"emails": 0}

    def fake_send_pairing_interest_emails(**_kwargs):
        calls["emails"] += 1
        return {}

    monkeypatch.setattr(pairing_service, "send_pairing_interest_emails", fake_send_pairing_interest_emails)

    result = create_public_tournament_partner_request(
        FakeSupabase(storage),
        club_id="club-1",
        edit_token="not-used",
        requester_selection_id="sel1",
        target_selection_id="sel2",
        website="bot field",
    )

    assert result["ok"] is True
    assert result["status"] == "accepted"
    assert calls["emails"] == 0
    assert storage["tournament_registration_partner_requests"] == []


def test_duplicate_interest_is_idempotent_and_does_not_repeat_email(monkeypatch) -> None:
    secret = "test-secret-for-partner-flow-1234567890"
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", secret)
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    supabase.rpc = None
    requester_registration_id, requester_selection_id = _submit(supabase, first_name="Alex", email="alex@example.com")
    _target_registration_id, target_selection_id = _submit(supabase, first_name="Casey", email="casey@example.com", mode="NEEDS_PARTNER", board=True)
    token = build_registration_edit_token(
        tournament_id="t1",
        registration_id=requester_registration_id,
        email="alex@example.com",
        secret=secret,
    )
    calls = {"count": 0}

    def fake_send_pairing_interest_emails(**_kwargs):
        calls["count"] += 1
        return {"player": {"status": "dry_run"}, "organizer": {"status": "dry_run"}}

    monkeypatch.setattr(pairing_service, "send_pairing_interest_emails", fake_send_pairing_interest_emails)

    first = create_public_tournament_partner_request(
        supabase,
        club_id="club-1",
        edit_token=token,
        requester_selection_id=requester_selection_id,
        target_selection_id=target_selection_id,
        tournament_id="t1",
    )
    retry = create_public_tournament_partner_request(
        supabase,
        club_id="club-1",
        edit_token=token,
        requester_selection_id=requester_selection_id,
        target_selection_id=target_selection_id,
        tournament_id="t1",
    )

    assert first["idempotent"] is False
    assert retry["idempotent"] is True
    assert retry["partner_request_id"] == first["partner_request_id"]
    assert retry["notification_status"] == {"player": "not_repeated", "organizer": "not_repeated"}
    assert calls["count"] == 1
    assert len(storage["tournament_registration_partner_requests"]) == 1


def test_interest_write_survives_notification_failure(monkeypatch) -> None:
    secret = "test-secret-for-partner-flow-1234567890"
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", secret)
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    supabase.rpc = None
    requester_registration_id, requester_selection_id = _submit(supabase, first_name="Alex", email="alex@example.com")
    _target_registration_id, target_selection_id = _submit(supabase, first_name="Casey", email="casey@example.com", mode="NEEDS_PARTNER", board=True)
    token = build_registration_edit_token(
        tournament_id="t1",
        registration_id=requester_registration_id,
        email="alex@example.com",
        secret=secret,
    )
    monkeypatch.setattr(
        pairing_service,
        "send_pairing_interest_emails",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("smtp unavailable")),
    )

    result = create_public_tournament_partner_request(
        supabase,
        club_id="club-1",
        edit_token=token,
        requester_selection_id=requester_selection_id,
        target_selection_id=target_selection_id,
        tournament_id="t1",
    )

    assert result["ok"] is True
    assert result["notification_status"] == {"player": "failed", "organizer": "failed"}
    assert len(storage["tournament_registration_partner_requests"]) == 1


def test_contact_denylist_suppresses_player_delivery_without_exposing_email(monkeypatch) -> None:
    secret = "test-secret-for-partner-flow-1234567890"
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", secret)
    monkeypatch.setenv("JUPR_TOURNAMENT_PARTNER_CONTACT_DENYLIST", "@example.com")
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    supabase.rpc = None
    requester_registration_id, requester_selection_id = _submit(supabase, first_name="Alex", email="alex@example.com")
    _target_registration_id, target_selection_id = _submit(supabase, first_name="Casey", email="casey@example.com", mode="NEEDS_PARTNER", board=True)
    token = build_registration_edit_token(
        tournament_id="t1",
        registration_id=requester_registration_id,
        email="alex@example.com",
        secret=secret,
    )
    captured = {}

    def fake_send_pairing_interest_emails(**kwargs):
        captured.update(kwargs)
        return {"player": {"status": "skipped"}, "organizer": {"status": "dry_run"}}

    monkeypatch.setattr(pairing_service, "send_pairing_interest_emails", fake_send_pairing_interest_emails)
    result = create_public_tournament_partner_request(
        supabase,
        club_id="club-1",
        edit_token=token,
        requester_selection_id=requester_selection_id,
        target_selection_id=target_selection_id,
        tournament_id="t1",
    )

    assert captured["target_email"] == ""
    assert "edit_token" not in str(result).lower()
    assert "casey@example.com" not in str(result).lower()
    assert result["notification_status"] == {"player": "skipped", "organizer": "dry_run"}


def test_disabled_partner_board_refuses_request_before_write_or_email(monkeypatch) -> None:
    secret = "test-secret-for-partner-flow-1234567890"
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", secret)
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    supabase.rpc = None
    requester_registration_id, requester_selection_id = _submit(
        supabase,
        first_name="Alex",
        email="alex@example.com",
    )
    _target_registration_id, target_selection_id = _submit(
        supabase,
        first_name="Casey",
        email="casey@example.com",
        mode="NEEDS_PARTNER",
        board=True,
    )
    storage["tournament_registration_settings"][0]["partner_board_enabled"] = False
    token = build_registration_edit_token(
        tournament_id="t1",
        registration_id=requester_registration_id,
        email="alex@example.com",
        secret=secret,
    )
    calls = {"count": 0}
    monkeypatch.setattr(
        pairing_service,
        "send_pairing_interest_emails",
        lambda **_kwargs: calls.update(count=calls["count"] + 1),
    )

    with pytest.raises(ValueError, match="partner board is not available"):
        create_public_tournament_partner_request(
            supabase,
            club_id="club-1",
            edit_token=token,
            requester_selection_id=requester_selection_id,
            target_selection_id=target_selection_id,
            tournament_id="t1",
        )

    assert calls["count"] == 0
    assert storage["tournament_registration_partner_requests"] == []
