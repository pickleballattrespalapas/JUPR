from __future__ import annotations

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
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": mode, "show_on_partner_board": board}],
        },
    )
    selection = next(row for row in supabase.storage["tournament_registration_selections"] if row["registration_id"] == result["registration_id"])
    return result["registration_id"], selection["id"]


def test_public_pairing_interest_emails_player_and_organizer(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_REGISTRATION_EDIT_SECRET", "test-secret")
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    requester_registration_id, requester_selection_id = _submit(supabase, first_name="Alex", email="alex@example.com")
    _target_registration_id, target_selection_id = _submit(supabase, first_name="Casey", email="casey@example.com", mode="NEEDS_PARTNER", board=True)
    token = build_registration_edit_token(
        tournament_id="t1",
        registration_id=requester_registration_id,
        email="alex@example.com",
        secret="test-secret",
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
        target_selection_id=target_selection_id,
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
