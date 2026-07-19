import pytest

from jupr_app.domain import tournament_partner_service as domain
from jupr_app.services import public_tournament_partner_request_service as public_service
from tests.test_tournament_partner_requests import _FakeSupabase, _storage as _base_storage


def _storage():
    storage = _base_storage()
    for registration in storage["tournament_registrations"]:
        registration.update({"status": "CONFIRMED", "wants_partner_board_contact": True})
    for selection in storage["tournament_registration_selections"]:
        if selection["id"] in {"sel_elizabeth", "sel_alice"}:
            selection["show_on_partner_board"] = True
    storage["tournament_registration_settings"] = [
        {"id": "settings_1", "tournament_id": "tour-1", "partner_board_enabled": True}
    ]
    storage["tournament_event_options"] = [
        {
            "id": "event-wd-35",
            "tournament_id": "tour-1",
            "enabled": True,
            "partner_board_enabled": True,
            "status": "confirmed",
        }
    ]
    return storage


def _verified_bundle_for(registration_id: str, selection_id: str):
    registration = {
        "id": registration_id,
        "email": f"{registration_id}@example.com",
        "display_name": registration_id.replace("reg_", "").title(),
        "wants_partner_board_contact": True,
        "status": "CONFIRMED",
    }
    return (
        {"tournament_id": "tour-1", "registration_id": registration_id, "email": registration["email"]},
        {
            "registration": registration,
            "settings": {"registration_slug": "spring"},
            "selections": [
                {
                    "id": selection_id,
                    "tournament_id": "tour-1",
                    "registration_id": registration_id,
                    "event_option_id": "event-wd-35",
                }
            ],
        },
    )


def test_atomic_fallback_is_idempotent_and_accept_cancels_competing_requests():
    storage = _storage()
    supabase = _FakeSupabase(storage)

    first = domain.create_partner_request_atomic(
        supabase,
        request_id="preq_first",
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        requester_selection_id="sel_mary",
        target_selection_id="sel_elizabeth",
        target_display_name_snapshot="Elizabeth Whelan",
        source="PUBLIC_PARTNER_BOARD",
    )
    retry = domain.create_partner_request_atomic(
        supabase,
        request_id="preq_retry",
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        requester_selection_id="sel_mary",
        target_selection_id="sel_elizabeth",
        target_display_name_snapshot="Elizabeth Whelan",
        source="PUBLIC_PARTNER_BOARD",
    )
    competing = domain.create_partner_request_atomic(
        supabase,
        request_id="preq_competing",
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        requester_selection_id="sel_alice",
        target_selection_id="sel_elizabeth",
        target_display_name_snapshot="Elizabeth Whelan",
        source="PUBLIC_PARTNER_BOARD",
    )

    assert first["outcome"] == "created"
    assert retry["id"] == first["id"]
    assert retry["idempotent"] is True
    assert len(storage["tournament_registration_partner_requests"]) == 2

    accepted = domain.transition_partner_request_atomic(
        supabase,
        request_id=first["id"],
        actor_selection_id="sel_elizabeth",
        action="accept",
    )
    accepted_retry = domain.transition_partner_request_atomic(
        supabase,
        request_id=first["id"],
        actor_selection_id="sel_elizabeth",
        action="accept",
    )
    stale = domain.transition_partner_request_atomic(
        supabase,
        request_id=competing["id"],
        actor_selection_id="sel_elizabeth",
        action="accept",
    )

    assert accepted["status"] == "ACCEPTED"
    assert accepted["cancelled_request_ids"] == [competing["id"]]
    assert accepted_retry["outcome"] == "idempotent"
    assert accepted_retry["team_link_id"] == accepted["team_link_id"]
    assert stale == {
        "outcome": "stale",
        "idempotent": False,
        "status": "CANCELLED",
        "partner_request_id": competing["id"],
        "team_link_id": None,
        "cancelled_request_ids": [],
    }
    assert len(storage["tournament_registration_team_links"]) == 1
    assert len(storage["tournament_registration_team_members"]) == 2
    accepted_selection_ids = {"sel_mary", "sel_elizabeth"}
    assert all(
        row.get("show_on_partner_board") is False
        for row in storage["tournament_registration_selections"]
        if row["id"] in accepted_selection_ids
    )


def test_decline_and_requester_cancel_are_owned_and_idempotent():
    storage = _storage()
    supabase = _FakeSupabase(storage)
    declined_request = domain.create_partner_request_atomic(
        supabase,
        request_id="preq_decline",
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        requester_selection_id="sel_mary",
        target_selection_id="sel_elizabeth",
        target_display_name_snapshot="Elizabeth Whelan",
        source="PUBLIC_PARTNER_BOARD",
    )

    with pytest.raises(ValueError, match="requested partner"):
        domain.transition_partner_request_atomic(
            supabase,
            request_id=declined_request["id"],
            actor_selection_id="sel_mary",
            action="decline",
        )

    declined = domain.transition_partner_request_atomic(
        supabase,
        request_id=declined_request["id"],
        actor_selection_id="sel_elizabeth",
        action="decline",
    )
    declined_retry = domain.transition_partner_request_atomic(
        supabase,
        request_id=declined_request["id"],
        actor_selection_id="sel_elizabeth",
        action="decline",
    )
    assert declined["status"] == "DECLINED"
    assert declined_retry["idempotent"] is True

    cancelled_request = domain.create_partner_request_atomic(
        supabase,
        request_id="preq_cancel",
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        requester_selection_id="sel_mary",
        target_selection_id="sel_elizabeth",
        target_display_name_snapshot="Elizabeth Whelan",
        source="PUBLIC_PARTNER_BOARD",
    )
    with pytest.raises(ValueError, match="requester"):
        domain.transition_partner_request_atomic(
            supabase,
            request_id=cancelled_request["id"],
            actor_selection_id="sel_elizabeth",
            action="cancel",
        )
    cancelled = domain.transition_partner_request_atomic(
        supabase,
        request_id=cancelled_request["id"],
        actor_selection_id="sel_mary",
        action="cancel",
    )
    cancelled_retry = domain.transition_partner_request_atomic(
        supabase,
        request_id=cancelled_request["id"],
        actor_selection_id="sel_mary",
        action="cancel",
    )
    assert cancelled["status"] == "CANCELLED"
    assert cancelled_retry["idempotent"] is True


def test_accept_cancels_stale_request_when_target_withdraws_contact_consent():
    storage = _storage()
    supabase = _FakeSupabase(storage)
    request = domain.create_partner_request_atomic(
        supabase,
        request_id="preq_stale_consent",
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        requester_selection_id="sel_mary",
        target_selection_id="sel_elizabeth",
        target_display_name_snapshot="Elizabeth Whelan",
        source="PUBLIC_PARTNER_BOARD",
    )
    target_registration = next(
        row for row in storage["tournament_registrations"] if row["id"] == "reg_elizabeth"
    )
    target_registration["wants_partner_board_contact"] = False

    result = domain.transition_partner_request_atomic(
        supabase,
        request_id=request["id"],
        actor_selection_id="sel_elizabeth",
        action="accept",
    )

    assert result["outcome"] == "stale"
    assert result["status"] == "CANCELLED"
    assert storage["tournament_registration_team_links"] == []
    assert storage["tournament_registration_team_members"] == []


def test_rpc_validation_error_is_reduced_to_safe_domain_error():
    class FailingCall:
        def execute(self):
            raise RuntimeError({"message": "JUPR_PARTNER_TRANSACTION: private database detail"})

    class RpcSupabase:
        def rpc(self, _name, _params):
            return FailingCall()

    with pytest.raises(ValueError, match="state changed or is invalid") as exc_info:
        domain.create_partner_request_atomic(
            RpcSupabase(),
            request_id="preq_error",
            tournament_id="tour-1",
            event_option_id="event-wd-35",
            requester_selection_id="sel_mary",
            target_selection_id="sel_elizabeth",
            target_display_name_snapshot="Elizabeth Whelan",
            source="PUBLIC_PARTNER_BOARD",
        )

    assert "private database detail" not in str(exc_info.value)


def test_request_review_projection_contains_actions_but_no_contact_fields(monkeypatch):
    storage = _storage()
    storage["tournament_registrations"][0]["email"] = "mary@example.com"
    storage["tournament_registrations"][0]["phone"] = "+1-secret"
    storage["tournament_registrations"][1]["email"] = "elizabeth@example.com"
    request = domain.create_partner_request(
        _FakeSupabase(storage),
        tournament_id="tour-1",
        event_option_id="event-wd-35",
        requester_selection_id="sel_mary",
        target_selection_id="sel_elizabeth",
        source="PUBLIC_PARTNER_BOARD",
    )
    monkeypatch.setattr(
        public_service,
        "_verified_bundle",
        lambda *_args, **_kwargs: _verified_bundle_for("reg_elizabeth", "sel_elizabeth"),
    )

    payload = public_service.list_public_tournament_partner_requests(
        _FakeSupabase(storage),
        club_id="club",
        edit_token="token",
        tournament_id="tour-1",
    )
    row = payload["incoming"][0]

    assert row["id"] == request["id"]
    assert row["available_actions"] == ["accept", "decline"]
    serialized = str(payload).lower()
    assert "mary@example.com" not in serialized
    assert "elizabeth@example.com" not in serialized
    assert "+1-secret" not in serialized
    assert "edit_token" not in serialized
