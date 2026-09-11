from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
import pytest

from jupr_app.services.public_tournament_registration_service import submit_public_tournament_registration
from jupr_app.services.public_tournament_roster_service import build_public_tournament_roster_page
from jupr_app.domain.tournament_partner_service import accept_partner_request, create_partner_request
from tests.test_public_tournament_registration_service import FakeSupabase, fake_storage


@pytest.fixture
def reserved_roster_storage():
    storage = fake_storage()
    storage["tournament_event_options"][0].update(event_type="DOUBLES", partner_required=True)
    storage["tournament_registrations"] = [dict(id="registered", tournament_id="t1", display_name="Alex Player",
        email="alex.private@example.com", status="CONFIRMED", wants_partner_board_contact=True)]
    storage["tournament_registration_selections"] = [dict(id="reserved-selection", tournament_id="t1", registration_id="registered",
        registration_day_id="day1", event_option_id="event1", partner_mode="NEEDS_PARTNER", show_on_partner_board=True)]
    storage["tournament_partner_invitations"] = [dict(id="private-invitation", tournament_id="t1", status="RESERVED",
        target_selection_id="reserved-selection", requester_name="Casey Guest", requester_email="casey.private@example.com",
        message="A private message", expires_at=(datetime.now(timezone.utc) + timedelta(days=2)).isoformat())]
    return storage


def test_accepted_guest_appears_as_pending_registration_without_becoming_a_registrant(reserved_roster_storage):
    payload = build_public_tournament_roster_page(FakeSupabase(reserved_roster_storage), club_id="club-1", registration_slug="tres-open")
    roster = payload["roster"]
    entry = roster["registrations_by_event"][0]
    assert entry["status"] == "Pending Registration"
    assert [member["display_name"] for member in entry["members"]] == ["Alex Player", "Casey Guest"]
    assert entry["members"][1]["registration_pending"] is True
    assert entry["entry_type"] == "Team" and entry["combined_rating"] is None
    assert roster["confirmed_teams"] == []
    assert roster["partner_board_entries"] == [] and roster["players_needing_partners"] == []
    assert payload["summary"]["total_players"] == 1 and payload["summary"]["total_registrations"] == 1
    serialized = json.dumps(roster)
    for private in ["casey.private@example.com", "alex.private@example.com", "private-invitation", "reserved-selection", "A private message"]:
        assert private not in serialized


@pytest.mark.parametrize("status,expired,other_tournament", [
    ("PENDING", False, False), ("DECLINED", False, False), ("CANCELLED", False, False),
    ("EXPIRED", False, False), ("RESERVED", True, False), ("RESERVED", False, True),
])
def test_only_current_accepted_reservations_show_on_roster(reserved_roster_storage, status, expired, other_tournament):
    invitation = reserved_roster_storage["tournament_partner_invitations"][0]
    invitation["status"] = status
    if expired: invitation["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    if other_tournament: invitation["tournament_id"] = "another-tournament"
    payload = build_public_tournament_roster_page(FakeSupabase(reserved_roster_storage), club_id="club-1", registration_slug="tres-open")
    assert "Casey Guest" not in json.dumps(payload["roster"])
    assert payload["roster"]["registrations_by_event"][0]["status"] == "Needs Partner"


def test_completed_registration_replaces_pending_display_with_one_confirmed_team(reserved_roster_storage):
    storage = reserved_roster_storage
    supabase = FakeSupabase(storage)
    before = build_public_tournament_roster_page(supabase, club_id="club-1", registration_slug="tres-open")
    assert before["roster"]["registrations_by_event"][0]["status"] == "Pending Registration"
    storage["tournament_registrations"].append(dict(id="guest-registered", tournament_id="t1",
        display_name="Casey Guest", email="casey.private@example.com", status="CONFIRMED"))
    storage["tournament_registration_selections"].append(dict(id="guest-selection", tournament_id="t1",
        registration_id="guest-registered", registration_day_id="day1", event_option_id="event1", partner_mode="NEEDS_PARTNER"))
    request = create_partner_request(supabase, tournament_id="t1", event_option_id="event1",
        requester_selection_id="guest-selection", target_selection_id="reserved-selection", source="NEEDS_PARTNER_LIST")
    accept_partner_request(supabase, request_id=request["id"], accepted_by_selection_id="reserved-selection")
    storage["tournament_partner_invitations"][0]["status"] = "COMPLETED"
    after = build_public_tournament_roster_page(supabase, club_id="club-1", registration_slug="tres-open")
    entries = after["roster"]["registrations_by_event"]
    assert len(entries) == 1 and entries[0]["status"] == "Registered"
    assert {member["display_name"] for member in entries[0]["members"]} == {"Alex Player", "Casey Guest"}
    assert all(not member.get("registration_pending") for member in entries[0]["members"])
    assert len(after["roster"]["confirmed_teams"]) == 1
    assert after["summary"]["total_registrations"] == 2 and after["summary"]["total_players"] == 2


@pytest.mark.parametrize("name", ["Guest guest.private@example.com", "+1 555 012 3456", ""])
def test_pending_names_do_not_publish_contact_details(reserved_roster_storage, name):
    reserved_roster_storage["tournament_partner_invitations"][0]["requester_name"] = name
    payload = build_public_tournament_roster_page(FakeSupabase(reserved_roster_storage), club_id="club-1", registration_slug="tres-open")
    assert payload["roster"]["registrations_by_event"][0]["members"][1]["display_name"] == "Player"


@pytest.mark.parametrize("event_type,partner_mode", [("SINGLES", "NONE"), ("DOUBLES", "NEEDS_PARTNER")])
def test_public_roster_counts_only_active_registrations(event_type, partner_mode):
    storage = fake_storage()
    storage["tournament_event_options"][0].update({"event_type": event_type, "partner_required": event_type != "SINGLES"})
    storage["tournament_registrations"] = [
        {"id": key, "tournament_id": "t1", "email": f"{key}@example.com", "display_name": name, "status": status, "submitted_at": "2026-06-01T10:00:00Z", "wants_partner_board_contact": True}
        for key, name, status in [("active", "Active Player", "confirmed"), ("cancelled", "Cancelled Player", "cancelled")]
    ]
    storage["tournament_registration_selections"] = [
        {"id": f"selection-{key}", "tournament_id": "t1", "registration_id": key, "registration_day_id": "day1", "event_option_id": "event1", "partner_mode": partner_mode, "show_on_partner_board": True}
        for key in ["active", "cancelled"]
    ]
    payload = build_public_tournament_roster_page(FakeSupabase(storage), club_id="club-1", registration_slug="tres-open")
    assert payload["summary"]["total_registrations"] == 1
    assert payload["summary"]["total_players"] == 1
    assert len(payload["roster"]["registrations_by_event"]) == 1
    assert "Cancelled Player" not in json.dumps(payload["roster"])


def test_public_tournament_roster_page_is_public_safe_after_registration() -> None:
    storage = fake_storage()
    storage["tournament_registration_settings"][0].update(
        {
            "registration_open_at": "2026-07-01T14:00:00Z",
            "registration_close_at": "2099-08-25T23:00:00Z",
        }
    )
    supabase = FakeSupabase(storage)

    submit_public_tournament_registration(
        supabase,
        club_id="club-1",
        payload={
            "registration_slug": "tres-open",
            "first_name": "Alex",
            "last_name": "Rivera",
            "email": "alex@example.com",
            "phone": "555-0100",
            "doubles_skill": 4.0,
            "dupr_id": "DUPR-123",
            "terms_accepted": True,
            "selections": [{"event_option_id": "event1", "partner_mode": "NONE"}],
        },
    )

    payload = build_public_tournament_roster_page(supabase, club_id="club-1", registration_slug="tres-open")

    assert payload["available"] is True
    assert payload["tournament"]["name"] == "Tres Palapas Open"
    assert payload["summary"]["total_registrations"] == 1
    assert payload["summary"]["total_players"] == 1
    assert payload["settings"]["registration_open_at"] == "2026-07-01T14:00:00Z"
    assert payload["settings"]["registration_close_at"] == "2099-08-25T23:00:00Z"
    assert payload["settings"]["weather_policy_markdown"] == "Unsafe conditions may delay or reschedule play."
    assert payload["events"][0]["scheduled_day_ids"] == ["day1", "day2"]
    roster_rows = payload["roster"]["registrations_by_event"]
    assert roster_rows[0]["event_family"] == "Doubles"
    assert roster_rows[0]["division"] == "Open"
    assert roster_rows[0]["members"][0]["display_name"] == "Alex Rivera"
    assert "email" not in roster_rows[0]["members"][0]
    assert "phone" not in roster_rows[0]["members"][0]
    assert "admin_notes" not in payload["tournament"]
    assert "internal_seed_notes" not in payload["events"][0]
    assert "builder_draft_json" not in payload["settings"]


@pytest.mark.parametrize("partner_note", [None, "", "Text me at +1 555-010-9988 or casey.private@example.com"])
def test_public_tournament_roster_projection_denies_private_fields_and_contact_values(partner_note) -> None:
    storage = fake_storage()
    supabase = FakeSupabase(storage)

    result = submit_public_tournament_registration(
        supabase,
        club_id="club-1",
        payload={
            "registration_slug": "tres-open",
            "first_name": "Casey",
            "last_name": "Court",
            "email": "casey.private@example.com",
            "phone": "+1 (555) 010-9988",
            "dupr_id": "PRIVATE-DUPR-42",
            "doubles_skill": 3.75,
            "age": 47,
            "notes": "Staff only: please discuss my scheduling accommodation privately.",
            "wants_partner_board_contact": True,
            "terms_accepted": True,
            "selections": [
                {
                    "event_option_id": "event1",
                    "partner_mode": "NEEDS_PARTNER",
                    "show_on_partner_board": True,
                    "partner_note": partner_note,
                }
            ],
        },
    )

    payload = build_public_tournament_roster_page(supabase, club_id="club-1", registration_slug="tres-open")
    roster = payload["roster"]
    serialized = json.dumps(roster, sort_keys=True)
    denied_keys = {
        "registration_id",
        "selection_id",
        "player_id",
        "event_option_id",
        "partner_request_id",
        "partner_link_id",
        "source_registration_ids",
        "source_selection_ids",
        "source_player_ids",
        "dupr_id",
        "email",
        "phone",
        "age",
    }

    def keys(value):
        if isinstance(value, dict):
            return set(value).union(*(keys(child) for child in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(child) for child in value)) if value else set()
        return set()

    assert keys(roster).isdisjoint(denied_keys)
    assert "casey.private@example.com" not in serialized
    assert "555-010-9988" not in serialized
    assert "PRIVATE-DUPR-42" not in serialized
    assert "Staff only:" not in serialized
    assert storage["tournament_registrations"][0]["notes"] == "Staff only: please discuss my scheduling accommodation privately."
    assert result["registration_id"] not in serialized
    board_entry = roster["players_needing_partners"][0]
    assert board_entry["board_entry_key"].startswith("tr_")
    assert board_entry["age_bracket"] == "40-49"
    if partner_note:
        assert board_entry["note"].count("[contact removed]") == 2
    else:
        assert board_entry["note"] == ""


def test_public_tournament_roster_reports_missing_schema() -> None:
    storage = fake_storage()

    class MissingSelectionsSupabase(FakeSupabase):
        def table(self, name):
            if name == "tournament_registration_selections":
                raise RuntimeError('relation "tournament_registration_selections" does not exist')
            return super().table(name)

    payload = build_public_tournament_roster_page(MissingSelectionsSupabase(storage), club_id="club-1", registration_slug="tres-open")

    assert payload["available"] is False
    assert payload["tournament"] is None
    assert "tournament_registration_selections" in str(payload["setup_error"])
