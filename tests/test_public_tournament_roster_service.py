from __future__ import annotations

import json

from jupr_app.services.public_tournament_registration_service import submit_public_tournament_registration
from jupr_app.services.public_tournament_roster_service import build_public_tournament_roster_page
from tests.test_public_tournament_registration_service import FakeSupabase, fake_storage


def test_public_tournament_roster_page_is_public_safe_after_registration() -> None:
    storage = fake_storage()
    storage["tournament_registration_settings"][0].update(
        {
            "registration_open_at": "2026-07-01T14:00:00Z",
            "registration_close_at": "2026-08-25T23:00:00Z",
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
    assert payload["settings"]["registration_close_at"] == "2026-08-25T23:00:00Z"
    roster_rows = payload["roster"]["registrations_by_event"]
    assert roster_rows[0]["event_family"] == "Doubles"
    assert roster_rows[0]["division"] == "Open"
    assert roster_rows[0]["members"][0]["display_name"] == "Alex Rivera"
    assert "email" not in roster_rows[0]["members"][0]
    assert "phone" not in roster_rows[0]["members"][0]
    assert "admin_notes" not in payload["tournament"]
    assert "internal_seed_notes" not in payload["events"][0]
    assert "builder_draft_json" not in payload["settings"]


def test_public_tournament_roster_projection_denies_private_fields_and_contact_values() -> None:
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
            "terms_accepted": True,
            "selections": [
                {
                    "event_option_id": "event1",
                    "partner_mode": "NEEDS_PARTNER",
                    "show_on_partner_board": True,
                    "partner_note": "Text me at +1 555-010-9988 or casey.private@example.com",
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
    assert result["registration_id"] not in serialized
    board_entry = roster["players_needing_partners"][0]
    assert board_entry["board_entry_key"].startswith("tr_")
    assert board_entry["age_bracket"] == "40-49"
    assert board_entry["note"].count("[contact removed]") == 2


def test_public_tournament_roster_reports_missing_schema() -> None:
    storage = fake_storage()
    del storage["tournament_registration_selections"]

    payload = build_public_tournament_roster_page(FakeSupabase(storage), club_id="club-1", registration_slug="tres-open")

    assert payload["available"] is False
    assert payload["tournament"] is None
    assert "tournament_registration_selections" in str(payload["setup_error"])
