from __future__ import annotations

from jupr_app.services.public_tournament_registration_service import submit_public_tournament_registration
from jupr_app.services.public_tournament_roster_service import build_public_tournament_roster_page
from tests.test_public_tournament_registration_service import FakeSupabase, fake_storage


def test_public_tournament_roster_page_is_public_safe_after_registration() -> None:
    storage = fake_storage()
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
    roster_rows = payload["roster"]["registrations_by_event"]
    assert roster_rows[0]["event_family"] == "Doubles"
    assert roster_rows[0]["division"] == "Open"
    assert roster_rows[0]["members"][0]["display_name"] == "Alex Rivera"
    assert "email" not in roster_rows[0]["members"][0]
    assert "phone" not in roster_rows[0]["members"][0]
    assert "admin_notes" not in payload["tournament"]
    assert "internal_seed_notes" not in payload["events"][0]
    assert "builder_draft_json" not in payload["settings"]


def test_public_tournament_roster_reports_missing_schema() -> None:
    storage = fake_storage()
    del storage["tournament_registration_selections"]

    payload = build_public_tournament_roster_page(FakeSupabase(storage), club_id="club-1", registration_slug="tres-open")

    assert payload["available"] is False
    assert payload["tournament"] is None
    assert "tournament_registration_selections" in str(payload["setup_error"])
