from jupr_app.domain.tournament_registration_compiler import compile_tournament_registration_state
from jupr_app.ui.pages import tournament_registration_admin_streamlined as admin_ui


def test_cancelled_registration_is_hidden_from_compiled_rosters_and_partner_board():
    state = compile_tournament_registration_state(
        tournament={"id": "t-1"},
        settings={},
        days=[{"id": "day-1", "label": "Day 1", "sort_order": 1}],
        event_options=[
            {
                "id": "wd-35",
                "registration_day_id": "day-1",
                "label": "Women's Doubles 3.5",
                "event_family_label": "Women's Doubles",
                "division_name": "3.5",
                "event_type": "GENDER_DOUBLES",
                "partner_required": True,
                "sort_order": 1,
            }
        ],
        registrations=[
            {"id": "reg-active", "display_name": "Active Player", "email": "active@example.com", "status": "confirmed", "submitted_at": "2026-06-01T10:00:00+00:00"},
            {"id": "reg-cancelled", "display_name": "Cancelled Player", "email": "cancelled@example.com", "status": "cancelled", "submitted_at": "2026-06-01T10:01:00+00:00"},
        ],
        selections=[
            {"id": "sel-active", "registration_id": "reg-active", "registration_day_id": "day-1", "event_option_id": "wd-35", "partner_mode": "NEEDS_PARTNER"},
            {"id": "sel-cancelled", "registration_id": "reg-cancelled", "registration_day_id": "day-1", "event_option_id": "wd-35", "partner_mode": "NEEDS_PARTNER"},
        ],
    )

    roster_entries = state["event_rosters"][0]["entries"]
    visible_names = [entry["members"][0]["display_name"] for entry in roster_entries]
    partner_board_names = [row["player"]["display_name"] for row in state["partner_board"]]

    assert visible_names == ["Active Player"]
    assert partner_board_names == ["Active Player"]


def test_bulk_hard_delete_only_deletes_cancelled_registrations(monkeypatch):
    calls = []

    def fake_delete_registration(_supabase, *, tournament_id, registration_id):
        calls.append((tournament_id, registration_id))
        if registration_id == "reg-locked":
            raise ValueError("Registration is already imported into a draw.")

    monkeypatch.setattr(admin_ui, "delete_registration", fake_delete_registration)

    changed, skipped = admin_ui._apply_bulk_action(
        supabase=object(),
        tournament_id="tour-1",
        selected_rows=[
            {"registration_id": "reg-cancelled", "registration_status": "cancelled", "label": "Cancelled Player"},
            {"registration_id": "reg-active", "registration_status": "confirmed", "label": "Active Player"},
            {"registration_id": "reg-locked", "registration_status": "cancelled", "label": "Locked Player"},
        ],
        action=admin_ui.BULK_DELETE_CANCELLED_ACTION,
        status_value="",
        payment_value="",
        partner_mode_value="",
        target_event_id="",
        event_lookup={},
        note_text="",
    )

    assert changed == 1
    assert calls == [("tour-1", "reg-cancelled"), ("tour-1", "reg-locked")]
    assert any("Active Player" in item and "not cancelled" in item for item in skipped)
    assert any("Locked Player" in item and "imported into a draw" in item for item in skipped)
