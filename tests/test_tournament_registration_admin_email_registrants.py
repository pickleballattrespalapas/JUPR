from pathlib import Path

from jupr_app.ui.pages import tournament_registration_admin_streamlined as admin_ui


def test_registration_management_has_email_registrants_tab():
    source = Path("jupr_app/ui/pages/tournament_registration_admin_streamlined/__init__.py").read_text(encoding="utf-8")

    assert "Email Registrants" in source
    assert "Send email to registrants" in source
    assert "registration_admin_view" in source
    assert "Recipients do not see the full recipient list" in source


def test_registrant_recipient_list_dedupes_and_excludes_cancelled_by_default(monkeypatch):
    monkeypatch.setattr(
        admin_ui,
        "list_registrations",
        lambda _supabase, tournament_id: [
            {"display_name": "Active One", "email": "One@Example.com", "status": "confirmed"},
            {"display_name": "Duplicate One", "email": "one@example.com", "status": "confirmed"},
            {"display_name": "Cancelled Two", "email": "two@example.com", "status": "cancelled"},
            {"display_name": "No Email", "email": "", "status": "confirmed"},
        ],
    )

    active_only = admin_ui._registrant_recipients(object(), tournament_id="tour-1", include_cancelled=False)
    with_cancelled = admin_ui._registrant_recipients(object(), tournament_id="tour-1", include_cancelled=True)

    assert active_only == [
        {"name": "Active One", "email": "one@example.com", "status": "confirmed", "payment_status": "unpaid"}
    ]
    assert [row["email"] for row in with_cancelled] == ["one@example.com", "two@example.com"]
