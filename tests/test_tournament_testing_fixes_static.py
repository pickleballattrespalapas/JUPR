from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_public_registration_leads_with_registration_and_links_details_home() -> None:
    registration = _read(
        "apps/web/app/clubs/[clubSlug]/tournament-registration/page.tsx"
    )
    home = _read("apps/web/app/clubs/[clubSlug]/tournaments/page.tsx")

    assert "Register now" in registration
    assert 'href="#registration-form"' in registration
    assert 'id="registration-form"' in registration
    assert registration.index('id="registration-form"') < registration.index(
        "Need tournament details?"
    )
    assert "Tournament information" in home
    for heading in ("Venue", "Rules", "Refund policy", "Weather policy"):
        assert heading in home


def test_admin_registration_editor_uses_per_event_cards_and_focused_dialogs() -> None:
    source = _read(
        "apps/web/app/admin/tournaments/registration/registrants/"
        "[registrationId]/TournamentRegistrantEditPanel.tsx"
    )

    assert "Event entries" in source
    assert "Edit entry" in source
    assert "Change partner" in source
    assert "Add another event entry" in source
    assert 'open={eventDialogOpen && Boolean(selectedSelection)}' in source
    assert 'open={partnerDialogOpen && Boolean(selectedSelection)}' in source
    assert "Manual partner details" in source
    assert "public roster shows one canonical team" in source


def test_manual_partner_write_is_one_transaction_and_one_canonical_team() -> None:
    sql = _read(
        "supabase/migrations/20261105000000_manual_partner_canonical_teams.sql"
    ).lower()
    repo = _read("jupr_app/domain/tournament_registration_repo.py")

    assert "canonicalize_tournament_manual_partner_team_v1" in sql
    assert "after insert or update of" in sql
    assert "insert into public.players" in sql
    assert "insert into public.tournament_registrations" in sql
    assert "insert into public.tournament_registration_selections" in sql
    assert "insert into public.tournament_registration_team_links" in sql
    assert "insert into public.tournament_registration_team_members" in sql
    assert "'admin_confirmed'" in sql
    assert "create_tournament_registration_canonical_v1" in sql
    assert "security definer" in sql
    assert "set search_path = ''" in sql
    assert "grant execute on function public.create_tournament_registration_canonical_v1" in sql
    assert "'manual-tournament-partner:' || new.tournament_id::text" in sql
    assert sql.count("perform public.refresh_initial_combined_rating_review_v1(") == 2
    assert "new.id::text" in sql
    assert "v_partner_selection.id::text" in sql
    assert 'PUBLIC_REGISTRATION_CANONICAL_CREATE_RPC = (' in repo
    assert 'ADMIN_REGISTRATION_CANONICAL_CREATE_RPC = (' in repo
    assert "else PUBLIC_REGISTRATION_CANONICAL_CREATE_RPC" in repo
    assert 'response = rpc(\n            canonical_create_rpc' in repo
    assert "trusted_admin_create=True" in repo


def test_manual_partner_ui_requires_identity_and_new_player_baseline() -> None:
    public_form = _read(
        "apps/web/app/clubs/[clubSlug]/tournament-registration/"
        "TournamentRegistrationForm.tsx"
    )
    admin_editor = _read(
        "apps/web/app/admin/tournaments/registration/registrants/"
        "[registrationId]/TournamentRegistrantEditPanel.tsx"
    )

    assert "partner name, email, age, gender, and starting skill are required" in public_form
    assert "Partner starting skill *" in public_form
    assert "Your partner does not need to register again" in public_form
    assert "manualPartnerFieldsValid" in admin_editor
    assert "Partner name *" in admin_editor
    assert "Partner email *" in admin_editor
    assert "Partner starting skill *" in admin_editor
