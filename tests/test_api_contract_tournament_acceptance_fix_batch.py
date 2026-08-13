from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "apps" / "web"


def read_web(path: str) -> str:
    return (WEB / path).read_text(encoding="utf-8")


def read_root(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_basics_supports_sponsor_reordering_and_draft_publication_clarity() -> None:
    panel = read_web("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")

    assert "function moveSponsor(index: number, direction: -1 | 1)" in panel
    assert "Move up" in panel
    assert "Move down" in panel
    assert "Unpublished setup draft" in panel
    assert "Public tournament pages continue using the currently published configuration" in panel
    assert "Review domain publishes the complete tournament" in panel
    assert "Publish reviewed tournament" in panel


def test_publish_result_stays_visible_in_the_confirmation_dialog() -> None:
    panel = read_web("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")

    assert 'import type { ConfirmActionSuccess } from "@/components/ConfirmAction";' in panel
    assert "Promise<ConfirmActionSuccess | void>" in panel
    assert 'title: "Tournament published"' in panel
    assert "The reviewed tournament setup is now published." in panel
    assert "Registration status was left unchanged." in panel
    assert 'closeLabel: "Done"' in panel
    assert "throw publishError" in panel


def test_tournament_days_are_date_locked_and_include_court_capacity_controls() -> None:
    panel = read_web("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    builder = read_web("app/admin/tournament-setup/tournamentSetupBuilder.ts")
    venue_model = read_web("app/admin/tournaments/setup/TournamentVenueModel.ts")
    migration = read_root(
        "supabase/migrations/20260805043000_tournament_venue_inventory.sql"
    ).lower()

    assert "syncTournamentDays" in panel
    assert "Venue and tournament days" in panel
    assert "Venue address" in panel
    assert "Total venue courts" in panel
    assert "Venue court inventory" in panel
    assert "Fixed tournament date" in panel
    assert "readOnly disabled" in panel
    assert "Which courts are available?" in panel
    assert "tournament-level court hours" in panel
    assert "court_open_time: null" in venue_model
    assert "court_close_time: null" in venue_model
    assert "FACILITY_COURT_LIMIT = 100" in builder
    assert "venue_courts_json" in migration
    assert "available_court_ids" in migration


def test_events_and_divisions_use_dialog_editors_and_read_only_cards() -> None:
    panel = read_web("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    event_dialog = read_web(
        "app/admin/tournaments/setup/TournamentSetupEventFamilyDialog.tsx"
    )
    event_card = read_web(
        "app/admin/tournaments/setup/TournamentSetupEventFamilyCard.tsx"
    )
    division_dialog = read_web(
        "app/admin/tournaments/setup/TournamentSetupDivisionDialog.tsx"
    )
    division_card = read_web(
        "app/admin/tournaments/setup/TournamentSetupDivisionCard.tsx"
    )

    assert "TournamentSetupEventFamilyDialog" in panel
    assert "TournamentSetupDivisionDialog" in panel
    assert "sortEventFamiliesByTournamentDay" in panel
    assert "sortDivisionsByEventAndName" in panel
    assert 'disabled={structure === "MIXED_DOUBLES" || structure === "FOUR_PLAYER_TEAM"}' in event_dialog
    assert 'gender_restriction: "MIXED"' in event_dialog
    assert "compact, read-only event card" in event_dialog
    assert "compact, read-only card" in division_dialog
    assert "skill_label" in division_dialog
    assert "datalist" in division_dialog
    assert "onClick={onEdit}" in event_card
    assert "onClick={onEdit}" in division_card
    assert "<input" not in event_card
    assert "<select" not in event_card
    assert "<input" not in division_card
    assert "<select" not in division_card


def test_flexible_bundles_and_extra_option_guidance_are_present() -> None:
    panel = read_web("app/admin/tournaments/commerce/TournamentCommercePanel.tsx")
    domain = read_root("jupr_app/domain/tournament_commerce.py")
    migration = read_root(
        "supabase/migrations/20261023000000_tournament_setup_courts_flexible_bundles.sql"
    )

    assert "Flexible event choice" in panel
    assert "Divisions required from eligible list" in panel
    assert "any {requiredCount} of {choiceComponents.length}" in panel
    assert "sizes, room types, meal choices, or pack" in panel
    assert '"EVENT_CHOICE"' in domain
    assert "combinations(eligible_keys, required_count)" in domain
    assert "EVENT_CHOICE" in migration


def test_review_has_actionable_blocked_change_resolution() -> None:
    panel = read_web("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    page = read_web("app/admin/tournaments/setup/TournamentSetupWizardPage.tsx")
    service = read_root("jupr_app/services/admin_tournament_setup_service.py")
    repo = read_root("jupr_app/domain/tournament_registration_repo.py")

    assert "Keep published value" in panel
    assert "Edit affected draft" in panel
    assert "keepPublishedValueForBlockedChange" in panel
    assert "resolveDivisionId" in panel
    assert "resolveDivisionId" in page
    assert "impact" in service
    assert '"blocked_details"' in repo


def test_registration_reports_keep_navigation_and_show_loading_state() -> None:
    reports = read_web("app/admin/tournaments/registrations/page.tsx")
    loading = read_web("app/admin/tournaments/registrations/loading.tsx")

    assert "TournamentPhaseNav" in reports
    assert 'phase="registration"' in reports
    assert "TournamentPhaseNav" in loading
    assert "Loading communications and reports" in loading
    assert 'aria-live="polite"' in loading


def test_registration_editor_supports_partner_and_extra_reconciliation() -> None:
    panel = read_web(
        "app/admin/tournaments/registration/registrants/[registrationId]/TournamentRegistrantEditPanel.tsx"
    )
    admin_routes = read_root("services/api/admin_tournament_routes.py")
    commerce_routes = read_root("services/api/admin_tournament_commerce_routes.py")
    commerce_service = read_root(
        "jupr_app/services/admin_tournament_commerce_service.py"
    )
    partner_service = read_root("jupr_app/services/admin_tournament_service.py")

    assert "Assigned partner" in panel
    assert "Save assigned partner" in panel
    assert 'confirmationText="SAVE PARTNER"' in panel
    assert "Review updated extras" in panel
    assert "Save reviewed extras" in panel
    assert "expected_quote_fingerprint" in panel
    assert "expected_order_updated_at" in panel
    assert "/partner" in admin_routes
    assert "replace_admin_tournament_selection_partner" in partner_service
    assert "/quote" in commerce_routes
    assert "replace_admin_tournament_commerce_order" in commerce_routes
    assert "quote_admin_tournament_commerce_order" in commerce_service
    assert "replace_admin_tournament_commerce_order" in commerce_service
