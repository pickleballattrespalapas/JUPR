from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (WEB / path).read_text()


def test_guided_setup_order_and_policy_merge() -> None:
    nav = read("components/TournamentSetupWizardNav.tsx")
    panel = read("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    assert 'label: "Tournament"' in nav
    assert 'key: "schedule"' in nav
    assert nav.index('key: "schedule"') < nav.index('key: "events"') < nav.index('key: "divisions"')
    assert '"registration-rules"' not in nav
    assert "Domain {domainDefinition.number} of 4" in panel
    assert "TournamentSetupPolicies" in panel
    assert 'goTo("schedule")' in panel
    assert 'saveDraftAndContinue("events")' in panel
    assert 'goTo("divisions")' in panel
    assert 'goTo("pricing")' in panel
    assert "Continue to Divisions" in panel
    assert "Continue to Commerce" in panel


def test_lifecycle_overview_and_phase_nav_match_refined_setup_flow() -> None:
    overview = read("app/admin/tournaments/TournamentLifecycleOverviewPanel.tsx")
    phase_nav = read("components/TournamentPhaseNav.tsx")
    nav = read("components/TournamentSetupWizardNav.tsx")
    for label in (
        "1. Tournament basics and policies",
        "2. Schedule and courts",
        "3. Events",
        "4. Divisions",
        "5. Pricing, extras, and fulfillment",
        "6. Review and open registration",
    ):
        assert label in overview
    for label in ("Tournament", "Competition", "Commerce", "Review"):
        assert f'label: "{label}"' in phase_nav
    assert 'label: "Venue and tournament days"' in nav
    assert 'label: "Events and event policies"' in nav
    assert 'label: "Fees, extras, bundles, and giveaways"' in nav
    assert 'label: "Preview, conflicts, publish, and registration"' in nav
    assert '"registration-rules"' not in phase_nav
    assert overview.index("1. Tournament basics and policies") < overview.index("2. Schedule and courts")
    assert overview.index("2. Schedule and courts") < overview.index("3. Events") < overview.index("4. Divisions")


def test_legacy_registration_rules_route_redirects_to_basics() -> None:
    route = read("app/admin/tournaments/setup/registration-rules/page.tsx")
    assert 'redirect(`/admin/tournaments/setup/basics?' in route
    assert 'params.set("tournament", tournament)' in route
    assert 'params.set("name", name)' in route


def test_policy_templates_and_weather_policy_are_required() -> None:
    policies = read("app/admin/tournaments/setup/TournamentSetupPolicies.tsx")
    panel = read("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    routes = (ROOT / "services/api/admin_tournament_setup_routes.py").read_text()
    service = (ROOT / "jupr_app/services/admin_tournament_setup_service.py").read_text()
    repo = (ROOT / "jupr_app/domain/tournament_registration_repo.py").read_text()
    assert policies.count('Write a custom policy') == 1
    assert 'Standard tournament rules' in policies
    assert 'Competitive and rating-verified' in policies
    assert 'Community and recreational' in policies
    assert 'Flexible refund policy' in policies
    assert 'Refunds until registration closes' in policies
    assert 'Registration fees are final' in policies
    assert 'Weather policy' in policies
    assert 'weather_policy_markdown' in panel
    assert 'weather_policy_markdown' in routes
    assert 'weather_policy_markdown' in service
    assert 'weather_policy_markdown' in repo


def test_add_division_uses_dialog_instead_of_appending_hidden_row() -> None:
    panel = read("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    dialog = read("app/admin/tournaments/setup/TournamentSetupDivisionDialog.tsx")
    division_card = read("app/admin/tournaments/setup/TournamentSetupDivisionCard.tsx")
    assert "TournamentSetupDivisionDialog" in panel
    assert 'onClick={() => setDivisionDialogKey(null)}' in panel
    assert 'mode={divisionDialogKey === null ? "add" : "edit"}' in panel
    assert 'onEdit={() => setDivisionDialogKey(row.key)}' in panel
    assert 'role="dialog"' in dialog
    assert '? "Add division"' in dialog
    assert '`Edit ${cleanString(draft.division_name ?? draft.label) || "division"}`' in dialog
    assert 'dialogMode === "add" ? "Add division" : "Save division"' in dialog
    assert 'onConfirm(draft)' in dialog
    assert 'setDivisionDialogKey(undefined)' in panel
    assert 'onClick={onEdit}' in division_card
    assert '>Edit<' not in division_card  # JSX includes line breaks around the label.
    assert 'Edit' in division_card
    assert '<input' not in division_card
    assert '<select' not in division_card


def test_event_and_division_schedules_support_multiple_days() -> None:
    builder = read("app/admin/tournament-setup/tournamentSetupBuilder.ts")
    event_dialog = read("app/admin/tournaments/setup/TournamentSetupEventFamilyDialog.tsx")
    division_dialog = read("app/admin/tournaments/setup/TournamentSetupDivisionDialog.tsx")
    event_card = read("app/admin/tournaments/setup/TournamentSetupEventFamilyCard.tsx")
    division_card = read("app/admin/tournaments/setup/TournamentSetupDivisionCard.tsx")
    service = (ROOT / "jupr_app/services/admin_tournament_setup_service.py").read_text()
    repo = (ROOT / "jupr_app/domain/tournament_registration_repo.py").read_text()
    migration = (ROOT / "supabase/migrations/20261022000000_tournament_setup_policies_multiday.sql").read_text()
    assert "eventDayReferences" in builder
    assert "setEventDayReferences" in builder
    assert "scheduled_day_ids" in builder
    assert "Select every day on which this event may be played" in event_dialog
    assert "Use every day selected for the parent event" in division_dialog
    assert "Choose specific event days" in division_dialog
    assert "No tournament days" in event_card
    assert "No tournament days" in division_card
    assert '"scheduled_day_ids"' in service
    assert '"scheduled_day_ids"' in repo
    assert "scheduled_day_ids jsonb" in migration
    assert "registration_day_id remains the primary grouping day" in migration


def test_multi_day_projection_preserves_existing_single_day_contracts() -> None:
    builder = read("app/admin/tournament-setup/tournamentSetupBuilder.ts")
    payload_tests = read("app/admin/tournament-setup/tournamentSetupBuilder.test.mjs")
    assert 'Object.prototype.hasOwnProperty.call(projected, "scheduled_day_ids")' in builder
    assert "scheduledDayIds.length > 1" in builder
    assert "const inferredScheduledDays = events" in builder
    assert "dayLabel(primaryDay)" in builder
    assert 'assert.deepEqual(draft.scheduled_day_ids, ["day-1"]);' in payload_tests
    assert 'scheduled_day_ids: ["day-friday"]' in payload_tests
    assert 'scheduled_day_ids: ["day-saturday"]' in payload_tests


def test_public_registration_displays_required_weather_policy() -> None:
    public_api = read("lib/tournamentRegistrationApi.ts")
    public_page = read("app/clubs/[clubSlug]/tournament-registration/page.tsx")
    assert "weather_policy_markdown?: string | null" in public_api
    assert "scheduled_day_ids?: string[] | null" in public_api
    assert "settings?.weather_policy_markdown" in public_page
    assert ">Weather policy<" in public_page


def test_public_services_project_weather_and_multi_day_fields() -> None:
    registration_service = (ROOT / "jupr_app/services/public_tournament_registration_service.py").read_text()
    roster_service = (ROOT / "jupr_app/services/public_tournament_roster_service.py").read_text()
    for source in (registration_service, roster_service):
        assert '"weather_policy_markdown"' in source
        assert '"scheduled_day_ids"' in source
