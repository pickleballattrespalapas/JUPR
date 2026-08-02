from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (WEB / path).read_text()


def test_guided_setup_order_and_policy_merge() -> None:
    nav = read("components/TournamentSetupWizardNav.tsx")
    panel = read("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    assert 'label: "Tournament basics and policies"' in nav
    assert 'key: "schedule"' in nav
    assert nav.index('key: "schedule"') < nav.index('key: "events"') < nav.index('key: "divisions"')
    assert '"registration-rules"' not in nav
    assert "Step {definition.number} of 6" in panel
    assert "TournamentSetupPolicies" in panel
    assert 'goTo("schedule")' in panel
    assert 'saveDraftAndContinue("events")' in panel
    assert 'saveDraftAndContinue("divisions")' in panel
    assert 'saveDraftAndContinue("pricing")' in panel


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
    assert "TournamentSetupDivisionDialog" in panel
    assert 'onClick={() => setDivisionDialogOpen(true)}' in panel
    assert 'role="dialog"' in dialog
    assert 'Add division' in dialog
    assert 'onConfirm(draft)' in dialog
    assert 'scrollIntoView' in panel


def test_event_and_division_schedules_support_multiple_days() -> None:
    builder = read("app/admin/tournament-setup/tournamentSetupBuilder.ts")
    event_card = read("app/admin/tournaments/setup/TournamentSetupEventFamilyCard.tsx")
    division_card = read("app/admin/tournaments/setup/TournamentSetupDivisionCard.tsx")
    service = (ROOT / "jupr_app/services/admin_tournament_setup_service.py").read_text()
    repo = (ROOT / "jupr_app/domain/tournament_registration_repo.py").read_text()
    migration = (ROOT / "supabase/migrations/20261022000000_tournament_setup_policies_multiday.sql").read_text()
    assert "eventDayReferences" in builder
    assert "setEventDayReferences" in builder
    assert "scheduled_day_ids" in builder
    assert "Select every day on which this event may be played" in event_card
    assert "Use every day selected for the parent event" in division_card
    assert "Choose specific event days" in division_card
    assert '"scheduled_day_ids"' in service
    assert '"scheduled_day_ids"' in repo
    assert "scheduled_day_ids jsonb" in migration
    assert "registration_day_id remains the primary grouping day" in migration
