from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (WEB / path).read_text(encoding="utf-8")


def test_setup_wizard_has_separate_event_and_division_steps() -> None:
    nav = read("components/TournamentSetupWizardNav.tsx")
    panel = read("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    assert '| "divisions"' in nav
    assert 'label: "Events and event policies"' in nav
    assert 'label: "Divisions"' in nav
    assert "Domain {domainDefinition.number} of 4" in panel
    assert "Section ${domainSectionIndex} of ${domainDefinition.steps.length}" in panel
    assert "TournamentSetupEventFamilyCard" in panel
    assert "TournamentSetupDivisionCard" in panel
    assert 'goTo("divisions")' in panel
    assert 'goTo("pricing")' in panel
    assert "Continue to Divisions" in panel
    assert "Continue to Commerce" in panel
    assert (WEB / "app/admin/tournaments/setup/divisions/page.tsx").exists()


def test_tournament_domain_separates_basics_from_venue_without_save_confirmation() -> None:
    panel = read("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    basics = panel.split("function renderBasics()", 1)[1].split("function renderEvents()", 1)[0]
    venue = panel.split("function renderSchedule()", 1)[1].split("function renderReview()", 1)[0]
    assert "Registration and public policies" in basics
    assert "Add sponsor" in basics
    assert "Sponsor name" in basics
    assert "Venue name" in venue
    assert "Timezone" in venue
    assert "Total venue courts" in venue
    assert "ConfirmAction" not in basics
    assert "void saveBasics()" in basics


def test_routine_save_and_continue_actions_do_not_require_confirmation_dialogs() -> None:
    panel = read("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    assert "Save registration rules?" not in panel
    assert "Save the tournament schedule?" not in panel
    assert 'onClick={() => void saveBasics()}' in panel
    assert 'onClick={() => void saveDraftAndContinue("events")}' in panel


def test_divisions_do_not_expose_registration_status() -> None:
    division = read("app/admin/tournaments/setup/TournamentSetupDivisionCard.tsx")
    rules = read("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    assert "Registration status" not in division
    assert "DIVISION_STATUSES" not in division
    assert "TournamentSetupPolicies" in rules
    assert "weather_policy_markdown" in rules
    assert "configurationWithGlobalStatus" in rules


def test_setup_metadata_is_structured_and_migrated() -> None:
    migration = (ROOT / "supabase/migrations/20261021000000_tournament_setup_basics_metadata.sql").read_text()
    routes = (ROOT / "services/api/admin_tournament_setup_routes.py").read_text()
    service = (ROOT / "jupr_app/services/admin_tournament_setup_service.py").read_text()
    repo = (ROOT / "jupr_app/domain/tournament_registration_repo.py").read_text()
    for field in ("location_name", "timezone", "sponsors_json"):
        assert field in migration
        assert field in routes
        assert field in service
        assert field in repo

def test_setup_status_badges_wait_for_loaded_detail() -> None:
    panel = read("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")
    publication_status = read(
        "app/admin/tournaments/setup/tournamentSetupPublicationStatus.ts"
    )
    assert "const states = detail ? setupState(basics, settings, configuration) : {};" in panel
    assert "states={states}" in panel
    assert 'aria-busy={publicationStatus === "checking"}' in panel
    assert "Checking published setup…" in panel
    assert 'if (detailLoadState !== "loaded") return "checking"' in publication_status
