from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (WEB / path).read_text(encoding="utf-8")


def test_top_level_tournament_navigation_uses_four_lifecycle_phases() -> None:
    nav = read("components/TournamentAdminNav.tsx")

    for label in (
        'label: "Tournament Home"',
        'label: "Tournament Builder"',
        'label: "Registration"',
        'label: "Live Operations"',
        'label: "Publish"',
    ):
        assert label in nav

    for old_label in (
        'label: "Bulk actions"',
        'label: "Extras & fulfillment"',
        'label: "Ratings & team play"',
        'label: "Operations"',
        'label: "Results"',
        'label: "Live runner"',
        'label: "Official publish"',
        'label: "Status & recovery"',
    ):
        assert old_label not in nav


def test_tournament_home_is_a_lifecycle_dashboard_with_next_action() -> None:
    panel = read("app/admin/tournaments/tournament/TournamentHomePanel.tsx")

    assert "Next action" in panel
    assert "Tournament workflow" in panel
    assert 'title: "Setup"' in panel
    assert 'title: "Registration"' in panel
    assert 'title: "Live Operations"' in panel
    assert 'title: "Publish"' in panel
    assert "Results become official only through Publish" in panel


def test_lifecycle_phase_overviews_and_subnavigation_exist() -> None:
  overview = read("app/admin/tournaments/TournamentLifecycleOverviewPanel.tsx")
  phase_nav = read("components/TournamentPhaseNav.tsx")
  setup_entry = read("app/admin/tournaments/setup/page.tsx")
  setup_nav = read("components/TournamentSetupWizardNav.tsx")
  setup_panel = read("app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")

  assert "tournamentSetupStepHref" in setup_entry
  assert '"basics"' in setup_entry
  assert 'redirect("/admin/tournaments")' in setup_entry

  registration = read("app/admin/tournaments/registration/page.tsx")
  assert "TournamentLifecycleOverviewPanel" in registration
  assert 'redirect("/admin/tournaments")' in registration

  live_shell = read("app/admin/tournaments/live-operations/TournamentLiveRoute.tsx")
  assert "readTournamentRouteContext" in live_shell
  assert "initialTournamentId={context.tournamentId}" in live_shell
  assert "initialDrawId={context.drawId}" in live_shell
  assert "SelectedTournamentPanelScope" not in live_shell
  for page in (
      "app/admin/tournaments/live-operations/page.tsx",
      "app/admin/tournaments/publish/page.tsx",
  ):
      source = read(page)
      assert "TournamentLiveRoute" in source
      assert "searchParams={searchParams}" in source

  for label in (
      "Tournament basics and policies",
      "Schedule and courts",
      "Events",
      "Divisions",
      "Pricing, extras, and fulfillment",
      "Review and open registration",
      "Partners and teams",
      "Payments and extras",
      "Preflight and check-in",
      "Draws and scheduling",
      "Live scoring",
      "Corrections and recovery",
      "Review results",
      "Publish divisions",
      "Tournament closeout",
  ):
      assert label in overview

  assert 'export type TournamentPhase = "setup" | "registration" | "live" | "publish";' in phase_nav
  for label in ("Tournament", "Competition", "Commerce", "Review"):
      assert f'label: "{label}"' in phase_nav

  for domain, steps in (
      ('key: "tournament"', 'steps: ["basics", "schedule"]'),
      ('key: "competition"', 'steps: ["events", "divisions"]'),
      ('key: "commerce"', 'steps: ["pricing"]'),
      ('key: "review"', 'steps: ["review"]'),
  ):
      assert domain in setup_nav
      assert steps in setup_nav

  for key in (
      'key: "basics"',
      'key: "schedule"',
      'key: "events"',
      'key: "divisions"',
      'key: "pricing"',
      'key: "review"',
  ):
      assert key in setup_nav

  assert 'goTo("schedule")' in setup_panel
  assert "Save draft and continue" in setup_panel
  assert "private admin draft" in setup_panel
  assert "Publish reviewed tournament" in setup_panel
  assert "Tournament preview" in setup_panel
  assert "Refresh review" in setup_panel
  assert "Open registration" in setup_panel


def test_admin_session_is_shared_across_admin_route_mounts() -> None:
    session = read("lib/useAdminSession.ts")

    assert "useSyncExternalStore" in session
    assert "let sharedSnapshot" in session
    assert "let restoreRequest" in session
    assert "const listeners = new Set" in session
    assert "if (restoreRequest) return restoreRequest;" in session
    assert "restoreAuthorizedAdminSession" in session
    assert "adminSessionIsFresh" in session


def test_commerce_is_selected_tournament_only_and_uses_edit_summaries() -> None:
    panel = read("app/admin/tournaments/commerce/TournamentCommercePanel.tsx")
    page = read("app/admin/tournaments/commerce/page.tsx")

    assert "tournamentId: string;" in panel
    assert "tournamentName: string;" in panel
    assert "listAdminTournamentCommerceTournaments" not in panel
    assert "adminSessionLabel" not in panel
    assert "Tournament extras workspace" not in panel
    assert "Catalog action is available" not in panel
    assert 'inputMode="decimal"' in panel
    assert "savedItemIds" in panel
    assert "savedBundleIds" in panel
    assert 'savedItemIds.has(item.id) ? "Edit" : "New extra"' in panel
    assert 'savedBundleIds.has(bundle.id) ? "Edit" : "New bundle"' in panel
    assert "draft.variants.length" in panel
    assert "TournamentPhaseNav" in page
    assert "readTournamentRouteContext(searchParams)" in page
    assert "tournamentId={context.tournamentId}" in page


def test_event_format_is_one_primary_choice_with_a_review_summary() -> None:
    panel = read("app/admin/tournaments/team-competition/TeamTournamentAdminPanel.tsx")
    page = read("app/admin/tournaments/team-competition/page.tsx")

    assert "initialTournamentId: string" in panel
    assert "listAdminTeamTournaments" not in panel
    assert "adminSessionLabel" not in panel
    assert "Event format" in panel
    assert "Standard singles or doubles" in panel
    assert "Combined-rating doubles" in panel
    assert "Four-player team · two men and two women" in panel
    assert "Maximum combined partner rating" in panel
    assert "Eligibility is strictly below this cap." in panel
    assert "Review:" in panel
    assert "friendlyWorkspaceWarning" in panel
    assert "APIError" not in panel
    assert "readTournamentRouteContext(searchParams)" in page
    assert "initialTournamentId={context.tournamentId}" in page
    assert "initialDrawId={context.drawId}" in page
    assert 'phase="setup"' in page


def test_registration_list_and_edit_are_separate_pages_with_richer_summary() -> None:
    list_panel = read(
        "app/admin/tournaments/registration/registrants/TournamentRegistrantListPanel.tsx"
    )
    edit_panel = read(
        "app/admin/tournaments/registration/registrants/[registrationId]/TournamentRegistrantEditPanel.tsx"
    )
    legacy_page = read("app/admin/tournaments/editor/page.tsx")

    assert "Amount" in list_panel
    assert "Events" in list_panel
    assert "Extras" in list_panel
    assert "Edit registration" in list_panel
    assert "getAdminTournamentCommerceDetail" in list_panel
    assert "← Back to registrations" in edit_panel
    assert "Save registration" in edit_panel
    assert "Save event entry" in edit_panel
    assert 'tournamentRouteHref("/admin/tournaments/registration/registrants", context)' in legacy_page


def test_ops_pages_are_selected_tournament_only_without_legacy_selector_shell() -> None:
    panel = read("app/admin/tournaments/ops/TournamentOpsPanel.tsx")
    workflow = read("app/admin/tournaments/ops/TournamentOpsWorkflowPage.tsx")

    assert "initialTournamentId: string" in panel
    assert "AdminTournamentListResponse" not in panel
    assert "AdminTournament," not in panel
    assert "loadTournaments" not in panel
    assert "Refresh tournaments" not in panel
    assert "Select tournament" not in panel
    assert "adminSessionLabel" not in panel
    assert "initialTournamentId={tournamentId}" in workflow
    assert "TournamentPhaseNav" in workflow


def test_selected_tournament_legacy_dom_scope_is_retired() -> None:
    assert not (
        ROOT
        / "apps"
        / "web"
        / "app"
        / "admin"
        / "tournaments"
        / "SelectedTournamentPanelScope.tsx"
    ).exists()


def test_temporary_patch_files_are_retired() -> None:
    assert not (ROOT / "scripts" / "tmp_patch_tournament_lifecycle.py").exists()
    assert not (
        ROOT
        / ".github"
        / "workflows"
        / "tmp-tournament-lifecycle-large-panel-patch.yml"
    ).exists()
