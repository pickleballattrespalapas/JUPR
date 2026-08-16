from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_admin_team_tournament_workspace_is_reachable_and_complete():
    nav = _read("apps/web/components/TournamentAdminNav.tsx")
    page = _read(
        "apps/web/app/admin/tournaments/team-competition/page.tsx"
    )
    panel = _read(
        "apps/web/app/admin/tournaments/team-competition/"
        "TeamTournamentAdminPanel.tsx"
    )

    assert '"/admin/tournaments/team-competition"' in nav
    assert "TeamTournamentAdminPanel" in page
    for workflow in (
        "SAVE COMPETITION",
        "VERIFY RATING",
        "SAVE RATING REVIEW",
        "CLOSE RATING REVIEW",
        "CREATE TEAM",
        "REISSUE INVITATION",
        "REPLACE ROSTER",
        "WITHDRAW TEAM",
        "BUILD TEAM SCHEDULE",
        "BUILD TEAM PLAYOFFS",
        "LOCK TEAM LINEUP",
        "SAVE TEAM SCORE",
        "RECONCILE TEAM SCORE",
        "SAVE TEAM PODIUM",
    ):
        assert workflow in panel
    assert "Team podium publication is unavailable" in panel
    assert "PUBLISH TEAM PODIUM" not in panel
    assert "publish: true" not in panel
    assert "/admin/tournaments/ops/publish" in panel
    assert "useAdminSession" in panel
    assert "useAuthenticatedAutoLoad" in panel
    assert "operationKeys" in panel
    ops_panel = _read(
        "apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx"
    )
    assert "Official publishing moved" in ops_panel
    assert "cannot publish official matches or four-player rating children" in ops_panel
    assert "next_team_tournament_child_publish" not in ops_panel


def test_team_tournament_surfaces_are_responsive_and_publicly_discoverable():
    css = _read(
        "apps/web/app/admin/tournaments/team-competition/"
        "TeamTournamentAdminPanel.module.css"
    )
    club = _read("apps/web/app/clubs/[clubSlug]/page.tsx")
    registration = _read(
        "apps/web/app/clubs/[clubSlug]/tournament-registration/page.tsx"
    )

    assert "@media (max-width: 760px)" in css
    assert "grid-template-columns: 1fr" in css
    assert 'href: `${base}/tournaments`' in club
    assert "tournament-team-results" in registration


def test_invitation_secret_is_removed_from_address_before_api_use():
    invite = _read(
        "apps/web/app/clubs/[clubSlug]/tournament-team-invitation/"
        "TeamInvitationReview.tsx"
    )

    hash_read = invite.index("window.location.hash.slice(1)")
    address_clear = invite.index("window.history.replaceState")
    resolve = invite.index("resolvePublicTeamInvitation")
    resolve_call = invite.index("resolvePublicTeamInvitation(", resolve + 1)
    assert hash_read < address_clear < resolve_call
    assert "searchParams" not in invite
    assert "token.current" in invite
    assert "idempotencyKeys" in invite


def test_team_tournament_routes_are_installed_and_public_origin_is_server_owned():
    main = _read("services/api/main.py")
    public_routes = _read("services/api/public_tournament_team_routes.py")
    admin_service = _read(
        "jupr_app/services/admin_tournament_team_competition_service.py"
    )

    assert "install_public_tournament_team_routes(app" in main
    assert "install_admin_tournament_team_competition_routes(app" in main
    assert "public_base_url" not in public_routes
    assert "Browser input never owns email links" in admin_service


def test_registration_team_setup_has_server_backed_interruption_recovery():
    form = _read(
        "apps/web/app/clubs/[clubSlug]/tournament-registration/"
        "TournamentRegistrationForm.tsx"
    )
    confirmation = _read(
        "apps/web/app/clubs/[clubSlug]/tournament-registration/"
        "confirmation/page.tsx"
    )
    recovery = _read(
        "apps/web/app/clubs/[clubSlug]/tournament-registration/"
        "confirmation/FourPlayerTeamSetupRecovery.tsx"
    )
    service = _read("jupr_app/services/public_tournament_team_service.py")
    routes = _read("services/api/public_tournament_team_routes.py")

    assert "window.history.replaceState" in form
    assert 'query.set("team_setup", "attention")' in form
    assert "FourPlayerTeamSetupRecovery" in confirmation
    assert "recoverPublicFourPlayerTeamSetup" in recovery
    assert "Always consult server state" in recovery
    assert "build_public_four_player_team_setup_recovery" in service
    assert "tournament_team_operations" in service
    assert "four-player-team/recover" in routes
