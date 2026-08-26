from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_check_in_routes_are_authenticated_and_installed() -> None:
    source = read("services/api/admin_tournament_checkin_routes.py")
    installer = read("services/api/admin_operations_routes.py")

    assert '@app.get("/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/check-in")' in source
    assert '@app.put(' in source
    assert '"/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/check-in/{registration_id}"' in source
    assert "_resolve_tournament_role_or_403" in source
    assert "PERMISSION_MANAGE_TOURNAMENTS" in source
    assert "install_admin_tournament_checkin_routes" in installer
    assert "day_id: str | None = Query" in source
    assert "day_id: str = Query" in source
    assert "registration_day_id=day_id" in source
    assert 'Literal["EXPECTED", "CHECKED_IN", "ABSENT"]' in source
    assert 'Literal["EXPECTED", "CHECKED_IN", "ABSENT"] = "EXPECTED"' not in source
    assert "operation_key: UUID" in source
    assert "TournamentCheckInIdempotencyConflictError" in source


def test_check_in_update_queries_only_real_registration_columns() -> None:
    source = read("jupr_app/services/admin_tournament_checkin_service.py")

    assert '.select("id,tournament_id,player_id,status")' in source
    assert '.select("id,tournament_id,status,registration_status")' not in source


def test_check_in_put_is_only_in_both_tournament_live_waves() -> None:
    from scripts.staging_write_waves import STAGING_WRITE_WAVE_ROUTES

    route = (
        "PUT",
        "/admin/clubs/{club_id}/tournament-live/tournaments/"
        "{tournament_id}/check-in/{registration_id}",
    )

    containing_waves = {
        wave for wave, routes in STAGING_WRITE_WAVE_ROUTES.items() if route in routes
    }
    assert containing_waves == {
        "tournament-live",
        "tournament-live-official-publish",
    }


def test_check_in_web_contract_has_real_controls_and_no_client_only_truth() -> None:
    page = read("apps/web/app/admin/tournaments/live-operations/check-in/page.tsx")
    panel = read(
        "apps/web/app/admin/tournaments/live-operations/check-in/TournamentCheckInPanel.tsx"
    )
    api = read("apps/web/lib/adminTournamentCheckInApi.ts")

    assert "TournamentCheckInPanel" in page
    assert "Checked in" in panel
    assert "Expected" in panel
    assert "Absent" in panel
    assert "Unresolved" in panel
    assert "Waiver" in panel
    assert "Approved substitute" not in panel
    assert "Search players" in panel
    assert "NEEDS_REVIEW" in panel
    assert "overflowX: \"auto\"" not in page
    assert "fetchAdminTournamentCheckIn" in api
    assert "updateAdminTournamentCheckIn" in api
    assert 'method: "PUT"' in api
    update_input = api.split("export type TournamentCheckInUpdate = {", 1)[1].split(
        "};", 1
    )[0]
    assert "approved_substitute_name" not in update_input
    assert "identity_current" in api
    assert "requires_reconfirmation" in api
    assert "Legacy saved substitute" in panel
    assert "Restore original registrant" in panel
    assert "Roster changes do not happen at check-in" in panel
    assert "authoritative draw or four-player team roster" in panel
    assert "SUBSTITUTE_ASSIGNMENT_ATOMICITY_UNAVAILABLE" in api
    assert "Registered but not rostered" in panel
    assert "registration_follow_up" in api


def test_check_in_api_rejects_name_only_substitutes_before_service_update() -> None:
    source = read("services/api/admin_tournament_checkin_routes.py")

    assert "approved_substitute_name" in source
    assert "approved_substitute_player_id is None" in source
    assert "Select an active club player as the approved substitute" in source
