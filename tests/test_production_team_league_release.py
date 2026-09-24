"""Scope and regression checks for the Tres-only team league release."""
import inspect

import pytest
from fastapi import HTTPException

from jupr_app.services import team_league_service as service
from jupr_app.services.staging_write_guard import (
    staging_admin_team_league_writes_enabled,
    staging_public_team_league_writes_enabled,
)
from scripts import deployment_verifier as verifier
from services.api import admin_team_league_routes as admin_routes
from services.api import public_team_league_routes as public_routes
from tests.test_api_contract_team_leagues import FakeApp
from tests.test_production_deployment_hardening import _health_payload
from tests.test_team_league_composition import _CompositionSupabase


@pytest.fixture
def production(monkeypatch):
    for key, value in {
        "JUPR_ENV": "production", "FLY_APP_NAME": "juprleagues-api",
        "SUPABASE_URL": "https://dnoockbwfenunhcibwfn.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "server-fixture",
        "JUPR_PRODUCTION_WRITE_POLICY": "enabled", "JUPR_STAGING_WRITE_WAVE": "none",
        "JUPR_ENABLE_TEAM_LEAGUES": "1",
        "JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES": "0",
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES": "0",
    }.items():
        monkeypatch.setenv(key, value)


def test_team_release_changes_only_team_flag_and_preserves_live_rollback():
    previous = _health_payload(feature_profile="pre_team_leagues")
    released = _health_payload(feature_profile="release")
    assert verifier.production_feature_profile_from_health(previous) == "pre_team_leagues"
    assert verifier.production_feature_profile_from_health(released) == "release"
    assert verifier.expected_production_email_mode(profile="pre_team_leagues") == "live"
    assert [flag for flag in previous["feature_flags"]
            if previous["feature_flags"][flag] != released["feature_flags"][flag]] == [
        "JUPR_ENABLE_TEAM_LEAGUES"
    ]


def test_production_team_writes_require_explicit_club(production):
    for guard in (staging_admin_team_league_writes_enabled, staging_public_team_league_writes_enabled):
        assert guard("tres_palapas")
        assert not guard("another_club")
        assert not guard()


@pytest.mark.parametrize("key,value", [
    ("JUPR_ENV", "staging"), ("JUPR_ENV", "unknown"),
    ("FLY_APP_NAME", "juprleagues-api-staging"),
    ("SUPABASE_URL", "https://sijpxjxvdtrehmqvirfi.supabase.co"),
    ("SUPABASE_SERVICE_ROLE_KEY", ""), ("JUPR_PRODUCTION_WRITE_POLICY", "disabled"),
    ("JUPR_STAGING_WRITE_WAVE", "open"), ("JUPR_ENABLE_TEAM_LEAGUES", "0"),
])
def test_team_write_gates_reject_environment_drift(production, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    assert not staging_admin_team_league_writes_enabled("tres_palapas")
    assert not staging_public_team_league_writes_enabled("tres_palapas")


@pytest.mark.parametrize("name", [
    "register_public_team_league", "confirm_public_team_league_partner",
    "save_admin_team_league_settings", "create_admin_team_league_team",
    "admin_team_league_roster_action", "admin_team_league_waitlist_action",
    "commit_admin_team_league_schedule", "score_admin_team_league_fixture",
    "reconcile_admin_team_league_fixture", "resolve_admin_team_league_operation",
])
def test_service_mutations_cannot_bypass_club_scope(production, name):
    function = getattr(service, name)
    arguments = {key: None for key, param in inspect.signature(function).parameters.items()
                 if param.default is inspect.Parameter.empty}
    arguments.update(supabase=object(), club_id="another_club")
    with pytest.raises(PermissionError, match="not enabled for this club"):
        function(**arguments)


@pytest.mark.parametrize("size", [2, 4])
def test_authorized_admin_can_save_each_requested_roster_size(production, monkeypatch, size):
    app, db = FakeApp(), _CompositionSupabase()
    admin_routes.install_admin_team_league_routes(app, get_supabase_client=lambda: db)
    authenticated = []

    def authenticated_role(_db, **kwargs):
        authenticated.append(kwargs["club_id"])
        return "admin@example.com", "club_owner"

    monkeypatch.setattr(admin_routes, "_role_or_403", authenticated_role)
    handler = app.routes[("PUT", "/admin/clubs/{club_id}/league-manager/team-leagues/{league_name}/settings")]
    result = handler("tres_palapas", "Season Teams", admin_routes.TeamLeagueSettingsRequest(
        settings={"team_size": size, "team_category": "open", "timezone": "America/Mazatlan"},
        expected_settings_version=0, idempotency_key=f"settings:production-test:{size}",
        confirmation_text="SAVE TEAM LEAGUE",
    ), "Bearer fixture")
    assert result["committed"] is True
    assert authenticated == ["tres_palapas"]
    name, params = db.rpc_calls[0]
    assert name == "team_league_save_settings_v2"
    assert params["p_club_id"] == "tres_palapas"
    assert params["p_settings"]["team_size"] == size
    assert params["p_actor_role"] == "club_owner"


def test_release_keeps_admin_authentication_required(production):
    app = FakeApp()
    admin_routes.install_admin_team_league_routes(app, get_supabase_client=lambda: object())
    handler = app.routes[("GET", "/admin/clubs/{club_id}/league-manager/team-leagues")]
    with pytest.raises(HTTPException) as error:
        handler("tres_palapas", None)
    assert error.value.status_code == 401


@pytest.mark.parametrize("resolved_club", ["tres_palapas", "another_club"])
def test_public_registration_checks_resolved_club(production, monkeypatch, resolved_club):
    app = FakeApp()
    received = []
    public_routes.install_public_team_league_routes(
        app, get_club=lambda slug: {"id": resolved_club, "slug": slug},
        get_supabase_client=lambda: object(), public_club_payload=lambda club, slug: club,
    )
    monkeypatch.setattr(public_routes, "get_next_web_base_url", lambda: "https://pickleballclubsandwich.com")
    monkeypatch.setattr(public_routes, "register_public_team_league", lambda db, **kw: received.append(kw) or {"ok": True})
    handler = app.routes[("POST", "/clubs/{club_slug}/team-leagues/{league_name}/registrations")]
    payload = public_routes.PublicTeamLeagueRegistrationRequest(
        signup_type="solo", player_id=1, contact_email="fixture@example.com",
        idempotency_key="public:production-test", confirmation_text="JOIN PARTNER WAITLIST",
    )
    if resolved_club == "tres_palapas":
        assert handler("tres-palapas", "Season Teams", payload)["ok"]
        assert received[0]["club_id"] == "tres_palapas"
    else:
        with pytest.raises(HTTPException) as error:
            handler("tres-palapas", "Season Teams", payload)
        assert error.value.status_code == 403
        assert received == []
