from __future__ import annotations

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")

from fastapi import HTTPException

from services.api import admin_team_league_routes as admin_routes
from services.api import public_team_league_routes as public_routes


class FakeApp:
    def __init__(self):
        self.routes: dict[tuple[str, str], object] = {}

    def _decorator(self, method: str, path: str):
        def decorate(function):
            self.routes[(method, path)] = function
            return function

        return decorate

    def get(self, path: str):
        return self._decorator("GET", path)

    def post(self, path: str):
        return self._decorator("POST", path)

    def put(self, path: str):
        return self._decorator("PUT", path)


def _public_app() -> FakeApp:
    app = FakeApp()
    public_routes.install_public_team_league_routes(
        app,
        get_club=lambda _slug: {
            "id": "club",
            "slug": "staging-club",
            "name": "Staging Club",
            "public_base_url": "https://juprleagues.com/clubs/production-club",
        },
        get_supabase_client=lambda: object(),
        public_club_payload=lambda club, slug: {
            "id": club["id"],
            "slug": slug,
        },
    )
    return app


def test_partner_invitation_link_uses_staging_next_origin_not_club_production_url(
    monkeypatch,
) -> None:
    app = _public_app()
    captured: dict[str, object] = {}
    monkeypatch.setenv("JUPR_ENV", "test")
    monkeypatch.setattr(
        public_routes,
        "get_next_web_base_url",
        lambda: "https://staging-web.example.test/",
    )

    def fake_register(_supabase, **kwargs):
        captured.update(kwargs)
        return {"ok": True, "payment_mode": "offline"}

    monkeypatch.setattr(public_routes, "register_public_team_league", fake_register)
    handler = app.routes[
        ("POST", "/clubs/{club_slug}/team-leagues/{league_name}/registrations")
    ]
    payload = public_routes.PublicTeamLeagueRegistrationRequest(
        signup_type="team",
        player_id=1,
        contact_email="captain@example.com",
        partner_player_id=2,
        partner_email="partner@example.com",
        team_name="Aces",
        idempotency_key="team-register:test",
        confirmation_text="REGISTER TEAM",
    )

    result = handler("ignored-request-slug", "Open", payload)

    assert result["ok"] is True
    assert captured["public_base_url"] == (
        "https://staging-web.example.test/clubs/staging-club"
    )
    assert "juprleagues.com" not in str(captured["public_base_url"])


def test_public_team_league_mutation_denies_production_before_service(
    monkeypatch,
) -> None:
    app = _public_app()
    called = {"service": False}
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("JUPR_ENABLE_TEAM_LEAGUES", "1")
    monkeypatch.setenv("JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES", "1")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "public-intake-auth")
    monkeypatch.setattr(
        public_routes,
        "register_public_team_league",
        lambda *_args, **_kwargs: called.update(service=True),
    )
    handler = app.routes[
        ("POST", "/clubs/{club_slug}/team-leagues/{league_name}/registrations")
    ]
    payload = public_routes.PublicTeamLeagueRegistrationRequest(
        signup_type="solo",
        player_id=1,
        contact_email="player@example.com",
        idempotency_key="team-register:production",
        confirmation_text="JOIN PARTNER WAITLIST",
    )

    try:
        handler("club", "Open", payload)
    except HTTPException as exc:
        assert exc.status_code == 403
        assert "staging-only" in str(exc.detail)
    else:
        raise AssertionError("production mutation was not denied")
    assert called == {"service": False}


def test_admin_team_league_mutation_denies_production_before_auth_or_service(
    monkeypatch,
) -> None:
    app = FakeApp()
    called = {"client": False}

    def forbidden_client():
        called["client"] = True
        raise AssertionError("client should not load")

    admin_routes.install_admin_team_league_routes(
        app, get_supabase_client=forbidden_client
    )
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("JUPR_ENABLE_TEAM_LEAGUES", "1")
    monkeypatch.setenv(
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES", "1"
    )
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "league-manager")
    handler = app.routes[
        (
            "PUT",
            "/admin/clubs/{club_id}/league-manager/team-leagues/"
            "{league_name}/settings",
        )
    ]
    payload = admin_routes.TeamLeagueSettingsRequest(
        settings={},
        expected_settings_version=0,
        idempotency_key="team-settings:production",
        confirmation_text="SAVE TEAM LEAGUE",
    )

    try:
        handler("club", "Open", payload, "Bearer ignored")
    except HTTPException as exc:
        assert exc.status_code == 403
        assert "staging-only" in str(exc.detail)
    else:
        raise AssertionError("production mutation was not denied")
    assert called == {"client": False}


def test_admin_create_team_route_is_wired_and_staging_only(monkeypatch) -> None:
    app = FakeApp()
    called = {"client": False}

    def forbidden_client():
        called["client"] = True
        raise AssertionError("client should not load")

    admin_routes.install_admin_team_league_routes(
        app, get_supabase_client=forbidden_client
    )
    route = (
        "POST",
        "/admin/clubs/{club_id}/league-manager/team-leagues/"
        "{league_name}/teams",
    )
    assert route in app.routes
    payload = admin_routes.TeamLeagueCreateTeamRequest(
        team_name="Forming Aces",
        captain_player_id=1,
        captain_contact_email="captain@example.com",
        initial_primary_player_id=2,
        expected_roster_version=0,
        idempotency_key="create:team:production",
        confirmation_text="CREATE TEAM",
    )
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("JUPR_ENABLE_TEAM_LEAGUES", "1")
    monkeypatch.setenv(
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES", "1"
    )
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "league-manager")

    try:
        app.routes[route]("club", "Open", payload, "Bearer ignored")
    except HTTPException as exc:
        assert exc.status_code == 403
        assert "staging-only" in str(exc.detail)
    else:
        raise AssertionError("production create-team mutation was not denied")
    assert called == {"client": False}


def test_public_team_league_read_is_disabled_before_club_or_client_in_production(
    monkeypatch,
) -> None:
    app = FakeApp()
    called = {"club": False, "client": False}

    def forbidden_club(_slug: str):
        called["club"] = True
        raise AssertionError("club should not load")

    def forbidden_client():
        called["client"] = True
        raise AssertionError("client should not load")

    public_routes.install_public_team_league_routes(
        app,
        get_club=forbidden_club,
        get_supabase_client=forbidden_client,
        public_club_payload=lambda club, slug: {"club": club, "slug": slug},
    )
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("JUPR_ENABLE_TEAM_LEAGUES", "0")
    handler = app.routes[("GET", "/clubs/{club_slug}/team-leagues")]

    try:
        handler("production-club")
    except HTTPException as exc:
        assert exc.status_code == 403
        assert "disabled" in str(exc.detail).lower()
    else:
        raise AssertionError("production read was not disabled")
    assert called == {"club": False, "client": False}


def test_admin_team_league_read_is_disabled_before_client_or_auth_in_production(
    monkeypatch,
) -> None:
    app = FakeApp()
    called = {"client": False}

    def forbidden_client():
        called["client"] = True
        raise AssertionError("client should not load")

    admin_routes.install_admin_team_league_routes(
        app, get_supabase_client=forbidden_client
    )
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("JUPR_ENABLE_TEAM_LEAGUES", "0")
    handler = app.routes[
        ("GET", "/admin/clubs/{club_id}/league-manager/team-leagues")
    ]

    try:
        handler("production-club", "Bearer ignored")
    except HTTPException as exc:
        assert exc.status_code == 403
        assert "disabled" in str(exc.detail).lower()
    else:
        raise AssertionError("production read was not disabled")
    assert called == {"client": False}
