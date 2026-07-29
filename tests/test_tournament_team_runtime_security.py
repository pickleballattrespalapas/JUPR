from types import SimpleNamespace

import pytest

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from jupr_app.services.admin_tournament_team_competition_service import (
    _validated_team_invitation_base_url,
    record_combined_rating_review,
    require_admin_team_tournament_runtime,
    resolve_team_invitation_base_url,
    tournament_team_creation_fingerprint,
)
from jupr_app.services.public_tournament_team_service import (
    _public_invitation_response,
    _public_team_creation_response,
    require_public_team_tournament_mutation_runtime,
)
from services.api import admin_tournament_team_competition_routes as admin_routes
from services.api import public_tournament_team_routes as public_routes
from services.api.public_tournament_team_routes import PublicTeamCreateRequest


class Query:
    def __init__(self, rows):
        self.rows = list(rows)
        self.filters = []

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, _value):
        return self

    def execute(self):
        rows = [
            row
            for row in self.rows
            if all(str(row.get(key)) == str(value) for key, value in self.filters)
        ]
        return SimpleNamespace(data=rows)


class ClubSupabase:
    def __init__(self, rows):
        self.rows = rows

    def table(self, name):
        return Query(self.rows if name == "clubs" else [])


class ReviewSupabase:
    def __init__(self):
        self.rpc_name = ""
        self.rpc_params = {}
        self.review = {
            "id": "initial-review",
            "tournament_id": "tournament-1",
            "event_option_id": "event-1",
            "selection_id": "selection-1",
            "registration_id": "registration-1",
            "partner_registration_id": "registration-2",
            "review_phase": "INITIAL",
            "state": "ELIGIBLE",
            "player_rating": 3.47,
            "partner_rating": 4.33,
            "combined_rating": 7.8,
            "player_rating_source": "PCS_LINKED",
            "partner_rating_source": "PCS_LINKED",
        }

    def table(self, name):
        return Query(
            [self.review]
            if name == "tournament_rating_eligibility_reviews"
            else []
        )

    def rpc(self, name, params):
        self.rpc_name = name
        self.rpc_params = params
        return SimpleNamespace(
            execute=lambda: SimpleNamespace(data={"ok": True})
        )


def _admin_app(monkeypatch, *, allowed: bool) -> tuple[TestClient, list[dict]]:
    app = FastAPI()
    calls: list[dict] = []
    monkeypatch.setattr(admin_routes, "is_admin_team_tournament_enabled", lambda: True)
    monkeypatch.setattr(
        admin_routes,
        "authenticate_bearer",
        lambda _authorization: SimpleNamespace(
            email="admin@example.com",
            user_id="user-1",
        ),
    )
    monkeypatch.setattr(
        admin_routes,
        "resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner" if allowed else "viewer"),
    )
    monkeypatch.setattr(
        admin_routes,
        "has_permission",
        lambda role, _permission: role == "club_owner",
    )
    monkeypatch.setattr(admin_routes, "write_admin_activity_log", lambda *_args: None)
    monkeypatch.setattr(
        admin_routes,
        "update_tournament_competition_config",
        lambda *_args, **kwargs: calls.append(kwargs) or {"ok": True},
    )
    admin_routes.install_admin_tournament_team_competition_routes(
        app,
        get_supabase_client=lambda: object(),
    )
    return TestClient(app), calls


def test_status_requires_manage_tournaments_permission(monkeypatch):
    client, _calls = _admin_app(monkeypatch, allowed=False)

    response = client.get(
        "/admin/clubs/club/tournaments/team-competition/status",
        headers={"Authorization": "Bearer denied"},
    )

    assert response.status_code == 403


@pytest.mark.parametrize(
    ("environment", "flag", "expected_status"),
    [
        ("production", "1", 403),
        ("staging", "", 403),
        ("staging", "1", 200),
    ],
)
def test_admin_mutation_requires_staging_surface_gate(
    monkeypatch,
    environment,
    flag,
    expected_status,
):
    monkeypatch.setenv("JUPR_ENV", environment)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")
    if flag:
        monkeypatch.setenv(
            "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_SETUP_MUTATIONS",
            flag,
        )
    else:
        monkeypatch.delenv(
            "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_SETUP_MUTATIONS",
            raising=False,
        )
    client, calls = _admin_app(monkeypatch, allowed=True)

    response = client.post(
        "/admin/clubs/club/tournaments/admin/tournaments/tournament-1/"
        "team-competition/events/event-1/config",
        headers={"Authorization": "Bearer allowed"},
        json={
            "idempotency_key": "config-request-1",
            "confirmation_text": "SAVE COMPETITION",
            "expected_updated_at": "2026-07-27T00:00:00Z",
            "patch": {"competition_format": "FOUR_PLAYER_TEAM"},
        },
    )

    assert response.status_code == expected_status
    assert len(calls) == (1 if expected_status == 200 else 0)


def test_public_team_writes_refuse_production(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("JUPR_ENABLE_TOURNAMENT_TEAM_COMPETITION", "1")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")

    with pytest.raises(PermissionError, match="staging-only"):
        require_public_team_tournament_mutation_runtime()


def test_admin_team_service_writes_refuse_production(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("JUPR_ENABLE_TOURNAMENT_TEAM_COMPETITION", "1")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")

    with pytest.raises(PermissionError, match="staging-only"):
        require_admin_team_tournament_runtime()


def test_closed_admin_write_surface_does_not_open_database(monkeypatch):
    app = FastAPI()
    opened = {"database": 0}
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")
    monkeypatch.setenv("JUPR_ENABLE_TOURNAMENT_TEAM_COMPETITION", "1")
    monkeypatch.delenv(
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_SETUP_MUTATIONS",
        raising=False,
    )
    monkeypatch.setattr(
        admin_routes,
        "is_admin_team_tournament_enabled",
        lambda: True,
    )

    def forbidden_database():
        opened["database"] += 1
        raise AssertionError("database should not open")

    admin_routes.install_admin_tournament_team_competition_routes(
        app,
        get_supabase_client=forbidden_database,
    )
    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tournament-1/"
        "team-competition/events/event-1/config",
        headers={"Authorization": "Bearer not-read"},
        json={
            "idempotency_key": "closed-surface-request",
            "confirmation_text": "SAVE COMPETITION",
            "expected_updated_at": "2026-07-27T00:00:00Z",
            "patch": {"competition_format": "FOUR_PLAYER_TEAM"},
        },
    )

    assert response.status_code == 403
    assert opened == {"database": 0}


def test_invalid_team_setup_recovery_token_is_rejected_before_connectors(
    monkeypatch,
):
    app = FastAPI()
    opened = {"club": 0, "database": 0}
    monkeypatch.setenv(
        "JUPR_REGISTRATION_CONFIRMATION_SECRET",
        "team-recovery-route-secret",
    )
    monkeypatch.setattr(
        public_routes, "is_admin_team_tournament_enabled", lambda: True
    )
    monkeypatch.setattr(
        public_routes, "public_team_tournament_runtime_ready", lambda: True
    )

    def forbidden_club(_slug):
        opened["club"] += 1
        raise AssertionError("club connector should not open")

    def forbidden_database():
        opened["database"] += 1
        raise AssertionError("database should not open")

    public_routes.install_public_tournament_team_routes(
        app,
        get_club=forbidden_club,
        get_supabase_client=forbidden_database,
        public_club_payload=lambda club, slug: club,
    )
    response = TestClient(app).post(
        "/clubs/club/tournament-registration/four-player-team/recover",
        json={
            "confirmation_token": "not-a-valid-confirmation-token-1234",
            "website": "",
        },
    )

    assert response.status_code == 403
    assert opened == {"club": 0, "database": 0}


def test_public_team_writes_allow_public_intake_or_permanent_open(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_ENABLE_TOURNAMENT_TEAM_COMPETITION", "1")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")
    monkeypatch.setenv("JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES", "1")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "tournament-registration")

    with pytest.raises(PermissionError, match="public-intake-auth"):
        require_public_team_tournament_mutation_runtime()

    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "public-intake-auth")
    require_public_team_tournament_mutation_runtime()

    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "open")
    require_public_team_tournament_mutation_runtime()


def test_caller_supplied_invitation_origin_is_not_part_of_public_contract():
    payload = PublicTeamCreateRequest.model_validate(
        {
            "tournament_id": "tournament-1",
            "event_option_id": "event-1",
            "team_name": "Safe team",
            "captain_registration_id": "registration-1",
            "confirmation_token": "x" * 32,
            "members": [
                {"slot": "MAN_1", "email": "a@example.com"},
                {"slot": "MAN_2", "email": "b@example.com"},
                {"slot": "WOMAN_1", "email": "c@example.com"},
                {"slot": "WOMAN_2", "email": "d@example.com"},
            ],
            "idempotency_key": "team-request-1",
            "public_base_url": "https://attacker.example",
        }
    )

    assert "public_base_url" not in payload.model_dump()


def test_invitation_origin_comes_from_server_owned_club_state(monkeypatch):
    monkeypatch.setenv("JUPR_WEB_BASE_URL", "https://fallback.example")
    supabase = ClubSupabase(
        [
            {
                "id": "club-1",
                "slug": "safe-club",
                "public_base_url": "https://clubs.example/clubs/safe-club",
            }
        ]
    )

    assert resolve_team_invitation_base_url(
        supabase,
        club_id="club-1",
    ) == "https://clubs.example/clubs/safe-club"
    with pytest.raises(ValueError):
        _validated_team_invitation_base_url("https://safe.example@attacker.example")


def test_staging_origin_wins_over_production_club_url(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_WEB_BASE_URL", "https://staging.example")
    supabase = ClubSupabase(
        [
            {
                "id": "club-1",
                "slug": "safe-club",
                "public_base_url": "https://production.example/clubs/safe-club",
            }
        ]
    )

    assert resolve_team_invitation_base_url(
        supabase,
        club_id="club-1",
    ) == "https://staging.example/clubs/safe-club"


def test_public_invitation_response_is_an_explicit_safe_projection():
    projection = _public_invitation_response(
        {
            "ok": True,
            "status": "ACCEPTED",
            "operation_key": "private-recovery-key",
            "audit": {"actor": "private@example.com"},
            "team": {
                "id": "team-1",
                "name": "Safe Team",
                "status": "CONFIRMED",
                "version": 2,
                "eligibility_state": "internal",
                "created_by": "private@example.com",
            },
            "member": {
                "id": "member-1",
                "slot": "MAN_1",
                "status": "ACCEPTED",
                "invitation_version": 3,
                "invited_email": "private@example.com",
                "invitation_token_hash": "secret",
                "registration_id": "private-registration",
            },
        }
    )

    assert projection == {
        "ok": True,
        "status": "ACCEPTED",
        "team": {
            "id": "team-1",
            "name": "Safe Team",
            "status": "CONFIRMED",
            "version": 2,
        },
        "invitation": {
            "member_id": "member-1",
            "slot": "MAN_1",
            "status": "ACCEPTED",
            "invitation_version": 3,
        },
    }
    serialized = str(projection).lower()
    for forbidden in (
        "email",
        "token",
        "audit",
        "recovery",
        "operation",
        "eligibility",
        "registration_id",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize("business_recovery", [False, True])
def test_public_team_creation_is_an_explicit_safe_projection(
    business_recovery,
):
    projection = _public_team_creation_response(
        {
            "ok": True,
            "operation_key": "private-operation-key",
            "recovered_by_business_identity": business_recovery,
            "audit": {"actor": "private@example.com"},
            "team": {
                "id": "team-1",
                "name": "Safe Team",
                "status": "FORMING",
                "version": 1,
                "eligibility_state": "REVIEW_REQUIRED",
                "creation_fingerprint": "private-fingerprint",
                "created_by": "private@example.com",
            },
            "members": [
                {
                    "id": "member-1",
                    "slot": "MAN_1",
                    "display_name_snapshot": "Captain",
                    "status": "ACCEPTED",
                    "invitation_version": 1,
                    "invited_email": "private@example.com",
                    "invitation_token_hash": "secret",
                    "registration_id": "private-registration",
                }
            ],
            "invitation_deliveries": [
                {
                    "member_id": "member-2",
                    "status": "completed",
                    "email_mode": "staging",
                    "provider_message_id": "private-provider-id",
                    "operation_key": "private-delivery-key",
                    "request_fingerprint": "private-request",
                }
            ],
        }
    )

    assert projection == {
        "ok": True,
        "team": {
            "id": "team-1",
            "name": "Safe Team",
            "status": "FORMING",
            "version": 1,
        },
        "members": [
            {
                "member_id": "member-1",
                "slot": "MAN_1",
                "display_name": "Captain",
                "status": "ACCEPTED",
                "invitation_version": 1,
            }
        ],
        "invitation_deliveries": [
            {"member_id": "member-2", "status": "completed"}
        ],
    }
    serialized = str(projection).lower()
    for forbidden in (
        "email",
        "token",
        "audit",
        "recovery",
        "operation",
        "eligibility",
        "registration_id",
        "fingerprint",
        "provider",
    ):
        assert forbidden not in serialized


def test_team_creation_business_identity_is_order_independent_and_change_sensitive():
    members = [
        {
            "slot": "MAN_1",
            "registration_id": "captain-registration",
            "email": "CAPTAIN@example.com",
            "display_name": "Captain",
            "gender": "Men",
        },
        {
            "slot": "MAN_2",
            "email": "man2@example.com",
            "display_name": "Man Two",
            "gender": "Men",
        },
        {
            "slot": "WOMAN_1",
            "email": "woman1@example.com",
            "display_name": "Woman One",
            "gender": "Women",
        },
        {
            "slot": "WOMAN_2",
            "email": "woman2@example.com",
            "display_name": "Woman Two",
            "gender": "Women",
        },
    ]
    fingerprint, normalized = tournament_team_creation_fingerprint(
        event_option_id="event-1",
        team_name="Safe Team",
        captain_registration_id="captain-registration",
        members=members,
    )
    reordered, reordered_members = tournament_team_creation_fingerprint(
        event_option_id="event-1",
        team_name="Safe Team",
        captain_registration_id="captain-registration",
        members=list(reversed(members)),
    )
    changed, _changed_members = tournament_team_creation_fingerprint(
        event_option_id="event-1",
        team_name="Safe Team",
        captain_registration_id="captain-registration",
        members=[
            *members[:-1],
            {**members[-1], "email": "different@example.com"},
        ],
    )

    assert fingerprint == reordered
    assert normalized == reordered_members
    assert changed != fingerprint
    assert normalized[0]["email"] == "captain@example.com"


def test_rating_review_uses_current_evidence_in_json_rpc_contract(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "test")
    monkeypatch.setenv("JUPR_ENABLE_TOURNAMENT_TEAM_COMPETITION", "1")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")
    supabase = ReviewSupabase()

    result = record_combined_rating_review(
        supabase,
        club_id="club-1",
        tournament_id="tournament-1",
        event_option_id="event-1",
        selection_id="selection-1",
        review_phase="REGISTRATION_CLOSE",
        override_state=None,
        override_reason=None,
        expected_selection_updated_at="2026-07-27T12:00:00Z",
        actor_email="admin@example.com",
        idempotency_key="rating-review-contract",
    )

    assert result == {"ok": True}
    assert supabase.rpc_name == "admin_record_tournament_rating_review_cas"
    review = supabase.rpc_params["p_review"]
    assert review["selection_id"] == "selection-1"
    assert review["combined_rating"] == 7.8
    assert review["review_phase"] == "REGISTRATION_CLOSE"
    assert review["expected_selection_updated_at"] == "2026-07-27T12:00:00Z"
    assert "p_selection_id" not in supabase.rpc_params


@pytest.mark.parametrize(
    ("method", "path", "json"),
    [
        ("get", "/clubs/safe/tournament-team-results", None),
        (
            "get",
            "/clubs/safe/tournament-team-results/tournament-1/draw-1",
            None,
        ),
        (
            "post",
            "/clubs/safe/tournament-team-invitation/resolve",
            {"token": "x" * 32},
        ),
        (
            "post",
            "/clubs/safe/tournament-team-invitation/respond",
            {
                "token": "x" * 32,
                "action": "ACCEPT",
                "registration_id": "registration-1",
                "idempotency_key": "invitation-response-1",
            },
        ),
    ],
)
def test_disabled_public_team_routes_do_not_open_clients(
    monkeypatch,
    method,
    path,
    json,
):
    app = FastAPI()
    opened = {"club": 0, "database": 0}
    monkeypatch.setattr(
        public_routes,
        "is_admin_team_tournament_enabled",
        lambda: False,
    )

    def forbidden_club(_slug):
        opened["club"] += 1
        raise AssertionError("club lookup should not run")

    def forbidden_database():
        opened["database"] += 1
        raise AssertionError("database should not open")

    public_routes.install_public_tournament_team_routes(
        app,
        get_club=forbidden_club,
        get_supabase_client=forbidden_database,
        public_club_payload=lambda club, slug: {"id": slug, **club},
    )
    client = TestClient(app)
    response = (
        client.get(path)
        if method == "get"
        else client.post(path, json=json)
    )

    assert response.status_code == 404
    assert opened == {"club": 0, "database": 0}
