from __future__ import annotations

import pytest

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from jupr_app.services import admin_tournament_match_publish_service as publish_service
from jupr_app.services import admin_tournament_team_competition_service as team_service
from jupr_app.services.admin_tournament_ops_service import (
    get_admin_tournament_ops_snapshot,
)
from jupr_app.services.admin_tournament_team_competition_service import (
    get_admin_team_tournament_snapshot,
)
from services.api import admin_tournament_routes
from services.api import admin_tournament_team_competition_routes as team_routes
from tests.test_admin_match_log_service import FakeSupabase


class TrackingSupabase(FakeSupabase):
    def __init__(self, tables):
        super().__init__(tables)
        self.opened_tables: list[str] = []

    def table(self, name):
        self.opened_tables.append(str(name))
        return super().table(name)


def _ops_tables():
    return {
        "tournaments": [
            {
                "id": "tournament-1",
                "club_id": "club-1",
                "name": "Team tournament",
                "status": "PUBLISHED",
            }
        ],
        "tournament_event_draws": [
            {
                "id": "standard-draw",
                "tournament_id": "tournament-1",
                "name": "Standard draw",
                "draw_kind": "STANDARD",
            },
            {
                "id": "child-draw",
                "tournament_id": "tournament-1",
                "name": "Protected child",
                "draw_kind": "TEAM_RATING_CHILD",
                "hidden_from_primary_ops": True,
            },
        ],
        "tournament_teams": [],
        "tournament_games": [
            {
                "id": "child-game",
                "tournament_id": "tournament-1",
                "draw_id": "child-draw",
                "team_match_game_id": "team-child",
                "score_a": 11,
                "score_b": 7,
            }
        ],
        "tournament_team_match_games": [
            {
                "id": "team-child",
                "tournament_id": "tournament-1",
                "rating_draw_id": "child-draw",
                "tournament_game_id": "child-game",
                "counts_for_rating": True,
                "status": "FINAL",
                "match_format": "DOUBLES",
                "team_a_player_ids": [1, 2],
                "team_b_player_ids": [3, 4],
                "score_a": 11,
                "score_b": 7,
            }
        ],
        "matches": [
            {
                "id": "official-child",
                "club_id": "club-1",
                "tournament_id": "tournament-1",
                "tournament_game_id": "child-game",
                "match_format": "doubles",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 7,
            }
        ],
    }


def test_disabled_team_feature_omits_child_queue_and_child_table_reads(
    monkeypatch,
):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENABLE_TOURNAMENT_TEAM_COMPETITION", "0")
    supabase = TrackingSupabase(_ops_tables())

    snapshot = get_admin_tournament_ops_snapshot(
        supabase,
        club_id="club-1",
        tournament_id="tournament-1",
    )

    assert snapshot["rating_child_draws"] == []
    assert snapshot["rating_child_publish_queue"] == []
    assert snapshot["summary"]["rating_children"] == 0
    assert [row["id"] for row in snapshot["draws"]] == ["standard-draw"]
    assert "tournament_team_match_games" not in supabase.opened_tables


def test_ops_and_team_snapshots_require_an_exact_canonical_child(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENABLE_TOURNAMENT_TEAM_COMPETITION", "1")
    tables = _ops_tables()

    ops_snapshot = get_admin_tournament_ops_snapshot(
        TrackingSupabase(tables),
        club_id="club-1",
        tournament_id="tournament-1",
    )
    team_snapshot = get_admin_team_tournament_snapshot(
        TrackingSupabase(tables),
        club_id="club-1",
        tournament_id="tournament-1",
    )

    assert ops_snapshot["rating_child_publish_queue"][0]["publish_state"] == "PUBLISHED"
    assert team_snapshot["game_publish_state"]["team-child"] == "PUBLISHED"

    tables["matches"][0]["score_t1"] = 10
    drifted_ops = get_admin_tournament_ops_snapshot(
        TrackingSupabase(tables),
        club_id="club-1",
        tournament_id="tournament-1",
    )
    drifted_team = get_admin_team_tournament_snapshot(
        TrackingSupabase(tables),
        club_id="club-1",
        tournament_id="tournament-1",
    )

    assert (
        drifted_ops["rating_child_publish_queue"][0]["publish_state"]
        == "RECONCILE_REQUIRED"
    )
    assert (
        drifted_team["game_publish_state"]["team-child"]
        == "RECONCILE_REQUIRED"
    )


def test_child_publish_service_refuses_disabled_feature_before_child_table_read(
    monkeypatch,
):
    monkeypatch.setattr(
        publish_service,
        "is_admin_team_tournament_enabled",
        lambda: False,
    )

    class ForbiddenSupabase:
        def table(self, _name):
            raise AssertionError("child table should not be read")

    with pytest.raises(PermissionError, match="disabled"):
        publish_service._validate_team_rating_child_publish_source(
            ForbiddenSupabase(),
            tournament_id="tournament-1",
            draw={"id": "child-draw", "draw_kind": "TEAM_RATING_CHILD"},
            teams=[],
            games=[],
            playoff_winner_bonus_elo=0,
        )


def test_protected_child_publish_fails_closed_without_canonical_tournament_readiness(
    monkeypatch,
):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setattr(
        publish_service,
        "is_admin_team_tournament_enabled",
        lambda: True,
    )
    tables = _ops_tables()
    tables["matches"] = []
    supabase = TrackingSupabase(tables)

    with pytest.raises(ValueError, match="official publishing is blocked"):
        publish_service.publish_admin_tournament_draw_matches(
            supabase,
            club_id="club-1",
            tournament_id="tournament-1",
            draw_id="child-draw",
            actor_email="owner@example.com",
            actor_role="club_owner",
            confirmation_text="PUBLISH MATCHES",
            source="next_team_tournament_child_publish",
        )

    assert tables["matches"] == []
    assert tables.get("tournament_admin_operations", []) == []
    assert tables.get("admin_activity_log", []) == []


def test_protected_child_publish_route_blocks_before_durable_intent(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setattr(
        admin_tournament_routes,
        "is_admin_tournament_admin_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        admin_tournament_routes,
        "is_admin_team_tournament_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        admin_tournament_routes,
        "_resolve_tournament_role_or_403",
        lambda **_kwargs: ("owner@example.com", "club_owner"),
    )
    tables = _ops_tables()
    tables["matches"] = []
    supabase = TrackingSupabase(tables)
    app = FastAPI()
    admin_tournament_routes.install_admin_tournament_routes(
        app,
        get_supabase_client=lambda: supabase,
    )

    response = TestClient(app).post(
        "/admin/clubs/club-1/tournaments/admin/tournaments/tournament-1/"
        "draws/child-draw/matches/publish",
        headers={"Authorization": "Bearer local"},
        json={
            "confirmation_text": "PUBLISH MATCHES",
            "source": "next_team_tournament_child_publish",
        },
    )

    assert response.status_code == 400
    assert "official publishing is blocked" in response.json()["detail"]
    assert "Standard draw has no tournament games" in response.json()["detail"]
    assert tables.get("tournament_admin_operations", []) == []
    assert tables.get("admin_activity_log", []) == []


def test_child_publish_route_feature_gate_precedes_database_client(
    monkeypatch,
):
    app = FastAPI()
    opened = {"database": 0}
    monkeypatch.setattr(
        admin_tournament_routes,
        "is_admin_tournament_admin_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        admin_tournament_routes,
        "is_admin_team_tournament_enabled",
        lambda: False,
    )

    def forbidden_database():
        opened["database"] += 1
        raise AssertionError("database should not open")

    admin_tournament_routes.install_admin_tournament_routes(
        app,
        get_supabase_client=forbidden_database,
    )
    response = TestClient(app).post(
        "/admin/clubs/club-1/tournaments/admin/tournaments/tournament-1/"
        "draws/child-draw/matches/publish",
        headers={"Authorization": "Bearer not-read"},
        json={
            "confirmation_text": "PUBLISH MATCHES",
            "source": "next_team_tournament_child_publish",
        },
    )

    assert response.status_code == 403
    assert opened == {"database": 0}


def test_team_podium_publish_is_retired_before_snapshot_operation_or_rpc(monkeypatch):
    monkeypatch.setattr(
        team_service,
        "require_admin_team_tournament_runtime",
        lambda: None,
    )

    class ForbiddenSupabase:
        def table(self, _name):
            raise AssertionError("retired team podium publish must not read or write tables")

        def rpc(self, _name, _params):
            raise AssertionError("retired team podium publish must not call an RPC")

    with pytest.raises(
        PermissionError,
        match="TEAM_PODIUM_CANONICAL_REVIEW_UNAVAILABLE",
    ):
        team_service.replace_team_podium(
            ForbiddenSupabase(),
            club_id="club-1",
            tournament_id="tournament-1",
            draw_id="team-parent",
            expected_draw_updated_at="2026-08-15T13:00:00Z",
            publish=True,
            reason="",
            actor_email="owner@example.com",
            idempotency_key="team-podium-retired",
        )


def test_team_podium_publish_route_returns_403_without_domain_calls(monkeypatch):
    monkeypatch.setattr(team_routes, "is_admin_team_tournament_enabled", lambda: True)
    monkeypatch.setattr(
        team_service,
        "require_admin_team_tournament_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        team_routes,
        "require_tournament_admin_mutation_runtime",
        lambda _surface: None,
    )
    monkeypatch.setattr(
        team_routes,
        "_resolve_manage_role_or_403",
        lambda **_kwargs: ("owner@example.com", "club_owner"),
    )

    class ForbiddenSupabase:
        def table(self, _name):
            raise AssertionError("retired team podium publish must not read or write tables")

        def rpc(self, _name, _params):
            raise AssertionError("retired team podium publish must not call an RPC")

    app = FastAPI()
    team_routes.install_admin_tournament_team_competition_routes(
        app,
        get_supabase_client=lambda: ForbiddenSupabase(),
    )
    response = TestClient(app).post(
        "/admin/clubs/club-1/tournaments/admin/tournaments/tournament-1/"
        "team-competition/draws/team-parent/podium",
        headers={"Authorization": "Bearer local"},
        json={
            "idempotency_key": "team-podium-retired",
            "confirmation_text": "PUBLISH TEAM PODIUM",
            "expected_draw_updated_at": "2026-08-15T13:00:00Z",
            "publish": True,
            "podium": [],
        },
    )

    assert response.status_code == 403
    assert "TEAM_PODIUM_CANONICAL_REVIEW_UNAVAILABLE" in response.json()["detail"]


@pytest.mark.parametrize(
    ("operations_enabled", "official_enabled"),
    [
        (False, True),
        (True, False),
    ],
)
def test_official_publish_runtime_gates_precede_database_client(
    monkeypatch,
    operations_enabled,
    official_enabled,
):
    app = FastAPI()
    opened = {"database": 0}
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")
    monkeypatch.setattr(
        admin_tournament_routes,
        "is_admin_tournament_admin_enabled",
        lambda: True,
    )
    if operations_enabled:
        monkeypatch.setenv(
            "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS",
            "1",
        )
    else:
        monkeypatch.delenv(
            "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS",
            raising=False,
        )
    if official_enabled:
        monkeypatch.setenv(
            "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH",
            "1",
        )
    else:
        monkeypatch.delenv(
            "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH",
            raising=False,
        )

    def forbidden_database():
        opened["database"] += 1
        raise AssertionError("database should not open")

    admin_tournament_routes.install_admin_tournament_routes(
        app,
        get_supabase_client=forbidden_database,
    )
    response = TestClient(app).post(
        "/admin/clubs/club-1/tournaments/admin/tournaments/tournament-1/"
        "draws/draw-1/matches/publish",
        headers={"Authorization": "Bearer not-read"},
        json={
            "confirmation_text": "PUBLISH MATCHES",
            "source": "next_tournament_ops_publish_matches",
        },
    )

    assert response.status_code == 403
    assert opened == {"database": 0}
