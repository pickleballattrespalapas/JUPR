from __future__ import annotations

import pytest

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from jupr_app.services import admin_tournament_match_publish_service as publish_service
from jupr_app.services.admin_tournament_ops_service import (
    get_admin_tournament_ops_snapshot,
)
from jupr_app.services.admin_tournament_team_competition_service import (
    get_admin_team_tournament_snapshot,
)
from services.api import admin_tournament_routes
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
