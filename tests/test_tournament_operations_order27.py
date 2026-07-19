from __future__ import annotations

from pathlib import Path

import pytest

from jupr_app.services.admin_tournament_game_service import (
    _insert_tournament_draw_games_atomic,
    _require_reviewed_row_versions,
)
from jupr_app.services.admin_tournament_team_service import write_admin_tournament_draw_teams_atomic
from jupr_app.services.admin_tournament_guarded_operation import (
    StaleTournamentAdminStateError,
    TournamentAdminRecoveryRequiredError,
)
from jupr_app.services.admin_tournament_match_publish_service import reconcile_admin_tournament_official_publish
from tests.test_admin_match_log_service import FakeSupabase


def _publish_plan() -> dict:
    return {
        "draw_id": "draw-1",
        "tournament_game_ids": ["game-1", "game-2"],
        "match_count": 2,
        "singles_match_count": 1,
        "doubles_match_count": 1,
        "playoff_winner_bonus_elo": 0.0,
        "bonus_tournament_game_ids": [],
    }


def _tables(matches: list[dict]) -> dict[str, list[dict]]:
    return {
        "tournament_games": [
            {"id": "game-1", "tournament_id": "tournament-1", "draw_id": "draw-1", "stage": "ROUND_ROBIN"},
            {"id": "game-2", "tournament_id": "tournament-1", "draw_id": "draw-1", "stage": "PLAYOFF"},
        ],
        "matches": matches,
    }


def test_official_publish_complete_set_reconstructs_result_without_a_write() -> None:
    tables = _tables(
        [
            {"id": "match-1", "club_id": "club-1", "tournament_id": "tournament-1", "tournament_game_id": "game-1"},
            {"id": "match-2", "club_id": "club-1", "tournament_id": "tournament-1", "tournament_game_id": "game-2"},
        ]
    )
    before = {name: [dict(row) for row in rows] for name, rows in tables.items()}

    result = reconcile_admin_tournament_official_publish(
        FakeSupabase(tables),
        club_id="club-1",
        tournament_id="tournament-1",
        draw_id="draw-1",
        expected_plan=_publish_plan(),
    )

    assert result["match_count"] == 2
    assert result["process_result"]["reconciled_from_authoritative_matches"] is True
    assert tables == before


@pytest.mark.parametrize(
    ("matches", "evidence"),
    [
        ([], "zero"),
        (
            [{"id": "match-1", "club_id": "club-1", "tournament_id": "tournament-1", "tournament_game_id": "game-1"}],
            "partial or duplicate",
        ),
        (
            [
                {"id": "match-1", "club_id": "club-1", "tournament_id": "tournament-1", "tournament_game_id": "game-1"},
                {"id": "match-2", "club_id": "club-1", "tournament_id": "tournament-1", "tournament_game_id": "game-1"},
            ],
            "partial or duplicate",
        ),
    ],
)
def test_official_publish_zero_partial_or_duplicate_set_stays_recovery_required(matches, evidence) -> None:
    with pytest.raises(TournamentAdminRecoveryRequiredError, match=evidence):
        reconcile_admin_tournament_official_publish(
            FakeSupabase(_tables(matches)),
            club_id="club-1",
            tournament_id="tournament-1",
            draw_id="draw-1",
            expected_plan=_publish_plan(),
        )


def test_official_publish_reconciliation_isolated_by_club_and_tournament() -> None:
    foreign_matches = [
        {"id": "foreign-club", "club_id": "club-2", "tournament_id": "tournament-1", "tournament_game_id": "game-1"},
        {"id": "foreign-tournament", "club_id": "club-1", "tournament_id": "tournament-2", "tournament_game_id": "game-2"},
    ]

    with pytest.raises(TournamentAdminRecoveryRequiredError, match="zero"):
        reconcile_admin_tournament_official_publish(
            FakeSupabase(_tables(foreign_matches)),
            club_id="club-1",
            tournament_id="tournament-1",
            draw_id="draw-1",
            expected_plan=_publish_plan(),
        )


def test_order27_migration_and_private_surface_static_contract() -> None:
    migration_path = Path("supabase/migrations/20260719204700_tournament_operations_guard_surface.sql")
    assert migration_path.exists()
    assert not Path("supabase/migrations/20260719204500_tournament_operations_guard_surface.sql").exists()
    sql = migration_path.read_text().lower()
    guard = Path("jupr_app/services/admin_tournament_guarded_operation.py").read_text()
    results_service = Path("jupr_app/services/admin_tournament_results_import_service.py").read_text()

    assert '"operations": "jupr_enable_next_admin_tournament_operations_mutations"' in guard.lower()
    assert "security invoker" in sql
    assert "enable row level security" in sql
    assert "revoke all on table public.tournament_games from public, anon, authenticated" in sql
    assert "grant select, insert, update, delete on table public.tournament_games to service_role" in sql
    assert "idx_matches_unique_tournament_game_id" in sql
    assert "lower(btrim(name))" in sql
    assert "tournament_games_nonnegative_scores" in sql
    assert "jupr_tournament_dependency_duplicate" in sql
    assert "order by g.id" in sql and "for update" in sql
    assert "p_new_players jsonb" in sql
    assert sql.index("insert into public.players") < sql.index("delete from public.tournament_games")
    assert "jupr_tournament_result_new_player_conflict" in sql
    assert "jupr_tournament_result_append_player_assigned" in sql
    assert "touch_tournament_draw_version_from_child" in sql
    assert "before insert or update or delete on public.tournament_games" in sql
    assert "touch_tournament_draw_version_from_badge" in sql
    assert "before insert or update or delete on public.player_badges" in sql
    assert "jupr_tournament_badge_podium_stale" in sql
    assert "jupr_tournament_podium_already_awarded" in sql
    assert "old.player1_id is distinct from new.player1_id" in sql
    assert "old.seed is distinct from new.seed" not in sql
    assert "jupr_tournament_downstream_score_lock" in sql
    assert "jupr_tournament_score_published_lock" in sql
    assert "p_expected_draw_updated_at timestamptz" in sql
    assert "p_expected_teams jsonb" in sql
    assert "p_expected_source_games jsonb" in sql
    assert "jupr_tournament_team_snapshot_stale" in sql
    assert "jupr_tournament_source_game_snapshot_stale" in sql
    assert "jupr_tournament_podium_snapshot_stale" in sql
    assert "order by team.id" in sql
    assert "order by source_game.id" in sql
    team_rpc = sql.split("create or replace function public.admin_write_tournament_draw_teams_cas", 1)[1].split(
        "create or replace function public.admin_insert_tournament_draw_games_cas", 1
    )[0]
    game_rpc = sql.split("create or replace function public.admin_insert_tournament_draw_games_cas", 1)[1].split(
        "create or replace function public.admin_replace_tournament_draw_podium_cas", 1
    )[0]
    podium_rpc = sql.split("create or replace function public.admin_replace_tournament_draw_podium_cas", 1)[1].split(
        "create or replace function public.admin_score_tournament_game_cas", 1
    )[0]
    score_rpc = sql.split("create or replace function public.admin_score_tournament_game_cas", 1)[1].split(
        "create or replace function public.admin_import_tournament_draw_results_cas", 1
    )[0]
    results_rpc = sql.split("create or replace function public.admin_import_tournament_draw_results_cas", 1)[1].split(
        "revoke all on function public.admin_write_tournament_draw_teams_cas", 1
    )[0]
    for rpc in (team_rpc, game_rpc, podium_rpc, results_rpc):
        assert rpc.index("for update;") < rpc.index("and d.updated_at = p_expected_draw_updated_at")
        assert "for no key update;" in rpc
    assert podium_rpc.index("perform existing_podium.id") < podium_rpc.index("and d.updated_at = p_expected_draw_updated_at")
    assert results_rpc.index("perform existing_podium.id") < results_rpc.index("and d.updated_at = p_expected_draw_updated_at")
    assert score_rpc.index("from unnest(array_append") < score_rpc.index("and d.updated_at = p_expected_draw_updated_at")
    assert "expected_updated_at timestamptz" in score_rpc
    assert "dependency.updated_at is distinct from expected.expected_updated_at" in score_rpc
    assert "drop function if exists public.admin_write_tournament_draw_teams_cas(text, text, text, boolean, jsonb)" in sql
    assert "admin_write_tournament_draw_teams_cas(text, text, text, timestamptz, boolean, jsonb)" in sql
    assert "drop function if exists public.admin_import_tournament_draw_results_cas(text, text, text, text" in sql
    assert "admin_import_tournament_draw_results_cas(text, text, text, timestamptz, text" in sql
    assert "drop function if exists public.admin_replace_tournament_draw_podium_cas(text, text, text, jsonb)" in sql
    assert "admin_score_tournament_game_cas(text, text, text, timestamptz, timestamptz, jsonb, jsonb)" in sql
    assert "jupr.tournament_results_import_structural_write" in results_rpc
    assert "order by case when upper(coalesce(x.stage, '')) = 'round_robin' then 0 else 1 end" in results_rpc
    assert "safe_add_player" not in results_service
    assert '"p_new_players": new_players' in results_service
    assert 'stage = "round_robin" if imported_stage == "round_robin" else "playoff"' in results_service.lower()


def test_order27_next_routes_and_recovery_copy_static_contract() -> None:
    panel = Path("apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx").read_text()
    api = Path("services/api/admin_tournament_routes.py").read_text()
    for suffix in ("draws", "import", "results", "publish"):
        assert Path(f"apps/web/app/admin/tournaments/ops/{suffix}/page.tsx").exists()
    assert "expected_state_fingerprint" in panel
    assert "recovery" in panel.lower()
    assert "publish_plan" in api
    assert "reconcile_admin_tournament_official_publish" in api
    assert "PERMISSION_ENTER_SCORES" in api
    assert "PERMISSION_MANAGE_MATCHES" in api
    assert panel.count("expected_team_versions: reviewedTeamVersions") == 3
    assert panel.count("expected_source_game_versions: reviewedSourceGameVersions") == 2
    assert panel.count("expected_draw_updated_at: reviewedDrawUpdatedAt") == 8
    assert api.count('"expected_draw_updated_at": payload.expected_draw_updated_at') == 16
    assert api.count('"expected_team_versions": [_dump_model(row) for row in payload.expected_team_versions]') == 6
    assert api.count('"expected_source_game_versions": [_dump_model(row) for row in payload.expected_source_game_versions]') == 4


@pytest.mark.parametrize("label", ["team set", "source game set"])
def test_stale_team_or_source_game_snapshot_is_rejected_before_atomic_write(label: str) -> None:
    current = [{"id": "game-1", "updated_at": "2026-07-19T12:00:01Z"}]
    reviewed = [{"id": "game-1", "updated_at": "2026-07-19T12:00:00Z"}]

    with pytest.raises(StaleTournamentAdminStateError, match=f"{label} changed"):
        _require_reviewed_row_versions(current, reviewed, label=label, atomic=True)


def test_atomic_game_rpc_receives_exact_snapshots_and_maps_sql_cas_conflict() -> None:
    class FailingRpc:
        def __init__(self) -> None:
            self.name = ""
            self.payload: dict = {}

        def rpc(self, name, payload):
            self.name = name
            self.payload = payload
            return self

        def execute(self):
            raise RuntimeError("JUPR_TOURNAMENT_SOURCE_GAME_SNAPSHOT_STALE")

    supabase = FailingRpc()
    teams = [{"id": "team-1", "updated_at": "2026-07-19T12:00:00Z"}]
    games = [{"id": "game-1", "updated_at": "2026-07-19T12:00:00Z"}]

    with pytest.raises(StaleTournamentAdminStateError, match="source game set changed"):
        _insert_tournament_draw_games_atomic(
            supabase,
            club_id="club-1",
            tournament_id="tournament-1",
            draw_id="draw-1",
            expected_draw_updated_at="2026-07-19T12:00:00Z",
            expected_team_versions=teams,
            expected_source_game_versions=games,
            mode="PLAYOFF",
            rows=[{"id": "playoff-1", "stage": "PLAYOFF"}],
        )

    assert supabase.name == "admin_insert_tournament_draw_games_cas"
    assert supabase.payload["p_expected_teams"] == teams
    assert supabase.payload["p_expected_source_games"] == games


def test_atomic_team_rpc_receives_draw_cas_and_maps_sql_stale() -> None:
    class FailingRpc:
        def __init__(self) -> None:
            self.payload: dict = {}

        def rpc(self, _name, payload):
            self.payload = payload
            return self

        def execute(self):
            raise RuntimeError("JUPR_TOURNAMENT_DRAW_STALE")

    supabase = FailingRpc()
    with pytest.raises(StaleTournamentAdminStateError, match="draw changed"):
        write_admin_tournament_draw_teams_atomic(
            supabase,
            club_id="club-1",
            tournament_id="tournament-1",
            draw_id="draw-1",
            expected_draw_updated_at="2026-07-19T12:00:00Z",
            rows=[{"id": "team-1", "team_number": 1}],
            replace=True,
        )

    assert supabase.payload["p_expected_draw_updated_at"] == "2026-07-19T12:00:00Z"
