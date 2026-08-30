from __future__ import annotations

from pathlib import Path


MIGRATION = Path(
    "supabase/migrations/20261108022500_tournament_team_retirement_results.sql"
)


def migration_sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_retirement_state_is_durable_and_internally_consistent() -> None:
    sql = migration_sql()

    assert "add column if not exists competition_status" in sql
    assert "add column if not exists retired_at" in sql
    assert "add column if not exists retired_by" in sql
    assert "add column if not exists retirement_max_score" in sql
    assert "competition_status in ('active', 'retired')" in sql
    assert "tournament_teams_retirement_state_chk" in sql


def test_operator_selects_non_playing_team_and_server_derives_winner() -> None:
    sql = migration_sql()
    start = sql.index(
        "create or replace function public.admin_record_tournament_day_non_played_result_cas"
    )
    rpc = sql[start:]

    assert "p_non_playing_team_id text" in rpc
    assert "v_derived_winner := case" in rpc
    assert "p_winner_team_id is distinct from v_derived_winner" in rpc
    assert "p_game_patch->>'loser_team_id'" in rpc
    assert "is distinct from p_non_playing_team_id" in rpc


def test_operator_note_is_optional_but_length_limited() -> None:
    sql = migration_sql()
    start = sql.index(
        "create or replace function public.admin_record_tournament_day_non_played_result_cas"
    )
    rpc = sql[start:]

    assert "char_length(coalesce(p_result_note, '')) > 500" in rpc
    assert "operator note are required" not in rpc
    assert "new.result_note := nullif" in sql


def test_retirement_preserves_played_results_and_cascades_only_unfinished_games() -> None:
    sql = migration_sql()

    assert "create or replace function public.cascade_tournament_team_retirement" in sql
    assert "if new.finalized_at is null and (v_a_retired or v_b_retired)" in sql
    assert "result_type = 'retirement'" in sql
    assert "'rating_publish_eligible', false" in sql
    assert "and game.finalized_at is null" in sql
    assert "and p_non_playing_team_id::uuid in (game.team_a_id, game.team_b_id)" in sql
    assert "downstream.team_a_source->>'winnerof'" in sql
    assert "downstream.team_b_source->>'loserof'" in sql


def test_retirement_cascade_releases_queue_claims_and_is_service_role_only() -> None:
    sql = migration_sql()

    assert "update public.tournament_day_live_participant_claims" in sql
    assert "set state = 'released'" in sql
    assert "update public.tournament_day_live_queue" in sql
    assert "set team_a_id = new.team_a_id" in sql
    assert "state = 'completed'" in sql
    assert "security definer" not in sql
    assert sql.count("security invoker") == 3
    assert "from public, anon, authenticated" in sql
    assert ") to service_role;" in sql
