from __future__ import annotations

from pathlib import Path


MIGRATION = Path(
    "supabase/migrations/"
    "20261108025000_tournament_playoff_round_scoring.sql"
)


def migration_sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def _function(sql: str, name: str, next_marker: str) -> str:
    start = sql.index(f"create or replace function public.{name}")
    end = sql.index(next_marker, start)
    return sql[start:end]


def test_game_level_scoring_format_is_nullable_but_constrained() -> None:
    sql = migration_sql()

    assert "add column if not exists scoring_format text null" in sql
    assert "tournament_games_scoring_format_chk" in sql
    for scoring_format in (
        "game_to_11",
        "game_to_15",
        "game_to_21",
        "best_2_of_3",
    ):
        assert f"'{scoring_format}'" in sql


def test_atomic_game_insert_parses_validates_and_persists_reviewed_format() -> None:
    sql = migration_sql()
    rpc = _function(
        sql,
        "admin_insert_tournament_draw_games_cas",
        "revoke all on function public.admin_insert_tournament_draw_games_cas",
    )

    assert "jsonb_to_recordset(p_games) as x(scoring_format text)" in rpc
    assert "jupr_tournament_game_scoring_format_invalid" in rpc
    assert "playoff_game_code text, playoff_round text, scoring_format text" in rpc
    assert "upper(nullif(btrim(x.scoring_format), ''))" in rpc
    # Compatibility callers can omit the field; an explicit unsupported value
    # is rejected before the insert and the table check is a second boundary.
    assert "nullif(btrim(x.scoring_format), '') is not null" in rpc
    format_validation = rpc[
        rpc.index("if exists (", rpc.index("scoring_format text")) :
    ]
    assert "v_mode = 'playoff'" not in format_validation
    assert "security invoker" in rpc
    assert "set search_path = public, pg_temp" in rpc


def test_day_and_ordinary_score_review_prefer_the_game_override() -> None:
    sql = migration_sql()
    metadata = _function(
        sql,
        "apply_tournament_day_result_metadata",
        "revoke all on function public.apply_tournament_day_result_metadata",
    )
    reviewed_score = _function(
        sql,
        "admin_score_tournament_game_result_cas",
        "revoke all on function public.admin_score_tournament_game_result_cas",
    )

    assert metadata.index("new.scoring_format") < metadata.index(
        "event.scoring_override"
    )
    assert reviewed_score.index("game.scoring_format") < reviewed_score.index(
        "event.scoring_override"
    )
    assert "jupr_tournament_score_format_stale" in metadata
    assert "jupr_tournament_score_format_stale" in reviewed_score


def test_future_retirement_results_use_each_playoff_round_format() -> None:
    sql = migration_sql()
    cascade = _function(
        sql,
        "cascade_tournament_team_retirement",
        "revoke all on function public.cascade_tournament_team_retirement",
    )

    assert "upper(coalesce(new.stage, '')) = 'playoff'" in cascade
    assert "new.scoring_format" in cascade
    assert "event.scoring_override" in cascade
    assert "when 'game_to_11' then 11" in cascade
    assert "when 'game_to_15' then 15" in cascade
    assert "when 'game_to_21' then 21" in cascade
    assert "when 'best_2_of_3' then 2" in cascade
    assert "'scoring_format', v_scoring_format" in cascade
    assert "'target_score', v_target" in cascade
    rr_branch = cascade[cascade.index("elsif pg_catalog.upper") :]
    assert "v_team_a.retirement_max_score" in rr_branch
    assert "configured round-robin max score is missing" in rr_branch
    assert "v_scoring_format := case v_target" in rr_branch
    assert "when 11 then 'game_to_11'" in rr_branch
    assert "when 15 then 'game_to_15'" in rr_branch
    assert "when 21 then 'game_to_21'" in rr_branch
    assert "when 2 then 'best_2_of_3'" in rr_branch
    assert "configured round-robin max score is unsupported" in rr_branch


def test_scoring_policy_change_participates_in_draw_derivation_locks() -> None:
    sql = migration_sql()
    trigger = _function(
        sql,
        "touch_tournament_draw_version_from_child",
        "revoke all on function public.touch_tournament_draw_version_from_child",
    )

    assert "old.scoring_format is distinct from new.scoring_format" in trigger
    assert trigger.index("old.scoring_format") < trigger.index(
        "jupr_tournament_score_podium_lock"
    )
    assert "if tg_table_name = 'tournament_games' then" in trigger


def test_redefined_private_functions_preserve_service_role_boundary() -> None:
    sql = migration_sql()

    assert "security definer" not in sql
    assert sql.count("security invoker") == 5
    for function_name in (
        "touch_tournament_draw_version_from_child",
        "admin_insert_tournament_draw_games_cas",
        "apply_tournament_day_result_metadata",
        "admin_score_tournament_game_result_cas",
        "cascade_tournament_team_retirement",
    ):
        assert f"revoke all on function public.{function_name}" in sql
        assert f"grant execute on function public.{function_name}" in sql
    assert "notify pgrst, 'reload schema';" in sql
