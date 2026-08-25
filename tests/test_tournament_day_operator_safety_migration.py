from __future__ import annotations

from pathlib import Path


MIGRATION = Path("supabase/migrations/20261108000000_tournament_day_operator_safety.sql")
REPAIR_MIGRATION = Path(
    "supabase/migrations/20261108013000_tournament_operator_special_form_repair.sql"
)
TOURNAMENT_MIGRATIONS = (
    MIGRATION,
    Path(
        "supabase/migrations/"
        "20261108003000_tournament_terminal_completion_and_schedule_recovery.sql"
    ),
    Path(
        "supabase/migrations/"
        "20261108010000_public_primary_player_canonicalization.sql"
    ),
    REPAIR_MIGRATION,
    Path(
        "supabase/migrations/"
        "20261108014000_tournament_registration_player_projection_backfill.sql"
    ),
    Path(
        "supabase/migrations/"
        "20261108015000_tournament_player_projection_finalized_review_repair.sql"
    ),
    Path(
        "supabase/migrations/"
        "20261108016000_standard_tournament_substitute_policy.sql"
    ),
)


def migration_sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_postgres_conditional_expressions_are_never_schema_qualified() -> None:
    for migration in TOURNAMENT_MIGRATIONS:
        sql = migration.read_text(encoding="utf-8").lower()
        for special_form in ("coalesce", "nullif", "least", "greatest"):
            assert f"pg_catalog.{special_form}" not in sql, (
                f"{migration} schema-qualifies PostgreSQL special form "
                f"{special_form}"
            )


def test_applied_score_wrappers_have_an_additive_runtime_repair() -> None:
    source = REPAIR_MIGRATION.read_text(encoding="utf-8").lower()
    wrappers = (
        "admin_score_tournament_game_result_cas",
        "admin_import_tournament_draw_results_with_metadata_cas",
        "admin_score_tournament_team_match_game_reviewed_cas",
    )

    for wrapper in wrappers:
        assert f"create or replace function public.{wrapper}" in source
        assert f"revoke all on function public.{wrapper}" in source
        assert f"grant execute on function public.{wrapper}" in source
    assert source.count("security invoker") == len(wrappers)
    assert source.count("set search_path = ''") == len(wrappers)
    assert "division_scoring" not in source
    assert "notify pgrst, 'reload schema';" in source


def test_non_played_rpc_is_invoker_scoped_and_service_role_only() -> None:
    sql = migration_sql()
    assert "admin_record_non_played_tournament_day_game_cas" in sql
    assert "security definer" not in sql
    assert "security invoker" in sql
    assert "set search_path = ''" in sql
    assert "from public.admin_role_assignments" in sql
    assert "current score-entry staff authorization is required" in sql
    assert "revoke all on function public.admin_record_non_played" in sql
    assert ") from public, anon, authenticated;" in sql
    assert ") to service_role;" in sql


def test_non_played_rpc_retains_cas_idempotency_and_lock_order() -> None:
    sql = migration_sql()
    assert "assert_tournament_day_live_operation" in sql
    assert "p_request_fingerprint" in sql
    assert "p_expected_run_version" in sql
    assert "p_expected_queue_version" in sql
    assert "p_expected_queue_entry_version" in sql
    assert "p_expected_court_version" in sql
    assert "p_expected_game_updated_at" in sql
    assert "for update of queue" in sql
    assert "game -> court -> participant lock order" in sql
    assert sql.index("admin_score_tournament_game_cas(") < sql.index("for update;\n    if not found then", sql.index("game -> court"))


def test_non_played_metadata_is_visible_and_rating_ineligible() -> None:
    sql = migration_sql()
    assert "result_type in ('played', 'forfeit', 'no_show', 'retirement')" in sql
    assert "new.result_type := v_outcome->>'result_type'" in sql
    assert "'synthetic_progression_score', true" in sql
    assert "'rating_publish_eligible', false" in sql
    assert "set state = 'completed'" in sql
    assert "fill_tournament_day_live_courts" in sql


def test_non_played_rpc_rejects_incomplete_or_contradictory_progression_evidence() -> None:
    sql = migration_sql()
    start = sql.index(
        "create or replace function public.admin_record_non_played_tournament_day_game_cas"
    )
    end = sql.index("revoke all on function public.admin_record_non_played", start)
    rpc = sql[start:end]

    assert "jsonb_typeof(p_game_patch) is distinct from 'object'" in rpc
    assert "nullif(p_winner_team_id, '') is null" in rpc
    assert "nullif(p_game_patch->>'score_a', '') is null" in rpc
    assert "nullif(p_game_patch->>'score_b', '') is null" in rpc
    assert "v_score_a := (p_game_patch->>'score_a')::integer" in rpc
    assert "v_score_b := (p_game_patch->>'score_b')::integer" in rpc
    assert "is distinct from case" not in rpc
    assert "p_winner_team_id = v_queue.team_a_id::text" in rpc
    assert "p_game_patch->>'loser_team_id' = v_queue.team_b_id::text" in rpc
    assert "v_score_a > v_score_b" in rpc
    assert "p_winner_team_id = v_queue.team_b_id::text" in rpc
    assert "p_game_patch->>'loser_team_id' = v_queue.team_a_id::text" in rpc
    assert "v_score_b > v_score_a" in rpc


def test_guard_allows_only_the_explicit_non_played_day_action() -> None:
    sql = migration_sql()
    guard_start = sql.index("create or replace function public.guard_tournament_game_day_live_mutation")
    metadata_start = sql.index("create or replace function public.apply_tournament_day_result_metadata")
    guard = sql[guard_start:metadata_start]
    assert "tournament_day_live_record_non_played_result" in guard
    assert "operation.status = 'intent'" in guard


def test_ordinary_atomic_score_wrapper_persists_played_result_metadata() -> None:
    sql = migration_sql()
    start = sql.index(
        "create or replace function public.admin_score_tournament_game_result_cas"
    )
    end = sql.index(
        "create or replace function public.admin_import_tournament_draw_results_with_metadata_cas",
        start,
    )
    score = sql[start:end]
    assert "admin_score_tournament_game_cas(" in score
    assert "result_type = p_game_patch ->> 'result_type'" in score
    assert "result_note = p_game_patch ->> 'result_note'" in score
    assert "result_recorded_by = p_game_patch ->> 'result_recorded_by'" in score
    assert "score_review_json = p_game_patch -> 'score_review_json'" in score
    assert "jupr_tournament_outcome_conversion_required" in score
    assert score.count("message = 'jupr_tournament_score_format_stale';") == 1
    assert "division_scoring" not in score
    assert "grant execute on function public.admin_score_tournament_game_result_cas" in sql


def test_results_import_atomic_wrapper_persists_score_review_evidence() -> None:
    sql = migration_sql()
    start = sql.index(
        "create or replace function public.admin_import_tournament_draw_results_with_metadata_cas"
    )
    results_import = sql[start:]
    assert "admin_import_tournament_draw_results_cas(" in results_import
    assert "result_type = v_game_patch ->> 'result_type'" in results_import
    assert "result_note = v_game_patch ->> 'result_note'" in results_import
    assert "result_recorded_by = v_game_patch ->> 'result_recorded_by'" in results_import
    assert "score_review_json = v_game_patch -> 'score_review_json'" in results_import
    assert "division_scoring" not in results_import
    assert "grant execute on function public.admin_import_tournament_draw_results_with_metadata_cas" in sql


def test_team_score_wrapper_persists_review_on_child_and_rating_game() -> None:
    sql = migration_sql()
    start = sql.index(
        "create or replace function public.admin_score_tournament_team_match_game_reviewed_cas"
    )
    team_score = sql[start:]
    assert "add column if not exists score_review_json" in sql
    assert "admin_score_tournament_team_match_game_cas(" in team_score
    assert "set score_review_json = p_score_review" in team_score
    assert "result_recorded_by = p_actor" in team_score
    assert "p_score_review ->> 'accepted' is distinct from 'true'" in team_score
    assert "p_score_review ->> 'acknowledged' is distinct from 'true'" in team_score
    assert "division_scoring" not in team_score
    assert (
        "grant execute on function public.admin_score_tournament_team_match_game_reviewed_cas"
        in team_score
    )
