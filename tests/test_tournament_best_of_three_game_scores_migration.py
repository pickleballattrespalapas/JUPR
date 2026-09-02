from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / (
    "20261108026000_tournament_best_of_three_game_scores.sql"
)


def _source() -> str:
    return MIGRATION.read_text(encoding="utf-8")


def test_best_of_three_children_have_durable_parent_identity() -> None:
    source = _source()

    assert "add column if not exists series_parent_game_id uuid null" in source
    assert "add column if not exists series_game_number smallint null" in source
    assert "tournament_games_series_parent_fk" in source
    assert "references public.tournament_games(id)" in source
    assert "on delete cascade" in source
    assert "tournament_games_series_identity_chk" in source
    assert "uq_tournament_games_series_parent_number" in source
    assert "series_game_number is not null" in source
    assert "series_game_number between 1 and 3" in source
    assert "series_parent_game_id <> id" in source
    assert "pg_catalog.upper(coalesce(stage, '')) <> 'SERIES_GAME'" in source
    assert "matches_tournament_game_fk" in source
    assert "foreign key (tournament_game_id)" in source
    assert "on delete restrict" in source


def test_legacy_finalized_best_of_three_rows_fail_before_normalization() -> None:
    source = _source()

    preflight = source.index("v_legacy_parent_count bigint")
    constraints = source.index("tournament_games_series_parent_fk")
    assert preflight < constraints
    preflight_source = source[preflight:constraints]
    assert "game.finalized_at is not null" in preflight_source
    assert "coalesce(game.result_type, 'PLAYED')" in preflight_source
    assert "event.scoring_override" in preflight_source
    assert "event.scoring_default" in preflight_source
    assert "event.tournament_id = game.tournament_id::text" in preflight_source
    assert "= 'BEST_2_OF_3'" in preflight_source
    assert "if v_legacy_parent_count > 0" in preflight_source
    assert (
        "JUPR_TOURNAMENT_BEST_OF_THREE_LEGACY_DATA_REQUIRES_REVIEW: count=%s"
        in preflight_source
    )


def test_legacy_series_and_orphans_remain_not_valid_until_owner_review() -> None:
    source = _source()

    assert "v_series_identity_violation_count bigint" in source
    assert "v_match_tournament_game_orphan_count bigint" in source
    assert "v_series_identity_violation_count = 0" in source
    assert "v_match_tournament_game_orphan_count = 0" in source
    assert (
        "validate constraint tournament_games_series_identity_chk" in source
    )
    assert "validate constraint matches_tournament_game_fk" in source
    assert (
        "JUPR_TOURNAMENT_SERIES_IDENTITY_LEGACY_ROWS_RETAINED:" in source
    )
    assert "JUPR_TOURNAMENT_MATCH_GAME_ORPHANS_RETAINED:" in source
    assert source.count("state=NOT_VALID owner_review_required=true") == 2

    validation = source.index("v_series_identity_violation_count bigint")
    comments = source.index("comment on column public.tournament_games.series_parent_game_id")
    validation_source = source[validation:comments]
    assert "game.series_game_number is null" in validation_source
    assert "game.series_game_number between 1 and 3" in validation_source
    assert ") is false" in validation_source
    assert "left join public.tournament_games as game" in validation_source
    assert "match_row.tournament_game_id is not null" in validation_source
    assert "game.id is null" in validation_source


def test_best_of_three_parent_validation_and_child_sync_are_atomic() -> None:
    source = _source()

    assert "validate_tournament_best_of_three_result" in source
    assert "JUPR_TOURNAMENT_BEST_OF_THREE_GAMES_INVALID" in source
    assert "JUPR_TOURNAMENT_BEST_OF_THREE_CHILD_INVALID" in source
    assert "new.parent_result_only := true" in source
    assert "sync_tournament_best_of_three_rating_games" in source
    assert "after insert or update on public.tournament_games" in source
    assert "'SERIES_GAME'" in source
    assert "'GAME_TO_11'" in source
    assert "on conflict (series_parent_game_id, series_game_number)" in source
    assert "JUPR_TOURNAMENT_SCORE_PUBLISHED_LOCK" in source
    assert "pg_catalog.jsonb_array_length(v_games) - v_game_number" in source
    assert "pg_catalog.exists" not in source
    assert "pg_catalog.greatest" not in source
    assert "pg_catalog.least" not in source


def test_best_of_three_final_shape_is_deferred_until_transaction_end() -> None:
    source = _source()

    assert "assert_tournament_best_of_three_final_state" in source
    assert (
        "create constraint trigger trg_90_tournament_games_best_of_three_final_state"
        in source
    )
    assert "after insert or update or delete on public.tournament_games" in source
    assert "deferrable initially deferred" in source
    assert "coalesce(old.series_parent_game_id, old.id)" in source
    assert "coalesce(new.series_parent_game_id, new.id)" in source
    assert "v_effective_format = 'BEST_2_OF_3'" in source
    assert "event.scoring_override" in source
    assert "event.scoring_default" in source
    assert "event.tournament_id = v_parent.tournament_id::text" in source
    assert "v_review_format = 'BEST_2_OF_3'" in source
    assert "v_child_count <> v_game_count" in source
    assert "v_child.score_review_json is distinct from v_review" in source
    assert "JUPR_TOURNAMENT_BEST_OF_THREE_FINAL_STATE_INVALID" in source
    assert "JUPR_TOURNAMENT_BEST_OF_THREE_CHILD_SET_INVALID" in source


def test_retirement_preserves_only_valid_completed_rating_games() -> None:
    source = _source()

    assert "v_parent_allows_rating_children" in source
    assert "v_preserve_retirement_games" in source
    assert source.count("retirement_completed_games_preserved") >= 5
    assert "v_result_type = 'RETIREMENT'" in source
    assert "v_is_retirement_series" in source
    assert "v_game_count not in (1, 2)" in source
    assert "elsif greatest(v_wins_a, v_wins_b) >= 2" in source
    assert "JUPR_TOURNAMENT_RETIREMENT_RATING_EVIDENCE_INVALID" in source
    assert "JUPR_TOURNAMENT_RETIREMENT_RATING_EVIDENCE_IMMUTABLE" in source
    assert "v_child.result_type is distinct from 'PLAYED'" in source


def test_series_children_remain_under_day_and_recovery_authority() -> None:
    source = _source()

    guard = source.index("create or replace function public.guard_tournament_game_day_live_mutation")
    validator = source.index("create or replace function public.validate_tournament_best_of_three_result")
    guarded_source = source[guard:validator]
    assert "coalesce(new.series_parent_game_id, new.id)" in guarded_source
    assert "coalesce(old.series_parent_game_id, old.id)" in guarded_source
    assert "tournament_day_live_queue" in guarded_source
    assert "'tournament_day_live_record_non_played_result'" in guarded_source

    activation = source.index(
        "create or replace function public.assert_tournament_day_live_draw_ready"
    )
    seeding = source.index(
        "create or replace function public.seed_tournament_day_live_draw"
    )
    recovery = source.index(
        "create or replace function private.assert_tournament_draw_recovery_snapshot"
    )
    assert "game.series_parent_game_id is null" in source[activation:seeding]
    assert source[seeding:recovery].count("game.series_parent_game_id is null") >= 2
    # Recovery fences every stored source version (parents plus rating leaves),
    # while all schedule validation/counting remains parent-only.
    recovery_end = source.index(
        "create or replace function public.admin_reconcile_tournament_round_robin_games_cas"
    )
    assert "game.series_parent_game_id is null" not in source[recovery:recovery_end]
    assert source[recovery_end:].count("game.series_parent_game_id is null") >= 6


def test_official_rating_children_cannot_be_mutated_from_generic_match_log() -> None:
    source = _source()

    assert "protect_official_tournament_match_source" in source
    assert "before insert or update or delete on public.matches" in source
    assert "jupr.official_publish_operation_key" in source
    assert "operation.status = 'intent'" in source
    assert "'ops_official_publish'" in source
    assert "'tournament_live_official_publish'" in source
    assert "#> '{payload,publish_plan,tournament_game_ids}'" in source
    assert "new.context_id = new.tournament_game_id::text" in source
    assert "old.tournament_game_id is null and new.tournament_game_id is null" in source
    assert "JUPR_TOURNAMENT_OFFICIAL_MATCH_ASSOCIATION_REQUIRES_PUBLISH" in source
    assert "JUPR_TOURNAMENT_OFFICIAL_MATCH_SOURCE_IMMUTABLE" in source
    assert "JUPR_TOURNAMENT_OFFICIAL_MATCH_HARD_DELETE_FORBIDDEN" in source
    for field in (
        "new.score_t1 is distinct from old.score_t1",
        "new.score_t2 is distinct from old.score_t2",
        "new.deleted_at is distinct from old.deleted_at",
        "new.tournament_game_id is distinct from old.tournament_game_id",
        "new.rating_bonus_elo is distinct from old.rating_bonus_elo",
    ):
        assert field in source


def test_migration_keeps_client_roles_read_only() -> None:
    source = _source()

    for function_name in (
        "guard_tournament_game_day_live_mutation",
        "validate_tournament_best_of_three_result",
        "sync_tournament_best_of_three_rating_games",
        "assert_tournament_best_of_three_final_state",
        "protect_official_tournament_match_source",
    ):
        assert f"revoke all on function public.{function_name}()" in source
        assert f"grant execute on function public.{function_name}()" in source
    assert "from public, anon, authenticated" in source
    assert "to service_role" in source
