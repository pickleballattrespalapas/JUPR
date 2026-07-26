from pathlib import Path


MIGRATION = Path(
    "supabase/migrations/20260725231000_challenge_ladder_public_results.sql"
)
ADMIN_SERVICE = Path("jupr_app/services/admin_challenge_ladder_service.py")
OPERATION_SERVICE = Path("jupr_app/services/admin_live_ladder_operation_service.py")
ADMIN_ROUTES = Path("services/api/admin_challenge_ladder_routes.py")


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def _function(name: str) -> str:
    sql = _sql()
    marker = f"create or replace function public.{name}("
    start = sql.index(marker)
    end = sql.index("\n$$;", start) + len("\n$$;")
    return sql[start:end]


def test_public_result_relation_uses_only_the_canonical_migration_source() -> None:
    assert MIGRATION.is_file()
    assert MIGRATION.parent.as_posix() == "supabase/migrations"
    assert not Path("migrations/20260725231000_challenge_ladder_public_results.sql").exists()
    assert "challenge_ladder_public_result_source_guard" in _sql()


def test_public_result_json_is_nullable_narrow_and_not_backfilled() -> None:
    sql = _sql()
    assert "add column if not exists updated_at timestamptz not null" in sql
    assert "add column if not exists public_result_json jsonb" in sql
    assert "('ladder_challenges', 'tier_id')" in sql
    assert "('ladder_challenges', 'forfeit_by')" in sql
    assert "('ladder_challenges', 'forfeit_reason')" in sql
    assert "validate_challenge_ladder_public_result" in sql
    assert "array['version', 'match_ids', 'rank_change']" in sql
    assert "array['a', 'b']" in sql
    assert "array['swapped', 'challenger', 'defender']" in sql
    assert "array['player_id', 'before', 'after']" in sql
    assert "result_json ?& array['version', 'match_ids', 'rank_change']" in sql
    assert "match_ids ?& array['a', 'b']" in sql
    assert "challenger ?& array['player_id', 'before', 'after']" in sql
    assert "jsonb_typeof(match_ids->'a') is distinct from 'number'" in sql
    assert "result_json->'version' is distinct from '1'::jsonb" in sql
    assert "trunc((challenger->>'player_id')::numeric)" in sql
    assert "null for legacy, imported, forfeit, or unlinked results" in sql
    assert "admin_activity_log" not in sql


def test_atomic_source_guard_requires_every_mutated_table_and_receipt_column() -> None:
    sql = _sql()

    for table in (
        "players",
        "league_ratings",
        "leagues_metadata",
        "matches",
        "live_ladder_admin_operations",
    ):
        assert f"to_regclass('public.{table}')" in sql
    for required in (
        "('players', 'rating')",
        "('players', 'wins')",
        "('players', 'losses')",
        "('players', 'matches_played')",
        "('players', 'last_game_at')",
        "('players', 'inactive_at')",
        "('players', 'active')",
        "('league_ratings', 'rating')",
        "('league_ratings', 'starting_rating')",
        "('leagues_metadata', 'k_factor')",
        "('matches', 'elo_delta')",
        "('matches', 't1_p1_r')",
        "('matches', 't1_p1_r_end')",
        "('live_ladder_admin_operations', 'request_json')",
        "('live_ladder_admin_operations', 'result_json')",
        "('live_ladder_admin_operations', 'error_text')",
        "('live_ladder_admin_operations', 'completed_at')",
        "('live_ladder_admin_operations', 'updated_at')",
    ):
        assert required in sql


def test_finalize_rpc_is_service_role_only_and_operation_bound() -> None:
    sql = _sql()
    signature = (
        "public.admin_finalize_challenge_ladder_result_v1(\n"
        "  text, bigint, text, bigint, timestamptz, text, jsonb, text, text\n"
        ")"
    )
    assert "security invoker" in sql
    assert "set search_path = pg_catalog" in sql
    assert f"revoke all on function {signature}" in sql
    assert f"grant execute on function {signature}" in sql
    assert "from public, anon, authenticated" in sql
    assert "to service_role" in sql
    assert "from public.live_ladder_admin_operations" in sql
    assert "operation_row.status is distinct from 'running'" in sql
    assert "operation_row.operation_type is distinct from 'publish_result'" in sql
    assert "operation_row.recovery_json->'match_context_ids'" in sql
    assert "is distinct from jsonb_build_array(p_match_context_a, p_match_context_b)" in sql
    finalize = _function("admin_finalize_challenge_ladder_result_v1")
    assert "current_setting(" in finalize
    assert "'jupr.challenge_ladder_atomic_core'" in finalize


def test_finalize_rpc_verifies_active_exact_matches_and_swaps_without_collision() -> None:
    sql = _sql()
    finalize = _function("admin_finalize_challenge_ladder_result_v1")
    compact_finalize = " ".join(finalize.split())
    assert (
        "select * into challenger_roster from public.ladder_roster "
        "where club_id = p_club_id and player_id = challenge_row.challenger_id"
        in compact_finalize
    )
    assert (
        "select * into defender_roster from public.ladder_roster "
        "where club_id = p_club_id and player_id = challenge_row.defender_id"
        in compact_finalize
    )
    assert compact_finalize.count("tier_id = challenge_row.tier_id") >= 2
    assert compact_finalize.count("and is_active is not false") >= 2
    assert "limit 1; select * select * into defender_roster" not in " ".join(
        sql.split()
    )
    assert finalize.count("from public.matches") == 2
    assert finalize.count("and deleted_at is null") == 2
    assert "and context_id = p_match_context_a" in finalize
    assert "and context_id = p_match_context_b" in finalize
    assert "do not match the ranked participants and swing-partner format" in finalize
    assert (
        "match_a.t1_p2::bigint is not distinct from "
        "challenge_row.challenger_id::bigint"
    ) in sql
    assert (
        "match_a.t2_p2::bigint is not distinct from "
        "challenge_row.defender_id::bigint"
    ) in sql
    assert "match_a.t1_p2 is not distinct from match_a.t2_p2" in sql
    assert "match_a.t1_p2 is distinct from match_b.t2_p2" in sql
    assert "match_a.t2_p2 is distinct from match_b.t1_p2" in sql
    assert "select coalesce(max(rank), 0) + 1" in finalize
    assert finalize.count("update public.ladder_roster") == 3
    assert "public_result_json = final_public_result" in finalize
    assert "returning * into updated_challenge" in finalize
    assert "notify pgrst, 'reload schema'" in sql


def test_atomic_result_rpc_is_server_only_and_exactly_bound_to_persisted_plan() -> None:
    sql = _sql()
    compact_sql = " ".join(sql.split())
    signature = (
        "public.admin_apply_challenge_ladder_result_atomic_v1( "
        "text, bigint, text, jsonb, text, bigint, timestamptz, text, "
        "jsonb, jsonb, jsonb, jsonb, text, text "
        ")"
    )
    function = _function("admin_apply_challenge_ladder_result_atomic_v1")
    compact_function = " ".join(function.split())

    assert "security invoker" in function
    assert "set search_path = pg_catalog" in function
    assert f"revoke all on function {signature}" in compact_sql
    assert f"grant execute on function {signature}" in compact_sql
    assert "from public, anon, authenticated" in sql
    assert "to service_role" in sql
    assert "operation_row.operation_type is distinct from 'publish_result'" in function
    assert "operation_row.status is distinct from 'running'" in function
    assert (
        "operation_row.request_json->'atomic_core' is distinct from p_atomic_core"
        in function
    )
    assert (
        "p_atomic_core->>'plan_fingerprint' is distinct from p_plan_fingerprint"
        in function
    )
    for binding in (
        "p_atomic_core #> '{write_plan,match_rows}' is distinct from p_match_rows",
        "p_atomic_core #> '{write_plan,player_updates}' is distinct from p_player_updates",
        "p_atomic_core #> '{write_plan,league_rating_updates}' is distinct from p_league_rating_updates",
        "p_atomic_core #> '{write_plan,league_metadata_expectations}' is distinct from p_league_metadata_expectations",
    ):
        assert binding in compact_function


def test_atomic_result_rpc_cas_checks_complete_challenge_roster_and_rating_state() -> None:
    function = _function("admin_apply_challenge_ladder_result_atomic_v1")

    assert "p_atomic_core->'challenge_expected'" in function
    assert "p_atomic_core->'tier_roster_expected'" in function
    assert "from public.ladder_challenges" in function
    assert "from public.ladder_roster" in function
    assert "order by" in function
    assert "for update" in function
    assert "from jsonb_to_recordset(p_player_updates)" in function
    assert "from public.players" in function
    assert "from jsonb_to_recordset(p_league_metadata_expectations)" in function
    assert "from public.leagues_metadata" in function
    assert "from jsonb_to_recordset(p_league_rating_updates)" in function
    assert "from public.league_ratings" in function
    assert "update public.players" in function
    assert (
        "insert into public.league_ratings" in function
        or "update public.league_ratings" in function
    )
    assert "get diagnostics" in function
    assert "row_count" in function


def test_atomic_result_rpc_rejects_all_context_collisions_and_inserts_once() -> None:
    function = _function("admin_apply_challenge_ladder_result_atomic_v1")
    collision_start = function.index("from public.matches")
    insert_start = function.index("insert into public.matches")
    collision_block = function[collision_start:insert_start]

    assert "context_type = 'challenge_ladder'" in collision_block
    assert "p_match_context_a" in collision_block
    assert "p_match_context_b" in collision_block
    assert "deleted_at is null" not in collision_block
    assert function.count("insert into public.matches") == 1
    assert "jsonb_array_length(p_match_rows) <> 2" in function
    assert "get diagnostics inserted_count = row_count" in function
    assert "inserted_count <> 2" in function
    assert "perform set_config(" in function
    assert "'jupr.challenge_ladder_atomic_core'" in function
    assert "p_operation_key" in function
    assert "admin_finalize_challenge_ladder_result_v1(" in function


def test_result_and_forfeit_finalizers_persist_response_loss_receipts() -> None:
    result = _function("admin_finalize_challenge_ladder_result_v1")
    forfeit = _function("admin_finalize_challenge_ladder_forfeit_v1")

    for function, mode in (
        (result, "challenge_ladder_result_core"),
        (forfeit, "challenge_ladder_forfeit_core"),
    ):
        assert "'core_committed', true" in function
        assert f"'{mode}'" in function
        assert "'post_processors'" in function
        assert "update public.live_ladder_admin_operations" in function
        assert "status = 'mutated'" in function
        assert "result_json =" in function
        assert "error_text = null" in function
        assert "completed_at = p_completed_at" in function
        assert "get diagnostics" in function
        assert "row_count" in function

    assert "'status'," in result
    assert "'pending'" in result
    assert "'complete'" in result
    assert "'status', 'complete'" in forfeit
    assert "'official_matches'" in result
    assert "'public_result_json', null" in forfeit


def test_atomic_result_claim_blocks_other_rating_and_match_writers() -> None:
    sql = _sql()
    guard = _function("block_rating_change_during_challenge_ladder_publish")

    assert "security invoker" in guard
    assert "live_ladder_admin_operations" in guard
    assert "operation_type = 'publish_result'" in guard
    for status in ("'intent'", "'running'", "'mutated'", "'recovery_required'"):
        assert status in guard
    assert "current_setting(" in guard
    assert "'jupr.challenge_ladder_atomic_core'" in guard
    assert "raise exception" in guard
    for table in ("matches", "players", "league_ratings", "leagues_metadata"):
        assert (
            f"before insert or update or delete on public.{table}" in sql
        )
        assert (
            "execute function "
            "public.block_rating_change_during_challenge_ladder_publish()"
            in sql
        )
    signature = "public.block_rating_change_during_challenge_ladder_publish()"
    assert f"revoke all on function {signature}" in sql
    assert f"grant execute on function {signature}" in sql


def test_forfeit_finalize_rpc_is_atomic_operation_bound_and_service_role_only() -> None:
    sql = _sql()
    signature = (
        "public.admin_finalize_challenge_ladder_forfeit_v1(\n"
        "  text, bigint, text, bigint, timestamptz, text\n"
        ")"
    )
    assert f"revoke all on function {signature}" in sql
    assert f"grant execute on function {signature}" in sql
    assert "operation_row.operation_type is distinct from 'record_forfeit'" in sql
    assert "running durable challenge forfeit operation not found" in sql
    assert "forfeited player must be a ranked challenge participant" in sql
    assert "set status = 'forfeited'" in sql
    assert "forfeit_by = p_forfeited_by_id" in sql
    assert "public_result_json = null" in sql
    assert "'public_result_json', null" in sql


def test_service_delegates_rank_mutations_to_the_operation_bound_rpcs() -> None:
    source = ADMIN_SERVICE.read_text(encoding="utf-8")

    assert "def _swap_ranks(" not in source
    assert "submit_match_batch" not in source
    assert "build_write_plan_only=True" in source
    assert 'CHALLENGE_LADDER_RESULT_FINALIZE_RPC = "admin_finalize_challenge_ladder_result_v1"' in source
    assert 'CHALLENGE_LADDER_RESULT_ATOMIC_RPC = "admin_apply_challenge_ladder_result_atomic_v1"' in source
    assert 'CHALLENGE_LADDER_FORFEIT_FINALIZE_RPC = "admin_finalize_challenge_ladder_forfeit_v1"' in source
    assert "supabase.rpc(CHALLENGE_LADDER_RESULT_ATOMIC_RPC, params)" in source
    assert "supabase.rpc(CHALLENGE_LADDER_RESULT_FINALIZE_RPC, params)" in source
    assert "supabase.rpc(CHALLENGE_LADDER_FORFEIT_FINALIZE_RPC, params)" in source


def test_result_forfeit_race_and_reconcile_paths_all_use_receipt_recovery() -> None:
    routes = ADMIN_ROUTES.read_text(encoding="utf-8")
    operations = OPERATION_SERVICE.read_text(encoding="utf-8")

    assert "def _build_challenge_recovery_callback(" in routes
    result_route = routes[
        routes.index("def post_admin_challenge_ladder_result("):
        routes.index("def post_admin_challenge_ladder_result_preview(")
    ]
    assert result_route.index("replay_durable_admin_operation_if_present(") < (
        result_route.index("preview_admin_challenge_ladder_result_for_challenge(")
    )
    assert result_route.count("recover_incomplete=recover_incomplete") >= 2

    forfeit_route = routes[
        routes.index("def post_admin_challenge_ladder_forfeit("):
        routes.index("def post_admin_challenge_ladder_pass(")
    ]
    assert "replay_durable_admin_operation_if_present(" in forfeit_route
    assert forfeit_route.count("recover_incomplete=recover_incomplete") >= 2

    reconcile_start = routes.index("def post_admin_challenge_ladder_reconcile(")
    reconcile_route = routes[reconcile_start:]
    assert "reconcile_durable_admin_operation(" in reconcile_route
    assert (
        "recover_incomplete=_build_challenge_recovery_callback(" in reconcile_route
    )

    assert "def recover_admin_challenge_ladder_operation_result(" in (
        ADMIN_SERVICE.read_text(encoding="utf-8")
    )
    assert operations.count("recovered_result = recover_incomplete(") >= 3
    assert "if raced and str(raced.get(\"request_fingerprint\")" in operations
