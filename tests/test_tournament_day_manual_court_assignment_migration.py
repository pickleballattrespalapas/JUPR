from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "supabase"
    / "migrations"
    / "20261108020000_manual_tournament_day_court_assignment.sql"
)


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def _function_body(sql: str, function_name: str) -> str:
    marker = f"create or replace function public.{function_name}("
    start = sql.index(marker)
    end = sql.index("$function$;", start) + len("$function$;")
    return sql[start:end]


def test_implicit_allocator_is_gated_to_the_explicit_legacy_command() -> None:
    sql = _sql()
    gate = _function_body(sql, "fill_tournament_day_live_courts")

    assert "fill_tournament_day_live_courts_explicit" in sql
    assert "tournament_day_live_auto_fill_courts" in gate
    assert "return '[]'::jsonb" in gate
    assert "fill_tournament_day_live_courts_explicit" in gate
    assert "security invoker" in gate
    assert "set search_path = ''" in gate
    assert "security definer" not in sql


def test_one_game_assignment_rechecks_durable_intent_and_exact_live_truth() -> None:
    sql = _sql()
    assign = _function_body(sql, "admin_assign_tournament_day_game_cas")

    for evidence in (
        "assert_tournament_day_live_operation",
        "p_expected_run_version",
        "p_expected_queue_version",
        "p_expected_queue_entry_version",
        "p_expected_game_updated_at",
        "p_expected_court_version",
        "queue.state = 'waiting'",
        "court.state = 'open'",
        "day_draw.state = 'active'",
        "tournament_day_live_players_ready",
        "earlier.priority < v_queue.priority",
        "for update",
        "for share",
    ):
        assert evidence in assign
    assert "assign_next_court" in assign
    assert "assign_game_to_court" in assign
    assert "state = 'on_court'" in assign
    assert "on conflict (queue_id, player_id) do nothing" in assign
    assert "claim.released_at is not null" in assign
    assert "claim.released_at is null" in assign
    assert "queue_version = run.queue_version + 1" in assign


def test_move_and_requeue_are_atomic_and_preserve_queue_priority() -> None:
    sql = _sql()
    reassign = _function_body(sql, "admin_reassign_tournament_day_game_cas")

    for evidence in (
        "p_expected_source_court_version",
        "p_expected_target_court_version",
        "queue.state = 'on_court'",
        "state = 'released'",
        "state = 'waiting'",
        "court_id = null",
        "court_id = v_target_court.id",
        "eligible_since = coalesce(queue.eligible_since, v_now)",
        "for update",
    ):
        assert evidence in reassign
    assert "priority =" not in reassign
    assert "target_court_id" in reassign
    assert "queue_version = run.queue_version + 1" in reassign


def test_new_assignment_rpcs_are_service_role_only() -> None:
    sql = _sql()

    for function_name in (
        "admin_assign_tournament_day_game_cas",
        "admin_reassign_tournament_day_game_cas",
    ):
        assert f"revoke execute on function public.{function_name}(" in sql
        assert f"grant execute on function public.{function_name}(" in sql
    assert "from public, anon, authenticated" in sql
    assert "to service_role" in sql
