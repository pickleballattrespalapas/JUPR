from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "supabase"
    / "migrations"
    / "20261108021000_tournament_day_court_reservations.sql"
)
FK_INDEX_MIGRATION = (
    ROOT
    / "supabase"
    / "migrations"
    / "20261108022000_tournament_day_court_reservations_fk_index.sql"
)


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def _function_body(sql: str, function_name: str) -> str:
    marker = f"create or replace function public.{function_name}("
    start = sql.index(marker)
    end = sql.index("$function$;", start) + len("$function$;")
    return sql[start:end]


def test_reservation_state_has_one_court_slot_and_reserved_claims() -> None:
    sql = _sql()

    assert "reserved_court_id uuid" in sql
    assert "state = 'reserved'" in sql
    assert "reserved_at is not null" in sql
    assert "uq_tournament_day_live_queue_reserved_court" in sql
    assert "'reserved', 'held', 'called', 'on_court', 'released'" in sql


def test_reserved_court_foreign_key_has_a_leading_column_index() -> None:
    sql = FK_INDEX_MIGRATION.read_text(encoding="utf-8").lower()

    assert "ix_tournament_day_live_queue_reserved_court_id" in sql
    assert "on public.tournament_day_live_queue (reserved_court_id)" in sql
    assert "where reserved_court_id is not null" in sql


def test_reservation_rpc_rechecks_intent_queue_court_game_and_players() -> None:
    reserve = _function_body(_sql(), "admin_reserve_tournament_day_game_cas")

    for evidence in (
        "tournament_day_live_reserve_game_for_court",
        "p_expected_run_version",
        "p_expected_queue_version",
        "p_expected_queue_entry_version",
        "p_expected_game_updated_at",
        "p_expected_court_version",
        "queue.state = 'waiting'",
        "day_draw.state = 'active'",
        "tournament_day_live_players_ready",
        "earlier.priority < v_queue.priority",
        "state = 'reserved'",
        "reserved_court_id = v_court.id",
        "for update",
        "for share",
    ):
        assert evidence in reserve
    assert "on conflict (queue_id, player_id) do nothing" in reserve
    assert "queue_version = run.queue_version + 1" in reserve


def test_release_trigger_promotes_only_the_selected_reserved_match() -> None:
    promote = _function_body(
        _sql(), "promote_tournament_day_court_reservation"
    )

    assert "old.court_id" in promote
    assert "queue.reserved_court_id = old.court_id" in promote
    assert "queue.state = 'reserved'" in promote
    assert "state = 'on_court'" in promote
    assert "court_id = old.court_id" in promote
    assert "reserved_court_id = null" in promote
    assert "claim.state = 'reserved'" in promote
    assert "set state = 'on_court'" in promote
    assert "trg_tournament_day_live_promote_reserved_court" in _sql()
    assert "after update of state, court_id, released_at" in _sql()


def test_cancel_wait_returns_original_priority_and_releases_claims() -> None:
    cancel = _function_body(
        _sql(), "admin_cancel_tournament_day_court_reservation_cas"
    )

    assert "tournament_day_live_requeue_game" in cancel
    assert "queue.state = 'reserved'" in cancel
    assert "set state = 'released'" in cancel
    assert "set state = 'waiting'" in cancel
    assert "reserved_court_id = null" in cancel
    assert "priority =" not in cancel


def test_reservation_rpcs_are_service_role_only_and_invoker_safe() -> None:
    sql = _sql()

    for function_name in (
        "admin_reserve_tournament_day_game_cas",
        "admin_cancel_tournament_day_court_reservation_cas",
    ):
        assert f"revoke execute on function public.{function_name}(" in sql
        assert f"grant execute on function public.{function_name}(" in sql
    assert "security definer" not in sql
    assert sql.count("security invoker") == 3
    assert sql.count("set search_path = ''") == 3
