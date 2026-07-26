from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "supabase"
    / "migrations"
    / "20260726222339_atomic_direct_match_entry.sql"
)


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_atomic_direct_match_rpc_is_least_privilege_security_invoker():
    sql = _sql()

    assert "security invoker" in sql
    assert "set search_path = ''" in sql
    assert (
        "revoke all on table public.admin_direct_match_entry_operations\n"
        "  from public, anon, authenticated;"
    ) in sql
    assert (
        "grant select, insert on table "
        "public.admin_direct_match_entry_operations\n"
        "  to service_role;"
    ) in sql
    assert (
        "revoke all on function "
        "public.admin_apply_direct_match_entry_atomic_v1("
    ) in sql
    assert ") from public, anon, authenticated;" in sql
    assert ") to service_role;" in sql


def test_one_rpc_owns_match_ratings_receipt_and_audit_transaction():
    sql = _sql()
    function = sql.split(
        "create or replace function "
        "public.admin_apply_direct_match_entry_atomic_v1(",
        1,
    )[1].split("revoke all on function", 1)[0]

    assert "insert into public.matches" in function
    assert "update public.players" in function
    assert "insert into public.league_ratings" in function
    assert "update public.league_ratings" in function
    assert (
        "insert into public.admin_direct_match_entry_operations" in function
    )
    assert "insert into public.admin_activity_log" in function
    assert "exception\n  when unique_violation then" in function
    assert "no part of this plan committed" in function


def test_retry_receipts_and_stale_compare_and_swap_are_explicit():
    sql = _sql()

    assert "unique (club_id, idempotency_key)" in sql
    assert "unique (club_id, request_fingerprint)" not in sql
    assert "jupr_direct_match_idempotency_conflict" in sql
    assert "return v_operation.result_json" in sql
    assert "'idempotent', true" in sql
    assert "for update" in sql
    assert "jupr_direct_match_player_stale" in sql
    assert "jupr_direct_match_league_metadata_stale" in sql
    assert "jupr_direct_match_league_rating_stale" in sql


def test_doubles_and_singles_contracts_are_both_guarded():
    sql = _sql()

    assert "v_match_format not in ('doubles', 'singles')" in sql
    assert "v_match_format = 'doubles'" in sql
    assert "v_match_format = 'singles'" in sql
    assert "singles_replay_managed" in sql
    assert "singles_rating" in sql
    assert "rating = (v_after->>'rating')::numeric(10,4)" in sql
