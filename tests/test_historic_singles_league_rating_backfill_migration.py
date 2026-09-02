from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "supabase"
    / "migrations"
    / "20261104000000_historic_singles_league_rating_backfill.sql"
)


def _sql() -> str:
    assert MIGRATION.exists()
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_replay_league_rating_writer_is_lease_fenced_and_preserves_lifecycle():
    sql = _sql()
    function = sql.split(
        "create or replace function "
        "public.apply_replay_league_rating_rows_atomic(",
        1,
    )[1].split("revoke all on function", 1)[0]

    assert "security invoker" in function
    assert "set search_path = ''" in function
    assert "public.assert_replay_write_fence_atomic(" in function
    assert "jsonb_array_length(v_rows)" in function
    assert "between 1 and 500" in function
    assert "lower(pg_catalog.btrim(data.league_name))" in function
    assert "on conflict (club_id, player_id, league_name) do update" in function
    assert "is_active = excluded.is_active" in function
    assert "inactive_at = excluded.inactive_at" in function
    assert "v_verified <> v_expected" in function
    assert "to service_role" in sql
    assert "from public, anon, authenticated" in sql


def test_roster_batch_v2_atomically_gates_only_new_operations():
    sql = _sql()
    function = sql.split(
        "create or replace function "
        "public.admin_apply_league_roster_batch_atomic_v2(",
        1,
    )[1].split("revoke all on function", 1)[0]

    assert "security invoker" in function
    assert "set search_path = ''" in function

    roster_lock = function.index("jupr:league-roster-batch:")
    replay_lock = function.index("jupr:replay-club:", roster_lock)
    operation_lookup = function.index(
        "from public.admin_league_roster_batch_operations as operation",
        replay_lock,
    )
    receipt_branch = function.index("if v_has_operation then", operation_lookup)
    metadata_lock = function.index(
        "from public.leagues_metadata as metadata", receipt_branch
    )
    assert roster_lock < replay_lock < operation_lookup < receipt_branch < metadata_lock
    assert (
        "v_result := public.admin_apply_league_roster_batch_atomic_v1("
        in function[receipt_branch:metadata_lock]
    )
    assert "activity.after_json -> 'league_ratings'" in function
    assert "activity.entity_id = v_result ->> 'operation_id'" in function
    assert "for update" in function[metadata_lock:]

    assert "when v_raw_league_status = 'paused' then 'paused'" in function
    assert "is distinct from (v_league_status = 'active')" in function
    lifecycle_gate = function.index(
        "if v_league_status not in ('draft', 'active') then", metadata_lock
    )
    final_v1 = function.index(
        "v_result := public.admin_apply_league_roster_batch_atomic_v1(",
        lifecycle_gate,
    )
    assert lifecycle_gate < final_v1
    assert "jupr_league_roster_batch_read_only" in function

    signature = """public.admin_apply_league_roster_batch_atomic_v2(
  uuid,
  text,
  text,
  text,
  text,
  text,
  jsonb,
  numeric,
  text,
  text,
  text
)"""
    assert f"revoke all on function {signature}" in sql
    assert f"grant execute on function {signature}" in sql
    assert "from public, anon, authenticated" in sql
    assert "to service_role" in sql


def test_historic_repair_is_exact_guarded_idempotent_and_snapshot_derived():
    sql = _sql()

    assert "'tres_palapas'" in sql
    assert "'acceptance singles league 0731'" in sql
    assert "v_match_id constant bigint := 49" in sql
    assert "if not exists (" in sql
    assert "from public.leagues_metadata as metadata" in sql
    assert "return;" in sql
    assert "coalesce(v_league.is_active, false) is not true" in sql
    assert "v_league.ended_at is not null" in sql
    assert "v_league.k_factor is distinct from 32" in sql

    roster_lock = sql.index("jupr:league-roster-batch:")
    replay_lock = sql.index("jupr:replay-club:", roster_lock)
    direct_lock = sql.index("jupr:direct-match-entry:", replay_lock)
    assert roster_lock < replay_lock < direct_lock
    assert "in ('pending', 'running')" in sql
    assert "for update" in sql

    for reviewed_value in (
        "v_match.t1_p1 is distinct from 22",
        "v_match.t2_p1 is distinct from 23",
        "v_match.score_t1 is distinct from 11",
        "v_match.score_t2 is distinct from 8",
        "v_match.date::date is distinct from date '2026-07-31'",
        "v_match.singles_replay_managed is not true",
        "v_match.deleted_at is not null",
        "is distinct from 1205.0526::numeric",
        "is distinct from 1194.9474::numeric",
        "player.name = 'test b'",
        "player.name = 'test c'",
    ):
        assert reviewed_value in sql

    assert "pg_catalog.round(v_match.t1_p1_r::numeric, 4)" in sql
    assert "pg_catalog.round(v_match.t2_p1_r::numeric, 4)" in sql
    assert "pg_catalog.round(v_match.t1_p1_r_end::numeric, 4)" in sql
    assert "pg_catalog.round(v_match.t2_p1_r_end::numeric, 4)" in sql
    assert "pg_catalog.round(v_match.elo_delta::numeric, 4)" in sql
    assert "on conflict (club_id, player_id, league_name) do nothing" in sql
    assert "existing normalized row has different state" in sql
    assert "exact repaired rows were not found" in sql


def test_historic_repair_records_one_stable_reviewable_audit():
    sql = _sql()

    assert "historic_singles_league_rating_backfill" in sql
    assert "league_rating_backfill" in sql
    assert "migration:20261104000000_historic_singles_league_rating_backfill" in sql
    assert "authoritative_final_match_snapshots" in sql
    assert "known_missing_player_ids" in sql
    assert "v_existing_audit_count > 1" in sql
    assert "activity.after_json is distinct from v_after" in sql
    assert "if v_existing_audit_count = 0 then" in sql
    assert "flagged_for_review" in sql
    assert "true\n    );" in sql
