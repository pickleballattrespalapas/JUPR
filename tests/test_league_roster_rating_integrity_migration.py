from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "supabase"
    / "migrations"
    / "20261101000000_league_roster_rating_integrity.sql"
)


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def _direct_match_function(sql: str) -> str:
    return sql.split(
        "create or replace function "
        "public.admin_apply_direct_match_entry_atomic_v1(",
        1,
    )[1].split("revoke all on function", 1)[0]


def _roster_function(sql: str) -> str:
    return sql.split(
        "create or replace function "
        "public.admin_apply_league_roster_batch_atomic_v1(",
        1,
    )[1].split("revoke all on function", 1)[0]


def test_forward_wrapper_requires_exact_active_official_league_memberships() -> None:
    sql = _sql()
    function = _direct_match_function(sql)

    assert "managed_leagues as" in function
    assert "jsonb_to_recordset(v_league_metadata_expectations)" in function
    assert "join managed_leagues as managed_league" in function
    assert "required_memberships as" in function
    assert "(match_row.t1_p1, true)" in function
    assert "(match_row.t2_p1, true)" in function
    assert "(match_row.t1_p2, v_match_format = 'doubles')" in function
    assert "(match_row.t2_p2, v_match_format = 'doubles')" in function
    assert "not in ('overall', 'popup', 'singles')" in function
    assert "symmetric_difference as" in function
    assert "update_row.after->>'is_active' is distinct from 'true'" in function
    assert "official league participants require exact active" in function


def test_recordset_aliases_do_not_declare_duplicate_columns() -> None:
    function = _direct_match_function(_sql())
    declarations = re.findall(
        r"jsonb_to_recordset\([^)]*\)\s+as\s+\w+\s*\((.*?)\)",
        function,
        flags=re.DOTALL,
    )

    assert declarations
    for declaration in declarations:
        columns = [
            field.strip().split()[0]
            for field in declaration.split(",")
            if field.strip()
        ]
        assert len(columns) == len(set(columns)), declaration


def test_singles_league_ratings_use_cas_in_the_match_transaction() -> None:
    sql = _sql()
    function = _direct_match_function(sql)

    assert "security invoker" in function
    assert "set search_path = ''" in function
    assert "for update" in function
    assert "jupr_direct_match_league_rating_stale" in function
    assert "insert into public.league_ratings" in function
    assert "update public.league_ratings" in function
    assert "update public.admin_direct_match_entry_operations" in function
    assert "update public.admin_activity_log" in function
    assert "result_json = v_result" in function
    assert "when unique_violation then" in function
    assert "no part of this plan committed" in function

    compatibility_call = function.index(
        "v_result := "
        "public.admin_apply_direct_match_entry_base_20261101"
    )
    league_rating_write = function.index("insert into public.league_ratings")
    assert compatibility_call < league_rating_write
    assert "p_player_updates,\n    '[]'::jsonb," in function


def test_forward_wrapper_is_replay_safe_and_casefolds_league_keys() -> None:
    sql = _sql()
    function = _direct_match_function(sql)

    assert "to_regprocedure(v_base_signature) is null" in sql
    assert "to_regprocedure(v_public_signature) is null" in sql
    assert "create or replace function " in sql
    assert "requires case-insensitive unique league-rating rows" in sql
    assert "create unique index if not exists " in sql
    assert "league_ratings_club_player_normalized_league_uidx" in sql
    assert "(pg_catalog.lower(pg_catalog.btrim(league_name)))" in sql
    assert "lower(pg_catalog.btrim(update_row.league_name))" in function
    assert "lower(pg_catalog.btrim(league_rating.league_name))" in function


def test_doubles_delegation_and_exact_retries_remain_compatible() -> None:
    sql = _sql()
    function = _direct_match_function(sql)

    retry = function.index("if found then")
    coverage = function.index("required_memberships as")
    doubles = function.index("if v_match_format = 'doubles' then")
    singles_cas = function.index("singles historically rejected")
    assert retry < coverage
    assert doubles < singles_cas
    assert "official_league_rating_update_count" in function
    base_name = "admin_apply_direct_match_entry_base_20261101"
    assert base_name in function
    assert len(base_name.encode("utf-8")) <= 63
    assert ") to service_role;" in sql


def test_roster_activation_uses_the_league_format_baseline() -> None:
    sql = _sql()
    function = _roster_function(sql)

    assert "v_league_match_format text" in function
    assert "coalesce(v_league.match_format, '')" in function
    assert "v_league_match_format not in ('doubles', 'singles')" in function
    assert "when v_league_match_format = 'singles'" in function
    assert "coalesce(player.singles_rating, 1200)::numeric" in function
    assert "coalesce(player.rating, 1200)::numeric" in function

    insert = function.split("insert into public.league_ratings (", 1)[1]
    explicit_override = insert.index("when p_starting_rating is not null")
    format_baseline = insert.index("when baseline.raw_rating <= 20")
    assert explicit_override < format_baseline
    assert "else baseline.raw_rating" in insert


def test_roster_reactivation_preserves_an_existing_rating() -> None:
    function = _roster_function(_sql())
    reactivation = function.split(
        "update public.league_ratings as rating\n"
        "       set is_active = true,",
        1,
    )[1].split("insert into public.league_ratings", 1)[0]

    assert "inactive_at = null" in reactivation
    assert "set rating" not in reactivation
    assert "starting_rating" not in reactivation
