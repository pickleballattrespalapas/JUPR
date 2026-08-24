from __future__ import annotations

from pathlib import Path
from uuid import UUID

from jupr_app.domain.league_live_publish import (
    normalize_league_live_publish_matches,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "supabase"
    / "migrations"
    / "20261107000000_league_live_direct_match_context.sql"
)


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_league_live_context_is_accepted_and_requires_its_durable_id() -> None:
    sql = _sql()
    validation = sql.split(
        "if exists (\n    select 1\n"
        "      from pg_catalog.jsonb_to_recordset(v_match_rows)",
        1,
    )[1].split(
        "raise exception using\n"
        "      errcode = '22023',\n"
        "      message = 'jupr_direct_match_rows_invalid",
        1,
    )[0]

    assert "context_type text,\n        context_id text," in validation
    assert "'event',\n             'league_live_session'" in validation
    assert (
        "pg_catalog.btrim(coalesce(match_row.context_type, ''))\n"
        "            = 'league_live_session'\n"
        "          and (\n"
        "            nullif(pg_catalog.btrim(match_row.context_id), '') is null"
    ) in validation
    assert (
        "'^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-"
        "[89ab][0-9a-f]{3}-[0-9a-f]{12}$'"
    ) in validation


def test_domain_context_round_trips_without_cross_domain_relabeling() -> None:
    match = normalize_league_live_publish_matches(
        [
            {
                "court": 1,
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 7,
            }
        ],
        session_id="d7221db1-db24-52d2-90fc-b667c16e0193",
        round_number=1,
        league_name="Acceptance Flex 0822A",
        week_tag="Week 1",
        match_date="2026-08-24",
        expected_match_count=1,
    )[0]
    sql = _sql()

    assert match["context_type"] == "league_live_session"
    UUID(match["context_id"])
    assert f"'{match['context_type']}'" in sql
    assert "nullif(pg_catalog.btrim(match_row.context_type), '')" in sql
    assert "nullif(pg_catalog.btrim(match_row.context_id), '')" in sql
    assert "disguise league live rows as event rows" in sql
    assert "replace(" not in sql


def test_replacement_keeps_atomic_rpc_least_privilege() -> None:
    sql = _sql()

    assert (
        "create or replace function public."
        "admin_apply_direct_match_entry_atomic_v1_base_20260727("
    ) in sql
    assert "security invoker" in sql
    assert "set search_path = ''" in sql
    assert (
        "revoke all on function public."
        "admin_apply_direct_match_entry_atomic_v1_base_20260727("
    ) in sql
    assert ") from public, anon, authenticated;" in sql
    assert ") to service_role;" in sql
