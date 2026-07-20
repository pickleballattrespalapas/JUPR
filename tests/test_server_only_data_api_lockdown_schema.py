from __future__ import annotations

import re
from pathlib import Path


LOCKDOWN = Path(
    "supabase/migrations/20260719155515_server_only_data_api_lockdown.sql"
)
CANONICAL_TABLES = Path(
    "supabase/migrations/20260719155737_canonicalize_server_only_tables.sql"
)


def _normalized(path: Path) -> str:
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8").lower()).strip()


def test_server_only_lockdown_is_a_canonical_supabase_migration() -> None:
    assert LOCKDOWN.is_file()
    assert LOCKDOWN.parent.as_posix() == "supabase/migrations"
    assert CANONICAL_TABLES.is_file()
    assert CANONICAL_TABLES.parent.as_posix() == "supabase/migrations"


def test_lockdown_removes_direct_browser_data_api_access() -> None:
    sql = _normalized(LOCKDOWN)

    assert "revoke all on schema public from public, anon, authenticated" in sql
    assert "alter table %i.%i enable row level security" in sql
    assert (
        "revoke all on table %i.%i from public, anon, authenticated" in sql
    )
    assert (
        "revoke execute on function %i.%i(%s) "
        "from public, anon, authenticated" in sql
    )
    assert (
        "revoke all on sequence %i.%i from public, anon, authenticated" in sql
    )


def test_lockdown_retains_explicit_fastapi_service_role_access() -> None:
    sql = _normalized(LOCKDOWN)

    assert "grant usage on schema public to service_role" in sql
    assert "grant all privileges on table %i.%i to service_role" in sql
    assert "grant select on table %i.%i to service_role" in sql
    assert (
        "grant usage, select, update on sequence %i.%i to service_role" in sql
    )
    assert "grant execute on function %i.%i(%s) to service_role" in sql


def test_lockdown_makes_future_public_exposure_opt_in() -> None:
    sql = _normalized(LOCKDOWN)

    assert (
        "alter default privileges for role postgres in schema public "
        "revoke all on tables from public, anon, authenticated"
    ) in sql
    assert (
        "alter default privileges for role postgres in schema public "
        "revoke execute on functions from public, anon, authenticated"
    ) in sql
    assert "notify pgrst, 'reload schema'" in sql


def test_legacy_server_tables_are_in_canonical_history_and_locked_down() -> None:
    sql = _normalized(CANONICAL_TABLES)
    expected = {
        "weekly_recaps",
        "public_support_requests",
        "league_live_sessions",
        "league_live_rounds",
        "league_live_courts",
    }

    for table in expected:
        assert f"create table if not exists public.{table}" in sql
        assert f"'{table}'" in sql

    assert "alter table public.%i enable row level security" in sql
    assert (
        "revoke all on table public.%i from public, anon, authenticated" in sql
    )
    assert "grant all privileges on table public.%i to service_role" in sql


def test_canonical_history_normalizes_match_context_id_to_text() -> None:
    sql = _normalized(CANONICAL_TABLES)

    assert "alter table public.matches" in sql
    assert "add column if not exists context_type text null" in sql
    assert "add column if not exists context_id text null" in sql
    assert "alter column context_id type text using context_id::text" in sql
    assert "if current_type <> 'text' then" in sql
