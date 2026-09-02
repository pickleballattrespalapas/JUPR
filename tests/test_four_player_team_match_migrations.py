from __future__ import annotations

import json
import re
from pathlib import Path


SCHEMA = Path(
    "supabase/migrations/20260831043522_four_player_team_match_schema.sql"
)
RPCS = Path(
    "supabase/migrations/"
    "20260831043613_four_player_team_match_management_rpcs.sql"
)
HARDENING = Path(
    "supabase/migrations/"
    "20261109000000_four_player_team_match_security_hardening.sql"
)
PRODUCTION_CONTRACT = Path("config/production_migration_contract.json")

TABLES = (
    "team_match_competitions",
    "team_match_teams",
    "team_match_team_members",
    "team_match_matchups",
    "team_match_games",
)

FUNCTION_SIGNATURES = {
    "admin_update_team_match_competition_v1": (
        "text, text, integer, jsonb, text"
    ),
    "admin_save_team_match_team_v1": (
        "text, text, text, text, text, jsonb, integer, integer, text"
    ),
    "admin_replace_team_match_schedule_v1": (
        "text, text, integer, jsonb, text"
    ),
}

REQUIRED_LEDGER_NAMES = {
    "four_player_team_match_schema",
    "four_player_team_match_management_rpcs",
    "four_player_team_match_security_hardening",
}


def _normalized(path: Path) -> str:
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8").lower()).strip()


def test_historical_team_match_sources_and_later_hardening_are_ordered() -> None:
    assert SCHEMA.is_file()
    assert RPCS.is_file()
    assert HARDENING.is_file()
    assert SCHEMA.name < RPCS.name < HARDENING.name


def test_team_match_tables_are_forced_rls_service_only_objects() -> None:
    sql = _normalized(SCHEMA)

    for table in TABLES:
        assert f"create table if not exists public.{table}" in sql
        assert f"alter table public.{table} enable row level security" in sql
        assert f"alter table public.{table} force row level security" in sql
        assert (
            f"revoke all on table public.{table} "
            "from public, anon, authenticated, service_role"
        ) in sql
        assert (
            "grant select, insert, update, delete "
            f"on table public.{table} to service_role"
        ) in sql

    assert "create policy" not in sql
    assert "to anon" not in sql
    assert "to authenticated" not in sql
    assert "drop table" not in sql


def test_privileged_team_match_rpcs_are_secure_at_creation() -> None:
    sql = _normalized(RPCS)

    assert sql.count("security definer") == len(FUNCTION_SIGNATURES)
    assert sql.count("set search_path = ''") == len(FUNCTION_SIGNATURES)
    for name, signature in FUNCTION_SIGNATURES.items():
        assert f"create or replace function public.{name}(" in sql
        assert (
            f"revoke execute on function public.{name}( {signature} ) "
            "from public, anon, authenticated"
        ) in sql
        assert (
            f"grant execute on function public.{name}( {signature} ) "
            "to service_role"
        ) in sql

    assert "grant execute on function" in sql
    assert re.search(r"grant execute on function .* to public(?:[ ;,])", sql) is None
    assert re.search(r"grant execute on function .* to anon(?:[ ;,])", sql) is None
    assert (
        re.search(
            r"grant execute on function .* to authenticated(?:[ ;,])",
            sql,
        )
        is None
    )


def test_post_ledger_hardening_revokes_default_public_execute() -> None:
    sql = _normalized(HARDENING)

    for name, signature in FUNCTION_SIGNATURES.items():
        assert (
            f"alter function public.{name}( {signature} ) security definer"
            in sql
        )
        assert (
            f"alter function public.{name}( {signature} ) "
            "set search_path = ''"
        ) in sql
        assert (
            f"revoke all on function public.{name}( {signature} ) "
            "from public, anon, authenticated"
        ) in sql
        assert (
            f"grant execute on function public.{name}( {signature} ) "
            "to service_role"
        ) in sql

    assert re.search(r"grant execute on function .* to public(?:[ ;,])", sql) is None
    assert re.search(r"grant execute on function .* to anon(?:[ ;,])", sql) is None
    assert (
        re.search(
            r"grant execute on function .* to authenticated(?:[ ;,])",
            sql,
        )
        is None
    )


def test_production_migration_contract_requires_recovered_security_history() -> None:
    contract = json.loads(PRODUCTION_CONTRACT.read_text(encoding="utf-8"))
    required = set(contract["required_ledger_names"])

    assert REQUIRED_LEDGER_NAMES <= required
