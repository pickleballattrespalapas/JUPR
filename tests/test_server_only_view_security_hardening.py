from __future__ import annotations

import json
import re
from pathlib import Path


MIGRATION = Path(
    "supabase/migrations/"
    "20261109001000_server_only_view_security_hardening.sql"
)
PRODUCTION_CONTRACT = Path("config/production_migration_contract.json")
VIEWS = ("league_settings", "public_leaderboards")


def _sql() -> str:
    return re.sub(
        r"\s+",
        " ",
        MIGRATION.read_text(encoding="utf-8").lower(),
    ).strip()


def test_view_hardening_is_catalog_safe_and_idempotent() -> None:
    sql = _sql()

    for view in VIEWS:
        assert f"'{view}'" in sql
    assert "pg_catalog.pg_class" in sql
    assert "pg_catalog.pg_namespace" in sql
    assert "if relation_kind is null then continue" in sql
    assert "if relation_kind <> 'v' then" in sql
    assert "alter view %i.%i set (security_invoker = true)" in sql
    assert "drop view" not in sql


def test_view_hardening_removes_client_access_and_retains_service_reads() -> None:
    sql = _sql()

    assert (
        "revoke all on table %i.%i from public, anon, authenticated, "
        "service_role"
    ) in sql
    assert "grant select on table %i.%i to service_role" in sql
    assert re.search(r"grant .* to (?:public|anon|authenticated)\b", sql) is None
    assert "notify pgrst, 'reload schema'" in sql


def test_production_contract_requires_view_security_hardening() -> None:
    contract = json.loads(PRODUCTION_CONTRACT.read_text(encoding="utf-8"))

    assert "server_only_view_security_hardening" in contract[
        "required_ledger_names"
    ]
