from __future__ import annotations

import os
from typing import Any


REQUIRED_PLAYER_BADGES_COLUMNS = {
    "awarded_by",
    "rule_version",
    "eval_run_id",
}
REQUIRED_REVOKED_COLUMNS = {
    "revoked_at",
    "revoked_by",
    "revoke_reason",
}
MIGRATION_HINT = (
    "DB schema out of date. Apply migrations/20260625_badge_recompute_runs.sql and "
    "migrations/20260630_player_badges_revocation.sql"
)


def ensure_badge_schema_preflight(supabase: Any) -> bool:
    if _should_skip_preflight():
        return True
    columns = _fetch_player_badges_columns(supabase)
    missing = sorted(REQUIRED_PLAYER_BADGES_COLUMNS - columns)
    missing_revoked = sorted(REQUIRED_REVOKED_COLUMNS - columns)
    if missing or missing_revoked:
        missing_all = ", ".join(sorted(set(missing + missing_revoked)))
        raise RuntimeError(
            f"{MIGRATION_HINT}. Missing player_badges columns: {missing_all}."
        )
    return True


def _fetch_player_badges_columns(supabase: Any) -> set[str]:
    resp = (
        supabase.schema("information_schema")
        .table("columns")
        .select("column_name")
        .eq("table_schema", "public")
        .eq("table_name", "player_badges")
        .execute()
    )
    return {row.get("column_name") for row in (resp.data or []) if row.get("column_name")}


def _should_skip_preflight() -> bool:
    return os.getenv("JUPR_SKIP_DB_PREFLIGHT", "0") == "1"
