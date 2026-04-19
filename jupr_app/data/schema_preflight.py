from __future__ import annotations

import logging
import os
from typing import Any

from postgrest.exceptions import APIError


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
    "migrations/20260630_player_badges_revocation.sql. If you just applied them in "
    "Supabase, run \"NOTIFY pgrst, 'reload schema';\" in the SQL editor to refresh the "
    "PostgREST schema cache"
)
REQUIRED_BADGE_TABLES = {"badge_eval_runs"}
OPTIONAL_BADGE_TABLES = {"badge_eval_queue"}
SKIP_PREFLIGHT_ENV = "JUPR_SKIP_BADGE_SCHEMA_PREFLIGHT"
_MISSING_COLUMN_CODES = {"PGRST204"}
_MISSING_TABLE_CODES = {"PGRST205", "42P01"}

logger = logging.getLogger(__name__)


def ensure_badge_schema_preflight(supabase: Any) -> bool:
    return ensure_badge_schema_preflight_strict(supabase)


def ensure_badge_schema_preflight_live(supabase: Any) -> bool:
    if _should_skip_preflight():
        return True
    missing_columns = _find_missing_player_badges_columns(supabase, include_revoked=False)
    if missing_columns:
        message_parts = [MIGRATION_HINT + "."]
        missing_all = ", ".join(sorted(missing_columns))
        message_parts.append(f"Missing player_badges columns: {missing_all}.")
        raise RuntimeError(" ".join(message_parts))
    return True


def ensure_badge_schema_preflight_strict(supabase: Any) -> bool:
    if _should_skip_preflight():
        return True
    missing_columns = _find_missing_player_badges_columns(supabase, include_revoked=True)
    missing_tables = _find_missing_tables(supabase, REQUIRED_BADGE_TABLES)
    missing_optional_tables = _find_missing_tables(supabase, OPTIONAL_BADGE_TABLES)
    if missing_columns or missing_tables:
        message_parts = [MIGRATION_HINT + "."]
        if missing_columns:
            missing_all = ", ".join(sorted(missing_columns))
            message_parts.append(f"Missing player_badges columns: {missing_all}.")
        if missing_tables:
            missing_tables_list = ", ".join(sorted(missing_tables))
            message_parts.append(f"Missing tables: {missing_tables_list}.")
        raise RuntimeError(" ".join(message_parts))
    if missing_optional_tables:
        logger.warning(
            "Optional badge tables missing: %s. Badge queue processing will be disabled until applied.",
            ", ".join(sorted(missing_optional_tables)),
        )
    return True


def _find_missing_player_badges_columns(supabase: Any, *, include_revoked: bool) -> set[str]:
    missing = set()
    columns = set(REQUIRED_PLAYER_BADGES_COLUMNS)
    if include_revoked:
        columns |= REQUIRED_REVOKED_COLUMNS
    for column in sorted(columns):
        if not _probe_column(supabase, "player_badges", column):
            missing.add(column)
    return missing


def _find_missing_tables(supabase: Any, tables: set[str]) -> set[str]:
    missing = set()
    for table in sorted(tables):
        if not _probe_table(supabase, table):
            missing.add(table)
    return missing


def _probe_column(supabase: Any, table: str, column: str) -> bool:
    try:
        supabase.table(table).select(column).limit(1).execute()
    except APIError as exc:
        code = _get_api_error_code(exc)
        message = _get_api_error_message(exc)
        logger.warning(
            "PostgREST error while checking %s.%s (code=%s message=%s)",
            table,
            column,
            code,
            message,
        )
        if code in _MISSING_COLUMN_CODES:
            return False
        raise RuntimeError(
            f"PostgREST error while checking {table}.{column} (code={code} message={message})."
        ) from exc
    return True


def _probe_table(supabase: Any, table: str) -> bool:
    try:
        supabase.table(table).select("id").limit(1).execute()
    except APIError as exc:
        code = _get_api_error_code(exc)
        message = _get_api_error_message(exc)
        logger.warning(
            "PostgREST error while checking %s table (code=%s message=%s)",
            table,
            code,
            message,
        )
        if code in _MISSING_TABLE_CODES:
            return False
        raise RuntimeError(
            f"PostgREST error while checking table {table} (code={code} message={message})."
        ) from exc
    return True


def _get_api_error_code(exc: APIError) -> str | None:
    code = getattr(exc, "code", None)
    if code:
        return code
    if exc.args and isinstance(exc.args[0], dict):
        return exc.args[0].get("code")
    return None


def _get_api_error_message(exc: APIError) -> str:
    message = getattr(exc, "message", None)
    if message:
        return message
    if exc.args and isinstance(exc.args[0], dict):
        return exc.args[0].get("message", str(exc))
    return str(exc)


def _should_skip_preflight() -> bool:
    return (
        os.getenv(SKIP_PREFLIGHT_ENV, "0") == "1"
        or os.getenv("JUPR_SKIP_DB_PREFLIGHT", "0") == "1"
    )
