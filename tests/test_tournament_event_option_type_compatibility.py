from __future__ import annotations

import re
from pathlib import Path


BASE_MIGRATION = Path(
    "supabase/migrations/"
    "20261108025000_tournament_playoff_round_scoring.sql"
)
FIX_MIGRATION = Path(
    "supabase/migrations/"
    "20261108027000_fix_tournament_day_score_id_types.sql"
)

FUNCTIONS = (
    "apply_tournament_day_result_metadata",
    "cascade_tournament_team_retirement",
)
UNCAST_EVENT_PREDICATE = re.compile(
    r"event\.tournament_id\s*=\s*new\.tournament_id(?!::text)"
)
CAST_EVENT_PREDICATE = re.compile(
    r"event\.tournament_id\s*=\s*new\.tournament_id::text"
)


def _sql(path: Path) -> str:
    return path.read_text(encoding="utf-8").lower()


def _function_definition(sql: str, name: str) -> str:
    start = sql.index(f"create or replace function public.{name}")
    terminator = "$function$;"
    end = sql.index(terminator, start) + len(terminator)
    return sql[start:end]


def test_redefined_functions_only_cast_the_legacy_event_tournament_id() -> None:
    base_sql = _sql(BASE_MIGRATION)
    fix_sql = _sql(FIX_MIGRATION)
    old_predicate = "event.tournament_id = new.tournament_id"
    fixed_predicate = f"{old_predicate}::text"

    for function_name in FUNCTIONS:
        base_function = _function_definition(base_sql, function_name)
        fixed_function = _function_definition(fix_sql, function_name)

        assert base_function.count(old_predicate) == 1
        assert fixed_function == base_function.replace(
            old_predicate,
            fixed_predicate,
            1,
        )

    assert len(CAST_EVENT_PREDICATE.findall(fix_sql)) == len(FUNCTIONS)
    assert UNCAST_EVENT_PREDICATE.search(fix_sql) is None


def test_redefined_functions_preserve_private_invoker_contract() -> None:
    fix_sql = _sql(FIX_MIGRATION)
    normalized_sql = " ".join(fix_sql.split())

    assert "security definer" not in fix_sql
    assert fix_sql.count("create or replace function public.") == len(FUNCTIONS)

    for function_name in FUNCTIONS:
        function = _function_definition(fix_sql, function_name)

        assert function.count("security invoker") == 1
        assert function.count("set search_path = ''") == 1
        assert (
            f"revoke all on function public.{function_name}() "
            "from public, anon, authenticated, service_role;"
        ) in normalized_sql
        assert (
            f"grant execute on function public.{function_name}() to service_role;"
            in normalized_sql
        )


def test_type_compatibility_fix_does_not_rewire_schema_objects() -> None:
    fix_sql = _sql(FIX_MIGRATION)

    assert not re.search(
        r"\b(?:create|alter|drop)\s+(?:constraint\s+)?trigger\b",
        fix_sql,
    )
    assert not re.search(
        r"\b(?:create(?:\s+(?:temporary|temp|unlogged))?|alter|drop)\s+table\b",
        fix_sql,
    )
