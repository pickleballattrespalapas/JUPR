from __future__ import annotations

from typing import Any


def forbid_raw_match_insert(stack: Any) -> None:
    raise RuntimeError(
        "Direct matches insert detected. Use record_match() instead."
    )


def install_match_insert_guard(supabase: Any) -> Any:
    """Monkey-patch table('matches').insert in development mode."""
    original_table = supabase.table

    def guarded_table(table_name: str, *args: Any, **kwargs: Any) -> Any:
        query = original_table(table_name, *args, **kwargs)
        if str(table_name) == "matches" and hasattr(query, "insert"):
            query.insert = forbid_raw_match_insert
        return query

    supabase.table = guarded_table
    return supabase
