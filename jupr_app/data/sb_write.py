from __future__ import annotations

# Match writes must go through match_pipeline.

# SCHEMA STRICT MODE ENABLED
# All environments must match migrations

from typing import Any, Dict


def sb_insert(supabase: Any, table: str, payload: Dict) -> Any:
    return (
        supabase
        .table(table)
        .insert(payload)
        .execute()
    )


def sb_upsert(
    supabase: Any,
    table: str,
    payload: Dict,
    *,
    conflict: str,
) -> Any:
    return (
        supabase
        .table(table)
        .upsert(
            payload,
            on_conflict=conflict,
        )
        .execute()
    )


def sb_update(
    supabase: Any,
    table: str,
    payload: Dict,
    *,
    filters: Dict,
) -> Any:
    query = supabase.table(table).update(payload)

    for col, value in filters.items():
        query = query.eq(col, value)

    return query.execute()


def sb_delete(
    supabase: Any,
    table: str,
    *,
    filters: Dict,
) -> Any:
    query = supabase.table(table).delete()

    for col, value in filters.items():
        query = query.eq(col, value)

    return query.execute()


def sb_rpc(supabase: Any, name: str, payload: Dict) -> Any:
    return supabase.rpc(name, payload).execute()
