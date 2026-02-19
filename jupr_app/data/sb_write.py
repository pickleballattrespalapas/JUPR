from __future__ import annotations

# Match writes must go through match_pipeline.

# SCHEMA STRICT MODE ENABLED
# All environments must match migrations

from typing import Any, Dict

from jupr_app.config import PRODUCTION_MODE


RATING_STATE_DERIVATION_NOTE = "Rating state is derived from match history only."


def _enforce_rating_state_policy(*, table: str, payload: Dict, derived_from_match_history: bool) -> None:
    """Raise on manual rating-state writes in production.

    Rating state is derived from match history only.
    """
    if not PRODUCTION_MODE or derived_from_match_history:
        return
    if str(table) not in {"players", "league_ratings"}:
        return
    forbidden_fields = {"rating", "starting_rating", "wins", "losses", "matches_played"}
    if any(field in payload for field in forbidden_fields):
        kind = "league rating" if str(table) == "league_ratings" else "rating"
        raise RuntimeError(
            f"Manual {kind} edits are disabled in PRODUCTION_MODE. {RATING_STATE_DERIVATION_NOTE}"
        )


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
    derived_from_match_history: bool = False,
) -> Any:
    _enforce_rating_state_policy(
        table=table,
        payload=payload,
        derived_from_match_history=derived_from_match_history,
    )
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
    derived_from_match_history: bool = False,
) -> Any:
    _enforce_rating_state_policy(
        table=table,
        payload=payload,
        derived_from_match_history=derived_from_match_history,
    )
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
