"""Canonical match submission pipeline (infrastructure stub only).

This module provides a single future-facing entry point (`submit_match`) for
consolidating match write logic across league, ladder, tournament, and admin
contexts.

Constraints for this initial infrastructure PR:
- Additive only: no integration with existing flows.
- No behavior changes to current application paths beyond canonical match insert.
- Non-insert helpers remain explicit stubs.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Literal, Optional

from jupr_app.data.client import make_supabase

_ALLOWED_CONTEXT_TYPES = {"league", "ladder", "tournament", "admin"}


def get_supabase_client() -> Any:
    """Build a Supabase client from environment credentials."""
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv(
        "SUPABASE_KEY", ""
    )

    if not supabase_url or not supabase_key:
        raise ValueError(
            "Supabase credentials are missing. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY)."
        )

    return make_supabase(supabase_url, supabase_key)


def submit_match(
    club_id: str,
    context_type: Literal["league", "ladder", "tournament", "admin"],
    context_id: Optional[str],
    match_payload: Dict[str, Any],
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate input and route to canonical pipeline steps.

    The canonical insert step writes to Supabase and enforces idempotency when
    an idempotency key is provided. Other downstream stages remain no-op stubs.
    """
    if not club_id or not isinstance(club_id, str):
        raise ValueError("club_id must be a non-empty string")

    if context_type not in _ALLOWED_CONTEXT_TYPES:
        raise ValueError(
            "context_type must be one of: league, ladder, tournament, admin"
        )

    if not isinstance(match_payload, dict):
        raise ValueError("match_payload must be a dict")

    insert_result = _insert_match_record(
        club_id=club_id,
        context_type=context_type,
        context_id=context_id,
        match_payload=match_payload,
        idempotency_key=idempotency_key,
    )
    rating_result = _apply_rating_engine_stub(match_payload=match_payload)
    hooks_result = _run_context_hooks_stub(
        context_type=context_type,
        context_id=context_id,
        match_payload=match_payload,
    )

    return {
        "ok": True,
        "status": "inserted_with_stubbed_post_steps",
        "message": "Canonical submit_match pipeline executed. Insert is live; downstream steps are currently stubs.",
        "club_id": club_id,
        "context_type": context_type,
        "context_id": context_id,
        "idempotency_key": idempotency_key,
        "insert": insert_result,
        "rating": rating_result,
        "hooks": hooks_result,
    }


def _insert_match_record(
    club_id: str,
    context_type: str,
    context_id: Optional[str],
    match_payload: Dict[str, Any],
    idempotency_key: Optional[str],
) -> Dict[str, Any]:
    """Insert a match row, returning the existing row when idempotency matches.

    This function checks for an existing `(club_id, idempotency_key)` match
    before insert. If one exists, it is returned directly and no insert is
    performed.
    """
    supabase = get_supabase_client()

    if idempotency_key:
        existing_response = (
            supabase.table("matches")
            .select("*")
            .eq("club_id", club_id)
            .eq("idempotency_key", idempotency_key)
            .limit(1)
            .execute()
        )
        existing_rows = getattr(existing_response, "data", None) or []
        if existing_rows:
            return dict(existing_rows[0])

    payload: Dict[str, Any] = dict(match_payload)
    payload["club_id"] = club_id
    payload["context_type"] = context_type
    payload["context_id"] = context_id
    if idempotency_key:
        payload["idempotency_key"] = idempotency_key

    insert_response = supabase.table("matches").insert(payload).execute()
    inserted_rows = getattr(insert_response, "data", None) or []
    if not inserted_rows:
        raise RuntimeError("Supabase insert returned no row data for matches insert")

    return dict(inserted_rows[0])


def _apply_rating_engine_stub(match_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stub: placeholder for future rating engine application (currently no-op)."""
    _ = match_payload
    return {
        "stub": True,
        "name": "apply_rating_engine",
        "action": "noop",
        "details": "No rating changes applied.",
    }


def _run_context_hooks_stub(
    context_type: str,
    context_id: Optional[str],
    match_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Stub: placeholder for context-specific hooks (currently no-op)."""
    _ = (context_id, match_payload)
    return {
        "stub": True,
        "name": "run_context_hooks",
        "context_type": context_type,
        "action": "noop",
        "details": "No context hook executed.",
    }
