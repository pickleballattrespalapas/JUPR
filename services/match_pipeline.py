"""Canonical match submission pipeline (infrastructure stub only).

This module provides a single future-facing entry point (`submit_match`) for
consolidating match write logic across league, ladder, tournament, and admin
contexts.

Constraints for this initial infrastructure PR:
- Additive only: no integration with existing flows.
- No behavior changes to current application paths.
- Internal helpers are explicit stubs and perform no database writes.
"""

from typing import Any, Dict, Literal, Optional

_ALLOWED_CONTEXT_TYPES = {"league", "ladder", "tournament", "admin"}


def submit_match(
    club_id: str,
    context_type: Literal["league", "ladder", "tournament", "admin"],
    context_id: Optional[str],
    match_payload: Dict[str, Any],
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate input and route to no-op pipeline stubs.

    This function is intentionally non-invasive and currently performs no writes
    or side effects. It exists as a canonical entry point for future
    consolidation.
    """
    if not club_id or not isinstance(club_id, str):
        raise ValueError("club_id must be a non-empty string")

    if context_type not in _ALLOWED_CONTEXT_TYPES:
        raise ValueError(
            "context_type must be one of: league, ladder, tournament, admin"
        )

    if not isinstance(match_payload, dict):
        raise ValueError("match_payload must be a dict")

    insert_result = _insert_match_stub(
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
        "status": "stub_noop",
        "message": "Canonical submit_match pipeline stub executed; no writes performed.",
        "club_id": club_id,
        "context_type": context_type,
        "context_id": context_id,
        "idempotency_key": idempotency_key,
        "insert": insert_result,
        "rating": rating_result,
        "hooks": hooks_result,
    }


def _insert_match_stub(
    club_id: str,
    context_type: str,
    context_id: Optional[str],
    match_payload: Dict[str, Any],
    idempotency_key: Optional[str],
) -> Dict[str, Any]:
    """Stub: placeholder for future match insert logic (currently no-op)."""
    _ = (club_id, context_type, context_id, match_payload, idempotency_key)
    return {
        "stub": True,
        "name": "insert_match",
        "action": "noop",
        "details": "No database operation performed.",
    }


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
