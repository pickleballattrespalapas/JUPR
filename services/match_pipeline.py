"""Compatibility wrapper around the canonical domain match pipeline."""

from __future__ import annotations

import os
import uuid
from typing import Any, Literal, Mapping

from jupr_app.data.client import make_supabase
from jupr_app.domain.match_pipeline import ingest_match_with_identity_resolution

_ALLOWED_CONTEXT_TYPES = {"league", "ladder", "tournament", "round_robin", "moneyball", "admin"}


def _coerce_uuid(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = str(value).strip()
    if not normalized:
        return None

    try:
        return str(uuid.UUID(normalized))
    except (ValueError, TypeError, AttributeError):
        return None


def get_supabase_client() -> Any:
    """Build a Supabase client from environment credentials."""
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv("SUPABASE_KEY", "")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "Supabase credentials are missing. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY)."
        )

    return make_supabase(supabase_url, supabase_key)


def submit_match(
    club_id: str,
    context_type: Literal["league", "ladder", "tournament", "round_robin", "moneyball", "admin"],
    context_id: str | None,
    match_payload: Mapping[str, Any],
    idempotency_key: str | None = None,
    run_context_hooks: bool = True,
) -> dict[str, Any]:
    """Delegate legacy service entrypoint to `jupr_app.domain.match_pipeline.record_match`."""
    if context_type not in _ALLOWED_CONTEXT_TYPES:
        raise ValueError(
            "context_type must be one of: league, ladder, tournament, round_robin, moneyball, admin"
        )

    payload = dict(match_payload or {})
    payload["context_type"] = context_type
    payload["context_id"] = _coerce_uuid(context_id)
    if idempotency_key is not None:
        payload["idempotency_key"] = idempotency_key

    supabase = get_supabase_client()
    result = ingest_match_with_identity_resolution(supabase=supabase, club_id=club_id, match_payload=payload)
    result["run_context_hooks"] = bool(run_context_hooks)
    return result


__all__ = ["submit_match", "get_supabase_client"]
