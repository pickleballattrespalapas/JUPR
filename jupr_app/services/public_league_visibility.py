from __future__ import annotations

from typing import Any, Mapping


ACTIVE_LEAGUE_VIEW = "active"
PAST_LEAGUE_VIEW = "past"
PUBLIC_LEAGUE_VIEWS = {ACTIVE_LEAGUE_VIEW, PAST_LEAGUE_VIEW}

_ACTIVE_STATUSES = {"active", "running", "live"}
_ENDED_STATUSES = {"ended", "completed", "complete", "done"}


def normalize_public_league_view(value: Any) -> str:
    """Return the supported public league collection, defaulting to active."""

    clean = str(value or ACTIVE_LEAGUE_VIEW).strip().lower()
    return clean if clean in PUBLIC_LEAGUE_VIEWS else ACTIVE_LEAGUE_VIEW


def public_league_view(row: Mapping[str, Any] | None) -> str | None:
    """Classify a manager league for public discovery.

    Public league discovery is deliberately fail-closed. A league must have a
    consistent manager lifecycle state: currently active leagues appear in the
    default collection, while ended leagues appear in the past collection.
    The manager can end only a previously started active/paused league, so the
    ended state is the durable signal that the finished league was published.
    Draft/inactive, paused, archived, inconsistent, and metadata-less records
    are not public league options.
    """

    if not row:
        return None
    name = str(row.get("league_name") or row.get("league") or "").strip()
    if not name or name.upper() in {"OVERALL", "POPUP"}:
        return None

    status = str(row.get("status") or "").strip().lower()
    is_active = row.get("is_active")
    if status in _ACTIVE_STATUSES and is_active is True:
        return ACTIVE_LEAGUE_VIEW
    if status in _ENDED_STATUSES and is_active is False:
        return PAST_LEAGUE_VIEW
    return None


def league_is_public(row: Mapping[str, Any] | None) -> bool:
    return public_league_view(row) is not None
