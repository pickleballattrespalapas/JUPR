from __future__ import annotations

import json
from typing import Any, Mapping


CURRENT_TOURNAMENT_VIEW = "current"
PAST_TOURNAMENT_VIEW = "past"
PUBLIC_TOURNAMENT_VIEWS = {CURRENT_TOURNAMENT_VIEW, PAST_TOURNAMENT_VIEW}

_CURRENT_STATUSES = {"ACTIVE"}
_PAST_STATUSES = {"COMPLETED"}


def normalize_public_tournament_view(value: Any) -> str:
    """Return a supported public collection, defaulting to current."""

    clean = str(value or CURRENT_TOURNAMENT_VIEW).strip().lower()
    return clean if clean in PUBLIC_TOURNAMENT_VIEWS else CURRENT_TOURNAMENT_VIEW


def tournament_setup_is_published(settings: Mapping[str, Any] | None) -> bool:
    """Require durable setup-publication evidence before public discovery.

    Registration status controls whether patrons may submit an entry. It is not
    a publication signal and must never make a draft tournament public.
    """

    if not settings:
        return False
    snapshot: Any = settings.get("builder_draft_json")
    if isinstance(snapshot, str) and snapshot.strip():
        try:
            snapshot = json.loads(snapshot)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
    return bool(
        isinstance(snapshot, Mapping)
        and str(snapshot.get("published_at") or "").strip()
    )


def public_tournament_view(
    tournament: Mapping[str, Any] | None,
    settings: Mapping[str, Any] | None,
    *,
    completion_receipt: bool = False,
) -> str | None:
    """Classify a tournament for fail-closed public discovery.

    Current tournaments must be ACTIVE and have a published setup snapshot.
    Past tournaments must be COMPLETED and have the atomic completion receipt
    produced only after official publication. DRAFT, PAUSED, INACTIVE, and
    ARCHIVED tournaments are never public, even through a direct id or slug.
    """

    if not tournament_setup_is_published(settings):
        return None
    status = str((tournament or {}).get("status") or "").strip().upper()
    if status in _CURRENT_STATUSES:
        return CURRENT_TOURNAMENT_VIEW
    if status in _PAST_STATUSES and completion_receipt:
        return PAST_TOURNAMENT_VIEW
    return None


def tournament_is_public(
    tournament: Mapping[str, Any] | None,
    settings: Mapping[str, Any] | None,
    *,
    completion_receipt: bool = False,
) -> bool:
    return (
        public_tournament_view(
            tournament,
            settings,
            completion_receipt=completion_receipt,
        )
        is not None
    )
