"""Registration divisions represented by one playing draw.

The primary event still owns games and teams. Source divisions retain their
registration records and are covered by the same schedule and podium.
"""
from __future__ import annotations

from typing import Any, Iterable


INACTIVE_STATUSES = {"cancelled", "canceled", "inactive", "archived", "disabled", "deleted", "void", "voided"}


def draw_source_event_option_ids(draw: dict[str, Any]) -> list[str]:
    primary = str(draw.get("event_option_id") or "").strip()
    values = draw.get("source_event_option_ids") or []
    if not isinstance(values, list):
        values = []
    return list(dict.fromkeys(value for value in [primary, *(str(value or "").strip() for value in values)] if value))


def draw_source_family_key(event: dict[str, Any]) -> tuple[str, str, str, bool, str]:
    family = " ".join(str(event.get("event_family_key") or event.get("event_family_label") or "").lower().split())
    return (
        family,
        str(event.get("event_type") or "").strip().upper(),
        str(event.get("gender_restriction") or "ANY").strip().upper(),
        str(event.get("partner_required") or "").strip().lower() in {"true", "1", "yes", "on"},
        str(event.get("competition_format") or "STANDARD").strip().upper(),
    )


def validate_draw_source_events(draw: dict[str, Any], events: Iterable[dict[str, Any]] | dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate a combined draw against all of its current registration events."""
    event_map = events if isinstance(events, dict) else {str(row.get("id") or ""): row for row in events}
    source_ids = draw_source_event_option_ids(draw)
    day_id = str(draw.get("registration_day_id") or "").strip()
    tournament_id = str(draw.get("tournament_id") or "").strip()
    if not source_ids or not day_id:
        raise ValueError("Choose a registration division and tournament day before combining divisions.")
    rows = []
    for source_id in source_ids:
        event = event_map.get(source_id)
        if not event or (tournament_id and str(event.get("tournament_id") or "") != tournament_id):
            raise ValueError("Every source division must belong to this tournament.")
        if event.get("enabled") is not True or str(event.get("status") or "").lower() in INACTIVE_STATUSES:
            raise ValueError("Every source division must be enabled and active.")
        if str(event.get("competition_format") or "STANDARD").upper() != "STANDARD":
            raise ValueError("Only standard singles or doubles divisions can be combined into a draw.")
        if any(str(event.get(field) or "").strip().upper() == "COMBINED_RATING_CAP" for field in ("eligibility_mode", "skill_mode")):
            raise ValueError("Combined-rating-cap events cannot be combined with other registration divisions.")
        if str(event.get("registration_day_id") or "") != day_id:
            raise ValueError("Combined divisions must belong to the same tournament day.")
        scheduled = event.get("scheduled_day_ids")
        if scheduled not in (None, []) and (not isinstance(scheduled, list) or day_id not in scheduled):
            raise ValueError("Every source division must be scheduled for this draw's day.")
        rows.append(event)
    if len({draw_source_family_key(row) for row in rows}) != 1:
        raise ValueError("Choose divisions from the same event family, gender, and singles or doubles format.")
    return rows
