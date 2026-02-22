from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

MEXICO_CITY_TZ = ZoneInfo("America/Mexico_City")
_TOURNAMENT_KEYWORDS = (
    "tournament",
    "championship",
    "classic",
    "open",
    "cup",
    "invitational",
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    error: str | None = None


def _coerce_dt(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=MEXICO_CITY_TZ)
    return dt.astimezone(MEXICO_CITY_TZ)


def _normalize_item(row: dict[str, Any], *, source: str) -> dict[str, Any]:
    title = str(row.get("title") or row.get("name") or "").strip()
    starts_at = _coerce_dt(row.get("starts_at") or row.get("datetime") or row.get("created_at"))
    item = {
        "event_id": row.get("id"),
        "title": title,
        "datetime": starts_at.isoformat() if starts_at else None,
        "location": row.get("location") or row.get("venue") or "",
        "reg_url": row.get("reg_url") or row.get("registration_url") or "",
        "results_link": row.get("results_link") or "",
        "winners": row.get("winners"),
        "event_type": row.get("event_type") or row.get("type"),
        "category": row.get("category"),
        "source": source,
        "raw": row,
    }
    item["is_tournament"] = is_tournament_item(item)
    return item


def is_tournament_item(item: dict[str, Any]) -> bool:
    source = str(item.get("source") or "").lower()
    if source == "tournaments":
        return True

    event_type = item.get("event_type")
    category = item.get("category")
    if event_type is not None or category is not None:
        type_blob = f"{event_type or ''} {category or ''}".strip().lower()
        return "tournament" in type_blob

    title = str(item.get("title") or "").lower()
    return any(keyword in title for keyword in _TOURNAMENT_KEYWORDS)


def _load_events_rows(club_id: str, *, supabase, events_rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if events_rows is not None:
        return [row for row in events_rows if str(row.get("club_id")) == str(club_id)]
    if supabase is None:
        return []
    response = (
        supabase.table("events")
        .select("id,club_id,name,event_type,starts_at,created_at,notes")
        .eq("club_id", club_id)
        .execute()
    )
    return response.data or []


def _load_tournaments_rows(club_id: str, *, supabase, tournaments_rows: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if tournaments_rows is not None:
        return [row for row in tournaments_rows if str(row.get("club_id")) == str(club_id)]
    if supabase is None:
        return []
    response = (
        supabase.table("tournaments")
        .select("id,club_id,name,status,created_at")
        .eq("club_id", club_id)
        .execute()
    )
    return response.data or []


def load_events_in_period(
    club_id: str,
    report_start: datetime,
    report_end: datetime,
    *,
    supabase=None,
    events_rows: list[dict[str, Any]] | None = None,
    tournaments_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    start_local = _coerce_dt(report_start)
    end_local = _coerce_dt(report_end)
    if start_local is None or end_local is None:
        return []

    items: list[dict[str, Any]] = []
    for row in _load_events_rows(club_id, supabase=supabase, events_rows=events_rows):
        item = _normalize_item(row, source="events")
        dt = _coerce_dt(item.get("datetime"))
        if dt is not None and start_local <= dt < end_local:
            items.append(item)
    for row in _load_tournaments_rows(club_id, supabase=supabase, tournaments_rows=tournaments_rows):
        item = _normalize_item(row, source="tournaments")
        dt = _coerce_dt(item.get("datetime"))
        if dt is not None and start_local <= dt < end_local:
            items.append(item)
    return sorted(items, key=lambda x: x.get("datetime") or "")


def load_events_upcoming(
    club_id: str,
    now: datetime,
    lookahead_days: int,
    *,
    supabase=None,
    events_rows: list[dict[str, Any]] | None = None,
    tournaments_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    now_local = _coerce_dt(now)
    if now_local is None:
        return []
    return load_events_in_period(
        club_id,
        now_local,
        now_local + timedelta(days=max(int(lookahead_days), 0)),
        supabase=supabase,
        events_rows=events_rows,
        tournaments_rows=tournaments_rows,
    )


def validate_featured_past_event(
    featured_past_event: dict[str, Any] | None,
    report_start: datetime,
    report_end: datetime,
) -> ValidationResult:
    if not featured_past_event:
        return ValidationResult(ok=True)
    featured_dt = _coerce_dt(featured_past_event.get("datetime"))
    if featured_dt is None:
        return ValidationResult(ok=True)
    start_local = _coerce_dt(report_start)
    end_local = _coerce_dt(report_end)
    if start_local is None or end_local is None:
        return ValidationResult(ok=False, error="Invalid report window")
    if start_local <= featured_dt < end_local:
        return ValidationResult(ok=True)
    return ValidationResult(ok=False, error="featured_past_event must be inside [report_start, report_end)")


def compute_recap(
    club_id: str,
    report_start: datetime,
    report_end: datetime,
    *,
    lookahead_days: int,
    now: datetime | None = None,
    featured_upcoming_event: dict[str, Any] | None = None,
    featured_past_event: dict[str, Any] | None = None,
    supabase=None,
    events_rows: list[dict[str, Any]] | None = None,
    tournaments_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    now_value = now or datetime.now(tz=MEXICO_CITY_TZ)
    in_period = load_events_in_period(
        club_id,
        report_start,
        report_end,
        supabase=supabase,
        events_rows=events_rows,
        tournaments_rows=tournaments_rows,
    )
    upcoming = load_events_upcoming(
        club_id,
        now_value,
        lookahead_days,
        supabase=supabase,
        events_rows=events_rows,
        tournaments_rows=tournaments_rows,
    )

    events_in_period = [item for item in in_period if not item.get("is_tournament")]
    tournaments_in_period = [item for item in in_period if item.get("is_tournament")]
    upcoming_events = [item for item in upcoming if not item.get("is_tournament")]
    upcoming_tournaments = [item for item in upcoming if item.get("is_tournament")]

    featured_past = dict(featured_past_event or {})
    featured_past_event_id = featured_past.get("event_id")
    if featured_past_event_id:
        matched = next((item for item in in_period if str(item.get("event_id")) == str(featured_past_event_id)), None)
        if matched:
            featured_past = {
                **matched,
                **featured_past,
            }

    return {
        "club_id": str(club_id),
        "report_start": _coerce_dt(report_start).isoformat() if _coerce_dt(report_start) else None,
        "report_end": _coerce_dt(report_end).isoformat() if _coerce_dt(report_end) else None,
        "events_in_period": events_in_period,
        "tournaments_in_period": tournaments_in_period,
        "upcoming_events": upcoming_events,
        "upcoming_tournaments": upcoming_tournaments,
        "featured_upcoming_event": featured_upcoming_event,
        "featured_past_event": featured_past or None,
        "featured_past_event_validation": validate_featured_past_event(featured_past or None, report_start, report_end).__dict__,
    }
