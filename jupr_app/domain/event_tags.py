from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

SKILL_LEVEL_OPTIONS = ["2.5", "3.0", "3.5", "4.0", "4.5", "5.0", "All"]


def _as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except Exception:
        return None


def _season_tag(value: date) -> str:
    month = int(value.month)
    if month in {12, 1, 2}:
        season = "Winter"
    elif month in {3, 4, 5}:
        season = "Spring"
    elif month in {6, 7, 8}:
        season = "Summer"
    else:
        season = "Fall"
    return f"{season} {value.year}"


def normalize_skill_levels(values: object, *, default_all: bool = False) -> list[str]:
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        raw_values = []

    normalized: list[str] = []
    for value in raw_values:
        text = str(value or "").strip()
        if not text or text not in SKILL_LEVEL_OPTIONS:
            continue
        if text not in normalized:
            normalized.append(text)

    if "All" in normalized:
        return ["All"]
    if normalized:
        return normalized
    return ["All"] if default_all else []


def normalize_date_tags(values: object) -> list[str]:
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        raw_values = []

    normalized: list[str] = []
    for value in raw_values:
        text = str(value or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def derive_default_date_tags(event_date: object = None, start_date: object = None, end_date: object = None) -> list[str]:
    event_day = _as_date(event_date)
    start_day = _as_date(start_date)
    end_day = _as_date(end_date)

    if event_day:
        monday = event_day - timedelta(days=event_day.weekday())
        return normalize_date_tags(
            [
                f"Week of {monday.isoformat()}",
                event_day.strftime("%B %Y"),
            ]
        )

    if not start_day and end_day:
        start_day = end_day
    if not end_day and start_day:
        end_day = start_day
    if not start_day or not end_day:
        return []

    if end_day < start_day:
        start_day, end_day = end_day, start_day

    month_tags: list[str] = []
    cursor = date(start_day.year, start_day.month, 1)
    end_cursor = date(end_day.year, end_day.month, 1)
    while cursor <= end_cursor:
        month_tags.append(cursor.strftime("%B %Y"))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)

    season_tags = []
    start_season = _season_tag(start_day)
    end_season = _season_tag(end_day)
    season_tags.append(start_season)
    if end_season != start_season:
        season_tags.append(end_season)

    return normalize_date_tags(month_tags + season_tags)


def normalize_event_tags(tags: object, *, default_skill_all: bool = False) -> dict[str, list[str]]:
    payload = tags if isinstance(tags, dict) else {}
    return {
        "skill_levels": normalize_skill_levels(payload.get("skill_levels"), default_all=default_skill_all),
        "date_tags": normalize_date_tags(payload.get("date_tags")),
    }


def merge_event_tags(existing: object, updates: object, *, default_skill_all: bool = False) -> dict[str, list[str]]:
    normalized_existing = normalize_event_tags(existing, default_skill_all=default_skill_all)
    update_payload = updates if isinstance(updates, dict) else {}

    if "skill_levels" in update_payload:
        normalized_existing["skill_levels"] = normalize_skill_levels(
            update_payload.get("skill_levels"),
            default_all=default_skill_all,
        )
    if "date_tags" in update_payload:
        normalized_existing["date_tags"] = normalize_date_tags(update_payload.get("date_tags"))
    return normalized_existing


def get_event_tags(payload: dict[str, Any], *, default_skill_all: bool = False) -> dict[str, list[str]]:
    data = payload if isinstance(payload, dict) else {}
    return normalize_event_tags(data.get("event_tags"), default_skill_all=default_skill_all)


def get_event_skill_levels(payload: dict[str, Any], default_all: bool = False) -> list[str]:
    return get_event_tags(payload, default_skill_all=default_all).get("skill_levels", [])


def get_event_date_tags(payload: dict[str, Any]) -> list[str]:
    return get_event_tags(payload).get("date_tags", [])
