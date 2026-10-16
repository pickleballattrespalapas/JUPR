"""Recurring event identities and clean configuration for a new season."""
from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from uuid import NAMESPACE_URL, uuid5


KINDS = {"interclub", "league", "tournament"}


def next_event_id(club_id: str, request_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"pcs-event-season:{club_id}:{request_id}"))


def new_season_template(kind: str, source: dict, *, name: str, start_date: str, end_date: str, new_id: str) -> dict:
    """Copy settings only. Never copy entrants, fixtures, results or publication."""
    if kind not in KINDS:
        raise ValueError("Choose a supported event type.")
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    if end < start:
        raise ValueError("The end date must be on or after the start date.")
    if kind == "interclub":
        old = source.get("setup") or {}
        return {"name": name, "start_date": start_date, "end_date": end_date,
                "timezone": old.get("timezone") or "America/Mazatlan",
                "divisions": deepcopy(old.get("divisions") or []), "club_ids": deepcopy(old.get("club_ids") or []),
                "registration_rules": deepcopy((source.get("event") or {}).get("rules") or old.get("registration_rules") or {}),
                "meets": [], "setup_step": 0}
    if kind == "league":
        old = source["event"]
        schedule = deepcopy(old.get("schedule_config") or {})
        for key in ("dates", "sessions", "weeks", "fixtures", "blackout_dates", "skip_dates"):
            if isinstance(schedule.get(key), (dict, list)):
                schedule.pop(key, None)
        schedule.update(start_date=start_date, end_date=end_date)
        return {"schedule_config": schedule}

    # Rebuild the builder draft from published days/divisions, with new IDs.
    # Sponsor assets belong to the old tournament and are reviewed afresh.
    settings = source.get("settings") or {}
    old_draft = settings.get("builder_draft_json") or {}
    days = deepcopy(source.get("days") or [])
    divisions = deepcopy(source.get("divisions") or [])
    families = deepcopy(old_draft.get("published_event_families") or [])
    id_map = {}
    for item in [*days, *divisions, *families]:
        if item.get("id"):
            id_map[str(item["id"])] = str(uuid5(NAMESPACE_URL, f"{new_id}:{item['id']}"))
    old_dates = [date.fromisoformat(str(day["event_date"])[:10]) for day in days if day.get("event_date")]
    first_day = min(old_dates) if old_dates else start
    for day in days:
        previous = date.fromisoformat(str(day["event_date"])[:10]) if day.get("event_date") else first_day
        next_day = start + timedelta(days=(previous - first_day).days)
        if next_day > end:
            raise ValueError("The new date range needs enough days for the copied tournament schedule.")
        day["event_date"] = day["date"] = next_day.isoformat()
    def remap(value):
        if isinstance(value, list):
            return [remap(item) for item in value]
        if isinstance(value, dict):
            return {key: (new_id if key == "tournament_id" else remap(item)) for key, item in value.items()
                    if key not in {"created_at", "updated_at", "published_at", "published_event_families"}}
        return id_map.get(value, value) if isinstance(value, str) else value
    safe_settings = {key: deepcopy(settings[key]) for key in (
        "locale", "waitlist_enabled", "partner_board_enabled", "rules_markdown", "refund_policy_markdown",
        "weather_policy_markdown", "location_name", "venue_address", "venue_directions", "venue_courts_json", "timezone",
    ) if key in settings}
    basics = {**safe_settings, "name": name, "start_date": start_date, "end_date": end_date, "sponsors_json": []}
    return {"version": 3, "saved_step": "basics", "published_at": None, "published_event_families": [],
            "days": remap(days), "event_families": remap(families), "divisions": remap(divisions),
            "basics": basics, "settings": {**safe_settings, "registration_status": "draft"}}


def retained_champion_seasons(editions: list[dict], completed_ids: set[str]) -> set[str]:
    """Only the latest completed edition in each series remains defending champion."""
    latest = {}
    for edition in editions:
        if edition["source_id"] in completed_ids:
            previous = latest.get(edition["series_id"])
            if previous is None or edition["position"] > previous["position"]:
                latest[edition["series_id"]] = edition
    return {edition["source_id"] for edition in latest.values()}
