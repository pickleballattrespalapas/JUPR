"""Personal notification controls; never mutate the club's underlying work."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import re
from typing import Any
from urllib.parse import urlsplit

from jupr_app.data.paged_reads import read_all_rows
from jupr_app.services.admin_notification_sources import (
    collect_admin_notification_sources, resolve_admin_notification_item,
)
from jupr_app.services.staging_write_guard import NON_STAGING_WRITE_ENVIRONMENTS, staging_write_wave_allows

HISTORY_DAYS = 30
ITEM_LIMIT = 200
BULK_CLEAR_LIMIT = 5000
PREFERENCES_TABLE = "admin_notification_preferences"
STATES_TABLE = "admin_notification_states"
STATES = {"new", "flagged", "cleared"}


class NotificationUnavailable(RuntimeError):
    pass


class NotificationConflict(ValueError):
    pass


def notification_key(item: dict[str, Any]) -> str:
    identity = [str(item[field]) for field in ("category", "source_id", "source_version")]
    return hashlib.sha256(json.dumps(identity, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _instant(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Notification time must include a timezone")
    return parsed.astimezone(timezone.utc)


def _safe_item(item: dict[str, Any]) -> dict[str, Any]:
    # The saved snapshot contains only notification copy and an internal link.
    # Never persist a source row, contacts, tokens, or private audit snapshots.
    limits = {"category": 80, "source_id": 512, "source_version": 512, "title": 240, "description": 1200, "href": 1500}
    clean: dict[str, Any] = {}
    for field, limit in limits.items():
        value = item.get(field)
        if not isinstance(value, str) or not value or len(value) > limit:
            raise ValueError("Invalid notification field")
        clean[field] = value
    href = urlsplit(clean["href"])
    if not (href.path == "/admin" or href.path.startswith("/admin/")) or href.scheme or href.netloc or "\\" in clean["href"] or any(ord(c) < 32 for c in clean["href"]):
        raise ValueError("Notification destination must be an admin page")
    if item.get("kind") not in {"action", "activity"}:
        raise ValueError("Invalid notification kind")
    clean["kind"] = item["kind"]
    clean["occurred_at"] = _instant(item["occurred_at"]).isoformat()
    clean["key"] = notification_key(clean)
    return clean


def _personal_rows(db, table: str, *, club_id: str, user_id: str, order: str) -> list[dict]:
    try:
        return read_all_rows(lambda: db.table(table).select("*").eq("club_id", club_id).eq("user_id", user_id), order=order)
    except Exception as exc:
        raise NotificationUnavailable("Your saved notification settings could not be loaded. Try again.") from exc


def _context(db, *, club_id: str, user_id: str, assignments: list[dict], now: datetime) -> dict:
    preferences = _personal_rows(db, PREFERENCES_TABLE, club_id=club_id, user_id=user_id, order="category_key")
    saved = _personal_rows(db, STATES_TABLE, club_id=club_id, user_id=user_id, order="notification_key")
    try:
        sources = collect_admin_notification_sources(db, club_id=club_id, assignments=assignments,
            now=now, history_days=HISTORY_DAYS, item_limit=ITEM_LIMIT)
    except Exception as exc:
        raise NotificationUnavailable("Notifications could not be checked. Try again.") from exc
    states = {row["notification_key"]: row for row in saved}
    enabled = {row["category_key"]: bool(row["enabled"]) for row in preferences}
    statuses = {row["key"]: dict(row) for row in sources["sources"]}
    categories = {row["key"]: {**row, "enabled": enabled.get(row["key"], True),
        "status": statuses.get(row["key"], {}).get("status", "unavailable"),
        "total_count": statuses.get(row["key"], {}).get("total_count")} for row in sources["categories"]}
    items = {}
    for raw in sources["items"]:
        category = raw.get("category")
        if category not in categories:
            continue
        try:
            item = _safe_item(raw)
            if item["kind"] != categories[category]["kind"]:
                raise ValueError("Source kind changed")
            items[item["key"]] = item
        except Exception:
            categories[category].update(status="unavailable", total_count=None)
    return {"categories": categories, "items": items, "states": states, "sources": statuses}


def _saved_snapshot(row: dict) -> dict:
    item = _safe_item(row["snapshot"])
    if item["key"] != row["notification_key"] or item["category"] != row["category_key"] or item["kind"] != row["kind"]:
        raise ValueError("Saved notification identity changed")
    return item


def _resolve(db, item: dict, *, club_id: str, assignments: list[dict], now: datetime) -> dict | None:
    current = resolve_admin_notification_item(db, club_id=club_id, assignments=assignments,
        category=item["category"], source_id=item["source_id"], source_version=item["source_version"], now=now)
    if current is None:
        return None
    current = _safe_item(current)
    return current if current["key"] == item["key"] else None


def _feed(db, context: dict, *, club_id: str, assignments: list[dict], now: datetime) -> dict:
    categories, items, states = context["categories"], dict(context["items"]), context["states"]
    cutoff = now - timedelta(days=HISTORY_DAYS)
    for key, row in states.items():
        category = categories.get(row["category_key"])
        if key in items or category is None or not category["enabled"] or row["state"] == "new":
            continue
        try:
            saved = _saved_snapshot(row)
            if saved["kind"] == "activity":
                if row["state"] == "flagged" or _instant(saved["occurred_at"]) >= cutoff:
                    items[key] = saved
            elif category["status"] == "ready":
                # Saved action flags/clears must disappear when work is resolved
                # even when the saved item is outside the latest source page.
                current = _resolve(db, saved, club_id=club_id, assignments=assignments, now=now)
                if current is not None:
                    items[key] = current
        except Exception:
            category.update(status="unavailable", total_count=None)
    public = []
    for key, item in items.items():
        category = categories[item["category"]]
        if not category["enabled"]:
            continue
        state = states.get(key, {}).get("state", "new")
        if item["kind"] == "activity" and state != "flagged" and _instant(item["occurred_at"]) < cutoff:
            continue
        public.append({field: item[field] for field in ("key", "category", "kind", "title", "description", "href", "occurred_at")} | {"state": state})
    public.sort(key=lambda item: (item["occurred_at"], item["key"]), reverse=True)
    truncated = any(category["enabled"] and context["sources"].get(key, {}).get("truncated", False)
                    for key, category in categories.items())
    return {"club_id": club_id, "checked_at": now.isoformat(), "history_days": HISTORY_DAYS,
            "categories": list(categories.values()), "items": public, "truncated": truncated}


def get_admin_notifications(db, *, club_id: str, user_id: str, assignments: list[dict], now: datetime | None = None) -> dict:
    instant = now or datetime.now(timezone.utc)
    context = _context(db, club_id=club_id, user_id=user_id, assignments=assignments, now=instant)
    return _feed(db, context, club_id=club_id, assignments=assignments, now=instant)


def _require_personal_writes() -> None:
    environment = os.getenv("JUPR_ENV", "").strip().lower()
    if environment in NON_STAGING_WRITE_ENVIRONMENTS:
        return
    if environment == "staging" and staging_write_wave_allows("admin-notifications"):
        return
    raise PermissionError("Saving notification preferences is temporarily paused.")


def update_admin_notification_preferences(db, *, club_id: str, user_id: str, assignments: list[dict], categories: dict[str, bool], now: datetime | None = None) -> dict:
    _require_personal_writes()
    instant = now or datetime.now(timezone.utc)
    context = _context(db, club_id=club_id, user_id=user_id, assignments=assignments, now=instant)
    if any(key not in context["categories"] or type(value) is not bool for key, value in categories.items()):
        raise PermissionError("This notification category is unavailable for your account.")
    if categories:
        rows = [{"club_id": club_id, "user_id": user_id, "category_key": key, "enabled": value, "updated_at": instant.isoformat()} for key, value in categories.items()]
        try:
            result = db.table(PREFERENCES_TABLE).upsert(rows, on_conflict="club_id,user_id,category_key").execute()
            if not isinstance(result.data, list) or len(result.data) != len(rows):
                raise ValueError("Preference save not confirmed")
        except Exception as exc:
            raise NotificationUnavailable("Could not confirm your notification preferences. Reload before retrying.") from exc
        for key, enabled in categories.items():
            context["categories"][key]["enabled"] = enabled
    return _feed(db, context, club_id=club_id, assignments=assignments, now=instant)


def _item_for_state_change(db, context: dict, *, club_id: str, assignments: list[dict], key: str, now: datetime) -> dict:
    item = context["items"].get(key)
    if item is None and key in context["states"]:
        item = _saved_snapshot(context["states"][key])
    if item is None or item["category"] not in context["categories"]:
        raise NotificationConflict("This notification is no longer available. Refresh your notifications.")
    category = context["categories"][item["category"]]
    if not category["enabled"]:
        raise NotificationConflict("Enable this notification category before changing its items.")
    if category["status"] != "ready":
        raise NotificationUnavailable("This notification could not be checked. Try again before changing it.")
    if item["kind"] == "action":
        try:
            current = _resolve(db, item, club_id=club_id, assignments=assignments, now=now)
        except Exception as exc:
            raise NotificationUnavailable("This notification could not be checked. Try again before changing it.") from exc
        if current is None:
            raise NotificationConflict("This task has changed or is already resolved. Refresh your notifications.")
        item = current
    elif _instant(item["occurred_at"]) < now - timedelta(days=HISTORY_DAYS) and context["states"].get(key, {}).get("state") != "flagged":
        raise NotificationConflict("This notification is outside your recent activity history.")
    return item


def _state_row(item: dict, *, club_id: str, user_id: str, state: str, now: datetime) -> dict:
    return {"club_id": club_id, "user_id": user_id, "notification_key": item["key"], "category_key": item["category"],
            "kind": item["kind"], "state": state, "snapshot": item, "occurred_at": item["occurred_at"], "updated_at": now.isoformat()}


def update_admin_notification_state(db, *, club_id: str, user_id: str, assignments: list[dict], key: str, state: str, now: datetime | None = None) -> dict:
    _require_personal_writes()
    if state not in STATES:
        raise ValueError("Choose a supported notification state.")
    instant = now or datetime.now(timezone.utc)
    context = _context(db, club_id=club_id, user_id=user_id, assignments=assignments, now=instant)
    item = _item_for_state_change(db, context, club_id=club_id, assignments=assignments, key=key, now=instant)
    row = _state_row(item, club_id=club_id, user_id=user_id, state=state, now=instant)
    try:
        result = db.table(STATES_TABLE).upsert(row, on_conflict="club_id,user_id,notification_key").execute()
        if not isinstance(result.data, list) or len(result.data) != 1 or result.data[0].get("notification_key") != key:
            raise ValueError("Notification save not confirmed")
    except Exception as exc:
        raise NotificationUnavailable("Could not confirm this notification change. Reload before retrying.") from exc
    context["states"][key] = row
    context["items"][key] = item
    return _feed(db, context, club_id=club_id, assignments=assignments, now=instant)


def clear_admin_notifications(db, *, club_id: str, user_id: str, assignments: list[dict], keys: list[str], now: datetime | None = None) -> dict:
    """Validate every selected notice before one atomic personal-state upsert.

    The client supplies identities only. Source records, permissions, snapshots,
    and the user/club scope are all resolved on the server. A stale or unavailable
    selected item rejects the entire selection without clearing the other items.
    """
    _require_personal_writes()
    if not isinstance(keys, list) or not 1 <= len(keys) <= BULK_CLEAR_LIMIT or any(
        not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{64}", key) for key in keys
    ):
        raise ValueError("Choose a supported number of notification keys.")
    keys = list(dict.fromkeys(keys))
    instant = now or datetime.now(timezone.utc)
    context = _context(db, club_id=club_id, user_id=user_id, assignments=assignments, now=instant)
    items = [_item_for_state_change(db, context, club_id=club_id, assignments=assignments, key=key, now=instant)
             for key in keys]
    rows = [_state_row(item, club_id=club_id, user_id=user_id, state="cleared", now=instant) for item in items]
    try:
        result = db.table(STATES_TABLE).upsert(rows, on_conflict="club_id,user_id,notification_key").execute()
        if not isinstance(result.data, list) or len(result.data) != len(keys) or {
            row.get("notification_key") for row in result.data
        } != set(keys):
            raise ValueError("Bulk notification save not confirmed")
    except Exception as exc:
        raise NotificationUnavailable("Could not confirm the selected notification changes. Reload before retrying.") from exc
    for item, row in zip(items, rows):
        context["states"][item["key"]] = row
        context["items"][item["key"]] = item
    return _feed(db, context, club_id=club_id, assignments=assignments, now=instant)
