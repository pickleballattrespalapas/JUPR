from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime
from typing import Any

BADGE_SELECT = (
    "badge_id,name,category,prestige,requirements,description_md,state,is_active,rarity,tier,icon_key,scope,created_at"
)
PLAYER_BADGE_SELECT = "club_id,player_id,badge_id,earned_at,created_at,context_type,context_id"
PLAYER_SELECT = "id,club_id,name,active,inactive_at"
SECTION_ORDER = ["Common", "Uncommon", "Rare", "Legendary", "Unclaimed", "Unranked", "Other"]

_HTML_TAG_RE = re.compile(r"<[^>]*>")


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _plain_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = _HTML_TAG_RE.sub("", text).replace("<", "").replace(">", "").strip()
    return re.sub(r"\s+", " ", text) or None


def _requirements_text(row: dict[str, Any]) -> str:
    text = _plain_text(row.get("requirements"))
    if text:
        return re.sub(r"^(requirements?)\s*:\s*", "", text, flags=re.IGNORECASE).strip() or "Requirements TBD"
    return "Requirements TBD"


def _badge_state(row: dict[str, Any]) -> str:
    state = str(row.get("state") or "").strip().lower()
    if state:
        return state
    if row.get("is_active") is False:
        return "deprecated"
    return "live"


def _badge_should_be_public(row: dict[str, Any], earners_count: int | None) -> bool:
    badge_id = str(row.get("badge_id") or "").strip()
    name = str(row.get("name") or "").strip()
    if not badge_id or not name:
        return False
    state = _badge_state(row)
    if state == "deprecated" and not earners_count:
        return False
    if row.get("is_active") is False and not earners_count:
        return False
    return True


def _section_for_badge(badge: dict[str, Any], *, has_category: bool) -> str:
    category = str(badge.get("category") or "").strip()
    if has_category and category:
        return category
    earners = badge.get("earners_count")
    if earners is None:
        return "Unranked"
    try:
        count = int(earners)
    except Exception:
        return "Unranked"
    if count <= 0:
        return "Unclaimed"
    if count >= 100:
        return "Common"
    if count >= 25:
        return "Uncommon"
    if count >= 5:
        return "Rare"
    return "Legendary"


def _sort_sections(sections: dict[str, list[dict[str, Any]]], *, has_category: bool) -> list[dict[str, Any]]:
    if has_category:
        keys = sorted(sections.keys(), key=lambda item: str(item).lower())
    else:
        order = {name: idx for idx, name in enumerate(SECTION_ORDER)}
        keys = sorted(sections.keys(), key=lambda item: (order.get(item, len(order)), str(item).lower()))
    return [
        {
            "name": key,
            "badges": sorted(sections[key], key=lambda item: str(item.get("name") or "").lower()),
        }
        for key in keys
    ]


def _fetch_badge_rows(supabase: Any) -> list[dict[str, Any]]:
    try:
        return _safe_rows(supabase.table("badges").select(BADGE_SELECT).execute())
    except Exception:
        try:
            return _safe_rows(supabase.table("badges").select("badge_id,name,category,prestige,state,is_active").execute())
        except Exception:
            return []


def _fetch_player_badge_rows(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(supabase.table("player_badges").select(PLAYER_BADGE_SELECT).eq("club_id", str(club_id)).execute())
    except Exception:
        return []


def _fetch_player_names(supabase: Any, *, club_id: str) -> dict[int, str]:
    try:
        rows = _safe_rows(supabase.table("players").select(PLAYER_SELECT).eq("club_id", str(club_id)).execute())
    except Exception:
        rows = []
    names: dict[int, str] = {}
    for row in rows:
        if row.get("active") is False or row.get("inactive_at"):
            continue
        pid = _safe_int(row.get("id"))
        name = str(row.get("name") or "").strip()
        if pid is not None and name:
            names[int(pid)] = name
    return names


def _badge_counts(player_badges: list[dict[str, Any]]) -> dict[str, int]:
    unique: set[tuple[str, int]] = set()
    for row in player_badges:
        badge_id = str(row.get("badge_id") or "").strip()
        player_id = _safe_int(row.get("player_id"))
        if badge_id and player_id is not None:
            unique.add((badge_id, int(player_id)))
    counts: dict[str, int] = defaultdict(int)
    for badge_id, _ in unique:
        counts[badge_id] += 1
    return dict(counts)


def _sort_earner_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> str:
        return str(row.get("earned_at") or row.get("created_at") or "")

    return sorted(rows, key=key, reverse=True)


def _recent_earners_for_badge(
    player_badges: list[dict[str, Any]],
    *,
    badge_id: str,
    player_names: dict[int, str],
    limit: int,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    filtered = [row for row in player_badges if str(row.get("badge_id") or "").strip() == str(badge_id)]
    seen: set[int] = set()
    earners: list[dict[str, Any]] = []
    for row in _sort_earner_rows(filtered):
        player_id = _safe_int(row.get("player_id"))
        if player_id is None or int(player_id) in seen:
            continue
        seen.add(int(player_id))
        earners.append(
            {
                "player_id": int(player_id),
                "player_name": player_names.get(int(player_id), f"Player {int(player_id)}"),
                "earned_at": _json_safe(row.get("earned_at") or row.get("created_at")),
            }
        )
    safe_offset = max(0, int(offset or 0))
    safe_limit = max(1, min(int(limit or 25), 100))
    return earners[safe_offset : safe_offset + safe_limit], len(earners)


def _public_badge_payload(
    row: dict[str, Any],
    *,
    earners_count: int | None,
    recent_earners: list[dict[str, Any]],
) -> dict[str, Any]:
    badge_id = str(row.get("badge_id") or "").strip()
    state = _badge_state(row)
    prestige = _safe_int(row.get("prestige"), 0) or 0
    return {
        "badge_id": badge_id,
        "name": str(row.get("name") or "Badge"),
        "category": _plain_text(row.get("category")) or "Other",
        "prestige": int(prestige),
        "rarity": _plain_text(row.get("rarity")) or _plain_text(row.get("tier")),
        "icon_key": _plain_text(row.get("icon_key")),
        "scope": _plain_text(row.get("scope")),
        "state": state,
        "description": _plain_text(row.get("description_md")),
        "requirements": _requirements_text(row),
        "earners_count": earners_count,
        "recent_earners": recent_earners,
    }


def build_public_badge_codex(supabase: Any, *, club_id: str, recent_earners_limit: int = 5) -> dict[str, Any]:
    """Build public-safe Badge Codex data for a club."""

    badge_rows = _fetch_badge_rows(supabase)
    player_badges = _fetch_player_badge_rows(supabase, club_id=str(club_id))
    player_names = _fetch_player_names(supabase, club_id=str(club_id))
    counts = _badge_counts(player_badges)

    badges: list[dict[str, Any]] = []
    for row in badge_rows:
        badge_id = str(row.get("badge_id") or "").strip()
        earners_count = counts.get(badge_id)
        if not _badge_should_be_public(row, earners_count):
            continue
        recent, _total = _recent_earners_for_badge(
            player_badges,
            badge_id=badge_id,
            player_names=player_names,
            limit=recent_earners_limit,
        )
        badges.append(_public_badge_payload(row, earners_count=earners_count if earners_count is not None else 0, recent_earners=recent))

    has_category = any(str(badge.get("category") or "").strip() for badge in badges)
    sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for badge in badges:
        section = _section_for_badge(badge, has_category=has_category)
        sections[section].append(badge)

    earned_badges = sum(1 for badge in badges if int(badge.get("earners_count") or 0) > 0)
    total_earners = sum(int(badge.get("earners_count") or 0) for badge in badges)
    return {
        "summary": {
            "badge_count": len(badges),
            "earned_badge_count": earned_badges,
            "unclaimed_badge_count": max(0, len(badges) - earned_badges),
            "total_unique_earners_by_badge": total_earners,
        },
        "sections": _sort_sections(sections, has_category=has_category),
    }


def get_public_badge_earners(
    supabase: Any,
    *,
    club_id: str,
    badge_id: str,
    offset: int = 0,
    limit: int = 25,
) -> dict[str, Any]:
    """Return a public-safe page of badge earners."""

    clean_badge_id = str(badge_id or "").strip()
    if not clean_badge_id:
        raise ValueError("badge_id is required")
    player_badges = _fetch_player_badge_rows(supabase, club_id=str(club_id))
    player_names = _fetch_player_names(supabase, club_id=str(club_id))
    rows, total = _recent_earners_for_badge(
        player_badges,
        badge_id=clean_badge_id,
        player_names=player_names,
        limit=limit,
        offset=offset,
    )
    safe_offset = max(0, int(offset or 0))
    safe_limit = max(1, min(int(limit or 25), 100))
    return {
        "badge_id": clean_badge_id,
        "earners": rows,
        "total": int(total),
        "offset": safe_offset,
        "limit": safe_limit,
        "has_more": safe_offset + len(rows) < int(total),
    }
