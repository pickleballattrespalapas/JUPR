from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime
from functools import lru_cache
from typing import Any

from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.badge_schema import load_badge_definitions
from jupr_app.domain.gamification.requirements import load_requirements_map
from jupr_app.domain.gamification.presentation import badge_category, badge_requirement, category_sort_key
from jupr_app.data.paged_reads import read_all_rows

BADGE_SELECT = "badge_id,name,category,prestige,state,is_active,rarity,tier,icon_key,scope,lore,hint,created_at"
PLAYER_BADGE_SELECT = "club_id,player_id,badge_id,earned_at,revoked_at,context_type,context_id"
PLAYER_SELECT = "id,club_id,name,active,inactive_at"
SECTION_ORDER = ["Common", "Uncommon", "Rare", "Legendary", "Unclaimed", "Unranked", "Other"]
CATALOG_BUCKET_ORDER = ["Live Now", "Seasonal / League Close", "Manual / Curated", "Tracked / Disabled"]
CATALOG_BUCKET_DESCRIPTIONS = {
    "Live Now": "Automatically evaluated from eligible activity as results are recorded.",
    "Seasonal / League Close": "Evaluated from final league or season results when staff closes the competition.",
    "Manual / Curated": "Awarded by an authorized operator for tournament placement, sportsmanship, or community moments.",
    "Tracked / Disabled": "Defined and tracked for history, but not currently awarded by the live badge worker.",
}
CANONICAL_CATEGORY_LABELS = {
    "dominance": "Dominance",
    "consistency": "Consistency",
    "performance": "Performance",
    "activity": "Activity",
    "community": "Community",
    "streaks": "Streaks",
    "special": "Special",
}

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
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_`#]", "", text)
    return re.sub(r"\s+", " ", text) or None


def _category_label(value: Any) -> str:
    label = _plain_text(value)
    if not label:
        return "Other"
    key = " ".join(label.split()).lower()
    return CANONICAL_CATEGORY_LABELS.get(key, label.title())


def _requirements_text(row: dict[str, Any]) -> str:
    text = _plain_text(row.get("requirements"))
    if text:
        return re.sub(r"^(requirements?)\s*:\s*", "", text, flags=re.IGNORECASE).strip() or "Requirements TBD"
    return "Requirements TBD"


def _badge_state(row: dict[str, Any]) -> str:
    if row.get("is_active") is False:
        return "deprecated"
    state = str(row.get("state") or "").strip().lower()
    if state:
        return state
    if row.get("is_active") is False:
        return "deprecated"
    return "live"


@lru_cache(maxsize=1)
def _canonical_badge_maps() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    catalog = {str(item.badge_id): item for item in BADGE_DEFINITIONS}
    schemas = {item.id: item for item in load_badge_definitions(BADGE_DEFINITIONS)}
    return catalog, schemas, load_requirements_map()


def _badge_authority(row: dict[str, Any]) -> dict[str, Any]:
    badge_id = str(row.get("badge_id") or "").strip()
    catalog, schemas, requirements = _canonical_badge_maps()
    definition = catalog.get(badge_id)
    schema = schemas.get(badge_id)
    lifecycle_state = _badge_state(row)

    badge_status = str(getattr(schema, "status", "retired") or "retired").strip().lower()
    award_timing = str(getattr(schema, "award_timing", "disabled") or "disabled").strip().lower()
    badge_scope = str(getattr(schema, "scope", "") or "").strip().lower() or None
    requirements_text = _plain_text(getattr(getattr(schema, "display", None), "requirements", None))
    if not requirements_text:
        requirements_text = _plain_text(requirements.get(badge_id)) or _requirements_text(row)

    description = requirements_text

    if lifecycle_state == "deprecated":
        catalog_bucket = "Tracked / Disabled"
        availability = "Deprecated; prior awards remain visible."
    elif lifecycle_state == "frozen":
        catalog_bucket = "Tracked / Disabled"
        availability = "Frozen; no new awards while prior awards remain visible."
    elif badge_status in {"tracked", "retired"} or award_timing == "disabled":
        catalog_bucket = "Tracked / Disabled"
        availability = "Tracked only; not currently awarded."
    elif badge_status == "seasonal" or award_timing == "on_league_close":
        catalog_bucket = "Seasonal / League Close"
        availability = "Awarded when authorized staff closes the applicable league or season."
    elif badge_status == "curated" or award_timing == "manual":
        catalog_bucket = "Manual / Curated"
        availability = "Awarded manually by authorized staff."
    else:
        catalog_bucket = "Live Now"
        availability = "Currently obtainable from eligible recorded activity."

    return {
        "badge_status": badge_status,
        "badge_award_timing": award_timing,
        "badge_scope": badge_scope,
        "catalog_bucket": catalog_bucket,
        "availability": availability,
        "requirements": requirements_text,
        "description": description,
    }


def _badge_should_be_public(row: dict[str, Any], earners_count: int | None) -> bool:
    badge_id = str(row.get("badge_id") or "").strip()
    if badge_id in {"league_champion", "league_runner_up", "league_third_place", "podium"}:
        return False  # Historical profile awards remain; these are no longer catalog goals.
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
        keys = sorted(sections.keys(), key=category_sort_key)
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
    return read_all_rows(lambda: supabase.table("badges").select(BADGE_SELECT), order="badge_id")

def _fetch_player_badge_rows(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    rows = read_all_rows(lambda: supabase.table("player_badges").select(PLAYER_BADGE_SELECT).eq("club_id", str(club_id)))
    return [row for row in rows if not row.get("revoked_at")]

def _fetch_player_names(supabase: Any, *, club_id: str) -> dict[int, str]:
    rows = read_all_rows(lambda: supabase.table("players").select(PLAYER_SELECT).eq("club_id", str(club_id)))
    names: dict[int, str] = {}
    for row in rows:
        if row.get("active") is False or row.get("inactive_at"):
            continue
        pid = _safe_int(row.get("id"))
        name = str(row.get("name") or "").strip()
        if pid is not None and name:
            names[int(pid)] = name
    return names


def _badge_counts(player_badges: list[dict[str, Any]], *, public_player_ids: set[int]) -> dict[str, int]:
    unique: set[tuple[str, int]] = set()
    for row in player_badges:
        badge_id = str(row.get("badge_id") or "").strip()
        player_id = _safe_int(row.get("player_id"))
        if badge_id and player_id is not None and int(player_id) in public_player_ids:
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
        if player_id is None or int(player_id) in seen or int(player_id) not in player_names:
            continue
        seen.add(int(player_id))
        earners.append(
            {
                "player_id": int(player_id),
                "player_name": player_names[int(player_id)],
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
    authority = _badge_authority(row)
    prestige = _safe_int(row.get("prestige"), 0) or 0
    return {
        "badge_id": badge_id,
        "name": str(row.get("name") or "Badge"),
        "category": badge_category(badge_id),
        "prestige": int(prestige),
        "rarity": _plain_text(row.get("rarity")) or _plain_text(row.get("tier")),
        "icon_key": _plain_text(row.get("icon_key")),
        "scope": _plain_text(row.get("scope")),
        "state": state,
        "lifecycle_state": state,
        **authority,
        "earners_count": earners_count,
        "recent_earners": recent_earners,
    }


def _catalog_buckets(badges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for badge in badges:
        grouped[str(badge.get("catalog_bucket") or "Tracked / Disabled")].append(badge)

    result: list[dict[str, Any]] = []
    for bucket_name in CATALOG_BUCKET_ORDER:
        bucket_badges = grouped.get(bucket_name, [])
        category_sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for badge in bucket_badges:
            category_sections[str(badge.get("category") or "Other")].append(badge)
        result.append(
            {
                "name": bucket_name,
                "description": CATALOG_BUCKET_DESCRIPTIONS[bucket_name],
                "badge_count": len(bucket_badges),
                "sections": _sort_sections(category_sections, has_category=True),
            }
        )
    return result


def _trophy_room(
    player_badges: list[dict[str, Any]],
    *,
    badges_by_id: dict[str, dict[str, Any]],
    player_names: dict[int, str],
    limit: int = 12,
) -> list[dict[str, Any]]:
    awards_by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _sort_earner_rows(player_badges):
        player_id = _safe_int(row.get("player_id"))
        badge_id = str(row.get("badge_id") or "").strip()
        if player_id is None or int(player_id) not in player_names or badge_id not in badges_by_id:
            continue
        awards_by_player[int(player_id)].append({**row, "badge_id": badge_id})

    entries: list[dict[str, Any]] = []
    for player_id, awards in awards_by_player.items():
        unique_ids = {str(row.get("badge_id") or "") for row in awards}
        latest: list[dict[str, Any]] = []
        seen_latest: set[str] = set()
        for award in awards:
            badge_id = str(award.get("badge_id") or "")
            if badge_id in seen_latest:
                continue
            seen_latest.add(badge_id)
            badge = badges_by_id[badge_id]
            latest.append(
                {
                    "badge_id": badge_id,
                    "badge_name": badge.get("name") or "Badge",
                    "earned_at": _json_safe(award.get("earned_at") or award.get("created_at")),
                }
            )
            if len(latest) >= 4:
                break
        entries.append(
            {
                "player_id": player_id,
                "player_name": player_names[player_id],
                "unique_badge_count": len(unique_ids),
                "award_count": len(awards),
                "prestige_total": sum(
                    int(badges_by_id[str(row.get("badge_id") or "")].get("prestige") or 0)
                    for row in awards
                ),
                "latest_earned_at": _json_safe(awards[0].get("earned_at") or awards[0].get("created_at")),
                "latest_badges": latest,
            }
        )
    entries.sort(
        key=lambda item: (
            -int(item.get("prestige_total") or 0),
            -int(item.get("award_count") or 0),
            str(item.get("player_name") or "").casefold(),
        )
    )
    return entries[: max(1, min(int(limit or 12), 50))]


def build_public_badge_codex(supabase: Any, *, club_id: str, recent_earners_limit: int = 5) -> dict[str, Any]:
    """Build public-safe Badge Codex data for a club."""

    badge_rows = _fetch_badge_rows(supabase)
    player_badges = _fetch_player_badge_rows(supabase, club_id=str(club_id))
    player_names = _fetch_player_names(supabase, club_id=str(club_id))
    counts = _badge_counts(player_badges, public_player_ids=set(player_names))

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
    badges_by_id = {str(badge.get("badge_id") or ""): badge for badge in badges}
    categories = sorted({str(badge.get("category") or "Other") for badge in badges}, key=category_sort_key)
    scopes = sorted({str(badge.get("badge_scope") or "") for badge in badges if badge.get("badge_scope")}, key=str.casefold)
    return {
        "seasons": read_all_rows(lambda: supabase.table("badge_seasons").select("id,name,start_date,end_date,timezone").eq("club_id", str(club_id)), order="start_date"),
        "summary": {
            "badge_count": len(badges),
            "earned_badge_count": earned_badges,
            "unclaimed_badge_count": max(0, len(badges) - earned_badges),
            "total_unique_earners_by_badge": total_earners,
            "unique_earner_count": len({int(row["player_id"]) for row in player_badges
                                       if _safe_int(row.get("player_id")) in player_names
                                       and str(row.get("badge_id")) in badges_by_id}),
            "complete_definition_count": sum(
                1
                for badge in badges
                if badge.get("requirements")
                and badge.get("description")
                and "requirements tbd" not in str(badge.get("requirements") or "").lower()
            ),
        },
        "sections": _sort_sections(sections, has_category=has_category),
        "catalog_buckets": _catalog_buckets(badges),
        "filters": {
            "categories": categories,
            "scopes": scopes,
            "statuses": sorted({str(badge.get("badge_status") or "") for badge in badges}, key=str.casefold),
            "award_timings": sorted({str(badge.get("badge_award_timing") or "") for badge in badges}, key=str.casefold),
        },
        "trophy_room": _trophy_room(
            player_badges,
            badges_by_id=badges_by_id,
            player_names=player_names,
        ),
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
    badge_rows = _fetch_badge_rows(supabase)
    player_badges = _fetch_player_badge_rows(supabase, club_id=str(club_id))
    player_names = _fetch_player_names(supabase, club_id=str(club_id))
    matching_row = next(
        (row for row in badge_rows if str(row.get("badge_id") or "").strip() == clean_badge_id),
        None,
    )
    if matching_row is None:
        raise ValueError("badge not found")
    public_count = _badge_counts(player_badges, public_player_ids=set(player_names)).get(clean_badge_id, 0)
    if not _badge_should_be_public(matching_row, public_count):
        raise ValueError("badge not found")
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
        "badge": _public_badge_payload(
            matching_row,
            earners_count=int(public_count),
            recent_earners=[],
        ),
        "earners": rows,
        "total": int(total),
        "offset": safe_offset,
        "limit": safe_limit,
        "has_more": safe_offset + len(rows) < int(total),
    }
