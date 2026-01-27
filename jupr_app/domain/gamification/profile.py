from __future__ import annotations

from typing import Any

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def r(row: Any, field: str, default: Any = None) -> Any:
    return getattr(row, field, default)


def select_featured_badges(
    unlocked_badges: list[dict[str, Any]],
    *,
    max_count: int = 5,
    sort_mode: str = "recent",
) -> list[dict[str, Any]]:
    if not unlocked_badges:
        return []

    def sort_key(badge: dict[str, Any]) -> tuple:
        last_earned = pd.to_datetime(badge.get("last_earned_at"), utc=True, errors="coerce")
        prestige = _coerce_int(badge.get("prestige"), 0)
        if sort_mode == "prestige":
            return (-prestige, last_earned if pd.notna(last_earned) else pd.Timestamp.min)
        return (
            -(last_earned.timestamp() if pd.notna(last_earned) else 0.0),
            -prestige,
        )

    ordered = sorted(unlocked_badges, key=sort_key)
    seen: set[str] = set()
    featured: list[dict[str, Any]] = []
    for badge in ordered:
        badge_id = str(badge.get("badge_id"))
        if badge_id in seen:
            continue
        seen.add(badge_id)
        featured.append(badge)
        if len(featured) >= max_count:
            break
    return featured


def build_gamification_summary(
    player_id: int,
    df_badges: pd.DataFrame,
    df_player_badges: pd.DataFrame,
) -> dict[str, Any]:
    badge_defs = df_badges.copy() if df_badges is not None else pd.DataFrame()
    pb = df_player_badges.copy() if df_player_badges is not None else pd.DataFrame()

    optional_columns = {
        "lore": "",
        "hint": "",
        "rarity": None,
        "icon_key": None,
        "tier": None,
        "scope": None,
        "category": "",
    }
    for col, default in optional_columns.items():
        if col not in badge_defs.columns:
            badge_defs[col] = default

    if __debug__:
        logger.debug("badge_defs columns: %s", list(badge_defs.columns))

    badge_defs["badge_id"] = badge_defs.get("badge_id", "").astype(str)
    badge_defs["prestige"] = pd.to_numeric(badge_defs.get("prestige", 0), errors="coerce").fillna(0)
    if "is_active" not in badge_defs.columns:
        badge_defs["is_active"] = True
    badge_defs["is_active"] = badge_defs["is_active"].fillna(True)
    rarity_fallback = False
    if "rarity" not in badge_defs.columns:
        rarity_fallback = True
        badge_defs["rarity"] = badge_defs["prestige"].apply(rarity_from_prestige)
    else:
        missing_rarity = badge_defs["rarity"].isna() | (
            badge_defs["rarity"].astype(str).str.strip() == ""
        )
        if missing_rarity.any():
            rarity_fallback = True
            badge_defs.loc[missing_rarity, "rarity"] = badge_defs.loc[
                missing_rarity, "prestige"
            ].apply(rarity_from_prestige)

    if __debug__:
        assert "rarity" in badge_defs.columns or rarity_fallback
    active_badges = badge_defs[badge_defs["is_active"] != False].copy()

    pb = pb[pb.get("player_id") == int(player_id)].copy() if not pb.empty else pd.DataFrame()
    if pb.empty or active_badges.empty:
        unlocked = []
    else:
        merged = pb.merge(active_badges, on="badge_id", how="left")
        merged["earned_at_dt"] = pd.to_datetime(merged.get("earned_at", None), utc=True, errors="coerce")
        merged["prestige"] = pd.to_numeric(merged.get("prestige", 0), errors="coerce").fillna(0)

        unlocked = []
        for badge_id, group in merged.groupby("badge_id"):
            group_sorted = group.sort_values("earned_at_dt")
            first = group_sorted.iloc[0]
            last = group_sorted.iloc[-1]
            latest_excerpt = ""
            try:
                value_json = last.get("value_json") or {}
                latest_excerpt = str(value_json.get("tape_excerpt") or "")
            except Exception:
                latest_excerpt = ""
            if not latest_excerpt.strip():
                latest_excerpt = _fallback_tape_excerpt(
                    str(first.get("name") or "Badge"),
                    str(first.get("category") or "season"),
                )

            unlocked.append(
                {
                    "badge_id": badge_id,
                    "name": first.get("name"),
                    "category": first.get("category", ""),
                    "rarity": _resolve_rarity(first.get("rarity"), first.get("prestige", 0)),
                    "prestige": int(first.get("prestige", 0) or 0),
                    "lore": first.get("lore", ""),
                    "icon_key": first.get("icon_key", None),
                    "tier": first.get("tier", None),
                    "scope": first.get("scope", None),
                    "stack_count": int(len(group_sorted)),
                    "first_earned_at": first.get("earned_at"),
                    "last_earned_at": last.get("earned_at"),
                    "latest_tape_excerpt": latest_excerpt,
                }
            )

    unlocked_ids = {u["badge_id"] for u in unlocked}
    locked = []
    for row in active_badges.itertuples(index=False):
        if str(row.badge_id) in unlocked_ids:
            continue
        locked.append(
            {
                "badge_id": row.badge_id,
                "name": row.name,
                "category": r(row, "category", ""),
                "rarity": _resolve_rarity(r(row, "rarity", None), r(row, "prestige", 0)),
                "prestige": int(r(row, "prestige", 0) or 0),
                "lore": r(row, "lore", ""),
                "hint": r(row, "hint", ""),
                "icon_key": r(row, "icon_key", None),
                "tier": r(row, "tier", None),
                "scope": r(row, "scope", None),
            }
        )

    prestige_total = int(sum(u.get("prestige", 0) * u.get("stack_count", 1) for u in unlocked))
    collected_unique_count = len(unlocked)
    total_active_badge_types = int(len(active_badges))
    collected_by_category: dict[str, int] = {}
    for badge in unlocked:
        category = badge.get("category") or "Other"
        collected_by_category[category] = collected_by_category.get(category, 0) + badge.get("stack_count", 1)

    summary = {
        "prestige_total": prestige_total,
        "collected_unique_count": collected_unique_count,
        "total_active_badge_types": total_active_badge_types,
        "unlocked_badges": unlocked,
        "locked_badges": locked,
        "collection_stats": {
            "collected_count": collected_unique_count,
            "total_count": total_active_badge_types,
            "collected_by_category": collected_by_category,
        },
    }
    return summary


def _fallback_tape_excerpt(name: str, category: str) -> str:
    safe_name = name.strip() or "The badge"
    safe_category = category.strip() or "season"
    lines = [
        f"{safe_name} lives in the tape room archives.",
        f"The {safe_category.lower()} reel keeps the moment on loop.",
    ]
    return "\n".join(lines)


def rarity_from_prestige(prestige: float | int) -> str:
    try:
        prestige_value = float(prestige)
    except (TypeError, ValueError):
        prestige_value = 0.0

    if prestige_value >= 90:
        return "legendary"
    if prestige_value >= 70:
        return "epic"
    if prestige_value >= 50:
        return "rare"
    if prestige_value >= 30:
        return "uncommon"
    return "common"


def _resolve_rarity(value: Any, prestige: Any) -> str:
    if value is None:
        return rarity_from_prestige(prestige)
    if isinstance(value, float) and pd.isna(value):
        return rarity_from_prestige(prestige)
    if isinstance(value, str) and not value.strip():
        return rarity_from_prestige(prestige)
    return str(value)


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        if value is None:
            return fallback
        return int(value)
    except Exception:
        return fallback
