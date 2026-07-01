from __future__ import annotations

from typing import Any

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def r(row: Any, field: str, default: Any = None) -> Any:
    return getattr(row, field, default)


def _is_missingish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set)):
        return False
    try:
        result = pd.isna(value)
        return bool(result) if not hasattr(result, "any") else False
    except Exception:
        return False


def _safe_text(value: Any, default: str = "") -> str:
    if _is_missingish(value):
        return default
    text = str(value).strip()
    return text if text else default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if _is_missingish(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    try:
        return bool(value)
    except Exception:
        return default


def _safe_value(value: Any, default: Any = None) -> Any:
    return default if _is_missingish(value) else value


def _coerce_int(value: Any, fallback: int) -> int:
    if _is_missingish(value):
        return int(fallback)
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(fallback)


def _coerce_timestamp(value: Any) -> pd.Timestamp:
    return pd.to_datetime(_safe_value(value, None), utc=True, errors="coerce")


def select_featured_badges(
    unlocked_badges: list[dict[str, Any]],
    *,
    max_count: int = 5,
    sort_mode: str = "recent",
) -> list[dict[str, Any]]:
    if not unlocked_badges:
        return []

    def sort_key(badge: dict[str, Any]) -> tuple:
        last_earned = _coerce_timestamp(badge.get("last_earned_at"))
        prestige = _coerce_int(badge.get("prestige"), 0)
        last_sort = last_earned.timestamp() if pd.notna(last_earned) else 0.0
        if sort_mode == "prestige":
            return (-prestige, -last_sort)
        return (-last_sort, -prestige)

    ordered = sorted(unlocked_badges, key=sort_key)
    seen: set[str] = set()
    featured: list[dict[str, Any]] = []
    for badge in ordered:
        badge_id = _safe_text(badge.get("badge_id"), "__unknown__")
        if badge_id in seen:
            continue
        seen.add(badge_id)
        featured.append(badge)
        if len(featured) >= max_count:
            break
    return featured


def _empty_summary() -> dict[str, Any]:
    return {
        "prestige_total": 0,
        "collected_unique_count": 0,
        "total_active_badge_types": 0,
        "unlocked_badges": [],
        "locked_badges": [],
        "collection_stats": {
            "collected_count": 0,
            "total_count": 0,
            "collected_by_category": {},
        },
    }


def _ensure_columns(df: pd.DataFrame, defaults: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    return out


def _series_or_default(df: pd.DataFrame, col: str, default: Any) -> pd.Series:
    if col in df.columns:
        return df[col]
    fallback_col = f"{col}_def"
    if fallback_col in df.columns:
        return df[fallback_col]
    return pd.Series([default] * len(df), index=df.index)


def _safe_badge_name(value: Any, badge_id: Any = None) -> str:
    name = _safe_text(value)
    if name:
        return name
    badge_key = _safe_text(badge_id)
    if badge_key:
        return " ".join(part for part in badge_key.replace("-", "_").split("_") if part).title()
    return "Badge"


def build_gamification_summary(
    player_id: int,
    df_badges: pd.DataFrame,
    df_player_badges: pd.DataFrame,
) -> dict[str, Any]:
    badge_defs = df_badges.copy() if df_badges is not None else pd.DataFrame()
    pb = df_player_badges.copy() if df_player_badges is not None else pd.DataFrame()

    if badge_defs.empty and pb.empty:
        return _empty_summary()

    optional_columns = {
        "requirements": "Requirements TBD",
        "rarity": None,
        "icon_key": None,
        "tier": None,
        "scope": None,
        "badge_scope": None,
        "badge_status": "live",
        "badge_award_timing": "live",
        "category": "",
        "is_stackable": False,
        "is_active": True,
        "prestige": 0,
        "name": "Badge",
    }
    badge_defs = _ensure_columns(badge_defs, {"badge_id": ""} | optional_columns)

    if __debug__:
        logger.debug("badge_defs columns: %s", list(badge_defs.columns))

    badge_defs["badge_id"] = badge_defs.get("badge_id", "").astype(str)
    badge_defs["prestige"] = pd.to_numeric(badge_defs.get("prestige", 0), errors="coerce").fillna(0)
    badge_defs["is_active"] = badge_defs["is_active"].fillna(True)

    if "rarity" not in badge_defs.columns:
        badge_defs["rarity"] = badge_defs["prestige"].apply(rarity_from_prestige)
    else:
        missing_rarity = badge_defs["rarity"].isna() | (badge_defs["rarity"].astype(str).str.strip() == "")
        if missing_rarity.any():
            badge_defs.loc[missing_rarity, "rarity"] = badge_defs.loc[missing_rarity, "prestige"].apply(rarity_from_prestige)

    active_badges = badge_defs[badge_defs["is_active"] != False].copy()

    if not pb.empty:
        if "player_id" not in pb.columns:
            pb = pd.DataFrame()
        else:
            pb_player_id = pd.to_numeric(pb.get("player_id"), errors="coerce")
            pb = pb.loc[pb_player_id.notna()].copy()
            if not pb.empty:
                pb_player_id = pb_player_id.loc[pb.index].astype(int)
                pb = pb.loc[pb_player_id == int(player_id)].copy()

    unlocked: list[dict[str, Any]] = []
    if not pb.empty:
        if "badge_id" not in pb.columns:
            pb["badge_id"] = ""
        pb["badge_id"] = pb["badge_id"].fillna("").astype(str)

        if active_badges.empty:
            merged = pb.copy()
        else:
            merged = pb.merge(active_badges, on="badge_id", how="left", suffixes=("", "_def"))
        merged["earned_at_dt"] = pd.to_datetime(merged.get("earned_at", None), utc=True, errors="coerce")
        merged["prestige"] = pd.to_numeric(_series_or_default(merged, "prestige", 0), errors="coerce").fillna(0)
        merged["name"] = _series_or_default(merged, "name", None)
        merged["category"] = _series_or_default(merged, "category", "")
        merged["rarity"] = _series_or_default(merged, "rarity", None)
        merged["requirements"] = _series_or_default(merged, "requirements", "Requirements TBD")
        merged["is_stackable"] = _series_or_default(merged, "is_stackable", False)
        merged["icon_key"] = _series_or_default(merged, "icon_key", None)
        merged["tier"] = _series_or_default(merged, "tier", None)
        merged["scope"] = _series_or_default(merged, "scope", None)
        merged["badge_scope"] = _series_or_default(merged, "badge_scope", None)
        merged["badge_status"] = _series_or_default(merged, "badge_status", "live")
        merged["badge_award_timing"] = _series_or_default(merged, "badge_award_timing", "live")

        for badge_id, group in merged.groupby("badge_id", dropna=False):
            group_sorted = group.sort_values("earned_at_dt", na_position="first")
            first = group_sorted.iloc[0]
            last = group_sorted.iloc[-1]
            latest_excerpt = ""
            try:
                value_json = last.get("value_json") or {}
                if isinstance(value_json, str):
                    import json

                    value_json = json.loads(value_json)
                latest_excerpt = _safe_text((value_json or {}).get("tape_excerpt")) if isinstance(value_json, dict) else ""
            except Exception:
                latest_excerpt = ""
            if not latest_excerpt.strip():
                latest_excerpt = _fallback_tape_excerpt(
                    _safe_badge_name(first.get("name"), badge_id),
                    _safe_text(first.get("category"), "season"),
                )

            unlocked.append(
                {
                    "badge_id": _safe_text(badge_id, "__unknown__"),
                    "name": _safe_badge_name(first.get("name"), badge_id),
                    "category": _safe_text(first.get("category"), "Other"),
                    "rarity": _resolve_rarity(first.get("rarity"), first.get("prestige", 0)),
                    "prestige": _coerce_int(first.get("prestige"), 0),
                    "requirements": _safe_text(first.get("requirements"), "Requirements TBD"),
                    "is_stackable": _safe_bool(first.get("is_stackable"), False),
                    "icon_key": _safe_value(first.get("icon_key"), None),
                    "tier": _safe_value(first.get("tier"), None),
                    "scope": _safe_value(first.get("scope"), None),
                    "badge_scope": _safe_value(first.get("badge_scope"), None),
                    "badge_status": _safe_text(first.get("badge_status"), "live"),
                    "badge_award_timing": _safe_text(first.get("badge_award_timing"), "live"),
                    "stack_count": max(_coerce_int(len(group_sorted), 1), 1),
                    "first_earned_at": _safe_value(first.get("earned_at"), None),
                    "last_earned_at": _safe_value(last.get("earned_at"), None),
                    "latest_tape_excerpt": latest_excerpt,
                }
            )

    unlocked_ids = {u["badge_id"] for u in unlocked}
    locked: list[dict[str, Any]] = []
    if not active_badges.empty:
        for row in active_badges.itertuples(index=False):
            badge_id = _safe_text(r(row, "badge_id", ""))
            if not badge_id or badge_id in unlocked_ids:
                continue
            locked.append(
                {
                    "badge_id": badge_id,
                    "name": _safe_badge_name(r(row, "name", None), badge_id),
                    "category": _safe_text(r(row, "category", ""), "Other"),
                    "rarity": _resolve_rarity(r(row, "rarity", None), r(row, "prestige", 0)),
                    "prestige": _coerce_int(r(row, "prestige", 0), 0),
                    "requirements": _safe_text(r(row, "requirements", "Requirements TBD"), "Requirements TBD"),
                    "is_stackable": _safe_bool(r(row, "is_stackable", False), False),
                    "icon_key": _safe_value(r(row, "icon_key", None), None),
                    "tier": _safe_value(r(row, "tier", None), None),
                    "scope": _safe_value(r(row, "scope", None), None),
                    "badge_scope": _safe_value(r(row, "badge_scope", None), None),
                    "badge_status": _safe_text(r(row, "badge_status", "live"), "live"),
                    "badge_award_timing": _safe_text(r(row, "badge_award_timing", "live"), "live"),
                }
            )

    prestige_total = int(sum(_coerce_int(u.get("prestige"), 0) * max(_coerce_int(u.get("stack_count"), 1), 1) for u in unlocked))
    collected_unique_count = len(unlocked)
    total_active_badge_types = int(len(active_badges)) if not active_badges.empty else collected_unique_count
    collected_by_category: dict[str, int] = {}
    for badge in unlocked:
        category = _safe_text(badge.get("category"), "Other")
        collected_by_category[category] = collected_by_category.get(category, 0) + max(_coerce_int(badge.get("stack_count"), 1), 1)

    return {
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
    if _is_missingish(value):
        return rarity_from_prestige(prestige)
    text = str(value).strip()
    return text or rarity_from_prestige(prestige)
