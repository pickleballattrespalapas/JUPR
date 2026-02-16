from __future__ import annotations

from jupr_app.data.sb_write import sb_update

from datetime import datetime, timezone
from typing import Any


USE_BADGE_ENGINE_V3 = False

_NUMERIC_OPERATORS = {
    ">": lambda left, right: left > right,
    ">=": lambda left, right: left >= right,
    "=": lambda left, right: left == right,
    "<": lambda left, right: left < right,
    "<=": lambda left, right: left <= right,
}


def evaluate_badges_v3(player_id: int, context: Any) -> list[str]:
    """Evaluate published V3 badges for a single player and award matches."""
    supabase = getattr(context, "supabase", None)
    club_id = str(getattr(context, "club_id", "") or "")
    context_id = str(getattr(context, "context_id", "overall") or "overall")
    if supabase is None or not club_id:
        return []

    badges_resp = supabase.table("badges").select("badge_id,award_count,status,is_locked").eq("status", "published").execute()
    badges = badges_resp.data or []
    if not badges:
        return []

    badge_ids = [str(row.get("badge_id") or "") for row in badges if row.get("badge_id")]
    conditions_resp = (
        supabase.table("badge_rule_conditions")
        .select("badge_id,fact_key,operator,value_numeric,value_boolean")
        .in_("badge_id", badge_ids)
        .execute()
    )
    conditions = conditions_resp.data or []
    by_badge: dict[str, list[dict[str, Any]]] = {}
    for row in conditions:
        badge_id = str(row.get("badge_id") or "")
        if not badge_id:
            continue
        by_badge.setdefault(badge_id, []).append(row)

    facts_resp = (
        supabase.table("player_badge_facts")
        .select("fact_key,fact_value_num,fact_value_bool")
        .eq("club_id", club_id)
        .eq("player_id", int(player_id))
        .eq("context_id", context_id)
        .execute()
    )
    fact_rows = facts_resp.data or []
    facts = {str(row.get("fact_key") or ""): row for row in fact_rows}

    awarded_badge_ids: list[str] = []
    now = datetime.now(timezone.utc).isoformat()

    for badge in badges:
        badge_id = str(badge.get("badge_id") or "")
        badge_conditions = by_badge.get(badge_id, [])
        if not badge_id or not badge_conditions:
            continue

        if not _conditions_match(badge_conditions, facts):
            continue

        already_awarded = (
            supabase.table("player_badges")
            .select("badge_id")
            .eq("club_id", club_id)
            .eq("player_id", int(player_id))
            .eq("badge_id", badge_id)
            .eq("context_id", context_id)
            .limit(1)
            .execute()
            .data
            or []
        )
        if already_awarded:
            continue

        supabase.table("player_badges").insert(
            {
                "club_id": club_id,
                "player_id": int(player_id),
                "badge_id": badge_id,
                "context_type": "overall",
                "context_id": context_id,
                "earned_at": now,
                "awarded_by": "engine_v3",
            }
        ).execute()

        current_awards = int(badge.get("award_count") or 0)
        new_award_count = current_awards + 1
        sb_update(
            supabase,
            "badges",
            {
                "award_count": new_award_count,
                "is_locked": True,
            },
            filters={"badge_id": badge_id},
        )
        awarded_badge_ids.append(badge_id)

    return awarded_badge_ids


def _conditions_match(conditions: list[dict[str, Any]], facts: dict[str, dict[str, Any]]) -> bool:
    for condition in conditions:
        fact_key = str(condition.get("fact_key") or "")
        operator = str(condition.get("operator") or "").strip()
        fact_row = facts.get(fact_key)
        if not fact_row:
            return False

        if operator == "is":
            actual = _coerce_bool(fact_row.get("fact_value_bool"))
            expected = _coerce_bool(condition.get("value_boolean"))
            if actual is None or expected is None or actual is not expected:
                return False
            continue

        compare = _NUMERIC_OPERATORS.get(operator)
        if compare is None:
            return False
        actual_num = _coerce_num(fact_row.get("fact_value_num"))
        expected_num = _coerce_num(condition.get("value_numeric"))
        if actual_num is None or expected_num is None:
            return False
        if not compare(actual_num, expected_num):
            return False
    return True


def _coerce_num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None
