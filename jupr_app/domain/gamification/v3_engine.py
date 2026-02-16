from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from postgrest.exceptions import APIError

from jupr_app.config import USE_BADGE_ENGINE_V3

_NUMERIC_OPERATORS = {
    ">": lambda left, right: left > right,
    ">=": lambda left, right: left >= right,
    "=": lambda left, right: left == right,
    "<": lambda left, right: left < right,
    "<=": lambda left, right: left <= right,
}


def evaluate_badges_v3_for_player(supabase: Any, club_id: str, player_id: int, context_id: str) -> list[str]:
    """Evaluate published V3 badges for a single player in one context using AND-only numeric rules."""
    if supabase is None:
        return []

    normalized_club_id = str(club_id or "")
    normalized_context_id = str(context_id or "overall")
    if not normalized_club_id:
        return []

    badges = (
        supabase.table("badges")
        .select("badge_id,status")
        .eq("club_id", normalized_club_id)
        .eq("status", "published")
        .execute()
        .data
        or []
    )

    awarded_badge_ids: list[str] = []
    for badge in badges:
        badge_id = str(badge.get("badge_id") or "")
        if not badge_id:
            continue

        conditions = (
            supabase.table("badge_rule_conditions")
            .select("fact_key,operator,value_numeric")
            .eq("badge_id", badge_id)
            .execute()
            .data
            or []
        )
        if not conditions:
            continue

        if not _all_conditions_pass(supabase, normalized_club_id, int(player_id), normalized_context_id, conditions):
            continue

        did_insert = _insert_player_badge(
            supabase=supabase,
            club_id=normalized_club_id,
            player_id=int(player_id),
            badge_id=badge_id,
            context_id=normalized_context_id,
        )
        if not did_insert:
            continue

        _increment_badge_awards(supabase, normalized_club_id, badge_id)
        awarded_badge_ids.append(badge_id)

    return awarded_badge_ids


def evaluate_badges_v3(player_id: int, context: Any, *, allowed_badge_ids: set[str] | None = None) -> list[str]:
    """Backward-compatible wrapper used by the queue worker."""
    supabase = getattr(context, "supabase", None)
    club_id = str(getattr(context, "club_id", "") or "")
    context_id = str(getattr(context, "context_id", "overall") or "overall")
    awarded = evaluate_badges_v3_for_player(supabase, club_id, int(player_id), context_id)
    if allowed_badge_ids is None:
        return awarded
    allowed = {str(badge_id) for badge_id in allowed_badge_ids}
    return [badge_id for badge_id in awarded if badge_id in allowed]


def _all_conditions_pass(
    supabase: Any,
    club_id: str,
    player_id: int,
    context_id: str,
    conditions: list[dict[str, Any]],
) -> bool:
    for condition in conditions:
        fact_key = str(condition.get("fact_key") or "")
        operator = str(condition.get("operator") or "").strip()
        expected_value = _coerce_num(condition.get("value_numeric"))
        compare = _NUMERIC_OPERATORS.get(operator)
        if not fact_key or compare is None or expected_value is None:
            return False

        fact_rows = (
            supabase.table("player_badge_facts")
            .select("fact_value_num")
            .eq("club_id", club_id)
            .eq("player_id", int(player_id))
            .eq("context_id", context_id)
            .eq("fact_key", fact_key)
            .limit(1)
            .execute()
            .data
            or []
        )
        if not fact_rows:
            return False

        actual_value = _coerce_num(fact_rows[0].get("fact_value_num"))
        if actual_value is None or not compare(actual_value, expected_value):
            return False
    return True


def _insert_player_badge(
    supabase: Any,
    club_id: str,
    player_id: int,
    badge_id: str,
    context_id: str,
) -> bool:
    try:
        supabase.table("player_badges").insert(
            {
                "club_id": club_id,
                "player_id": int(player_id),
                "badge_id": badge_id,
                "context_id": context_id,
                "earned_at": datetime.now(timezone.utc).isoformat(),
            }
        ).execute()
        return True
    except APIError as exc:
        if getattr(exc, "code", None) == "23505":
            return False
        raise


def _increment_badge_awards(supabase: Any, club_id: str, badge_id: str) -> None:
    rows = (
        supabase.table("badges")
        .select("award_count")
        .eq("club_id", club_id)
        .eq("badge_id", badge_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    current_award_count = int((rows[0] or {}).get("award_count") or 0) if rows else 0
    supabase.table("badges").update(
        {
            "award_count": current_award_count + 1,
            "is_locked": True,
        }
    ).eq("club_id", club_id).eq("badge_id", badge_id).execute()


def _coerce_num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
