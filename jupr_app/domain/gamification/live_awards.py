from __future__ import annotations

from typing import Any

from jupr_app.domain.gamification.badge_engine import compute_candidates_for_player
from jupr_app.domain.gamification.badges_repo import upsert_player_badges
from jupr_app.domain.gamification.badge_worker import _badge_ids_for_trigger, _resolve_context


def run_live_badge_awards(
    supabase: Any,
    *,
    club_id: str,
    player_ids: list[int] | set[int],
    event_type: str,
    ctx: Any | None = None,
    match_limit: int = 5000,
) -> dict[str, Any]:
    normalized_player_ids = sorted({int(pid) for pid in (player_ids or [])})
    if not normalized_player_ids:
        return {"awarded_count": 0, "candidate_count": 0, "badge_ids": [], "mode": "inline"}

    context = _resolve_context(ctx, supabase, str(club_id), match_limit)
    eligible_badge_ids = sorted(_badge_ids_for_trigger(context, event_type))
    if not eligible_badge_ids:
        return {"awarded_count": 0, "candidate_count": 0, "badge_ids": [], "mode": "inline"}

    candidates = []
    for pid in normalized_player_ids:
        player_candidates = compute_candidates_for_player(
            str(club_id),
            int(pid),
            ctx=context,
            status="live",
            award_timing="live",
        )
        candidates.extend([c for c in player_candidates if str(c.badge_id) in set(eligible_badge_ids)])

    created = upsert_player_badges(
        supabase,
        str(club_id),
        candidates,
        awarded_by="engine",
    )
    return {
        "awarded_count": len(created),
        "candidate_count": len(candidates),
        "badge_ids": eligible_badge_ids,
        "mode": "inline",
    }
