from __future__ import annotations

from jupr_app.domain.gamification.badge_policy import badges_enabled, is_formal_trophy_candidate

import logging
from types import SimpleNamespace
from typing import Any

from jupr_app.domain.gamification.badge_engine import compute_candidates_for_club
from jupr_app.domain.gamification.badges_repo import upsert_player_badges


logger = logging.getLogger(__name__)


def ensure_badges(
    ctx: Any,
    *,
    club_id: str | None = None,
    league_id: str | None = None,
    as_of=None,
    tournament_id: str | None = None,
    tournament_name: str | None = None,
    status: str = "live",
    award_timing: str = "live",
    allow_non_live: bool = False,
) -> list:
    # League close still mints its formal podium/top-performer trophies.
    if not badges_enabled() and award_timing != "on_league_close":
        return []
    if bool(getattr(ctx, "public_mode", False)):
        return []

    supabase = getattr(ctx, "supabase", None)
    resolved_club_id = str(club_id or getattr(ctx, "club_id", "") or "")
    if supabase is None or not resolved_club_id:
        return []

    evaluation_ctx = ctx
    if tournament_id or tournament_name:
        attrs = dict(vars(ctx)) if hasattr(ctx, "__dict__") else {}
        attrs.update({"tournament_id": tournament_id, "tournament_name": tournament_name})
        evaluation_ctx = SimpleNamespace(**attrs)

    try:
        candidates = list(
            compute_candidates_for_club(
                club_id=resolved_club_id,
                league_id=league_id,
                as_of=as_of,
                ctx=evaluation_ctx,
                status=status,
                award_timing=award_timing,
                allow_non_live=allow_non_live,
            )
        )
        if not badges_enabled():
            candidates = [candidate for candidate in candidates
                          if is_formal_trophy_candidate(candidate)]
        if not candidates:
            return []
        return upsert_player_badges(supabase, resolved_club_id, candidates)
    except Exception:
        logger.exception("ensure_badges failed")
        return []
