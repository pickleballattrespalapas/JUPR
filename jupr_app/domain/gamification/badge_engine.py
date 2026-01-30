from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime
import logging
from typing import Any

from jupr_app.domain.gamification.badge_registry import is_badge_active, registry
from jupr_app.domain.gamification.badge_types import BadgeCandidate
from jupr_app.domain.gamification.evaluators import build_evaluation_context


logger = logging.getLogger(__name__)


def compute_candidates_for_player(
    club_id: str,
    player_id: int,
    league_id: str | None = None,
    as_of: datetime | None = None,
    ctx: Any | None = None,
    *,
    status: str = "live",
    award_timing: str = "live",
) -> list[BadgeCandidate]:
    if ctx is None:
        raise ValueError("compute_candidates_for_player requires a context with match data")
    evaluation = build_evaluation_context(ctx, club_id, league_id, as_of)
    candidates: list[BadgeCandidate] = []
    for spec in registry().values():
        if not is_badge_active(spec.badge_id, status=status, award_timing=award_timing):
            continue
        try:
            for candidate in spec.evaluator(evaluation):
                if int(candidate.player_id) != int(player_id):
                    continue
                candidates.append(candidate)
        except Exception:
            logger.exception("Badge evaluator failed for %s", spec.badge_id)
    return candidates


def compute_candidates_for_club(
    club_id: str,
    league_id: str | None = None,
    as_of: datetime | None = None,
    ctx: Any | None = None,
    *,
    status: str = "live",
    award_timing: str = "live",
) -> Iterator[BadgeCandidate]:
    if ctx is None:
        raise ValueError("compute_candidates_for_club requires a context with match data")
    evaluation = build_evaluation_context(ctx, club_id, league_id, as_of)
    for spec in registry().values():
        if not is_badge_active(spec.badge_id, status=status, award_timing=award_timing):
            continue
        try:
            yield from spec.evaluator(evaluation)
        except Exception:
            logger.exception("Badge evaluator failed for %s", spec.badge_id)
