from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import replace
from datetime import datetime
import logging
from typing import Any

from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
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
    allow_non_live: bool = False,
    strict: bool = False,
) -> list[BadgeCandidate]:
    if ctx is None:
        raise ValueError("compute_candidates_for_player requires a context with match data")
    evaluation = build_evaluation_context(ctx, club_id, league_id, as_of)
    state_map = _badge_state_map(ctx)
    candidates: list[BadgeCandidate] = []
    for spec in registry().values():
        if not allow_non_live and not _is_badge_live(spec.badge_id, state_map):
            continue
        if not is_badge_active(spec.badge_id, status=status, award_timing=award_timing):
            continue
        eval_for_badge = _context_for_spec(evaluation, spec.metric_source_policy)
        try:
            for candidate in spec.evaluator(eval_for_badge):
                if int(candidate.player_id) != int(player_id):
                    continue
                candidates.append(candidate)
        except Exception:
            if strict:
                raise
            logger.exception(
                "Badge evaluator failed for %s (club_id=%s league_id=%s)",
                spec.badge_id,
                club_id,
                league_id,
                extra={"badge_id": spec.badge_id, "club_id": club_id, "league_id": league_id},
            )
    return candidates


def compute_candidates_for_club(
    club_id: str,
    league_id: str | None = None,
    as_of: datetime | None = None,
    ctx: Any | None = None,
    *,
    status: str = "live",
    award_timing: str = "live",
    allow_non_live: bool = False,
    strict: bool = False,
) -> Iterator[BadgeCandidate]:
    if ctx is None:
        raise ValueError("compute_candidates_for_club requires a context with match data")
    evaluation = build_evaluation_context(ctx, club_id, league_id, as_of)
    state_map = _badge_state_map(ctx)
    for spec in registry().values():
        if not allow_non_live and not _is_badge_live(spec.badge_id, state_map):
            continue
        if not is_badge_active(spec.badge_id, status=status, award_timing=award_timing):
            continue
        eval_for_badge = _context_for_spec(evaluation, spec.metric_source_policy)
        try:
            yield from spec.evaluator(eval_for_badge)
        except Exception:
            if strict:
                raise
            logger.exception(
                "Badge evaluator failed for %s (club_id=%s league_id=%s)",
                spec.badge_id,
                club_id,
                league_id,
                extra={"badge_id": spec.badge_id, "club_id": club_id, "league_id": league_id},
            )


def _badge_state_map(ctx: Any | None) -> dict[str, str]:
    if ctx is None:
        return {badge.badge_id: badge.state if badge.is_active else "deprecated" for badge in BADGE_DEFINITIONS}
    df_badges = getattr(ctx, "df_badges", None)
    if df_badges is None or getattr(df_badges, "empty", True):
        return {badge.badge_id: badge.state if badge.is_active else "deprecated" for badge in BADGE_DEFINITIONS}
    if "badge_id" not in df_badges.columns:
        return {badge.badge_id: badge.state if badge.is_active else "deprecated" for badge in BADGE_DEFINITIONS}
    return {
        str(row.badge_id): "deprecated" if getattr(row, "is_active", True) == False else str(getattr(row, "state", "") or "live")
        for row in df_badges.itertuples(index=False)
    }


def _is_badge_live(badge_id: str, state_map: dict[str, str]) -> bool:
    state = state_map.get(str(badge_id), "live")
    if not state:
        return True
    return str(state).strip().lower() == "live"


def _context_for_spec(evaluation, metric_source_policy: str):
    if metric_source_policy == "match_facts_hybrid":
        return replace(evaluation, facts=evaluation.facts_hybrid if evaluation.facts_hybrid is not None else evaluation.facts)
    if metric_source_policy == "match_facts_canonical":
        return replace(evaluation, facts=evaluation.facts_canonical if evaluation.facts_canonical is not None else evaluation.facts)
    if metric_source_policy in {"standings_overall", "league_ratings", "non_match"}:
        return replace(evaluation, facts=evaluation.facts_canonical if evaluation.facts_canonical is not None else evaluation.facts)
    return replace(evaluation, facts=evaluation.facts_canonical if evaluation.facts_canonical is not None else evaluation.facts)
