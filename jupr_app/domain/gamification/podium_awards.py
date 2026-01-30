from __future__ import annotations

"""DEPRECATED: league podium badges are awarded via the badge engine."""

from jupr_app.domain.gamification.ensure_badges import ensure_badges


def ensure_podium_awards_exist(ctx, league_id: str) -> None:
    """DEPRECATED: use ensure_badges (badge engine) instead."""
    ensure_badges(ctx, league_id=league_id, status="seasonal", award_timing="on_league_close")


def award_league_podium_badges(ctx, league_id: str) -> None:
    """DEPRECATED: use ensure_badges (badge engine) instead."""
    ensure_badges(ctx, league_id=league_id, status="seasonal", award_timing="on_league_close")
