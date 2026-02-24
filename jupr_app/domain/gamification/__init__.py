from __future__ import annotations

from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS, BadgeDefinition


def ensure_badges(*args, **kwargs):
    from jupr_app.domain.gamification.ensure_badges import ensure_badges as _ensure_badges

    return _ensure_badges(*args, **kwargs)


def build_gamification_summary(*args, **kwargs):
    from jupr_app.domain.gamification.profile import build_gamification_summary as _build_gamification_summary

    return _build_gamification_summary(*args, **kwargs)


def ensure_player_stories(*args, **kwargs):
    from jupr_app.domain.gamification.story_engine import ensure_player_stories as _ensure_player_stories

    return _ensure_player_stories(*args, **kwargs)


__all__ = [
    "BADGE_DEFINITIONS",
    "BadgeDefinition",
    "build_gamification_summary",
    "ensure_badges",
    "ensure_player_stories",
]
