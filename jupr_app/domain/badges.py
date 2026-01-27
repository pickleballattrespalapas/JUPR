from __future__ import annotations

from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS, BadgeDefinition
from jupr_app.domain.gamification.badge_rules import ensure_badges

__all__ = ["BADGE_DEFINITIONS", "BadgeDefinition", "ensure_badges"]
