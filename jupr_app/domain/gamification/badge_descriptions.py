from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.presentation import badge_requirement

BADGE_DESCRIPTIONS_MD = {badge.badge_id: badge_requirement(badge.badge_id) for badge in BADGE_DEFINITIONS}
