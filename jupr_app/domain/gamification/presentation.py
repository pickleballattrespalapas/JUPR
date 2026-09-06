"""One player-facing category and earning requirement for every badge."""
import re

from jupr_app.domain.gamification.requirements import requirement_for

CATEGORY_ORDER = ("Participation", "Improvement", "Partnerships", "Match Achievements", "Trophies")
_GROUPS = {
    "Participation": "participant dedicated_participant_50 lifetime_participant_200 first_win weekly_regular iron_week marathon_month battle_tested consistency swiss_army_knife",
    "Improvement": "level_up rocket_start most_improved_monthly mountain_climber breakthrough bounce_back",
    "Partnerships": "social_butterfly network_builder draft_master good_sport community_builder mentor",
    "Trophies": "league_champion league_runner_up league_third_place podium tournament_champion tournament_runner_up tournament_third_place top_performer_highest_rating top_performer_most_improved top_performer_best_win_pct top_performer_most_wins",
}
_CATEGORIES = {badge_id: category for category, ids in _GROUPS.items() for badge_id in ids.split()}


def badge_category(badge_id: str) -> str:
    return _CATEGORIES.get(str(badge_id), "Match Achievements")


def category_sort_key(category: str) -> tuple[int, str]:
    return (CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else len(CATEGORY_ORDER), category)


def badge_requirement(badge_id: str) -> str:
    text = requirement_for(str(badge_id))
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return re.sub(r"\s+", " ", re.sub(r"[*_`#]", "", text)).strip()
