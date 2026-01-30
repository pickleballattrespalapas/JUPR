from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.badge_types import BadgeCandidate, BadgeEvaluationContext
from jupr_app.domain.gamification.evaluators import (
    evaluate_above_expectations,
    evaluate_battle_tested,
    evaluate_blowout_artist,
    evaluate_bounce_back,
    evaluate_breakthrough,
    evaluate_clean_sweep_week,
    evaluate_clutch_performer,
    evaluate_community_builder,
    evaluate_consistency,
    evaluate_david_vs_goliath,
    evaluate_dominant_run,
    evaluate_draft_master,
    evaluate_dedicated_participant_50,
    evaluate_first_win,
    evaluate_giant_slayer,
    evaluate_good_sport,
    evaluate_hall_of_fame_night,
    evaluate_high_output,
    evaluate_high_roller,
    evaluate_hot_streak,
    evaluate_ice_in_veins,
    evaluate_iron_week,
    evaluate_league_champion,
    evaluate_legendary_upset,
    evaluate_lifetime_participant_200,
    evaluate_level_up,
    evaluate_marathon_month,
    evaluate_mentor,
    evaluate_most_improved_monthly,
    evaluate_mountain_climber,
    evaluate_mr_reliable,
    evaluate_network_builder,
    evaluate_nemesis_found,
    evaluate_participant,
    evaluate_pickle_perfection,
    evaluate_podium,
    evaluate_rocket_start,
    evaluate_rivalry_streak,
    evaluate_rivalry_win,
    evaluate_settled_the_score,
    evaluate_social_butterfly,
    evaluate_steady_hand,
    evaluate_swiss_army_knife,
    evaluate_top_performer_best_win_pct,
    evaluate_top_performer_highest_rating,
    evaluate_top_performer_most_improved,
    evaluate_top_performer_most_wins,
    evaluate_tournament_champion,
    evaluate_tournament_runner_up,
    evaluate_tournament_third_place,
    evaluate_untouchable,
    evaluate_upset_champion,
    evaluate_weekly_regular,
)


Evaluator = Callable[[BadgeEvaluationContext], Iterable[BadgeCandidate]]


@dataclass(frozen=True)
class BadgeSpec:
    badge_id: str
    evaluator: Evaluator
    context_type: str
    is_stackable: bool


def registry() -> dict[str, BadgeSpec]:
    specs = [
        BadgeSpec("participant", evaluate_participant, "overall", False),
        BadgeSpec("dedicated_participant_50", evaluate_dedicated_participant_50, "overall", False),
        BadgeSpec("lifetime_participant_200", evaluate_lifetime_participant_200, "overall", False),
        BadgeSpec("first_win", evaluate_first_win, "overall", False),
        BadgeSpec("weekly_regular", evaluate_weekly_regular, "league", False),
        BadgeSpec("iron_week", evaluate_iron_week, "week", True),
        BadgeSpec("marathon_month", evaluate_marathon_month, "month", True),
        BadgeSpec("level_up", evaluate_level_up, "league", True),
        BadgeSpec("rocket_start", evaluate_rocket_start, "league", False),
        BadgeSpec("most_improved_monthly", evaluate_most_improved_monthly, "month", True),
        BadgeSpec("mountain_climber", evaluate_mountain_climber, "league", True),
        BadgeSpec("hot_streak", evaluate_hot_streak, "league", True),
        BadgeSpec("bounce_back", evaluate_bounce_back, "match", True),
        BadgeSpec("clutch_performer", evaluate_clutch_performer, "overall", False),
        BadgeSpec("ice_in_veins", evaluate_ice_in_veins, "overall", False),
        BadgeSpec("pickle_perfection", evaluate_pickle_perfection, "match", True),
        BadgeSpec("blowout_artist", evaluate_blowout_artist, "match", True),
        BadgeSpec("untouchable", evaluate_untouchable, "overall", True),
        BadgeSpec("clean_sweep_week", evaluate_clean_sweep_week, "week", True),
        BadgeSpec("high_roller", evaluate_high_roller, "match", True),
        BadgeSpec("social_butterfly", evaluate_social_butterfly, "overall", False),
        BadgeSpec("network_builder", evaluate_network_builder, "overall", False),
        BadgeSpec("draft_master", evaluate_draft_master, "month", True),
        BadgeSpec("swiss_army_knife", evaluate_swiss_army_knife, "season", True),
        BadgeSpec("giant_slayer", evaluate_giant_slayer, "match", True),
        BadgeSpec("david_vs_goliath", evaluate_david_vs_goliath, "match", True),
        BadgeSpec("upset_champion", evaluate_upset_champion, "match", True),
        BadgeSpec("hall_of_fame_night", evaluate_hall_of_fame_night, "league", True),
        BadgeSpec("legendary_upset", evaluate_legendary_upset, "match", True),
        BadgeSpec("nemesis_found", evaluate_nemesis_found, "opponent", False),
        BadgeSpec("rivalry_win", evaluate_rivalry_win, "match", True),
        BadgeSpec("rivalry_streak", evaluate_rivalry_streak, "opponent", True),
        BadgeSpec("settled_the_score", evaluate_settled_the_score, "opponent", True),
        BadgeSpec("steady_hand", evaluate_steady_hand, "season", True),
        BadgeSpec("mr_reliable", evaluate_mr_reliable, "season", True),
        BadgeSpec("league_champion", evaluate_league_champion, "season", True),
        BadgeSpec("podium", evaluate_podium, "season", True),
        BadgeSpec("good_sport", evaluate_good_sport, "season", True),
        BadgeSpec("community_builder", evaluate_community_builder, "season", True),
        BadgeSpec("mentor", evaluate_mentor, "season", True),
        BadgeSpec("breakthrough", evaluate_breakthrough, "overall", False),
        BadgeSpec("above_expectations", evaluate_above_expectations, "overall", True),
        BadgeSpec("dominant_run", evaluate_dominant_run, "league", True),
        BadgeSpec("high_output", evaluate_high_output, "match", True),
        BadgeSpec("battle_tested", evaluate_battle_tested, "season", True),
        BadgeSpec("consistency", evaluate_consistency, "season", True),
        BadgeSpec("top_performer_highest_rating", evaluate_top_performer_highest_rating, "league", True),
        BadgeSpec("top_performer_most_improved", evaluate_top_performer_most_improved, "league", True),
        BadgeSpec("top_performer_best_win_pct", evaluate_top_performer_best_win_pct, "league", True),
        BadgeSpec("top_performer_most_wins", evaluate_top_performer_most_wins, "league", True),
        BadgeSpec("tournament_champion", evaluate_tournament_champion, "tournament", False),
        BadgeSpec("tournament_runner_up", evaluate_tournament_runner_up, "tournament", False),
        BadgeSpec("tournament_third_place", evaluate_tournament_third_place, "tournament", False),
    ]
    return {spec.badge_id: spec for spec in specs}


def all_badge_ids() -> list[str]:
    return list(registry().keys())


_BADGE_DEFINITIONS_BY_ID = {b.badge_id: b for b in BADGE_DEFINITIONS}


def active_badge_ids() -> set[str]:
    return {b.badge_id for b in _BADGE_DEFINITIONS_BY_ID.values() if b.is_active}


def is_badge_active(badge_id: str) -> bool:
    badge = _BADGE_DEFINITIONS_BY_ID.get(badge_id)
    if badge is None:
        return True
    return bool(badge.is_active)
