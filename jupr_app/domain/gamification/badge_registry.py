from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS
from jupr_app.domain.gamification.badge_schema import BadgeDefinitionSchema, load_badge_definitions
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
    metric_source_policy: str = "match_facts_canonical"
    match_source_policy: str = "canonical_only"


def registry() -> dict[str, BadgeSpec]:
    specs = [
        BadgeSpec("participant", evaluate_participant, "overall", False, "standings_overall"),
        BadgeSpec("dedicated_participant_50", evaluate_dedicated_participant_50, "overall", False, "standings_overall"),
        BadgeSpec("lifetime_participant_200", evaluate_lifetime_participant_200, "overall", False, "standings_overall"),
        BadgeSpec("first_win", evaluate_first_win, "overall", False, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("weekly_regular", evaluate_weekly_regular, "league", False, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("iron_week", evaluate_iron_week, "week", True, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("marathon_month", evaluate_marathon_month, "month", True, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("level_up", evaluate_level_up, "league", True, "league_ratings", "non_match"),
        BadgeSpec("rocket_start", evaluate_rocket_start, "league", False, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("most_improved_monthly", evaluate_most_improved_monthly, "month", True, "match_facts_canonical", "canonical_only"),
        BadgeSpec("mountain_climber", evaluate_mountain_climber, "league", True, "league_ratings", "non_match"),
        BadgeSpec("hot_streak", evaluate_hot_streak, "league", True, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("bounce_back", evaluate_bounce_back, "match", True, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("clutch_performer", evaluate_clutch_performer, "overall", False),
        BadgeSpec("ice_in_veins", evaluate_ice_in_veins, "overall", False, "match_facts_canonical", "canonical_only"),
        BadgeSpec("pickle_perfection", evaluate_pickle_perfection, "match", True, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("blowout_artist", evaluate_blowout_artist, "match", True, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("untouchable", evaluate_untouchable, "overall", True, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("clean_sweep_week", evaluate_clean_sweep_week, "week", True, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("high_roller", evaluate_high_roller, "overall", True, "standings_overall"),
        BadgeSpec("social_butterfly", evaluate_social_butterfly, "overall", False, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("network_builder", evaluate_network_builder, "overall", False, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("draft_master", evaluate_draft_master, "week", True, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("swiss_army_knife", evaluate_swiss_army_knife, "season", True, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("giant_slayer", evaluate_giant_slayer, "match", True, "match_facts_canonical", "canonical_only"),
        BadgeSpec("david_vs_goliath", evaluate_david_vs_goliath, "match", True, "match_facts_canonical", "canonical_only"),
        BadgeSpec("upset_champion", evaluate_upset_champion, "match", True, "match_facts_canonical", "canonical_only"),
        BadgeSpec("hall_of_fame_night", evaluate_hall_of_fame_night, "league", True, "match_facts_canonical", "canonical_only"),
        BadgeSpec("legendary_upset", evaluate_legendary_upset, "match", True, "match_facts_canonical", "canonical_only"),
        BadgeSpec("nemesis_found", evaluate_nemesis_found, "opponent", False),
        BadgeSpec("rivalry_win", evaluate_rivalry_win, "match", True),
        BadgeSpec("rivalry_streak", evaluate_rivalry_streak, "opponent", True),
        BadgeSpec("settled_the_score", evaluate_settled_the_score, "opponent", True),
        BadgeSpec("steady_hand", evaluate_steady_hand, "season", True, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("mr_reliable", evaluate_mr_reliable, "season", True),
        BadgeSpec("league_champion", evaluate_league_champion, "season", True),
        BadgeSpec("podium", evaluate_podium, "season", True),
        BadgeSpec("good_sport", evaluate_good_sport, "season", True),
        BadgeSpec("community_builder", evaluate_community_builder, "season", True),
        BadgeSpec("mentor", evaluate_mentor, "season", True),
        BadgeSpec("breakthrough", evaluate_breakthrough, "overall", False, "league_ratings", "non_match"),
        BadgeSpec("above_expectations", evaluate_above_expectations, "overall", True, "match_facts_canonical", "canonical_only"),
        BadgeSpec("dominant_run", evaluate_dominant_run, "league", True, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("high_output", evaluate_high_output, "match", True, "match_facts_hybrid", "hybrid_safe"),
        BadgeSpec("battle_tested", evaluate_battle_tested, "season", True),
        BadgeSpec("consistency", evaluate_consistency, "season", True),
        BadgeSpec("top_performer_highest_rating", evaluate_top_performer_highest_rating, "league", True, "league_ratings", "non_match"),
        BadgeSpec("top_performer_most_improved", evaluate_top_performer_most_improved, "league", True, "league_ratings", "non_match"),
        BadgeSpec("top_performer_best_win_pct", evaluate_top_performer_best_win_pct, "league", True, "league_ratings", "non_match"),
        BadgeSpec("top_performer_most_wins", evaluate_top_performer_most_wins, "league", True, "league_ratings", "non_match"),
        BadgeSpec("tournament_champion", evaluate_tournament_champion, "tournament", False, "non_match", "non_match"),
        BadgeSpec("tournament_runner_up", evaluate_tournament_runner_up, "tournament", False, "non_match", "non_match"),
        BadgeSpec("tournament_third_place", evaluate_tournament_third_place, "tournament", False, "non_match", "non_match"),
    ]
    return {spec.badge_id: spec for spec in specs}


def all_badge_ids() -> list[str]:
    return list(registry().keys())


_CANONICAL_BADGES: list[BadgeDefinitionSchema] | None = None


def _build_rule_map() -> dict[str, dict[str, object]]:
    return {
        spec.badge_id: {
            "rule": spec.evaluator.__name__,
            "params": {
                "context_type": spec.context_type,
                "is_stackable": spec.is_stackable,
                "metric_source_policy": spec.metric_source_policy,
                "match_source_policy": spec.match_source_policy,
            },
        }
        for spec in registry().values()
    }


def canonical_badge_definitions() -> list[BadgeDefinitionSchema]:
    global _CANONICAL_BADGES
    if _CANONICAL_BADGES is None:
        _CANONICAL_BADGES = load_badge_definitions(BADGE_DEFINITIONS, rules=_build_rule_map())
    return list(_CANONICAL_BADGES)


def badge_schema_by_id() -> dict[str, BadgeDefinitionSchema]:
    return {badge.id: badge for badge in canonical_badge_definitions()}


def awardable_badge_ids(*, status: str = "live", award_timing: str = "live") -> set[str]:
    return {
        badge.id
        for badge in canonical_badge_definitions()
        if badge.status == status and badge.award_timing == award_timing
    }


def active_badge_ids() -> set[str]:
    return awardable_badge_ids(status="live", award_timing="live")


def is_badge_active(badge_id: str, *, status: str = "live", award_timing: str = "live") -> bool:
    badge = badge_schema_by_id().get(str(badge_id))
    if badge is None:
        return False
    return badge.status == status and badge.award_timing == award_timing
