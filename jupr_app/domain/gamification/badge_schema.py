from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Literal

from jupr_app.domain.gamification.badge_catalog import BADGE_DEFINITIONS, BadgeDefinition
from jupr_app.domain.gamification.requirements import load_requirements_map


BadgeStatus = Literal["live", "tracked", "seasonal", "curated", "retired"]
BadgeScope = Literal["match", "week", "month", "season", "league", "lifetime"]
AwardTiming = Literal["live", "on_league_close", "manual", "disabled"]


@dataclass(frozen=True)
class BadgeDisplay:
    requirements: str
    flavor: str


@dataclass(frozen=True)
class BadgeCompute:
    rule: str
    params: dict[str, Any]
    internal_terms_allowed: bool = False


@dataclass(frozen=True)
class BadgeDefinitionSchema:
    id: str
    title: str
    prestige: int
    status: BadgeStatus
    scope: BadgeScope
    award_timing: AwardTiming
    display: BadgeDisplay
    compute: BadgeCompute


def _build_badge_metadata() -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}

    def assign(badge_ids: Iterable[str], *, status: BadgeStatus, scope: BadgeScope, award_timing: AwardTiming) -> None:
        for badge_id in badge_ids:
            metadata[str(badge_id)] = {
                "status": status,
                "scope": scope,
                "award_timing": award_timing,
            }

    assign(
        [
            "participant",
            "dedicated_participant_50",
            "lifetime_participant_200",
            "first_win",
        ],
        status="live",
        scope="lifetime",
        award_timing="live",
    )
    assign(["weekly_regular"], status="live", scope="league", award_timing="live")
    assign(["iron_week", "clean_sweep_week"], status="live", scope="week", award_timing="live")
    assign(["marathon_month", "most_improved_monthly", "upset_champion"], status="live", scope="month", award_timing="live")
    assign(
        ["level_up", "rocket_start", "mountain_climber", "hot_streak"],
        status="live",
        scope="league",
        award_timing="live",
    )
    assign(
        [
            "bounce_back",
            "pickle_perfection",
            "blowout_artist",
            "giant_slayer",
            "david_vs_goliath",
            "legendary_upset",
        ],
        status="live",
        scope="match",
        award_timing="live",
    )
    assign(["ice_in_veins", "social_butterfly", "network_builder", "untouchable", "high_roller"], status="live", scope="lifetime", award_timing="live")
    assign(["draft_master"], status="live", scope="week", award_timing="live")
    assign(["swiss_army_knife", "steady_hand"], status="live", scope="season", award_timing="live")
    assign(["hall_of_fame_night"], status="live", scope="match", award_timing="live")
    assign(
        [
            "tournament_champion",
            "tournament_runner_up",
            "tournament_third_place",
        ],
        status="live",
        scope="season",
        award_timing="manual",
    )

    assign(
        [
            "top_performer_highest_rating",
            "top_performer_most_improved",
            "top_performer_best_win_pct",
            "top_performer_most_wins",
            "league_champion",
            "league_runner_up",
            "league_third_place",
            "podium",
        ],
        status="seasonal",
        scope="season",
        award_timing="on_league_close",
    )

    assign(["breakthrough", "above_expectations", "clutch_performer"], status="tracked", scope="lifetime", award_timing="disabled")
    assign(["dominant_run"], status="tracked", scope="league", award_timing="disabled")
    assign(["high_output", "rivalry_win"], status="tracked", scope="match", award_timing="disabled")
    assign(["nemesis_found", "rivalry_streak", "settled_the_score"], status="tracked", scope="lifetime", award_timing="disabled")
    assign(["battle_tested", "consistency", "mr_reliable"], status="tracked", scope="season", award_timing="disabled")
    assign(["good_sport", "community_builder"], status="curated", scope="lifetime", award_timing="manual")
    assign(["mentor"], status="curated", scope="match", award_timing="manual")

    return metadata


BADGE_METADATA = _build_badge_metadata()

_VALID_STATUS = {"live", "tracked", "seasonal", "curated", "retired"}
_VALID_SCOPE = {"match", "week", "month", "season", "league", "lifetime"}
_VALID_AWARD_TIMING = {"live", "on_league_close", "manual", "disabled"}

_REQUIRED_BADGE_TITLES = [
    "Level Up",
    "Most Improved",
    "Mountain Climber",
    "Hot Streak",
    "Untouchable",
    "Weekly Regular",
    "Iron Week",
    "Clean Sweep Week",
    "High Roller",
    "Draft Master",
    "Giant Slayer",
    "Participant",
    "Dedicated Participant",
    "Lifetime Participant",
    "First Win",
    "Marathon Month",
    "Rocket Start",
    "Blowout Artist",
    "Pickle Perfection",
    "Bounce Back",
    "Ice in Veins",
    "David vs Goliath",
    "Legendary Upset",
    "Upset Champion",
    "Hall of Fame Night",
    "Social Butterfly",
    "Network Builder",
    "Nemesis Found",
    "Rivalry Win",
    "Rivalry Streak",
    "Settled the Score",
    "Swiss Army Knife",
    "Steady Hand",
    "Tournament Champion",
    "Tournament Runner-Up",
    "Tournament Third Place",
    "Top Performer: Highest Rating",
    "Top Performer: Most Wins",
    "Top Performer: Best Win %",
    "Top Performer: Most Improved",
]


def _validate_requirements(requirements: dict[str, str], badges: list[BadgeDefinition]) -> None:
    errors: list[str] = []
    badge_titles = {badge.name for badge in badges}
    missing_titles = [title for title in _REQUIRED_BADGE_TITLES if title not in badge_titles]
    if missing_titles:
        errors.append(f"missing badge titles: {', '.join(missing_titles)}")

    bad_requirements = [
        badge_id
        for badge_id, text in requirements.items()
        if re.search(r"elo", text or "", flags=re.IGNORECASE)
    ]
    if bad_requirements:
        errors.append(f"requirements contain forbidden term 'Elo': {', '.join(sorted(bad_requirements))}")

    if errors:
        raise ValueError(f"Badge requirement validation failed: {'; '.join(errors)}")


def _map_legacy_scope(scope: str | None) -> BadgeScope | None:
    if not scope:
        return None
    normalized = str(scope).strip().lower()
    if normalized in _VALID_SCOPE:
        return normalized  # type: ignore[return-value]
    if normalized == "overall":
        return "lifetime"
    if normalized == "opponent":
        return "lifetime"
    if normalized == "tournament":
        return "season"
    return None


def load_badge_definitions(
    raw_badges: Iterable[BadgeDefinition] | None = None,
    *,
    requirements_map: dict[str, str] | None = None,
    rules: dict[str, dict[str, Any]] | None = None,
) -> list[BadgeDefinitionSchema]:
    badges = list(raw_badges or BADGE_DEFINITIONS)
    requirements = requirements_map or load_requirements_map()
    rules_map = rules or {}

    _validate_requirements(requirements, badges)

    definitions: list[BadgeDefinitionSchema] = []
    errors: list[str] = []

    for badge in badges:
        badge_id = str(badge.badge_id)
        meta = BADGE_METADATA.get(badge_id, {})
        status = meta.get("status", "live")
        award_timing = meta.get("award_timing", "live")
        scope = meta.get("scope") or _map_legacy_scope(getattr(badge, "scope", None)) or "match"

        requirements_text = str(requirements.get(badge_id, "Requirements TBD") or "Requirements TBD").strip()
        flavor_text = str(getattr(badge, "lore", "") or "").strip()
        rule = str(rules_map.get(badge_id, {}).get("rule", "unregistered"))
        params = dict(rules_map.get(badge_id, {}).get("params", {}) or {})

        if not badge_id or status not in _VALID_STATUS:
            errors.append(f"{badge_id}: invalid status")
        if scope not in _VALID_SCOPE:
            errors.append(f"{badge_id}: invalid scope")
        if award_timing not in _VALID_AWARD_TIMING:
            errors.append(f"{badge_id}: invalid award_timing")
        if not requirements_text:
            errors.append(f"{badge_id}: missing requirements")

        definitions.append(
            BadgeDefinitionSchema(
                id=badge_id,
                title=str(badge.name),
                prestige=int(badge.prestige),
                status=status,  # type: ignore[arg-type]
                scope=scope,  # type: ignore[arg-type]
                award_timing=award_timing,  # type: ignore[arg-type]
                display=BadgeDisplay(requirements=requirements_text, flavor=flavor_text),
                compute=BadgeCompute(rule=rule, params=params, internal_terms_allowed=False),
            )
        )

    if errors:
        raise ValueError(f"Invalid badge definitions: {', '.join(errors)}")

    return definitions
