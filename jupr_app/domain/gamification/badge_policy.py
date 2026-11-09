"""Product policy: engagement badges are retired; formal trophies remain.

Changing deployment flags cannot accidentally restart badge computation.
Historical records and the pure evaluators are retained for archival use.
"""

BADGES_ENABLED = False

FORMAL_TROPHY_BADGE_IDS = frozenset({
    "podium", "tournament_champion", "tournament_runner_up", "tournament_third_place",
    "league_champion", "league_runner_up", "league_third_place",
    "top_performer_highest_rating", "top_performer_most_improved",
    "top_performer_best_win_pct", "top_performer_most_wins",
})


def badges_enabled() -> bool:
    return BADGES_ENABLED


def is_formal_trophy_candidate(candidate) -> bool:
    """Keep only explicit top-three league-end or tournament trophy awards."""
    if candidate.badge_id not in FORMAL_TROPHY_BADGE_IDS:
        return False
    if candidate.context_type not in {
        "league", "league_award", "league_end", "league_end_award", "season_award",
        "tournament", "tournament_award", "tournament_podium", "podium",
    }:
        return False
    values = candidate.value_json or {}
    for key in ("placement", "podium_place", "rank"):
        value = values.get(key)
        if value is not None:
            try:
                return not isinstance(value, bool) and float(value) in {1.0, 2.0, 3.0}
            except (TypeError, ValueError):
                return False
    if candidate.badge_id.startswith("top_performer_"):
        # Historical top-performer contexts encode the award rank last.
        suffix = str(candidate.context_id or "").rsplit(":", 1)[-1]
        return suffix in {"1", "2", "3"}
    if candidate.badge_id == "podium":
        return candidate.value_num in {1, 2, 3}
    return True
