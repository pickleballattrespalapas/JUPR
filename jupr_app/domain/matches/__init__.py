from .normalization import as_player_id, extract_scores, normalize_rating_scope, collect_seed_candidates
from .rating_engine import is_popup_match, should_update_island, compute_team_deltas, compute_outcomes
from .persistence import build_match_row, insert_match_chunks_with_rating_scope_fallback
from .side_effects import run_badge_side_effects, queue_player_updates

__all__ = [
    "as_player_id",
    "extract_scores",
    "normalize_rating_scope",
    "collect_seed_candidates",
    "is_popup_match",
    "should_update_island",
    "compute_team_deltas",
    "compute_outcomes",
    "build_match_row",
    "insert_match_chunks_with_rating_scope_fallback",
    "run_badge_side_effects",
    "queue_player_updates",
]
