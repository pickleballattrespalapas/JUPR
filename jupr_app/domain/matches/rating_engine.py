from __future__ import annotations

from jupr_app.domain.ratings import calculate_hybrid_elo


def is_popup_match(match_type: str, is_popup_flag: bool) -> bool:
    return bool(is_popup_flag) or (match_type == "PopUp")


def should_update_island(*, is_popup: bool, rating_scope: str) -> bool:
    return (not is_popup) and (rating_scope != "overall_only") and (rating_scope != "unrated")


def compute_team_deltas(r1: float, r2: float, s1: int, s2: int, *, k_factor: float, min_win_delta: float, cap_loser_gain: float | None):
    return calculate_hybrid_elo(
        r1,
        r2,
        s1,
        s2,
        k_factor=float(k_factor),
        min_win_delta=float(min_win_delta),
        cap_loser_gain=cap_loser_gain,
    )


def compute_outcomes(s1: int, s2: int):
    if s1 == s2:
        return None, None
    return s1 > s2, s2 > s1
