from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.badge_engine import compute_candidates_for_club
from jupr_app.domain.gamification.badge_state import can_transition_badge_state


def _build_ctx(state: str):
    df_matches = pd.DataFrame(
        [
            {
                "id": 1,
                "club_id": "club",
                "date": "2024-01-01T00:00:00Z",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 4,
            }
        ]
    )
    df_badges = pd.DataFrame([{"badge_id": "participant", "state": state}])
    return SimpleNamespace(
        club_id="club",
        df_matches=df_matches,
        df_players_all=pd.DataFrame(),
        df_badges=df_badges,
    )


def _has_participant(candidates):
    return any(c.badge_id == "participant" for c in candidates)


def test_frozen_badges_not_awarded():
    ctx = _build_ctx("frozen")
    candidates = list(compute_candidates_for_club("club", ctx=ctx))
    assert not _has_participant(candidates)


def test_deprecated_badges_not_awarded():
    ctx = _build_ctx("deprecated")
    candidates = list(compute_candidates_for_club("club", ctx=ctx))
    assert not _has_participant(candidates)


def test_badge_state_transitions():
    assert can_transition_badge_state("live", "frozen").allowed
    assert can_transition_badge_state("frozen", "deprecated").allowed
    assert not can_transition_badge_state("live", "deprecated").allowed
    assert not can_transition_badge_state("deprecated", "frozen").allowed
    assert not can_transition_badge_state("live", "live").allowed
    assert can_transition_badge_state("deprecated", "live", force=True).allowed
