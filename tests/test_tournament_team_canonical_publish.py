from __future__ import annotations

from copy import deepcopy

import pytest

from jupr_app.domain.tournament_team_canonical_publish import (
    NOT_RATED,
    NOT_READY,
    PUBLISHED,
    READY_TO_PUBLISH,
    RECONCILE_REQUIRED,
    classify_team_child_publish_state,
)


def _child(**patch):
    row = {
        "id": "child-1",
        "counts_for_rating": True,
        "status": "FINAL",
        "tournament_game_id": "game-1",
        "match_format": "DOUBLES",
        "team_a_player_ids": [1, 2],
        "team_b_player_ids": [3, 4],
        "score_a": 11,
        "score_b": 7,
    }
    row.update(patch)
    return row


def _game(**patch):
    row = {
        "id": "game-1",
        "team_match_game_id": "child-1",
        "parent_result_only": False,
        "score_a": 11,
        "score_b": 7,
    }
    row.update(patch)
    return row


def _official(**patch):
    row = {
        "id": "match-1",
        "tournament_game_id": "game-1",
        "match_format": "doubles",
        "t1_p1": 1,
        "t1_p2": 2,
        "t2_p1": 3,
        "t2_p2": 4,
        "score_t1": 11,
        "score_t2": 7,
        "deleted_at": None,
        "excluded_from_ratings": False,
    }
    row.update(patch)
    return row


def _state(
    *,
    child=None,
    game=None,
    official=None,
):
    canonical = [] if official is None else official
    if isinstance(canonical, dict):
        canonical = [canonical]
    return classify_team_child_publish_state(
        child=deepcopy(child or _child()),
        tournament_game=deepcopy(_game() if game is None else game),
        canonical_matches=deepcopy(canonical),
    )


def test_no_canonical_history_is_ready_and_exact_history_is_published():
    assert _state() == READY_TO_PUBLISH
    assert _state(official=_official()) == PUBLISHED


def test_exact_singles_history_is_published():
    child = _child(
        match_format="SINGLES",
        team_a_player_ids=[1],
        team_b_player_ids=[3],
    )
    official = _official(
        match_format="singles",
        t1_p2=None,
        t2_p2=None,
    )

    assert _state(child=child, official=official) == PUBLISHED


@pytest.mark.parametrize(
    "official",
    [
        _official(deleted_at="2026-07-27T00:00:00Z"),
        _official(excluded_from_ratings=True),
        _official(match_format="singles", t1_p2=None, t2_p2=None),
        _official(t1_p2=None),
        _official(score_t1=10),
        _official(
            t1_p1=3,
            t1_p2=4,
            t2_p1=1,
            t2_p2=2,
            score_t1=7,
            score_t2=11,
        ),
        _official(t1_p1=99),
    ],
)
def test_any_deleted_excluded_shape_score_or_side_drift_requires_reconciliation(
    official,
):
    assert _state(official=official) == RECONCILE_REQUIRED


def test_duplicate_history_requires_reconciliation_even_with_one_exact_active_row():
    assert _state(
        official=[
            _official(),
            _official(
                id="match-deleted",
                deleted_at="2026-07-27T00:00:00Z",
            ),
        ]
    ) == RECONCILE_REQUIRED


@pytest.mark.parametrize(
    "game",
    [
        {},
        _game(team_match_game_id="other-child"),
        _game(parent_result_only=True),
        _game(score_a=7, score_b=11),
    ],
)
def test_broken_protected_tournament_game_requires_reconciliation(game):
    assert _state(game=game) == RECONCILE_REQUIRED


@pytest.mark.parametrize(
    "child",
    [
        _child(match_format="UNKNOWN"),
        _child(team_a_player_ids=[1]),
        _child(team_a_player_ids=[1, 1]),
        _child(team_b_player_ids=[2, 4]),
    ],
)
def test_invalid_child_source_shape_requires_reconciliation_without_history(
    child,
):
    assert _state(child=child) == RECONCILE_REQUIRED


def test_nonrated_and_nonfinal_children_never_look_publishable():
    assert _state(child=_child(counts_for_rating=False)) == NOT_RATED
    assert _state(child=_child(status="PENDING")) == NOT_READY
    assert (
        _state(
            child=_child(counts_for_rating=False),
            official=_official(),
        )
        == RECONCILE_REQUIRED
    )
    assert (
        _state(
            child=_child(status="PENDING"),
            official=_official(),
        )
        == RECONCILE_REQUIRED
    )
