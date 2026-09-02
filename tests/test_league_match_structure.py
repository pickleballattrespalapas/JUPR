from __future__ import annotations

import pytest

from jupr_app.domain.league_match_structure import (
    LeagueMatchStructureError,
    normalize_league_match_structure,
    validate_league_series_matches,
)


BASE = {
    "court": 1,
    "t1_p1": 1,
    "t1_p2": 2,
    "t2_p1": 3,
    "t2_p2": 4,
    "series_key": "court-1-match-1",
}


def _game(number: int, score_one: int, score_two: int, *, kind: str, games: int):
    return {
        **BASE,
        "series_kind": kind,
        "series_games": games,
        "game_number": number,
        "score_t1": score_one,
        "score_t2": score_two,
    }


def test_fixed_series_requires_every_game_and_allows_repeated_scores() -> None:
    structure = {"kind": "fixed_games", "games": 2}
    rows = validate_league_series_matches(
        [
            _game(1, 11, 5, kind="fixed_games", games=2),
            _game(2, 11, 5, kind="fixed_games", games=2),
        ],
        match_structure=structure,
    )

    assert len(rows) == 2
    assert [row["game_number"] for row in rows] == [1, 2]
    assert all(row["series_key"] == "court-1-match-1" for row in rows)

    with pytest.raises(LeagueMatchStructureError, match="requires all 2"):
        validate_league_series_matches(
            [_game(1, 11, 5, kind="fixed_games", games=2)],
            match_structure=structure,
        )


def test_best_of_series_stops_at_clinch_and_each_played_game_counts() -> None:
    structure = normalize_league_match_structure({"kind": "best_of", "games": 3})
    assert structure == {
        "kind": "best_of",
        "games": 3,
        "result_counting": "each_game",
        "completion": "clinch",
    }

    swept = validate_league_series_matches(
        [
            _game(1, 11, 7, kind="best_of", games=3),
            _game(2, 11, 9, kind="best_of", games=3),
        ],
        match_structure=structure,
    )
    assert len(swept) == 2

    full = validate_league_series_matches(
        [
            _game(1, 11, 7, kind="best_of", games=3),
            _game(2, 8, 11, kind="best_of", games=3),
            _game(3, 11, 9, kind="best_of", games=3),
        ],
        match_structure=structure,
    )
    assert len(full) == 3

    with pytest.raises(LeagueMatchStructureError, match="after the series was already clinched"):
        validate_league_series_matches(
            [
                _game(1, 11, 7, kind="best_of", games=3),
                _game(2, 11, 9, kind="best_of", games=3),
                _game(3, 6, 11, kind="best_of", games=3),
            ],
            match_structure=structure,
        )


def test_series_rejects_stale_settings_or_changed_teams() -> None:
    with pytest.raises(LeagueMatchStructureError, match="does not match the league settings"):
        validate_league_series_matches(
            [_game(1, 11, 7, kind="best_of", games=3)],
            match_structure={"kind": "fixed_games", "games": 1},
        )

    changed_team = _game(2, 11, 8, kind="fixed_games", games=2)
    changed_team["t2_p2"] = 5
    with pytest.raises(LeagueMatchStructureError, match="changes teams or courts"):
        validate_league_series_matches(
            [
                _game(1, 11, 7, kind="fixed_games", games=2),
                changed_team,
            ],
            match_structure={"kind": "fixed_games", "games": 2},
        )
