from copy import deepcopy

import pytest

from jupr_app.services.league_live_score_readback import read_league_round_scores
from tests.test_admin_match_log_service import FakeSupabase


def fixture():
    games = [
        {"context_id": f"context-{i}", "court": 7, "series_key": f"slot-{i}", "game_number": 1,
         "t1_p1": 1, "t1_p2": partner, "t2_p1": opponents[0], "t2_p2": opponents[1],
         "score_t1": 11 if i != 2 else 9, "score_t2": 8 if i != 2 else 11}
        for i, (partner, opponents) in enumerate([(2, [3, 4]), (3, [2, 4]), (4, [2, 3])])
    ]
    saved = {"status": "submitted", "matches_json": games, "submitted_match_ids": [101, 102, 103]}
    official = [{**game, "id": 101 + i, "club_id": "club", "context_type": "league_live_session", "row_version": 1} for i, game in enumerate(games)]
    return saved, official


def test_corrected_scores_are_current_without_changing_publish_snapshot():
    saved, official = fixture()
    retained = deepcopy(saved)
    official[2].update(score_t1=11, score_t2=9, row_version=2)
    # Database order and submitted ID order must never identify the game.
    db = FakeSupabase({"matches": list(reversed(official))})
    result = read_league_round_scores(db, club_id="club", round_row=saved)
    assert result["score_readback_status"] == "current"
    assert result["score_correction_count"] == 1
    assert sum(row["score_t1"] > row["score_t2"] for row in result["matches_json"]) == 3
    assert result["matches_json"][2]["official_match_id"] == 103
    assert result["matches_json"][2]["official_row_version"] == 2
    assert saved == retained
    assert db.operations == []


@pytest.mark.parametrize("damage", ["missing", "duplicate", "wrong_club", "wrong_context", "wrong_id"])
def test_missing_or_ambiguous_official_results_are_not_silently_trusted(damage):
    saved, official = fixture()
    if damage == "missing": official.pop()
    elif damage == "duplicate": official.append(dict(official[0]))
    elif damage == "wrong_club": official[0]["club_id"] = "another-club"
    elif damage == "wrong_context": official[0]["context_type"] = "tournament"
    else: official[0]["id"] = 999
    result = read_league_round_scores(FakeSupabase({"matches": official}), club_id="club", round_row=saved)
    assert result["score_readback_status"] == "unavailable"
    assert "could not be verified" in result["score_readback_message"]
    assert all("official_match_id" not in row for row in result["matches_json"])


@pytest.mark.parametrize("patch", [{"is_active": False}, {"deleted_at": "2026-10-25"}, {"score_t1": None}, {"score_t1": 8}])
def test_excluded_or_invalid_scores_block_new_movement(patch):
    saved, official = fixture()
    official[0].update(patch)
    result = read_league_round_scores(FakeSupabase({"matches": official}), club_id="club", round_row=saved)
    assert result["score_readback_status"] == "blocked"
    assert result["matches_json"][0]["score_excluded"] is True


def test_old_rounds_without_official_references_are_labelled_as_saved_scores():
    saved, _ = fixture()
    for row in saved["matches_json"]: row.pop("context_id")
    result = read_league_round_scores(FakeSupabase({}), club_id="club", round_row=saved)
    assert result["score_readback_status"] == "snapshot"
    assert "older round" in result["score_readback_message"]
