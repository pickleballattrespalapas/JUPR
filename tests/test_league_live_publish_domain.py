from __future__ import annotations

import csv
import io

import pytest

from jupr_app.domain.league_live_orchestration import build_league_live_round_plan
from jupr_app.domain.league_live_publish import (
    LeagueLivePublishError,
    build_league_live_publish_request,
    build_rating_review,
    league_live_match_context_id,
    league_live_unusual_score_reason,
    normalize_league_live_publish_matches,
    rows_to_safe_csv,
    validate_league_live_game_slots,
)


MATCH = {
    "court": 1,
    "t1_p1": 1,
    "t1_p2": 2,
    "t2_p1": 3,
    "t2_p2": 4,
    "score_t1": 11,
    "score_t2": 8,
}


def test_publish_requires_every_generated_match() -> None:
    with pytest.raises(LeagueLivePublishError, match="1 of 2"):
        normalize_league_live_publish_matches(
            [MATCH],
            session_id="session-1",
            round_number=2,
            league_name="Tuesday",
            week_tag="Week 1",
            match_date="2026-07-19",
            expected_match_count=2,
        )


def test_publish_contexts_and_fingerprint_are_deterministic() -> None:
    kwargs = {
        "session_id": "session-1",
        "round_number": 2,
        "league_name": "Tuesday",
        "week_tag": "Week 1",
        "match_date": "2026-07-19",
        "matches": [MATCH],
        "expected_match_count": 1,
        "expected_updated_at": "2026-07-19T12:00:00+00:00",
        "expected_operation_key": "a" * 64,
    }
    first = build_league_live_publish_request(**kwargs)
    second = build_league_live_publish_request(**kwargs)

    assert first["request_fingerprint"] == second["request_fingerprint"]
    assert first["match_context_ids"] == second["match_context_ids"]
    assert first["match_context_ids"][0] == league_live_match_context_id(
        session_id="session-1", round_number=2, match_index=1
    )
    assert first["matches"][0]["context_type"] == "league_live_session"


def test_unusual_score_requires_explicit_operator_acknowledgement() -> None:
    unusual = {**MATCH, "score_t1": 11, "score_t2": 76}
    kwargs = {
        "session_id": "session-1",
        "round_number": 2,
        "league_name": "Tuesday",
        "week_tag": "Week 1",
        "match_date": "2026-07-19",
        "matches": [unusual],
        "expected_match_count": 1,
        "expected_updated_at": "2026-07-19T12:00:00+00:00",
        "expected_operation_key": "a" * 64,
    }

    assert league_live_unusual_score_reason(11, 76)
    with pytest.raises(LeagueLivePublishError, match="unusual score 11-76"):
        build_league_live_publish_request(**kwargs)

    request = build_league_live_publish_request(**kwargs, unusual_score_acknowledgement=True)
    assert request["unusual_score_acknowledgement"] is True
    assert request["unusual_score_findings"][0]["score_t2"] == 76


def test_movement_operation_key_ignores_publish_only_metadata() -> None:
    common = {
        "session_id": "session-1",
        "round_number": 1,
        "total_rounds": 2,
        "session_updated_at": "v1",
        "roster": [
            {"player_id": 1, "player_name": "Alex", "rating": 1400},
            {"player_id": 2, "player_name": "Blair", "rating": 1380},
            {"player_id": 3, "player_name": "Casey", "rating": 1360},
            {"player_id": 4, "player_name": "Devon", "rating": 1340},
        ],
        "courts": [
            {
                "court_number": 1,
                "format_type": "4-Player",
                "player_names": ["Alex", "Blair", "Casey", "Devon"],
            }
        ],
    }
    preview = build_league_live_round_plan(matches=[MATCH], **common)
    publish = build_league_live_round_plan(
        matches=[
            {
                **MATCH,
                "date": "2026-07-19",
                "league": "Tuesday",
                "week_tag": "Week 1",
                "match_type": "League Manager Live",
                "context_type": "league_live_session",
                "context_id": "deterministic-context",
            }
        ],
        **common,
    )

    assert publish["operation_key"] == preview["operation_key"]


def test_rating_review_surfaces_missing_readback() -> None:
    review = build_rating_review(
        before_rows=[{"id": 1, "name": "Alex", "rating": 1400, "matches_played": 3}],
        after_rows=[],
        expected_player_ids=[1],
        published_match_count=1,
    )

    assert review["status"] == "review_required"
    assert review["requires_replay_review"] is True
    assert review["warnings"]


def test_csv_export_neutralizes_spreadsheet_formulas() -> None:
    text = rows_to_safe_csv([{"name": "=HYPERLINK(\"https://bad.example\")", "score": 11}])
    row = next(csv.DictReader(io.StringIO(text)))

    assert row["name"].startswith("'=")
    assert row["score"] == "11"


def _flexible_request(**overrides):
    preview = {
        "courts": [{"court": 1, "matches": [{**MATCH, "row_id": "rr-1-1"}]}],
        "match_count": 1,
    }
    games = [{**MATCH, "series_key": "rr-1-1", "game_number": number} for number in range(1, 4)]
    return build_league_live_publish_request(**{
        "session_id": "session-flex", "round_number": 1,
        "league_name": "Sunday Ladder", "week_tag": "Week 1", "match_date": "2026-10-18",
        "matches": games, "expected_match_count": 3, "expected_updated_at": "v1",
        "expected_operation_key": "a" * 64, "match_structure": {"kind": "fixed_games", "games": 3},
        "preview": preview, **overrides,
    })


def test_one_game_substitution_preserves_actual_players_and_scheduled_slots() -> None:
    games = [{**MATCH, "series_key": "rr-1-1", "game_number": number} for number in range(1, 4)]
    games[1].update({"t1_p1": 5, "scheduled_player_ids": [1, 2, 3, 4], "substitution_reason": "Heat break"})
    request = _flexible_request(matches=games)

    assert [row["t1_p1"] for row in request["matches"]] == [1, 5, 1]
    assert request["matches"][1]["scheduled_player_ids"] == [1, 2, 3, 4]
    assert request["matches"][1]["substitution_reason"] == "Heat break"
    assert "scheduled_player_ids" not in request["matches"][0]
    assert "unplayed_games" not in request


def test_unfinished_game_is_recorded_without_publishing_a_score() -> None:
    games = [{**MATCH, "series_key": "rr-1-1", "game_number": number} for number in (1, 3)]
    skipped = [{"series_key": "rr-1-1", "game_number": 2, "court": 1, "reason": "Heat interruption"}]
    request = _flexible_request(matches=games, unplayed_games=skipped)

    assert request["expected_match_count"] == 3
    assert len(request["match_context_ids"]) == 2
    assert [row["game_number"] for row in request["matches"]] == [1, 3]
    assert request["unplayed_games"] == skipped
    assert request["preview"]["unplayed_games"] == skipped
    assert _flexible_request(matches=games, unplayed_games=skipped)["request_fingerprint"] == request["request_fingerprint"]
    assert _flexible_request(matches=games, unplayed_games=[{**skipped[0], "reason": "Injury"}])["request_fingerprint"] != request["request_fingerprint"]


@pytest.mark.parametrize("change,error", [
    ({"scheduled_player_ids": [6, 2, 3, 4]}, "generated scheduled players"),
    ({"scheduled_player_ids": [1, 2, 3, 3]}, "four distinct players"),
    ({"scheduled_player_ids": [1, 2, 3]}, "four ordered player IDs"),
    ({"substitution_reason": "  "}, "substitution reason"),
    ({"court": 2}, "generated court"),
    ({"series_key": "fake-match"}, "generated slot"),
    ({"game_number": 4}, "generated slot"),
    ({"game_number": 1.5}, "positive whole number"),
    ({"series_games": 2}, "league settings"),
    ({"t1_p1": 2}, "four distinct players"),
])
def test_substitution_rejects_invalid_slot_identity_or_missing_reason(change, error) -> None:
    games = [{**MATCH, "series_key": "rr-1-1", "game_number": number} for number in range(1, 4)]
    games[1].update({"t1_p1": 5, "scheduled_player_ids": [1, 2, 3, 4], "substitution_reason": "Heat break", **change})
    with pytest.raises(LeagueLivePublishError, match=error):
        _flexible_request(matches=games)


@pytest.mark.parametrize("skipped,error", [
    ([{"series_key": "rr-1-1", "game_number": 2, "court": 1, "reason": ""}], "needs a reason"),
    ([{"series_key": "rr-1-1", "game_number": 2, "court": 1, "reason": "Heat", "score_t1": 0}], "cannot have scores"),
    ([{"series_key": "rr-1-1", "game_number": 1, "court": 1, "reason": "Heat"}], "both scored and unplayed"),
    ([{"series_key": "rr-1-1", "game_number": 2, "court": 2, "reason": "Heat"}], "generated court"),
])
def test_unplayed_games_reject_invalid_or_conflicting_exclusions(skipped, error) -> None:
    with pytest.raises(LeagueLivePublishError, match=error):
        _flexible_request(matches=[{**MATCH, "series_key": "rr-1-1", "game_number": 1}], unplayed_games=skipped)


def test_unplayed_games_cannot_hide_missing_generated_slots() -> None:
    with pytest.raises(LeagueLivePublishError, match="Every generated game"):
        _flexible_request(
            matches=[{**MATCH, "series_key": "rr-1-1", "game_number": 1}],
            unplayed_games=[{"series_key": "rr-1-1", "game_number": 2, "court": 1, "reason": "Heat"}],
            expected_match_count=2,
        )


def test_explicit_empty_exclusions_still_require_every_generated_slot() -> None:
    with pytest.raises(LeagueLivePublishError, match="Every generated game"):
        _flexible_request(
            matches=[{**MATCH, "series_key": "rr-1-1", "game_number": 1}],
            unplayed_games=[], expected_match_count=1,
        )


def test_same_slot_cannot_be_scored_twice_with_different_results() -> None:
    row = {**MATCH, "series_key": "rr-1-1", "game_number": 1}
    with pytest.raises(LeagueLivePublishError, match="duplicated"):
        _flexible_request(matches=[row, {**row, "score_t2": 9}], unplayed_games=[], expected_match_count=2)


def test_entirely_abandoned_round_requires_cancellation() -> None:
    with pytest.raises(LeagueLivePublishError, match="At least one completed game"):
        _flexible_request(matches=[], unplayed_games=[
            {"series_key": "rr-1-1", "game_number": number, "court": 1, "reason": "Rain"}
            for number in range(1, 4)
        ])


def test_flexible_request_requires_verified_preview() -> None:
    with pytest.raises(LeagueLivePublishError, match="verified generated preview"):
        _flexible_request(preview=None, matches=[{
            **MATCH, "series_key": "rr-1-1", "game_number": 1,
            "scheduled_player_ids": [1, 2, 3, 4],
        }])


def test_partial_plan_can_validate_substitution_without_final_round_coverage() -> None:
    row = {**MATCH, "series_key": "rr-1-1", "game_number": 2, "t1_p1": 5,
           "scheduled_player_ids": [1, 2, 3, 4], "substitution_reason": "Heat"}
    rows, unplayed = validate_league_live_game_slots(
        [row], preview={"courts": [{"court": 1, "matches": [{**MATCH, "row_id": "rr-1-1"}]}]},
        match_structure={"kind": "fixed_games", "games": 3}, require_complete=False,
    )
    assert rows[0]["t1_p1"] == 5
    assert rows[0]["game_number"] == 2
    assert unplayed == []


def test_best_of_substitution_preserves_clinch_rules() -> None:
    games = [{**MATCH, "series_key": "rr-1-1", "game_number": number} for number in (1, 2)]
    games[1].update({"t1_p1": 5, "scheduled_player_ids": [1, 2, 3, 4], "substitution_reason": "Heat"})
    request = _flexible_request(matches=games, expected_match_count=2, match_structure={"kind": "best_of", "games": 3})
    assert [row["t1_p1"] for row in request["matches"]] == [1, 5]
    with pytest.raises(ValueError, match="not complete"):
        _flexible_request(matches=[{**games[1], "game_number": 1}], expected_match_count=1, match_structure={"kind": "best_of", "games": 3})
    with pytest.raises(LeagueLivePublishError, match="Best-of series must finish normally"):
        _flexible_request(matches=games, unplayed_games=[{"series_key": "rr-1-1", "game_number": 3, "court": 1, "reason": "Heat"}],
                          match_structure={"kind": "best_of", "games": 3})
