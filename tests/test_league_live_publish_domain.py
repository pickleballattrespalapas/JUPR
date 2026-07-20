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
    normalize_league_live_publish_matches,
    rows_to_safe_csv,
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
