from __future__ import annotations

from jupr_app.domain.idempotency import build_match_idempotency_key_v1


def test_build_match_idempotency_key_v1_ignores_source_metadata():
    base = {
        "club_id": "club-1",
        "date": "2026-01-01T12:00:00+00:00",
        "context_type": "league",
        "context_id": "league-1",
        "t1_p1": 1,
        "t1_p2": 2,
        "t2_p1": 3,
        "t2_p2": 4,
        "score_t1": 11,
        "score_t2": 9,
        "match_type": "League",
    }
    key_1 = build_match_idempotency_key_v1({**base, "source": "uploader", "submitted_by": "alice"})
    key_2 = build_match_idempotency_key_v1({**base, "source": "moneyball", "import_batch_id": "batch-7"})
    assert key_1 == key_2


def test_build_match_idempotency_key_v1_sorts_players_within_each_team():
    key_1 = build_match_idempotency_key_v1(
        {
            "club_id": "club-1",
            "date": "2026-01-01T12:00:00+00:00",
            "context_type": "league",
            "context_id": "league-1",
            "t1_p1": 2,
            "t1_p2": 1,
            "t2_p1": 4,
            "t2_p2": 3,
            "score_t1": 11,
            "score_t2": 8,
        }
    )
    key_2 = build_match_idempotency_key_v1(
        {
            "club_id": "club-1",
            "date": "2026-01-01T12:00:00+00:00",
            "context_type": "league",
            "context_id": "league-1",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "score_t1": 11,
            "score_t2": 8,
        }
    )
    assert key_1 == key_2
