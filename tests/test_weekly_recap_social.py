from datetime import datetime, timezone

import pandas as pd

from jupr_app.domain.recaps.weekly_recap import (
    _build_numbers,
    _build_social_spotlight_candidates,
    _compute_social_stats,
)
from jupr_app.ui.components.weekly_recap_layout import ACCENT_CLASS_BY_KEY


def _social_fixture():
    return {
        "events": [
            {"id": "evt-1", "name": "Morning Social", "event_date": "2026-03-20"},
        ],
        "participants": [
            {"id": "a", "event_id": "evt-1", "club_person_id": "cp-a", "linked_player_id": 10, "display_name_snapshot": "Alex"},
            {"id": "b", "event_id": "evt-1", "club_person_id": "cp-b", "linked_player_id": None, "display_name_snapshot": "Ben"},
            {"id": "c", "event_id": "evt-1", "club_person_id": "cp-c", "linked_player_id": None, "display_name_snapshot": "Cara"},
            {"id": "d", "event_id": "evt-1", "club_person_id": "cp-d", "linked_player_id": None, "display_name_snapshot": "Dina"},
        ],
        "matches": [
            {"event_id": "evt-1", "score_t1": 11, "score_t2": 7, "t1_p1_participant_id": "a", "t1_p2_participant_id": "b", "t2_p1_participant_id": "c", "t2_p2_participant_id": "d"},
            {"event_id": "evt-1", "score_t1": 11, "score_t2": 8, "t1_p1_participant_id": "a", "t1_p2_participant_id": "b", "t2_p1_participant_id": "c", "t2_p2_participant_id": "d"},
            {"event_id": "evt-1", "score_t1": 11, "score_t2": 9, "t1_p1_participant_id": "a", "t1_p2_participant_id": "b", "t2_p1_participant_id": "c", "t2_p2_participant_id": "d"},
            {"event_id": "evt-1", "score_t1": 11, "score_t2": 10, "t1_p1_participant_id": "a", "t1_p2_participant_id": "b", "t2_p1_participant_id": "c", "t2_p2_participant_id": "d"},
            {"event_id": "evt-1", "score_t1": 11, "score_t2": 4, "t1_p1_participant_id": "a", "t1_p2_participant_id": "c", "t2_p1_participant_id": "b", "t2_p2_participant_id": "d"},
        ],
    }


def test_social_spotlight_candidates_rank_correctly():
    stats, _, _ = _compute_social_stats(_social_fixture(), {10: "Alex Linked"})
    candidates = _build_social_spotlight_candidates(stats)

    assert candidates["COMMUNITY_STANDOUT_WEEK"][0].display.startswith("Alex Linked — 5-0")
    assert candidates["SOCIAL_GRIND_WEEK"][0].display == "Alex Linked — 5 games"


def test_linked_player_identity_dedupes_total_players():
    social_stats, _, social_meta = _compute_social_stats(_social_fixture(), {10: "Alex Linked"})
    assert social_stats[("player", 10)]["display_name"] == "Alex Linked"

    numbers = _build_numbers(
        df_week=pd.DataFrame([{"id": 1}]),
        stats={10: {"games": 1}},
        event_meta={"leagues": {}, "round_robins": {}},
        supabase=None,
        club_id="club-1",
        start_dt=datetime(2026, 3, 1, tzinfo=timezone.utc),
        end_dt=datetime(2026, 3, 31, tzinfo=timezone.utc),
        social_match_count=social_meta["social_match_count"],
        social_event_count=social_meta["social_event_count"],
        social_canonical_identities=social_meta["canonical_identities"],
    )

    assert numbers["players"] == 4


def test_new_spotlight_accents_exist_and_numbers_cards_shape():
    assert ACCENT_CLASS_BY_KEY["COMMUNITY_STANDOUT_WEEK"]
    assert ACCENT_CLASS_BY_KEY["SOCIAL_GRIND_WEEK"]

    numbers_cards = [
        {"key": "matches", "label": "Matches", "value": 10},
        {"key": "players", "label": "Players", "value": 8},
        {"key": "leagues", "label": "Leagues", "value": 2},
        {"key": "round_robins", "label": "Pop-Ups", "value": 3},
        {"key": "social_round_robins", "label": "Social RRs", "value": 1},
        {"key": "new_faces", "label": "New Faces", "value": 2},
    ]

    assert len(numbers_cards) == 6
    assert numbers_cards[-2]["label"] == "Social RRs"
