from datetime import datetime, timezone

import pandas as pd

from datetime import date

from jupr_app.domain.recaps.weekly_recap import (
    _build_numbers,
    _build_numbers_cards,
    _build_social_around_club,
    _build_social_spotlight_candidates,
    _compute_social_stats,
    _load_social_live_data,
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
    assert ("person", "cp-a") not in social_stats

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

    numbers_cards = _build_numbers_cards(
        {
            "matches": 10,
            "players": 8,
            "leagues": 2,
            "round_robins": 3,
            "community_events": 1,
            "new_faces": 2,
        }
    )
    assert len(numbers_cards) == 6
    assert numbers_cards[-2]["label"] == "Community Events"


class _Resp:
    def __init__(self, data):
        self.data = data


class _Q:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.filters: list[tuple[str, object]] = []

    def select(self, _cols: str):
        return self

    def eq(self, key: str, value: object):
        self.filters.append((key, value))
        return self

    def gte(self, key: str, value: object):
        self.filters.append((f"{key}__gte", value))
        return self

    def lte(self, key: str, value: object):
        self.filters.append((f"{key}__lte", value))
        return self

    def in_(self, key: str, values: list[object]):
        self.filters.append((f"{key}__in", set(values)))
        return self

    def execute(self):
        result = list(self.rows)
        for key, value in self.filters:
            if key.endswith("__gte"):
                col = key[:-5]
                result = [row for row in result if str(row.get(col) or "") >= str(value)]
            elif key.endswith("__lte"):
                col = key[:-5]
                result = [row for row in result if str(row.get(col) or "") <= str(value)]
            elif key.endswith("__in"):
                col = key[:-4]
                result = [row for row in result if row.get(col) in value]
            else:
                result = [row for row in result if row.get(key) == value]
        return _Resp(result)


class _Supabase:
    def __init__(self):
        self.tables = {
            "live_events": [
                {
                    "id": "evt-saved",
                    "club_id": "club-1",
                    "name": "Saved Social",
                    "event_type": "round_robin",
                    "event_date": "2026-03-21",
                    "result_mode": "social_unrated",
                    "status": "saved",
                },
                {
                    "id": "evt-league",
                    "club_id": "club-1",
                    "name": "Saved Ladder",
                    "event_type": "league",
                    "event_date": "2026-03-22",
                    "result_mode": "social_unrated",
                    "status": "saved",
                },
                {
                    "id": "evt-rejected",
                    "club_id": "club-1",
                    "name": "Rejected Social",
                    "event_type": "round_robin",
                    "event_date": "2026-03-21",
                    "result_mode": "social_unrated",
                    "status": "rejected",
                },
                {
                    "id": "evt-pending",
                    "club_id": "club-1",
                    "name": "Pending Social",
                    "event_type": "league",
                    "event_date": "2026-03-23",
                    "result_mode": "social_unrated",
                    "status": "pending",
                },
            ],
            "live_event_participants": [],
            "live_event_matches": [],
        }

    def table(self, name: str):
        return _Q(list(self.tables.get(name, [])))


def test_load_social_data_ignores_rejected_events():
    supabase = _Supabase()
    loaded = _load_social_live_data(supabase, "club-1", date(2026, 3, 1), date(2026, 3, 31))
    event_ids = {row["id"] for row in loaded["events"]}
    assert event_ids == {"evt-saved", "evt-league"}


def test_build_social_around_club_handles_rr_and_league_labels():
    rows = _build_social_around_club(
        {
            "evt-rr": {
                ("player", 1): {
                    "display_name": "Amy",
                    "games": 6,
                    "wins": 5,
                    "losses": 1,
                    "points_for": 66,
                    "points_against": 40,
                    "differential": 26,
                }
            },
            "evt-lg": {
                ("player", 2): {
                    "display_name": "Ben",
                    "games": 7,
                    "wins": 4,
                    "losses": 3,
                    "points_for": 70,
                    "points_against": 61,
                    "differential": 9,
                }
            },
        },
        {
            "evt-rr": {"name": "Morning RR", "event_type": "round_robin", "event_date": "2026-03-21"},
            "evt-lg": {"name": "Night Ladder", "event_type": "league", "event_date": "2026-03-22"},
        },
    )
    labels = {row["event_name"]: row["event_type_label"] for row in rows}
    assert labels["Morning RR"] == "Social RR"
    assert labels["Night Ladder"] == "Social Ladder"
