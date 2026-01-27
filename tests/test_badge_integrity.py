from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.badge_integrity import dedupe_player_badges_rows
from jupr_app.domain.gamification.badge_rules import BadgeAward, _insert_badges
from jupr_app.domain.badges_participation import _insert_badges as insert_participation_badges


class CaptureTable:
    def __init__(self):
        self.on_conflict = None
        self.rows = []

    def upsert(self, rows, on_conflict=None):
        self.on_conflict = on_conflict
        self.rows.extend(rows)
        return self

    def execute(self):
        return SimpleNamespace(data=self.rows)


class CaptureSupabase:
    def __init__(self):
        self.table_ref = CaptureTable()

    def table(self, _name):
        return self.table_ref


def test_dedupe_player_badges_rows_keeps_earliest():
    rows = [
        {
            "id": "1",
            "club_id": "club",
            "player_id": 9,
            "badge_id": "giant_slayer",
            "context_id": "match:1",
            "earned_at": "2024-01-05T00:00:00Z",
        },
        {
            "id": "2",
            "club_id": "club",
            "player_id": 9,
            "badge_id": "giant_slayer",
            "context_id": "match:1",
            "earned_at": "2024-01-06T00:00:00Z",
        },
        {
            "id": "3",
            "club_id": "club",
            "player_id": 9,
            "badge_id": "first_win",
            "context_id": "overall",
            "earned_at": "2024-01-02T00:00:00Z",
        },
    ]
    deduped = dedupe_player_badges_rows(rows)
    assert len(deduped) == 2
    assert set(deduped["id"]) == {"1", "3"}


def test_badge_awards_insert_uses_upsert_conflict():
    supabase = CaptureSupabase()
    awards = [
        BadgeAward(
            player_id=1,
            badge_id="first_win",
            context_type="overall",
            context_id="first_win",
            match_id="m1",
            value_num=None,
            value_json={"tape_excerpt": "The first win hit the archive."},
        )
    ]
    _insert_badges(supabase, "club", awards)
    assert supabase.table_ref.on_conflict == "club_id,player_id,badge_id,context_id"


def test_participation_insert_uses_upsert_conflict():
    supabase = CaptureSupabase()
    rows = [
        {
            "id": "1",
            "club_id": "club",
            "player_id": 1,
            "badge_id": "participant",
            "earned_at": "2024-01-01T00:00:00Z",
            "context_type": "overall",
            "context_id": None,
            "value_num": 5,
        }
    ]
    insert_participation_badges(supabase, rows)
    assert supabase.table_ref.on_conflict == "club_id,player_id,badge_id,context_id"
