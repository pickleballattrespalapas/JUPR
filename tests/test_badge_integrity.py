from types import SimpleNamespace

import pandas as pd
import pytest

from jupr_app.domain.gamification.badge_integrity import dedupe_player_badges_rows
from jupr_app.domain.contracts.badges import BadgeCandidate
from jupr_app.domain.gamification.badges_repo import upsert_player_badges


class CaptureTable:
    def __init__(self):
        self.on_conflict = None
        self.rows = []
        self.filters = []

    def select(self, _cols):
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column, values):
        self.filters.append(("in", column, set(values)))
        return self

    def upsert(self, rows, on_conflict=None):
        self.on_conflict = on_conflict
        self.rows.extend(rows)
        return self

    def execute(self):
        data = list(self.rows)
        for op, column, value in self.filters:
            if op == "eq":
                data = [row for row in data if str(row.get(column)) == str(value)]
            elif op == "in":
                data = [row for row in data if row.get(column) in value]
        return SimpleNamespace(data=data)


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
    candidates = [
        BadgeCandidate(
            badge_id="first_win",
            player_id=1,
            club_id="club",
            context_type="overall",
            context_id="first_win",
            match_id="m1",
            value_json={"tape_excerpt": "The first win hit the archive."},
        )
    ]
    upsert_player_badges(supabase, "club", candidates)
    assert supabase.table_ref.on_conflict == "club_id,player_id,badge_id,context_id"


def test_participation_insert_uses_upsert_conflict():
    supabase = CaptureSupabase()
    candidates = [
        BadgeCandidate(
            badge_id="participant",
            player_id=1,
            club_id="club",
            context_type="overall",
            context_id="overall",
            match_id=None,
            value_num=5,
        )
    ]
    upsert_player_badges(supabase, "club", candidates)
    assert supabase.table_ref.on_conflict == "club_id,player_id,badge_id,context_id"


def test_upsert_rejects_missing_context_id():
    supabase = CaptureSupabase()
    candidates = [
        BadgeCandidate(
            badge_id="participant",
            player_id=1,
            club_id="club",
            context_type="overall",
            context_id=None,
            match_id=None,
            value_num=5,
        )
    ]
    with pytest.raises(ValueError):
        upsert_player_badges(supabase, "club", candidates)
