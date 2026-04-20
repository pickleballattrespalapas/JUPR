from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from jupr_app.domain.match_canonical_audit import build_match_canonical_audit
from jupr_app.domain.match_canonical_migration import normalize_legacy_matches_for_canonical


@dataclass
class Ctx:
    club_id: str
    df_matches: pd.DataFrame
    df_players_all: pd.DataFrame


class FakeExecute:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, table_rows):
        self.table_rows = table_rows
        self._patch = {}
        self._club_id = None
        self._id = None

    def update(self, patch):
        self._patch = dict(patch)
        return self

    def eq(self, column, value):
        if column == "club_id":
            self._club_id = str(value)
        if column == "id":
            self._id = int(value)
        return self

    def execute(self):
        for row in self.table_rows:
            if str(row.get("club_id")) == self._club_id and int(row.get("id")) == self._id:
                row.update(self._patch)
        return FakeExecute([])


class FakeSupabase:
    def __init__(self, rows):
        self.rows = rows

    def table(self, name):
        assert name == "matches"
        return FakeQuery(self.rows)


def _ctx_from_rows(rows):
    return Ctx(
        club_id="club",
        df_matches=pd.DataFrame(rows),
        df_players_all=pd.DataFrame([{"id": 1, "rating": 1200.0}, {"id": 2, "rating": 1200.0}, {"id": 3, "rating": 1200.0}, {"id": 4, "rating": 1200.0}]),
    )


def test_audit_detects_profile_visible_but_not_canonical_with_reasons():
    rows = [
        {"id": 10, "club_id": "club", "date": "2026-01-01", "league": "L1", "match_type": "League", "score_t1": 11, "score_t2": 7, "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4},
        {"id": 11, "club_id": "club", "date": "2026-01-02", "league": "L1", "match_type": "PopUp", "score_t1": 11, "score_t2": 8, "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4},
        {"id": 12, "club_id": "club", "date": "2026-01-03", "league": "L1", "match_type": "League", "score_t1": 0, "score_t2": 0, "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4},
        {"id": 13, "club_id": "club", "date": "2026-01-04", "league": "L1", "match_type": "Tournament", "score_t1": 11, "score_t2": 3, "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4},
    ]
    report = build_match_canonical_audit(_ctx_from_rows(rows), club_id="club", player_id=1)

    assert report["shared_ids"] == ["10"]
    assert sorted(report["only_in_profile"]) == ["11", "12", "13"]
    reasons_by_id = {int(row["match_id"]): set(row["exclusion_reasons"]) for row in report["excluded_only_in_profile"]}
    assert "popup" in reasons_by_id[11]
    assert "invalid/zero score" in reasons_by_id[12]
    assert "tournament match_type" in reasons_by_id[13]


def test_dry_run_normalization_reports_changes_without_updating_rows():
    rows = [
        {
            "id": 21,
            "club_id": "club",
            "date": "2026-01-04",
            "league": "L1",
            "match_type": "League",
            "score_t1": 0,
            "score_t2": 0,
            "team1_score": 11,
            "team2_score": 9,
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
        },
    ]
    ctx = _ctx_from_rows(rows)
    fake_supabase = FakeSupabase(rows)

    result = normalize_legacy_matches_for_canonical(fake_supabase, ctx=ctx, club_id="club", player_id=1, dry_run=True)

    assert result["proposed_update_count"] == 1
    assert result["applied_update_count"] == 0
    assert rows[0]["score_t1"] == 0
    patch = result["proposals"][0]["patch"]
    assert patch["score_t1"] == 11


def test_apply_normalization_updates_rows_and_leaves_shared_untouched():
    rows = [
        {
            "id": 31,
            "club_id": "club",
            "date": "2026-01-05",
            "league": "L1",
            "match_type": "League",
            "score_t1": 0,
            "score_t2": 0,
            "score1": 11,
            "score2": 6,
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
        },
        {
            "id": 32,
            "club_id": "club",
            "date": "2026-01-06",
            "league": "L1",
            "match_type": "League",
            "score_t1": 11,
            "score_t2": 8,
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
        },
    ]
    ctx = _ctx_from_rows(rows)
    fake_supabase = FakeSupabase(rows)

    result = normalize_legacy_matches_for_canonical(fake_supabase, ctx=ctx, club_id="club", player_id=1, dry_run=False)

    assert result["applied_update_count"] == 1
    assert rows[0]["score_t1"] == 11
    assert rows[1]["score_t1"] == 11
    assert rows[1]["score_t2"] == 8
