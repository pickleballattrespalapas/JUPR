from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.badge_audit import build_badge_audit_report
from jupr_app.domain.gamification.badge_types import BadgeCandidate


class FakeTable:
    def __init__(self, storage, name):
        self.storage = storage
        self.name = name
        self.filters = []

    def select(self, _cols):
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def execute(self):
        data = list(self.storage.get(self.name, []))
        for op, column, value in self.filters:
            if op == "eq":
                data = [row for row in data if str(row.get(column)) == str(value)]
        return SimpleNamespace(data=data)


class FakeSupabase:
    def __init__(self, storage=None):
        self.storage = storage if storage is not None else {}

    def table(self, name):
        return FakeTable(self.storage, name)


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        club_id="club",
        df_matches=pd.DataFrame(),
        df_players_all=pd.DataFrame(),
        df_players_active=pd.DataFrame(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
        df_badges=pd.DataFrame([{"badge_id": "high_roller", "state": "live"}]),
        df_player_badges=pd.DataFrame(),
        schema_degraded=False,
        schema_degraded_reason=None,
    )


def test_badge_audit_detects_stale_legacy_row(monkeypatch):
    storage = {
        "player_badges": [
            {
                "id": "pb1",
                "club_id": "club",
                "player_id": 10,
                "badge_id": "high_roller",
                "context_id": "match123:high_roller",
                "revoked_at": None,
            }
        ]
    }

    monkeypatch.setattr(
        "jupr_app.domain.gamification.badge_audit.compute_candidates_for_club",
        lambda **kwargs: [],
    )

    report = build_badge_audit_report(FakeSupabase(storage), club_id="club", ctx=_ctx())
    assert report["counts"]["stale_count"] == 1
    assert report["per_badge_summary"][0]["badge_id"] == "high_roller"
    assert report["per_badge_summary"][0]["stale_count"] == 1


def test_badge_audit_detects_missing_row(monkeypatch):
    candidate = BadgeCandidate(
        badge_id="high_roller",
        player_id=5,
        club_id="club",
        context_type="overall",
        context_id="lifetime_wins_100",
        match_id=None,
        value_json={"wins": 120},
    )
    monkeypatch.setattr(
        "jupr_app.domain.gamification.badge_audit.compute_candidates_for_club",
        lambda **kwargs: [candidate],
    )

    report = build_badge_audit_report(FakeSupabase({"player_badges": []}), club_id="club", ctx=_ctx())
    assert report["counts"]["missing_count"] == 1
    assert report["per_badge_summary"][0]["badge_id"] == "high_roller"
    assert report["per_badge_summary"][0]["missing_count"] == 1


def test_badge_audit_detects_duplicates_and_revoked_not_active(monkeypatch):
    storage = {
        "player_badges": [
            {
                "id": "a1",
                "club_id": "club",
                "player_id": 1,
                "badge_id": "participant",
                "context_id": "overall",
                "revoked_at": None,
            },
            {
                "id": "a2",
                "club_id": "club",
                "player_id": 1,
                "badge_id": "participant",
                "context_id": "overall",
                "revoked_at": "2026-01-01T00:00:00Z",
            },
        ]
    }
    monkeypatch.setattr(
        "jupr_app.domain.gamification.badge_audit.compute_candidates_for_club",
        lambda **kwargs: [],
    )

    report = build_badge_audit_report(FakeSupabase(storage), club_id="club", ctx=_ctx(), include_revoked=True)
    assert report["counts"]["duplicate_count"] == 2
    assert report["counts"]["revoked_count"] == 1
    assert report["counts"]["actual_active_count"] == 1


def test_badge_audit_include_non_live_passthrough(monkeypatch):
    seen = {}

    def _fake_compute(**kwargs):
        seen["allow_non_live"] = kwargs.get("allow_non_live")
        return []

    monkeypatch.setattr("jupr_app.domain.gamification.badge_audit.compute_candidates_for_club", _fake_compute)

    build_badge_audit_report(
        FakeSupabase({"player_badges": []}),
        club_id="club",
        ctx=_ctx(),
        include_non_live=False,
    )
    assert seen["allow_non_live"] is False

    build_badge_audit_report(
        FakeSupabase({"player_badges": []}),
        club_id="club",
        ctx=_ctx(),
        include_non_live=True,
    )
    assert seen["allow_non_live"] is True
