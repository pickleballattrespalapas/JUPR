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
        df_players_all=pd.DataFrame([{"id": 11, "wins": 112, "losses": 12, "matches_played": 124}]),
        df_players_active=pd.DataFrame(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
        df_badges=pd.DataFrame([{"badge_id": "high_roller", "state": "live"}]),
        df_player_badges=pd.DataFrame(),
        schema_degraded=False,
        schema_degraded_reason=None,
    )


def test_badge_audit_soft_match_context_drift(monkeypatch):
    storage = {
        "player_badges": [
            {
                "id": "pb1",
                "club_id": "club",
                "player_id": 7,
                "badge_id": "high_roller",
                "context_type": "overall",
                "context_id": "old_match_rule_context",
                "revoked_at": None,
            }
        ]
    }
    candidate = BadgeCandidate(
        badge_id="high_roller",
        player_id=7,
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

    report = build_badge_audit_report(FakeSupabase(storage), club_id="club", ctx=_ctx())
    assert report["counts"]["missing_exact_count"] == 1
    assert report["counts"]["stale_exact_count"] == 1
    assert report["counts"]["missing_soft_count"] == 0
    assert report["counts"]["stale_soft_count"] == 0
    assert report["counts"]["context_drift_soft_key_count"] == 1
    assert report["per_badge_summary"][0]["badge_id"] == "high_roller"
    assert report["per_badge_summary"][0]["context_drift_count"] == 1


def test_badge_audit_treats_context_type_as_part_of_exact_identity(monkeypatch):
    storage = {
        "player_badges": [
            {
                "id": "pb-context-type",
                "club_id": "club",
                "player_id": 7,
                "badge_id": "high_roller",
                "context_type": "league",
                "context_id": "shared-context",
                "revoked_at": None,
            }
        ]
    }
    candidate = BadgeCandidate(
        badge_id="high_roller",
        player_id=7,
        club_id="club",
        context_type="overall",
        context_id="shared-context",
        match_id=None,
        value_json={"wins": 120},
    )
    monkeypatch.setattr(
        "jupr_app.domain.gamification.badge_audit.compute_candidates_for_club",
        lambda **kwargs: [candidate],
    )

    report = build_badge_audit_report(FakeSupabase(storage), club_id="club", ctx=_ctx())

    assert report["counts"]["missing_exact_count"] == 1
    assert report["counts"]["stale_exact_count"] == 1
    assert report["counts"]["missing_soft_count"] == 0
    assert report["counts"]["stale_soft_count"] == 0
    assert report["counts"]["context_drift_soft_key_count"] == 1
    assert report["counts"]["duplicate_group_count"] == 0
    assert report["context_drift_rows"][0]["expected_contexts"] == [
        {"context_type": "overall", "context_id": "shared-context"}
    ]
    assert report["context_drift_rows"][0]["actual_contexts"] == [
        {"context_type": "league", "context_id": "shared-context"}
    ]


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
    assert report["counts"]["missing_exact_count"] == 1
    assert report["counts"]["missing_soft_count"] == 1
    assert report["per_badge_summary"][0]["badge_id"] == "high_roller"
    assert report["per_badge_summary"][0]["missing_exact_count"] == 1
    assert report["per_badge_summary"][0]["missing_soft_count"] == 1


def test_badge_audit_detects_true_stale_on_both_layers(monkeypatch):
    storage = {
        "player_badges": [
            {
                "id": "pb1",
                "club_id": "club",
                "player_id": 9,
                "badge_id": "high_roller",
                "context_type": "overall",
                "context_id": "legacy_context",
                "revoked_at": None,
            }
        ]
    }
    monkeypatch.setattr(
        "jupr_app.domain.gamification.badge_audit.compute_candidates_for_club",
        lambda **kwargs: [],
    )

    report = build_badge_audit_report(FakeSupabase(storage), club_id="club", ctx=_ctx())
    assert report["counts"]["stale_exact_count"] == 1
    assert report["counts"]["stale_soft_count"] == 1


def test_badge_audit_detects_duplicates_and_revoked_not_active(monkeypatch):
    storage = {
        "player_badges": [
            {
                "id": "a1",
                "club_id": "club",
                "player_id": 1,
                "badge_id": "high_roller",
                "context_type": "overall",
                "context_id": "overall",
                "revoked_at": None,
            },
            {
                "id": "a2",
                "club_id": "club",
                "player_id": 1,
                "badge_id": "high_roller",
                "context_type": "overall",
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
    assert report["counts"]["actual_active_exact_count"] == 1


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


def test_badge_audit_filters_non_live_actual_rows_by_default(monkeypatch):
    storage = {
        "player_badges": [
            {
                "id": "l1",
                "club_id": "club",
                "player_id": 1,
                "badge_id": "high_roller",
                "context_type": "overall",
                "context_id": "overall",
                "revoked_at": None,
            },
            {
                "id": "n1",
                "club_id": "club",
                "player_id": 1,
                "badge_id": "manual_only",
                "context_type": "manual",
                "context_id": "admin_manual",
                "revoked_at": None,
            },
        ]
    }
    monkeypatch.setattr(
        "jupr_app.domain.gamification.badge_audit.compute_candidates_for_club",
        lambda **kwargs: [],
    )
    ctx = _ctx()
    ctx.df_badges = pd.DataFrame(
        [
            {"badge_id": "high_roller", "state": "live"},
            {"badge_id": "manual_only", "state": "inactive"},
        ]
    )

    report_live_only = build_badge_audit_report(FakeSupabase(storage), club_id="club", ctx=ctx, include_non_live=False)
    assert report_live_only["counts"]["actual_active_exact_count"] == 1

    report_with_non_live = build_badge_audit_report(FakeSupabase(storage), club_id="club", ctx=ctx, include_non_live=True)
    assert report_with_non_live["counts"]["actual_active_exact_count"] == 2


def test_badge_audit_high_roller_debug_payload(monkeypatch):
    storage = {
        "player_badges": [
            {
                "id": "hr1",
                "club_id": "club",
                "player_id": 11,
                "badge_id": "high_roller",
                "context_type": "overall",
                "context_id": "lifetime_wins_100",
                "revoked_at": None,
            }
        ]
    }
    candidate = BadgeCandidate(
        badge_id="high_roller",
        player_id=11,
        club_id="club",
        context_type="overall",
        context_id="lifetime_wins_100",
        match_id=None,
        value_json={"wins": 112},
    )
    monkeypatch.setattr(
        "jupr_app.domain.gamification.badge_audit.compute_candidates_for_club",
        lambda **kwargs: [candidate],
    )
    monkeypatch.setattr(
        "jupr_app.domain.gamification.badge_audit.build_hybrid_player_match_facts",
        lambda *_args, **_kwargs: pd.DataFrame(
            [{"player_id": 11, "match_id": f"m-{i}", "win": True} for i in range(112)]
        ),
    )

    report = build_badge_audit_report(
        FakeSupabase(storage),
        club_id="club",
        ctx=_ctx(),
        badge_id="high_roller",
        player_id=11,
    )
    debug = report["high_roller_debug"]
    assert debug["player_standings_wins"] == 112
    assert debug["player_badge_evaluated_wins"] == 112
    assert debug["player_qualifies"] is True
    assert len(debug["player_existing_badge_rows"]) == 1
