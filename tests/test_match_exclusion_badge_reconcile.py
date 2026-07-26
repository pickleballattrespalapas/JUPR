from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from jupr_app.domain.gamification.badge_registry import active_badge_ids
from jupr_app.domain.gamification.badge_types import (
    BadgeCandidate,
    BadgeEvaluationContext,
)
from jupr_app.domain.gamification.match_exclusion_reconcile import (
    MATCH_EXCLUSION_BADGE_APPLY_RPC,
    MATCH_EXCLUSION_BADGE_CLAIM_RPC,
    MATCH_EXCLUSION_BADGE_CONTRACT_VERSION,
    MATCH_EXCLUSION_BADGE_FAIL_RPC,
    MATCH_EXCLUSION_BADGE_IDS,
    _assert_complete_active_match_load,
    build_match_exclusion_badge_plan,
    reconcile_match_exclusion_badges,
    resolve_match_exclusion_badge_ids,
)


class _CountQuery:
    def __init__(self, exact_count: int | None):
        self.exact_count = exact_count

    def select(self, _columns: str, *, count: str):
        assert count == "exact"
        return self

    def eq(self, _column: str, _value: Any):
        return self

    def is_(self, _column: str, _value: Any):
        return self

    def limit(self, _value: int):
        return self

    def execute(self):
        return SimpleNamespace(data=[], count=self.exact_count)


class _CountSupabase:
    def __init__(self, exact_count: int | None):
        self.exact_count = exact_count

    def table(self, name: str):
        assert name == "matches"
        return _CountQuery(self.exact_count)


class _RpcResult:
    def __init__(self, data: Any = None, error: Exception | None = None):
        self.data = data
        self.error = error

    def execute(self):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(data=self.data)


class _ReconcileSupabase:
    def __init__(self, *, apply_error: Exception | None = None):
        self.apply_error = apply_error
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.claimed = False

    def rpc(self, name: str, params: dict[str, Any]):
        self.calls.append((name, dict(params)))
        if name == MATCH_EXCLUSION_BADGE_CLAIM_RPC:
            if self.claimed:
                return _RpcResult({"ok": True, "claimed": False, "code": "empty"})
            self.claimed = True
            return _RpcResult(
                {
                    "ok": True,
                    "claimed": True,
                    "operation_id": "operation-1",
                    "progress_id": "progress-1",
                    "club_id": "club",
                    "player_id": 7,
                    "lease_token": "lease-1",
                    "badge_ids": ["first_win"],
                    "badge_contract_version": MATCH_EXCLUSION_BADGE_CONTRACT_VERSION,
                }
            )
        if name == MATCH_EXCLUSION_BADGE_APPLY_RPC:
            if self.apply_error is not None:
                return _RpcResult(error=self.apply_error)
            desired = list(params.get("p_desired_badges") or [])
            return _RpcResult(
                {
                    "ok": True,
                    "operation_id": "operation-1",
                    "progress_id": "progress-1",
                    "player_id": 7,
                    "status": "succeeded",
                    "desired_count": len(desired),
                    "inserted_count": 1,
                    "updated_count": 0,
                    "revoked_count": 2,
                    "idempotent": False,
                }
            )
        if name == MATCH_EXCLUSION_BADGE_FAIL_RPC:
            return _RpcResult({"ok": True, "status": "failed"})
        raise AssertionError(f"unexpected RPC: {name}")


def _safe_ctx(*, badges: list[dict[str, Any]] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        club_id="club",
        schema_degraded=False,
        schema_degraded_reason=None,
        df_matches=pd.DataFrame(
            [{"id": "m1", "club_id": "club", "deleted_at": None}]
        ),
        df_badges=pd.DataFrame(badges or []),
    )


def _evaluation(ctx: Any) -> BadgeEvaluationContext:
    canonical = pd.DataFrame([{"player_id": 7, "source": "canonical"}])
    hybrid = pd.DataFrame([{"player_id": 7, "source": "legacy"}])
    return BadgeEvaluationContext(
        club_id="club",
        league_id=None,
        as_of=None,
        ctx=ctx,
        facts=hybrid,
        matches=hybrid,
        facts_canonical=canonical,
        facts_hybrid=hybrid,
    )


def test_frozen_v1_allowlist_is_exactly_the_code_live_live_scope():
    assert set(MATCH_EXCLUSION_BADGE_IDS) == active_badge_ids()
    assert "tournament_champion" not in MATCH_EXCLUSION_BADGE_IDS
    assert "league_champion" not in MATCH_EXCLUSION_BADGE_IDS
    assert "above_expectations" not in MATCH_EXCLUSION_BADGE_IDS
    assert len(MATCH_EXCLUSION_BADGE_IDS) == len(set(MATCH_EXCLUSION_BADGE_IDS))


def test_pre_mutation_resolver_intersects_frozen_ids_with_current_catalog(
    monkeypatch,
):
    ctx = _safe_ctx(
        badges=[
            {
                "badge_id": "first_win",
                "state": "live",
                "is_active": True,
                "eval_triggers": ["match_recorded", "match_updated"],
                "badge_status": "live",
                "badge_award_timing": "live",
            },
            {
                "badge_id": "hot_streak",
                "state": "retired",
                "is_active": True,
                "eval_triggers": ["match_updated"],
                "badge_status": "live",
                "badge_award_timing": "live",
            },
            {
                "badge_id": "participant",
                "state": "live",
                "is_active": True,
                "eval_triggers": ["match_recorded"],
                "badge_status": "live",
                "badge_award_timing": "live",
            },
            {
                "badge_id": "tournament_champion",
                "state": "live",
                "is_active": True,
                "eval_triggers": ["match_updated"],
                "badge_status": "live",
                "badge_award_timing": "manual",
            },
        ]
    )
    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile._load_reconciliation_context",
        lambda *_args, **_kwargs: ctx,
    )

    resolved = resolve_match_exclusion_badge_ids(object(), club_id="club")

    assert resolved == ["first_win"]


def test_canonical_context_refuses_exact_count_mismatch():
    with pytest.raises(RuntimeError, match="incomplete or truncated"):
        _assert_complete_active_match_load(
            _CountSupabase(exact_count=2),
            club_id="club",
            df_matches=pd.DataFrame([{"id": 1}]),
            match_limit=5000,
        )


def test_canonical_context_refuses_history_over_match_limit():
    with pytest.raises(RuntimeError, match="exceeds match_limit"):
        _assert_complete_active_match_load(
            _CountSupabase(exact_count=6),
            club_id="club",
            df_matches=pd.DataFrame([{"id": value} for value in range(5)]),
            match_limit=5,
        )


def test_plan_forces_canonical_facts_and_emits_unique_json_safe_rows(
    monkeypatch,
):
    ctx = _safe_ctx()
    evaluation = _evaluation(ctx)
    seen_sources: list[str] = []

    def evaluator(current: BadgeEvaluationContext):
        seen_sources.extend(current.facts["source"].tolist())
        return [
            BadgeCandidate(
                badge_id="first_win",
                player_id=7,
                club_id="club",
                context_type="overall",
                context_id="first_win",
                match_id="m1",
                value_num=float("nan"),
                value_json={"played_at": pd.Timestamp("2026-07-26T12:00:00Z")},
            )
        ]

    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile.build_evaluation_context",
        lambda *_args, **_kwargs: evaluation,
    )
    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile.registry",
        lambda: {"first_win": SimpleNamespace(evaluator=evaluator)},
    )

    plan = build_match_exclusion_badge_plan(
        ctx=ctx,
        club_id="club",
        player_id=7,
        badge_ids=["first_win"],
    )

    assert seen_sources == ["canonical"]
    assert len(plan) == 1
    assert {
        key: value for key, value in plan[0].items() if key != "value_json"
    } == {
        "badge_id": "first_win",
        "context_type": "overall",
        "context_id": "first_win",
        "match_id": "m1",
        "value_num": None,
        "rule_version": MATCH_EXCLUSION_BADGE_CONTRACT_VERSION,
    }
    assert plan[0]["value_json"]["badge_id"] == "first_win"
    assert plan[0]["value_json"]["played_at"] == "2026-07-26T12:00:00+00:00"
    assert plan[0]["value_json"]["tape_excerpt"]
    assert plan[0]["value_json"]["tape_title"]


def test_plan_fails_closed_on_duplicate_player_badge_context_key(monkeypatch):
    ctx = _safe_ctx()
    evaluation = _evaluation(ctx)
    duplicate = BadgeCandidate(
        badge_id="first_win",
        player_id=7,
        club_id="club",
        context_type="overall",
        context_id="first_win",
        match_id="m1",
    )
    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile.build_evaluation_context",
        lambda *_args, **_kwargs: evaluation,
    )
    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile.registry",
        lambda: {
            "first_win": SimpleNamespace(evaluator=lambda _ctx: [duplicate, duplicate])
        },
    )

    with pytest.raises(RuntimeError, match="duplicate"):
        build_match_exclusion_badge_plan(
            ctx=ctx,
            club_id="club",
            player_id=7,
            badge_ids=["first_win"],
        )


def test_reconcile_claims_and_atomically_applies_exact_player_plan(monkeypatch):
    supabase = _ReconcileSupabase()
    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile._load_reconciliation_context",
        lambda *_args, **_kwargs: _safe_ctx(),
    )
    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile._canonical_evaluation_context",
        lambda *_args, **_kwargs: object(),
    )
    desired = [
        {
            "badge_id": "first_win",
            "context_type": "overall",
            "context_id": "first_win",
            "match_id": "m1",
            "value_num": None,
            "value_json": {},
            "rule_version": MATCH_EXCLUSION_BADGE_CONTRACT_VERSION,
        }
    ]
    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile._build_player_plan",
        lambda **_kwargs: desired,
    )

    result = reconcile_match_exclusion_badges(
        supabase,
        club_id="club",
        operation_id="operation-1",
        player_ids=[7],
        actor_email="admin@example.test",
    )

    assert result["ok"] is True
    assert result["processed_player_ids"] == [7]
    assert result["badge_ids"] == ["first_win"]
    assert result["desired_count"] == 1
    assert result["awarded_count"] == 1
    assert result["revoked_count"] == 2
    assert result["unchanged_count"] == 0
    apply_call = next(call for call in supabase.calls if call[0] == MATCH_EXCLUSION_BADGE_APPLY_RPC)
    assert apply_call[1]["p_desired_badges"] == desired
    assert apply_call[1]["p_actor_email"] == "admin@example.test"


def test_reconcile_marks_claimed_progress_failed_when_apply_errors(monkeypatch):
    supabase = _ReconcileSupabase(apply_error=RuntimeError("database rejected plan"))
    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile._load_reconciliation_context",
        lambda *_args, **_kwargs: _safe_ctx(),
    )
    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile._canonical_evaluation_context",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile._build_player_plan",
        lambda **_kwargs: [],
    )

    with pytest.raises(RuntimeError, match="database rejected plan"):
        reconcile_match_exclusion_badges(
            supabase,
            club_id="club",
            operation_id="operation-1",
            player_ids=[7],
        )

    fail_call = next(call for call in supabase.calls if call[0] == MATCH_EXCLUSION_BADGE_FAIL_RPC)
    assert fail_call[1]["p_progress_id"] == "progress-1"
    assert fail_call[1]["p_lease_token"] == "lease-1"
    assert "database rejected plan" in fail_call[1]["p_error_text"]


def test_reconcile_refuses_operation_allowlist_outside_frozen_v1(monkeypatch):
    supabase = _ReconcileSupabase()
    original_rpc = supabase.rpc

    def rpc(name: str, params: dict[str, Any]):
        result = original_rpc(name, params)
        if name == MATCH_EXCLUSION_BADGE_CLAIM_RPC:
            result.data["badge_ids"] = ["first_win", "future_badge"]
        return result

    supabase.rpc = rpc  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="exceeds"):
        reconcile_match_exclusion_badges(
            supabase,
            club_id="club",
            operation_id="operation-1",
            player_ids=[7],
        )

    assert any(name == MATCH_EXCLUSION_BADGE_FAIL_RPC for name, _params in supabase.calls)
