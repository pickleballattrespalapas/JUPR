from __future__ import annotations

from types import SimpleNamespace

import pytest
from postgrest.exceptions import APIError

from jupr_app.services import match_exclusion_durability_service as service


OPERATION_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
REPLAY_JOB_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
IDEMPOTENCY_KEY = "cccccccc-cccc-cccc-cccc-cccccccccccc"


class FakeQuery:
    def __init__(self, rows):
        self.rows = rows
        self.filters = []

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def limit(self, _value):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def execute(self):
        rows = list(self.rows)
        for key, value in self.filters:
            rows = [
                row for row in rows if str(row.get(key)) == str(value)
            ]
        return SimpleNamespace(data=[dict(row) for row in rows])


class FakeRpc:
    def __init__(self, owner, name, params):
        self.owner = owner
        self.name = name
        self.params = dict(params)

    def execute(self):
        self.owner.rpc_calls.append((self.name, self.params))
        handler = self.owner.rpc_handlers[self.name]
        result = handler(self.params) if callable(handler) else handler
        if isinstance(result, Exception):
            raise result
        if self.name == "finalize_match_exclusion_operation":
            operation = next(
                (
                    row
                    for row in self.owner.operations
                    if str(row.get("id"))
                    == str(self.params.get("p_operation_id"))
                ),
                None,
            )
            if operation is None:
                operation = {
                    "id": self.params["p_operation_id"],
                    "club_id": self.params["p_club_id"],
                    "mode": result.get("mode") or "exclude",
                    "excluded_match_ids": result.get("excluded_ids") or [7],
                    "affected_player_ids": result.get("affected_player_ids")
                    or [1, 2, 3, 4],
                    "replay_job_id": result.get("replay_job_id")
                    or REPLAY_JOB_ID,
                    "replay_result_json": {
                        "singles_replay_supported": True
                    },
                }
                self.owner.operations.append(operation)
            operation.update(
                {
                    "status": "succeeded",
                    "result_json": dict(result),
                    "recovery_stage": None,
                }
            )
        return SimpleNamespace(data=dict(result))


class FakeSupabase:
    def __init__(self, *, operations=None, rpc_handlers=None):
        self.operations = list(operations or [])
        self.rpc_handlers = dict(rpc_handlers or {})
        self.rpc_calls = []

    def table(self, name):
        if name == "match_edit_operations":
            return FakeQuery([])
        if name != "match_exclusion_operations":
            raise AssertionError(f"Unexpected table {name}")
        return FakeQuery(self.operations)

    def rpc(self, name, params):
        if name not in self.rpc_handlers:
            raise AssertionError(f"Unexpected RPC {name}")
        return FakeRpc(self, name, params)


def _pending_apply(*, idempotent=False):
    return {
        "ok": True,
        "operation_id": OPERATION_ID,
        "operation_status": "pending_replay",
        "mode": "exclude",
        "excluded_ids": [7],
        "excluded_count": 1,
        "affected_player_ids": [1, 2, 3, 4],
        "replay_job_id": REPLAY_JOB_ID,
        "replay_target": service.FULL_RESET_LABEL,
        "badge_ids": ["first_win"],
        "badge_contract_version": "jupr:match-exclusion-badges:v1",
        "idempotent": idempotent,
    }


def _pending_badges():
    return {
        **_pending_apply(),
        "operation_status": "pending_badge_reconcile",
    }


def _succeeded():
    return {
        **_pending_apply(),
        "operation_status": "succeeded",
        "result_json": {
            "replay": {
                "job_id": REPLAY_JOB_ID,
                "job_status": "succeeded",
                "result": {"singles_replay_supported": True},
            }
        },
    }


def _patch_badges(monkeypatch):
    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile.resolve_match_exclusion_badge_ids",
        lambda *_args, **_kwargs: ["first_win"],
    )
    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile.reconcile_match_exclusion_badges",
        lambda *_args, **kwargs: {
            "ok": True,
            "operation_id": kwargs["operation_id"],
            "player_ids": kwargs["player_ids"],
            "badge_ids": ["first_win"],
            "awarded_count": 0,
            "revoked_count": 1,
        },
    )


def test_apply_exclusion_freezes_scope_replays_reconciles_and_finalizes(
    monkeypatch,
):
    _patch_badges(monkeypatch)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", "1")
    monkeypatch.setattr(
        service,
        "uuid4",
        lambda: OPERATION_ID,
    )
    monkeypatch.setattr(
        service,
        "run_admin_replay_history",
        lambda *_args, **_kwargs: {
            "ok": True,
            "job_id": REPLAY_JOB_ID,
            "job_status": "succeeded",
            "result": {"singles_replay_supported": True},
            "warnings": [],
        },
    )
    supabase = FakeSupabase(
        rpc_handlers={
            "apply_match_exclusions_atomic": _pending_apply(),
            "transition_match_exclusion_after_replay": _pending_badges(),
            "finalize_match_exclusion_operation": _succeeded(),
        }
    )

    result = service.apply_atomic_match_exclusions(
        supabase,
        club_id="club",
        targets=[{"match_id": 7, "expected_row_version": 3}],
        actor_email="admin@example.com",
        actor_role="super_admin",
        source="test",
        note="Wrong row",
        idempotency_key=IDEMPOTENCY_KEY,
    )

    assert result["ok"] is True
    assert result["operation_status"] == "succeeded"
    assert result["excluded_ids"] == [7]
    assert result["replay_job_id"] == REPLAY_JOB_ID
    assert result["badge_reconcile"]["revoked_count"] == 1
    assert [name for name, _params in supabase.rpc_calls] == [
        "apply_match_exclusions_atomic",
        "transition_match_exclusion_after_replay",
        "finalize_match_exclusion_operation",
    ]
    apply_params = supabase.rpc_calls[0][1]
    assert apply_params["p_operation_id"] == OPERATION_ID
    assert apply_params["p_targets"] == [
        {"match_id": 7, "expected_row_version": 3}
    ]
    assert apply_params["p_badge_ids"] == ["first_win"]
    assert (
        apply_params["p_badge_contract_version"]
        == "jupr:match-exclusion-badges:v1"
    )


def test_response_loss_retry_reuses_stored_badge_contract_before_rpc(
    monkeypatch,
):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", "1")
    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile.resolve_match_exclusion_badge_ids",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("resolver must not run for a stored operation")
        ),
    )
    stored = {
        "id": OPERATION_ID,
        "club_id": "club",
        "idempotency_key": IDEMPOTENCY_KEY,
        "mode": "exclude",
        "status": "succeeded",
        "targets_json": [{"match_id": 7, "expected_row_version": 3}],
        "badge_ids": ["first_win"],
        "badge_contract_version": "jupr:match-exclusion-badges:v1",
        "excluded_match_ids": [7],
        "affected_player_ids": [1, 2, 3, 4],
        "replay_job_id": REPLAY_JOB_ID,
        "result_json": {},
    }
    supabase = FakeSupabase(
        operations=[stored],
        rpc_handlers={
            "apply_match_exclusions_atomic": {
                **_succeeded(),
                "idempotent": True,
            }
        },
    )

    result = service.apply_atomic_match_exclusions(
        supabase,
        club_id="club",
        targets=[{"match_id": 7, "expected_row_version": 3}],
        actor_email="admin@example.com",
        actor_role="super_admin",
        source="test",
        note="Wrong row",
        idempotency_key=IDEMPOTENCY_KEY,
    )

    assert result["ok"] is True
    assert result["idempotent"] is True
    assert supabase.rpc_calls[0][1]["p_operation_id"] == OPERATION_ID
    assert supabase.rpc_calls[0][1]["p_badge_ids"] == ["first_win"]


@pytest.mark.parametrize(
    ("error_code", "error_message", "expected_exception"),
    [
        (
            "P0001",
            "JUPR_MATCH_EXCLUSION_STALE: match 7 changed.",
            service.MatchExclusionStaleError,
        ),
        (
            "40001",
            "JUPR_MATCH_EXCLUSION_STALE: match 7 changed.",
            service.MatchExclusionStaleError,
        ),
        (
            "23505",
            "JUPR_MATCH_EXCLUSION_IDEMPOTENCY_CONFLICT: changed body.",
            service.MatchExclusionIdempotencyConflict,
        ),
    ],
)
def test_atomic_rpc_conflicts_are_typed(
    monkeypatch,
    error_code,
    error_message,
    expected_exception,
):
    _patch_badges(monkeypatch)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", "1")
    supabase = FakeSupabase(
        rpc_handlers={
            "apply_match_exclusions_atomic": Exception(
                {"code": error_code, "message": error_message}
            )
        }
    )

    with pytest.raises(expected_exception):
        service.apply_atomic_match_exclusions(
            supabase,
            club_id="club",
            targets=[{"match_id": 7, "expected_row_version": 3}],
            actor_email="admin@example.com",
            actor_role="super_admin",
            source="test",
            note="Wrong row",
            idempotency_key=IDEMPOTENCY_KEY,
        )


def test_postgrest_stale_detail_is_typed(monkeypatch):
    _patch_badges(monkeypatch)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", "1")
    supabase = FakeSupabase(
        rpc_handlers={
            "apply_match_exclusions_atomic": APIError(
                {
                    "code": "P0001",
                    "message": "Database transaction failed.",
                    "details": (
                        "JUPR_MATCH_EXCLUSION_STALE: match 7 expected "
                        "row_version 4 but is 3."
                    ),
                    "hint": None,
                }
            )
        }
    )

    with pytest.raises(service.MatchExclusionStaleError):
        service.apply_atomic_match_exclusions(
            supabase,
            club_id="club",
            targets=[{"match_id": 7, "expected_row_version": 4}],
            actor_email="admin@example.com",
            actor_role="super_admin",
            source="test",
            note="Wrong row",
            idempotency_key=IDEMPOTENCY_KEY,
        )


def test_active_club_replay_is_a_typed_409_condition(monkeypatch):
    _patch_badges(monkeypatch)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", "1")
    supabase = FakeSupabase(
        rpc_handlers={
            "apply_match_exclusions_atomic": {
                "ok": False,
                "code": "MATCH_EXCLUSION_REPLAY_IN_PROGRESS",
                "message": "A replay is already running.",
            }
        }
    )

    with pytest.raises(service.MatchExclusionReplayActive):
        service.apply_atomic_match_exclusions(
            supabase,
            club_id="club",
            targets=[{"match_id": 7, "expected_row_version": 3}],
            actor_email="admin@example.com",
            actor_role="super_admin",
            source="test",
            note="Wrong row",
            idempotency_key=IDEMPOTENCY_KEY,
        )


def test_active_replay_lease_does_not_mark_operation_for_recovery(monkeypatch):
    _patch_badges(monkeypatch)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", "1")
    monkeypatch.setattr(service, "uuid4", lambda: OPERATION_ID)
    monkeypatch.setattr(
        service,
        "run_admin_replay_history",
        lambda *_args, **_kwargs: {
            "ok": False,
            "job_id": REPLAY_JOB_ID,
            "job_status": "running",
        },
    )
    supabase = FakeSupabase(
        rpc_handlers={
            "apply_match_exclusions_atomic": _pending_apply(),
        }
    )

    with pytest.raises(service.MatchExclusionWorkActive) as exc_info:
        service.apply_atomic_match_exclusions(
            supabase,
            club_id="club",
            targets=[{"match_id": 7, "expected_row_version": 3}],
            actor_email="admin@example.com",
            actor_role="super_admin",
            source="test",
            note="Wrong row",
            idempotency_key=IDEMPOTENCY_KEY,
        )

    assert isinstance(exc_info.value, service.MatchExclusionReplayActive)
    assert exc_info.value.operation_id == OPERATION_ID
    assert exc_info.value.replay_job_id == REPLAY_JOB_ID
    assert exc_info.value.recovery_stage == "replay"
    assert exc_info.value.operation_status == "pending_replay"
    assert [name for name, _params in supabase.rpc_calls] == [
        "apply_match_exclusions_atomic"
    ]


def test_active_badge_lease_does_not_mark_operation_for_recovery(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", "1")
    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile.resolve_match_exclusion_badge_ids",
        lambda *_args, **_kwargs: ["first_win"],
    )
    monkeypatch.setattr(
        "jupr_app.domain.gamification.match_exclusion_reconcile.reconcile_match_exclusion_badges",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("BADGE_RECONCILE_IN_PROGRESS")
        ),
    )
    monkeypatch.setattr(service, "uuid4", lambda: OPERATION_ID)
    supabase = FakeSupabase(
        rpc_handlers={
            "apply_match_exclusions_atomic": _pending_badges(),
        }
    )

    with pytest.raises(service.MatchExclusionWorkActive) as exc_info:
        service.apply_atomic_match_exclusions(
            supabase,
            club_id="club",
            targets=[{"match_id": 7, "expected_row_version": 3}],
            actor_email="admin@example.com",
            actor_role="super_admin",
            source="test",
            note="Wrong row",
            idempotency_key=IDEMPOTENCY_KEY,
        )

    assert not isinstance(exc_info.value, service.MatchExclusionReplayActive)
    assert exc_info.value.operation_id == OPERATION_ID
    assert exc_info.value.replay_job_id == REPLAY_JOB_ID
    assert exc_info.value.recovery_stage == "badge_reconcile"
    assert exc_info.value.operation_status == "pending_badge_reconcile"
    assert [name for name, _params in supabase.rpc_calls] == [
        "apply_match_exclusions_atomic"
    ]


def test_failed_replay_persists_recovery_stage(monkeypatch):
    _patch_badges(monkeypatch)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", "1")
    monkeypatch.setattr(
        service,
        "run_admin_replay_history",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("replay unavailable")
        ),
    )
    recovery_calls = []
    supabase = FakeSupabase(
        rpc_handlers={
            "apply_match_exclusions_atomic": _pending_apply(),
            "mark_match_exclusion_recovery_required": (
                lambda params: recovery_calls.append(params)
                or {
                    "ok": True,
                    "operation_id": OPERATION_ID,
                    "operation_status": "recovery_required",
                    "recovery_stage": "replay",
                }
            ),
        }
    )

    with pytest.raises(
        service.MatchExclusionRecoveryRequired
    ) as exc_info:
        service.apply_atomic_match_exclusions(
            supabase,
            club_id="club",
            targets=[{"match_id": 7, "expected_row_version": 3}],
            actor_email="admin@example.com",
            actor_role="super_admin",
            source="test",
            note="Wrong row",
            idempotency_key=IDEMPOTENCY_KEY,
        )

    assert exc_info.value.operation_id == OPERATION_ID
    assert exc_info.value.replay_job_id == REPLAY_JOB_ID
    assert exc_info.value.recovery_stage == "replay"
    assert recovery_calls[0]["p_recovery_stage"] == "replay"
    assert "replay unavailable" in recovery_calls[0]["p_error_text"]


def test_failed_replay_status_is_recovery_required_not_active(monkeypatch):
    _patch_badges(monkeypatch)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", "1")
    monkeypatch.setattr(service, "uuid4", lambda: OPERATION_ID)
    monkeypatch.setattr(
        service,
        "run_admin_replay_history",
        lambda *_args, **_kwargs: {
            "ok": False,
            "job_id": REPLAY_JOB_ID,
            "job_status": "failed",
        },
    )
    recovery_calls = []
    supabase = FakeSupabase(
        rpc_handlers={
            "apply_match_exclusions_atomic": _pending_apply(),
            "mark_match_exclusion_recovery_required": (
                lambda params: recovery_calls.append(params)
                or {
                    "ok": True,
                    "operation_id": OPERATION_ID,
                    "operation_status": "recovery_required",
                    "recovery_stage": "replay",
                }
            ),
        }
    )

    with pytest.raises(service.MatchExclusionRecoveryRequired) as exc_info:
        service.apply_atomic_match_exclusions(
            supabase,
            club_id="club",
            targets=[{"match_id": 7, "expected_row_version": 3}],
            actor_email="admin@example.com",
            actor_role="super_admin",
            source="test",
            note="Wrong row",
            idempotency_key=IDEMPOTENCY_KEY,
        )

    assert not isinstance(exc_info.value, service.MatchExclusionWorkActive)
    assert exc_info.value.recovery_stage == "replay"
    assert len(recovery_calls) == 1
    assert "job status: failed" in recovery_calls[0]["p_error_text"]


def test_recover_badge_stage_is_idempotent_and_finalizes(monkeypatch):
    _patch_badges(monkeypatch)
    operation = {
        "id": OPERATION_ID,
        "club_id": "club",
        "idempotency_key": IDEMPOTENCY_KEY,
        "mode": "exclude",
        "status": "recovery_required",
        "recovery_stage": "badge_reconcile",
        "excluded_match_ids": [7],
        "affected_player_ids": [1, 2, 3, 4],
        "badge_ids": ["first_win"],
        "badge_contract_version": "jupr:match-exclusion-badges:v1",
        "replay_job_id": REPLAY_JOB_ID,
        "result_json": {},
    }
    supabase = FakeSupabase(
        operations=[operation],
        rpc_handlers={
            "finalize_match_exclusion_operation": {
                **_succeeded(),
                "inserted_count": 2,
                "updated_count": 1,
                "revoked_count": 3,
                "badge_results": [
                    {"player_id": 1, "status": "succeeded"}
                ],
            }
        },
    )

    result = service.recover_atomic_match_exclusion(
        supabase,
        club_id="club",
        operation_id=OPERATION_ID,
        actor_email="admin@example.com",
        actor_role="super_admin",
        source="test-recovery",
    )

    assert result["ok"] is True
    assert result["mode"] == "match_exclusion_recovered"
    assert result["operation_status"] == "succeeded"
    assert result["badge_reconcile"]["inserted_count"] == 2
    assert result["badge_reconcile"]["updated_count"] == 1
    assert result["badge_reconcile"]["revoked_count"] == 3
    assert (
        result["badge_reconcile"]["contract_version"]
        == "jupr:match-exclusion-badges:v1"
    )
    assert result["badge_reconcile"]["player_ids"] == [1, 2, 3, 4]
    assert result["badge_reconcile"]["processed_player_ids"] == [1, 2, 3, 4]


def test_targets_require_exact_unique_positive_versions(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", "1")
    supabase = FakeSupabase()

    with pytest.raises(ValueError, match="only once"):
        service.apply_atomic_match_exclusions(
            supabase,
            club_id="club",
            targets=[
                {"match_id": 7, "expected_row_version": 3},
                {"match_id": 7, "expected_row_version": 3},
            ],
            actor_email="admin@example.com",
            actor_role="super_admin",
            source="test",
            note=None,
            idempotency_key=IDEMPOTENCY_KEY,
        )
