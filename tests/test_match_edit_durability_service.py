from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.services import match_edit_durability_service as service


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.update_payload = None

    def select(self, *_args, **_kwargs):
        return self

    def update(self, payload):
        self.update_payload = dict(payload)
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def limit(self, _value):
        return self

    def execute(self):
        rows = list(self.storage.setdefault(self.table_name, []))
        for key, value in self.filters:
            rows = [row for row in rows if str(row.get(key)) == str(value)]
        if self.update_payload is not None:
            for row in rows:
                row.update(self.update_payload)
        return SimpleNamespace(data=rows)


class FakeRpc:
    def __init__(self, owner, name, params):
        self.owner = owner
        self.name = name
        self.params = params

    def execute(self):
        self.owner.rpc_calls.append((self.name, self.params))
        return SimpleNamespace(data=dict(self.owner.rpc_result))


class FakeSupabase:
    def __init__(self, rpc_result, storage=None):
        self.rpc_result = rpc_result
        self.storage = storage or {"match_edit_operations": []}
        self.rpc_calls = []

    def rpc(self, name, params):
        return FakeRpc(self, name, params)

    def table(self, name):
        return FakeQuery(self.storage, name)


def test_metadata_only_edit_is_atomic_without_replay(monkeypatch):
    supabase = FakeSupabase({
        "operation_id": "op-1",
        "status": "succeeded",
        "updated_ids": [4, 5],
        "updated_count": 2,
        "recompute_scope": {"standings": False, "ratings": False},
        "idempotent": False,
    })
    monkeypatch.setattr(service, "is_admin_replay_enabled", lambda: False)

    result = service.apply_atomic_match_edits(
        supabase,
        club_id="club",
        patches=[{"id": 4, "notes": "reviewed"}, {"id": 5, "notes": "reviewed"}],
        actor_email="admin@example.com",
        actor_role="club_owner",
        correction_note="Bulk note",
        source="test",
        idempotency_key="same-request",
    )

    assert result["ok"] is True
    assert result["atomic"] is True
    assert result["updated_ids"] == [4, 5]
    assert supabase.rpc_calls[0][0] == "apply_match_log_patches_atomic"


def test_rating_edit_requires_replay_flag_before_rpc(monkeypatch):
    supabase = FakeSupabase({})
    monkeypatch.setattr(service, "is_admin_replay_enabled", lambda: False)

    with pytest.raises(PermissionError, match="require JUPR_ENABLE_NEXT_ADMIN_REPLAY"):
        service.apply_atomic_match_edits(
            supabase,
            club_id="club",
            patches=[{"id": 4, "score_t1": 11}],
            actor_email="admin@example.com",
            actor_role="club_owner",
            correction_note=None,
            source="test",
            idempotency_key="rating-request",
        )

    assert supabase.rpc_calls == []


def test_rating_edit_reports_success_only_after_tracked_replay(monkeypatch):
    storage = {"match_edit_operations": [{"id": "op-2", "status": "pending_replay"}]}
    supabase = FakeSupabase({
        "operation_id": "op-2",
        "status": "pending_replay",
        "updated_ids": [4],
        "updated_count": 1,
        "recompute_scope": {"standings": True, "ratings": True},
        "replay_target": service.FULL_RESET_LABEL,
        "replay_job_id": "job-2",
        "idempotent": False,
    }, storage)
    monkeypatch.setattr(service, "is_admin_replay_enabled", lambda: True)
    monkeypatch.setattr(service, "run_admin_replay_history", lambda *_args, **_kwargs: {
        "job_id": "job-2", "job_status": "succeeded", "result": {"matches_rewritten": 1}, "warnings": []
    })

    result = service.apply_atomic_match_edits(
        supabase,
        club_id="club",
        patches=[{"id": 4, "score_t1": 11}],
        actor_email="admin@example.com",
        actor_role="club_owner",
        correction_note="Correct score",
        source="test",
        idempotency_key="rating-request",
    )

    assert result["ok"] is True
    assert result["mode"] == "applied_and_replayed"
    assert result["replay_job_id"] == "job-2"
    assert storage["match_edit_operations"][0]["status"] == "succeeded"


def test_failed_mandatory_replay_persists_recovery_state(monkeypatch):
    storage = {"match_edit_operations": [{"id": "op-3", "status": "pending_replay"}]}
    supabase = FakeSupabase({
        "operation_id": "op-3",
        "status": "pending_replay",
        "updated_ids": [4],
        "updated_count": 1,
        "recompute_scope": {"standings": True, "ratings": True},
        "replay_target": service.FULL_RESET_LABEL,
        "replay_job_id": "job-3",
    }, storage)
    monkeypatch.setattr(service, "is_admin_replay_enabled", lambda: True)
    monkeypatch.setattr(service, "run_admin_replay_history", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(service.MatchEditRecoveryRequired) as exc_info:
        service.apply_atomic_match_edits(
            supabase,
            club_id="club",
            patches=[{"id": 4, "score_t1": 11}],
            actor_email="admin@example.com",
            actor_role="club_owner",
            correction_note="Correct score",
            source="test",
            idempotency_key="rating-request",
        )

    assert exc_info.value.operation_id == "op-3"
    assert storage["match_edit_operations"][0]["status"] == "recovery_required"
    assert "boom" in storage["match_edit_operations"][0]["error_text"]


def test_idempotent_succeeded_edit_does_not_run_or_audit_replay_again(monkeypatch):
    supabase = FakeSupabase({
        "operation_id": "op-4",
        "status": "succeeded",
        "updated_ids": [4],
        "updated_count": 1,
        "recompute_scope": {"standings": True, "ratings": True},
        "replay_job_id": "job-4",
        "result_json": {"replay": {"job_id": "job-4", "job_status": "succeeded", "result": {"matches_rewritten": 1}}},
        "idempotent": True,
    })
    monkeypatch.setattr(service, "is_admin_replay_enabled", lambda: True)
    monkeypatch.setattr(service, "run_admin_replay_history", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not replay")))

    result = service.apply_atomic_match_edits(
        supabase,
        club_id="club",
        patches=[{"id": 4, "score_t1": 11}],
        actor_email="admin@example.com",
        actor_role="club_owner",
        correction_note="Correct score",
        source="test",
        idempotency_key="rating-request",
    )

    assert result["ok"] is True
    assert result["idempotent"] is True
    assert result["replay_job_id"] == "job-4"
