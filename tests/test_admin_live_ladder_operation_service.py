from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from uuid import UUID

import pytest

from jupr_app.services.admin_live_ladder_operation_service import (
    LiveLadderConflictError,
    LiveLadderPersistenceError,
    LiveLadderUncertainError,
    deterministic_match_context_id,
    deterministic_operation_key,
    is_staging_write_gate_enabled,
    operation_recovery_handoff,
    reconcile_durable_admin_operation,
    replay_durable_admin_operation_if_present,
    run_durable_admin_operation,
)


class FakeQuery:
    def __init__(self, owner, table_name):
        self.owner = owner
        self.table_name = table_name
        self.filters: list[tuple[str, object]] = []
        self.in_filters: list[tuple[str, list[object]]] = []
        self.limit_value: int | None = None
        self.insert_payload = None
        self.update_payload = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def in_(self, key, values):
        self.in_filters.append((key, list(values)))
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def insert(self, payload):
        self.insert_payload = deepcopy(payload)
        return self

    def update(self, payload):
        self.update_payload = deepcopy(payload)
        return self

    def execute(self):
        rows = self.owner.storage.setdefault(self.table_name, [])
        if self.insert_payload is not None:
            if self.table_name == "admin_activity_log" and self.insert_payload.get("action_type") in self.owner.fail_audit_actions:
                raise RuntimeError("audit unavailable")
            if self.table_name == "live_ladder_admin_operations" and self.owner.fail_operation_insert:
                raise RuntimeError("operation ledger unavailable")
            row = deepcopy(self.insert_payload)
            rows.append(row)
            return SimpleNamespace(data=[deepcopy(row)])
        scoped = list(rows)
        for key, expected in self.filters:
            scoped = [row for row in scoped if str(row.get(key)) == str(expected)]
        for key, expected in self.in_filters:
            scoped = [row for row in scoped if row.get(key) in expected]
        if self.update_payload is not None:
            updated = []
            for row in rows:
                if row in scoped:
                    row.update(deepcopy(self.update_payload))
                    updated.append(deepcopy(row))
            return SimpleNamespace(data=updated)
        if self.limit_value is not None:
            scoped = scoped[: self.limit_value]
        return SimpleNamespace(data=[deepcopy(row) for row in scoped])


class FakeSupabase:
    def __init__(self):
        self.storage = {"live_ladder_admin_operations": [], "admin_activity_log": []}
        self.fail_audit_actions: set[str] = set()
        self.fail_operation_insert = False

    def table(self, name):
        return FakeQuery(self, name)


def _run(supabase, mutate, *, payload=None, expected="version-1", resolver=None):
    context_id = deterministic_match_context_id(
        operation_key=deterministic_operation_key(
            club_id="club",
            surface="moneyball",
            operation_type="official_publish",
            entity_id="night-1",
            idempotency_key="stable-key-123",
        ),
        slot=1,
    )
    return run_durable_admin_operation(
        supabase,
        club_id="club",
        surface="moneyball",
        operation_type="official_publish",
        entity_id="night-1",
        idempotency_key="stable-key-123",
        expected_version=expected,
        current_version="version-1",
        request_payload=payload or {"scores": [11, 7]},
        recovery={"match_log_url": "/admin/match-log", "match_context_ids": [context_id]},
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        source="test_live_ladder_operation",
        mutate=mutate,
        current_version_resolver=resolver or (lambda: "version-1"),
    )


def test_durable_operation_persists_intent_completion_and_replays_once():
    supabase = FakeSupabase()
    mutation_calls = []

    first = _run(supabase, lambda: mutation_calls.append("called") or {"ok": True, "submitted_count": 1})
    replay = _run(supabase, lambda: mutation_calls.append("duplicate") or {"ok": True})

    assert mutation_calls == ["called"]
    assert first["idempotent_replay"] is False
    assert replay["idempotent_replay"] is True
    assert replay["submitted_count"] == 1
    assert supabase.storage["live_ladder_admin_operations"][0]["status"] == "completed"
    assert [row["action_type"] for row in supabase.storage["admin_activity_log"]] == [
        "intent_live_ladder_operation_admin",
        "complete_live_ladder_operation_admin",
    ]


def test_recovery_handoff_filters_match_log_but_labels_replay_as_global():
    recovery = operation_recovery_handoff(
        surface="moneyball",
        entity_id="night-1",
        match_context_ids=["context-1"],
    )

    assert recovery["match_log_url"] == (
        "/admin/match-log?context_type=moneyball&context_id=context-1"
    )
    assert recovery["replay_history_url"] == "/admin/replay-history"


def test_recovery_handoff_links_every_unique_match_context():
    recovery = operation_recovery_handoff(
        surface="moneyball",
        entity_id="night-1",
        match_context_ids=["context-1", "context 2", "context-1"],
    )

    assert recovery["match_context_ids"] == ["context-1", "context 2"]
    assert recovery["match_log_url"] == (
        "/admin/match-log?"
        "context_type=moneyball&context_ids=context-1%2Ccontext+2"
    )
    assert recovery["replay_history_url"] == "/admin/replay-history"


def test_match_context_is_stable_uuid_v5_for_canonical_matches_column():
    operation_key = deterministic_operation_key(
        club_id="club",
        surface="moneyball",
        operation_type="official_publish",
        entity_id="night-1",
        idempotency_key="stable-key-123",
    )

    first = deterministic_match_context_id(operation_key=operation_key, slot=1)
    second = deterministic_match_context_id(operation_key=operation_key, slot=1)

    assert first == second
    assert UUID(first).version == 5


def test_durable_operation_rejects_stale_state_without_any_write_or_mutation():
    supabase = FakeSupabase()
    before = deepcopy(supabase.storage)
    mutation_calls = []

    with pytest.raises(LiveLadderConflictError, match="authoritative state changed"):
        _run(supabase, lambda: mutation_calls.append("called") or {"ok": True}, expected="stale-version")

    assert mutation_calls == []
    assert supabase.storage == before


def test_required_intent_audit_failure_blocks_domain_mutation():
    supabase = FakeSupabase()
    supabase.fail_audit_actions.add("intent_live_ladder_operation_admin")
    mutation_calls = []

    with pytest.raises(LiveLadderPersistenceError, match="No domain mutation"):
        _run(supabase, lambda: mutation_calls.append("called") or {"ok": True})

    assert mutation_calls == []
    assert supabase.storage["live_ladder_admin_operations"] == []


def test_mutation_failure_is_audited_and_requires_recovery():
    supabase = FakeSupabase()

    def fail():
        raise RuntimeError("connection lost after request")

    with pytest.raises(LiveLadderUncertainError, match="may have changed domain state"):
        _run(supabase, fail)

    operation = supabase.storage["live_ladder_admin_operations"][0]
    assert operation["status"] == "recovery_required"
    assert operation["result_json"] == {}
    assert supabase.storage["admin_activity_log"][-1]["action_type"] == "fail_live_ladder_operation_admin"
    context_id = supabase.storage["admin_activity_log"][-1]["after_json"]["recovery"]["match_context_ids"][0]
    assert deterministic_match_context_id(
        operation_key=deterministic_operation_key(
            club_id="club",
            surface="moneyball",
            operation_type="official_publish",
            entity_id="night-1",
            idempotency_key="stable-key-123",
        ),
        slot=1,
    ) == context_id


def test_completion_response_loss_reconciles_result_without_second_mutation():
    supabase = FakeSupabase()
    supabase.fail_audit_actions.add("complete_live_ladder_operation_admin")
    mutation_calls = []

    with pytest.raises(LiveLadderUncertainError):
        _run(supabase, lambda: mutation_calls.append("called") or {"ok": True, "submitted_count": 2})

    supabase.fail_audit_actions.clear()
    operation_key = deterministic_operation_key(
        club_id="club",
        surface="moneyball",
        operation_type="official_publish",
        entity_id="night-1",
        idempotency_key="stable-key-123",
    )
    recovered = reconcile_durable_admin_operation(
        supabase,
        club_id="club",
        operation_key=operation_key,
        surface="moneyball",
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        confirmation_text="RECONCILE MONEYBALL",
        expected_confirmation="RECONCILE MONEYBALL",
        source="test_reconcile",
    )

    assert mutation_calls == ["called"]
    assert recovered["idempotent_replay"] is True
    assert recovered["submitted_count"] == 2
    assert supabase.storage["live_ladder_admin_operations"][0]["status"] == "completed"


def test_second_version_check_fails_cleanly_before_mutation():
    supabase = FakeSupabase()
    mutation_calls = []

    with pytest.raises(LiveLadderConflictError, match="write lease"):
        _run(
            supabase,
            lambda: mutation_calls.append("called") or {"ok": True},
            resolver=lambda: "version-2",
        )

    assert mutation_calls == []
    assert supabase.storage["live_ladder_admin_operations"][0]["status"] == "failed"
    assert supabase.storage["admin_activity_log"][-1]["after_json"]["domain_mutation_attempted"] is False


def test_idempotency_key_cannot_be_reused_with_different_payload():
    supabase = FakeSupabase()
    _run(supabase, lambda: {"ok": True}, payload={"scores": [11, 7]})

    with pytest.raises(LiveLadderConflictError, match="different request"):
        _run(supabase, lambda: {"ok": True}, payload={"scores": [11, 8]})


def test_early_replay_returns_stored_result_before_domain_preview_rebuild():
    supabase = FakeSupabase()
    _run(supabase, lambda: {"ok": True, "submitted_count": 1})

    replay = replay_durable_admin_operation_if_present(
        supabase,
        club_id="club",
        surface="moneyball",
        operation_type="official_publish",
        entity_id="night-1",
        idempotency_key="stable-key-123",
        request_payload={"scores": [11, 7]},
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        source="test_early_replay",
    )

    assert replay is not None
    assert replay["submitted_count"] == 1
    assert replay["idempotent_replay"] is True


def test_write_gate_requires_both_staging_and_surface_flag(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_STAGING_NEXT_ADMIN_MONEYBALL_WRITES", "1")
    monkeypatch.setenv("JUPR_ENV", "production")
    assert not is_staging_write_gate_enabled("JUPR_ENABLE_STAGING_NEXT_ADMIN_MONEYBALL_WRITES")

    monkeypatch.setenv("JUPR_ENV", "staging")
    assert is_staging_write_gate_enabled("JUPR_ENABLE_STAGING_NEXT_ADMIN_MONEYBALL_WRITES")

    monkeypatch.setenv("JUPR_ENABLE_STAGING_NEXT_ADMIN_MONEYBALL_WRITES", "0")
    assert not is_staging_write_gate_enabled("JUPR_ENABLE_STAGING_NEXT_ADMIN_MONEYBALL_WRITES")
