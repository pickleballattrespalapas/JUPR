from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from jupr_app.domain.tournament_admin_operations import (
    build_tournament_admin_operation_request,
    stable_tournament_admin_fingerprint,
)
from jupr_app.domain.admin_activity_log import ActivityLogWriteResult
from jupr_app.domain.tournament_registration_repo import publish_registration_configuration
from jupr_app.services.admin_tournament_guarded_operation import (
    StaleTournamentAdminStateError,
    TournamentAdminMutationNotAppliedError,
    TournamentAdminRecoveryRequiredError,
    reconcile_tournament_admin_guarded_operation,
    require_tournament_admin_mutation_runtime,
    run_tournament_admin_guarded_operation,
)
from jupr_app.services.admin_tournament_setup_service import (
    get_admin_tournament_setup_detail,
    publish_admin_tournament_setup,
    review_admin_tournament_setup_impact,
)
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_api_contract_admin_tournament_setup import FakeSupabase as SetupFakeSupabase


def _enable_staging(monkeypatch, surface: str = "registration") -> None:
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test-key")
    flag = {
        "registration": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_REGISTRATION_MUTATIONS",
        "tournament": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_MUTATIONS",
        "setup": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_SETUP_MUTATIONS",
        "operations": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS",
    }[surface]
    monkeypatch.setenv(flag, "1")


def _run(
    supabase,
    *,
    mutate,
    expected_state: str = "v1",
    preflight=None,
    current_state=None,
    reconcile=None,
    idempotency_key: str | None = None,
):
    return run_tournament_admin_guarded_operation(
        supabase,
        club_id="club",
        surface="registration",
        action="tournament_registration_update",
        entity_type="tournament_registration",
        entity_id="registration-1",
        expected_state=expected_state,
        current_state=current_state or (lambda: "v1"),
        payload={"status": "waitlist"},
        actor_email="admin@example.com",
        actor_role="club_owner",
        source="test_tournament_admin_guard",
        preflight=preflight,
        reconcile=reconcile,
        mutate=mutate,
        idempotency_key=idempotency_key,
    )


def test_generic_reconcile_never_verifies_an_in_flight_intent(monkeypatch) -> None:
    _enable_staging(monkeypatch)
    request = build_tournament_admin_operation_request(
        club_id="club",
        surface="registration",
        action="tournament_registration_update",
        entity_type="tournament_registration",
        entity_id="registration-1",
        expected_state="v1",
        payload={"status": "waitlist"},
    )
    operation = {
        **request,
        "status": "intent",
        "request_json": request,
        "result_json": {},
        "attempt_count": 1,
    }
    supabase = FakeSupabase(
        {"tournament_admin_operations": [operation], "admin_activity_log": []}
    )
    verifier_called = False

    def verify(_operation):
        nonlocal verifier_called
        verifier_called = True
        return {"status": "not_applied", "evidence": {}}

    with pytest.raises(TournamentAdminRecoveryRequiredError, match="may still be in flight"):
        reconcile_tournament_admin_guarded_operation(
            supabase,
            club_id="club",
            surface="registration",
            operation_key=request["operation_key"],
            entity_type="tournament_registration",
            entity_id="registration-1",
            actor_email="admin@example.com",
            actor_role="club_owner",
            source="test_generic_intent_reconcile",
            verify_outcome=verify,
        )

    assert verifier_called is False
    assert operation["status"] == "intent"
    assert supabase.tables["admin_activity_log"] == []


def test_operation_identity_is_deterministic_and_payload_sensitive() -> None:
    first = build_tournament_admin_operation_request(
        club_id="club",
        surface="registration",
        action="update",
        entity_type="registration",
        entity_id="r1",
        expected_state="v1",
        payload={"b": 2, "a": 1},
    )
    second = build_tournament_admin_operation_request(
        club_id="club",
        surface="registration",
        action="update",
        entity_type="registration",
        entity_id="r1",
        expected_state="v1",
        payload={"a": 1, "b": 2},
    )
    changed = build_tournament_admin_operation_request(
        club_id="club",
        surface="registration",
        action="update",
        entity_type="registration",
        entity_id="r1",
        expected_state="v2",
        payload={"a": 1, "b": 2},
    )

    assert first["request_fingerprint"] == second["request_fingerprint"]
    assert first["operation_key"] == second["operation_key"]
    assert len(first["operation_key"]) == 64
    assert changed["operation_key"] != first["operation_key"]
    assert stable_tournament_admin_fingerprint({"z": 1, "a": 2}) == stable_tournament_admin_fingerprint({"a": 2, "z": 1})


def test_staging_runtime_requires_surface_flag_and_server_only_service_role(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_REGISTRATION_MUTATIONS", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    with pytest.raises(PermissionError, match="REGISTRATION_MUTATIONS"):
        require_tournament_admin_mutation_runtime("registration")

    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_REGISTRATION_MUTATIONS", "1")
    with pytest.raises(RuntimeError, match="SUPABASE_SERVICE_ROLE_KEY"):
        require_tournament_admin_mutation_runtime("registration")

    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test-key")
    require_tournament_admin_mutation_runtime("registration")

    monkeypatch.setenv("JUPR_ENV", "production")
    with pytest.raises(PermissionError, match="staging-only"):
        require_tournament_admin_mutation_runtime("registration")


def test_guard_writes_intent_before_mutation_and_replays_without_second_write(monkeypatch) -> None:
    _enable_staging(monkeypatch)
    tables = {"tournament_admin_operations": [], "admin_activity_log": [], "domain_rows": []}
    supabase = FakeSupabase(tables)
    mutation_calls = 0
    preflight_calls = 0

    def preflight():
        nonlocal preflight_calls
        preflight_calls += 1

    def mutate():
        nonlocal mutation_calls
        mutation_calls += 1
        assert tables["admin_activity_log"][-1]["action_type"].endswith("_intent")
        tables["domain_rows"].append({"id": "row-1", "status": "waitlist"})
        return {"ok": True, "registration": dict(tables["domain_rows"][0])}

    first = _run(supabase, mutate=mutate, preflight=preflight)
    replay = _run(supabase, mutate=mutate, preflight=preflight)

    assert mutation_calls == 1
    assert preflight_calls == 1
    assert first["idempotent_replay"] is False
    assert replay["idempotent_replay"] is True
    assert replay["operation_key"] == first["operation_key"]
    assert tables["tournament_admin_operations"][0]["status"] == "completed"
    assert [row["action_type"] for row in tables["admin_activity_log"]] == [
        "tournament_registration_update_intent",
        "tournament_registration_update_completion",
    ]


def test_same_uuid_insert_race_refetches_winner_and_stays_recovery_required(
    monkeypatch,
) -> None:
    _enable_staging(monkeypatch, "operations")
    client_key = "11111111-1111-4111-8111-111111111111"
    payload = {
        "draw_id": "draw-1",
        "import_mode": "REPLACE",
        "expected_draw_updated_at": "2026-08-15T10:00:00+00:00",
    }
    request = build_tournament_admin_operation_request(
        club_id="club",
        surface="operations",
        action="ops_registration_import",
        entity_type="tournament_event_draw",
        entity_id="draw-1",
        lock_scope="tournament-1",
        expected_state="v1",
        payload=payload,
        idempotency_key=client_key,
    )
    tables = {
        "tournament_admin_operations": [],
        "admin_activity_log": [],
        "domain_rows": [],
    }
    supabase = FakeSupabase(tables)

    def lose_insert(_supabase, _payload):
        tables["tournament_admin_operations"].append(
            {
                **request,
                "client_idempotency_key": client_key,
                "status": "intent",
                "request_json": dict(request),
                "result_json": {},
                "attempt_count": 1,
            }
        )
        raise StaleTournamentAdminStateError("duplicate key")

    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_guarded_operation._insert_operation",
        lose_insert,
    )

    with pytest.raises(
        TournamentAdminRecoveryRequiredError,
        match="won the durable idempotency race and may still be in flight",
    ):
        run_tournament_admin_guarded_operation(
            supabase,
            club_id="club",
            surface="operations",
            action="ops_registration_import",
            entity_type="tournament_event_draw",
            entity_id="draw-1",
            lock_scope="tournament-1",
            expected_state="v1",
            current_state=lambda: "v1",
            payload=payload,
            actor_email="admin@example.com",
            actor_role="club_owner",
            source="next_tournament_ops_import_registrations",
            mutate=lambda: tables["domain_rows"].append({"id": "unsafe"}),
            idempotency_key=client_key,
        )

    assert tables["domain_rows"] == []
    assert tables["tournament_admin_operations"][0]["status"] == "intent"
    assert tables["admin_activity_log"] == []


def test_unrelated_active_lock_insert_collision_remains_definite_and_never_mutates(
    monkeypatch,
) -> None:
    _enable_staging(monkeypatch, "operations")
    tables = {
        "tournament_admin_operations": [],
        "admin_activity_log": [],
        "domain_rows": [],
    }
    supabase = FakeSupabase(tables)

    def lose_to_unrelated_lock(_supabase, _payload):
        tables["tournament_admin_operations"].append(
            {
                "operation_key": "another-operation",
                "request_fingerprint": "another-request",
                "client_idempotency_key": "22222222-2222-4222-8222-222222222222",
                "club_id": "club",
                "surface": "operations",
                "action": "ops_create_draw",
                "entity_type": "tournament_event_draw",
                "entity_id": "another-draw",
                "lock_scope": "tournament-1",
                "expected_state": "v1",
                "status": "intent",
                "request_json": {},
                "result_json": {},
            }
        )
        raise StaleTournamentAdminStateError("duplicate key")

    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_guarded_operation._insert_operation",
        lose_to_unrelated_lock,
    )

    with pytest.raises(
        StaleTournamentAdminStateError,
        match="duplicate key",
    ):
        run_tournament_admin_guarded_operation(
            supabase,
            club_id="club",
            surface="operations",
            action="ops_registration_import",
            entity_type="tournament_event_draw",
            entity_id="draw-1",
            lock_scope="tournament-1",
            expected_state="v1",
            current_state=lambda: "v1",
            payload={
                "draw_id": "draw-1",
                "import_mode": "REPLACE",
                "expected_draw_updated_at": "2026-08-15T10:00:00+00:00",
            },
            actor_email="admin@example.com",
            actor_role="club_owner",
            source="next_tournament_ops_import_registrations",
            mutate=lambda: tables["domain_rows"].append({"id": "unsafe"}),
            idempotency_key="11111111-1111-4111-8111-111111111111",
        )

    assert tables["domain_rows"] == []
    assert tables["admin_activity_log"] == []


def test_stale_state_refuses_operation_audit_and_domain_mutation(monkeypatch) -> None:
    _enable_staging(monkeypatch)
    tables = {"tournament_admin_operations": [], "admin_activity_log": [], "domain_rows": []}
    supabase = FakeSupabase(tables)

    with pytest.raises(StaleTournamentAdminStateError, match="changed after it was loaded"):
        _run(supabase, expected_state="stale", mutate=lambda: tables["domain_rows"].append({"id": "unsafe"}))

    assert tables["tournament_admin_operations"] == []
    assert tables["admin_activity_log"] == []
    assert tables["domain_rows"] == []


def test_state_is_rechecked_after_atomic_lock_acquisition_before_intent_audit(monkeypatch) -> None:
    _enable_staging(monkeypatch)
    tables = {"tournament_admin_operations": [], "admin_activity_log": [], "domain_rows": []}
    supabase = FakeSupabase(tables)
    states = iter(["v1", "v2"])

    with pytest.raises(StaleTournamentAdminStateError, match="lock was being acquired"):
        _run(
            supabase,
            current_state=lambda: next(states),
            mutate=lambda: tables["domain_rows"].append({"id": "unsafe"}),
        )

    assert tables["tournament_admin_operations"][0]["status"] == "failed"
    assert tables["admin_activity_log"] == []
    assert tables["domain_rows"] == []


def test_sql_cas_stale_after_intent_marks_failed_and_releases_recovery_lock(monkeypatch) -> None:
    _enable_staging(monkeypatch)
    tables = {"tournament_admin_operations": [], "admin_activity_log": [], "domain_rows": []}
    supabase = FakeSupabase(tables)

    def stale_cas():
        raise StaleTournamentAdminStateError("exact child snapshot changed under lock")

    with pytest.raises(StaleTournamentAdminStateError, match="child snapshot changed"):
        _run(supabase, mutate=stale_cas)

    assert tables["domain_rows"] == []
    assert tables["tournament_admin_operations"][0]["status"] == "failed"
    assert [row["action_type"] for row in tables["admin_activity_log"]] == [
        "tournament_registration_update_intent",
        "tournament_registration_update_failure",
    ]


def test_partial_mutation_exception_is_recovery_required_and_never_blindly_retried(monkeypatch) -> None:
    _enable_staging(monkeypatch)
    tables = {"tournament_admin_operations": [], "admin_activity_log": [], "domain_rows": []}
    supabase = FakeSupabase(tables)
    mutation_calls = 0

    def response_lost_after_write():
        nonlocal mutation_calls
        mutation_calls += 1
        tables["domain_rows"].append({"id": "possibly-committed"})
        raise TimeoutError("response lost after domain write")

    with pytest.raises(TournamentAdminRecoveryRequiredError, match="partial or response-lost"):
        _run(supabase, mutate=response_lost_after_write)

    assert mutation_calls == 1
    assert tables["domain_rows"] == [{"id": "possibly-committed"}]
    assert tables["tournament_admin_operations"][0]["status"] == "recovery_required"
    assert tables["admin_activity_log"][-1]["action_type"] == "tournament_registration_update_failure"

    with pytest.raises(TournamentAdminRecoveryRequiredError, match="unresolved recovery state"):
        _run(supabase, mutate=response_lost_after_write)
    assert mutation_calls == 1


def test_server_rejected_atomic_mutation_is_failed_not_recovery_required(
    monkeypatch,
) -> None:
    _enable_staging(monkeypatch)
    tables = {
        "tournament_admin_operations": [],
        "admin_activity_log": [],
        "domain_rows": [],
    }
    supabase = FakeSupabase(tables)

    def server_rejected_without_write():
        raise TournamentAdminMutationNotAppliedError(
            "The database rejected the atomic game schedule; no games were created."
        )

    with pytest.raises(
        TournamentAdminMutationNotAppliedError,
        match="no games were created",
    ):
        _run(supabase, mutate=server_rejected_without_write)

    assert tables["domain_rows"] == []
    assert tables["tournament_admin_operations"][0]["status"] == "failed"
    assert [row["action_type"] for row in tables["admin_activity_log"]] == [
        "tournament_registration_update_intent",
        "tournament_registration_update_failure",
    ]


@pytest.mark.parametrize(
    "mutation_error",
    [
        StaleTournamentAdminStateError("the reviewed snapshot changed"),
        TournamentAdminMutationNotAppliedError(
            "The database rejected the atomic game schedule; no games were created."
        ),
    ],
)
def test_definite_no_write_keeps_lock_when_failure_audit_is_missing(
    monkeypatch,
    mutation_error,
) -> None:
    _enable_staging(monkeypatch)
    tables = {
        "tournament_admin_operations": [],
        "admin_activity_log": [],
        "domain_rows": [],
    }
    supabase = FakeSupabase(tables)
    audit_attempts = 0

    def fail_only_failure_audit(_supabase, _payload):
        nonlocal audit_attempts
        audit_attempts += 1
        return ActivityLogWriteResult(ok=audit_attempts == 1)

    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_guarded_operation.write_admin_activity_log",
        fail_only_failure_audit,
    )

    def server_rejected_without_write():
        raise mutation_error

    with pytest.raises(
        TournamentAdminRecoveryRequiredError,
        match="active operation lock remains in place",
    ):
        _run(supabase, mutate=server_rejected_without_write)

    assert tables["domain_rows"] == []
    assert tables["tournament_admin_operations"][0]["status"] == "intent"
    assert audit_attempts == 2


def test_definite_failure_blocks_same_uuid_but_fresh_uuid_can_mutate_unchanged_state(
    monkeypatch,
) -> None:
    _enable_staging(monkeypatch)
    tables = {
        "tournament_admin_operations": [],
        "admin_activity_log": [],
        "domain_rows": [],
    }
    supabase = FakeSupabase(tables)
    first_key = "11111111-1111-4111-8111-111111111111"
    fresh_key = "22222222-2222-4222-8222-222222222222"
    mutation_calls = 0

    def rejected_without_write():
        nonlocal mutation_calls
        mutation_calls += 1
        raise TournamentAdminMutationNotAppliedError(
            "The database rejected the atomic game schedule; no games were created."
        )

    with pytest.raises(TournamentAdminMutationNotAppliedError) as rejected:
        _run(
            supabase,
            mutate=rejected_without_write,
            idempotency_key=first_key,
        )

    failed_operation = tables["tournament_admin_operations"][0]
    assert rejected.value.operation_key == failed_operation["operation_key"]
    assert failed_operation["status"] == "failed"

    with pytest.raises(
        TournamentAdminRecoveryRequiredError,
        match="unresolved recovery state",
    ):
        _run(
            supabase,
            mutate=lambda: tables["domain_rows"].append({"id": "unsafe"}),
            idempotency_key=first_key,
        )

    def successful_fresh_attempt():
        nonlocal mutation_calls
        mutation_calls += 1
        tables["domain_rows"].append({"id": "schedule-1"})
        return {"ok": True, "game_count": 36}

    result = _run(
        supabase,
        mutate=successful_fresh_attempt,
        idempotency_key=fresh_key,
    )

    assert mutation_calls == 2
    assert tables["domain_rows"] == [{"id": "schedule-1"}]
    assert result["game_count"] == 36
    assert result["client_idempotency_key"] == fresh_key
    assert result["operation_key"] != failed_operation["operation_key"]
    assert [row["status"] for row in tables["tournament_admin_operations"]] == [
        "failed",
        "completed",
    ]


def test_empty_recovery_result_reconciles_only_from_callback_without_second_mutation(monkeypatch) -> None:
    _enable_staging(monkeypatch)
    tables = {"tournament_admin_operations": [], "admin_activity_log": [], "domain_rows": []}
    supabase = FakeSupabase(tables)
    mutation_calls = 0
    reconcile_calls = 0

    def response_lost_after_write():
        nonlocal mutation_calls
        mutation_calls += 1
        tables["domain_rows"].append({"id": "official-1"})
        raise TimeoutError("response lost")

    with pytest.raises(TournamentAdminRecoveryRequiredError):
        _run(supabase, mutate=response_lost_after_write)

    def exact_readback(_operation):
        nonlocal reconcile_calls
        reconcile_calls += 1
        assert tables["domain_rows"] == [{"id": "official-1"}]
        return {"ok": True, "match_count": 1}

    reconciled = _run(supabase, mutate=response_lost_after_write, reconcile=exact_readback)

    assert reconciled["reconciled"] is True
    assert reconciled["idempotent_replay"] is True
    assert mutation_calls == 1
    assert reconcile_calls == 1
    assert tables["tournament_admin_operations"][0]["status"] == "completed"


def test_empty_or_partial_reconciliation_evidence_stays_recovery_required(monkeypatch) -> None:
    _enable_staging(monkeypatch)
    tables = {"tournament_admin_operations": [], "admin_activity_log": []}
    supabase = FakeSupabase(tables)
    mutation_calls = 0

    def ambiguous_mutation():
        nonlocal mutation_calls
        mutation_calls += 1
        raise TimeoutError("unknown outcome")

    with pytest.raises(TournamentAdminRecoveryRequiredError):
        _run(supabase, mutate=ambiguous_mutation)

    def partial_readback(_operation):
        raise TournamentAdminRecoveryRequiredError("partial evidence")

    with pytest.raises(TournamentAdminRecoveryRequiredError, match="partial evidence"):
        _run(supabase, mutate=ambiguous_mutation, reconcile=partial_readback)
    assert mutation_calls == 1
    assert tables["tournament_admin_operations"][0]["status"] == "recovery_required"


def test_recovery_replay_refuses_tampered_request_fingerprint(monkeypatch) -> None:
    _enable_staging(monkeypatch)
    tables = {"tournament_admin_operations": [], "admin_activity_log": []}
    supabase = FakeSupabase(tables)

    with pytest.raises(TournamentAdminRecoveryRequiredError):
        _run(supabase, mutate=lambda: (_ for _ in ()).throw(TimeoutError("unknown outcome")))
    tables["tournament_admin_operations"][0]["request_fingerprint"] = "different-request"

    with pytest.raises(ValueError, match="conflicts with a different request"):
        _run(supabase, mutate=lambda: {"ok": True}, reconcile=lambda _operation: {"ok": True})


def test_post_intent_failure_audit_is_attempted_when_recovery_state_update_is_lost(monkeypatch) -> None:
    _enable_staging(monkeypatch)
    tables = {"tournament_admin_operations": [], "admin_activity_log": [], "domain_rows": []}
    supabase = FakeSupabase(tables)

    def lose_recovery_state(**_kwargs):
        raise TournamentAdminRecoveryRequiredError("operation store unavailable")

    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_guarded_operation._update_operation",
        lose_recovery_state,
    )

    def response_lost():
        raise TimeoutError("response lost")

    with pytest.raises(TournamentAdminRecoveryRequiredError, match="recovery state could not be persisted"):
        _run(supabase, mutate=response_lost)

    assert [row["action_type"] for row in tables["admin_activity_log"]] == [
        "tournament_registration_update_intent",
        "tournament_registration_update_failure",
    ]


def test_domain_success_with_lost_mutated_marker_attempts_failure_audit(monkeypatch) -> None:
    _enable_staging(monkeypatch)
    tables = {"tournament_admin_operations": [], "admin_activity_log": [], "domain_rows": []}
    supabase = FakeSupabase(tables)

    def lose_mutated_marker(**kwargs):
        assert kwargs["patch"]["status"] == "mutated"
        raise TournamentAdminRecoveryRequiredError("operation store unavailable")

    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_guarded_operation._update_operation",
        lose_mutated_marker,
    )

    with pytest.raises(TournamentAdminRecoveryRequiredError, match="durable result did not persist"):
        _run(
            supabase,
            mutate=lambda: tables["domain_rows"].append({"id": "committed"}) or {"ok": True},
        )

    assert tables["domain_rows"] == [{"id": "committed"}]
    assert tables["tournament_admin_operations"][0]["status"] == "intent"
    assert [row["action_type"] for row in tables["admin_activity_log"]] == [
        "tournament_registration_update_intent",
        "tournament_registration_update_failure",
    ]


def test_completion_audit_response_loss_reconciles_stored_result_without_second_mutation(monkeypatch) -> None:
    _enable_staging(monkeypatch)
    tables = {"tournament_admin_operations": [], "admin_activity_log": []}
    supabase = FakeSupabase(tables)
    audit_calls = 0
    mutation_calls = 0

    def flaky_audit(_supabase, _payload):
        nonlocal audit_calls
        audit_calls += 1
        return ActivityLogWriteResult(ok=audit_calls != 2, warning="lost" if audit_calls == 2 else None)

    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_guarded_operation.write_admin_activity_log",
        flaky_audit,
    )

    def mutate():
        nonlocal mutation_calls
        mutation_calls += 1
        return {"ok": True, "registration": {"id": "registration-1", "status": "waitlist"}}

    with pytest.raises(TournamentAdminRecoveryRequiredError, match="completion audit failed"):
        _run(supabase, mutate=mutate)
    assert tables["tournament_admin_operations"][0]["status"] == "recovery_required"

    reconciled = _run(supabase, mutate=mutate)
    assert reconciled["idempotent_replay"] is True
    assert reconciled["reconciled"] is True
    assert mutation_calls == 1
    assert tables["tournament_admin_operations"][0]["status"] == "completed"


def test_setup_impact_review_is_deterministic_no_write(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    supabase = SetupFakeSupabase()
    detail = get_admin_tournament_setup_detail(supabase, club_id="club", tournament_id="t1")
    template = detail["templates"][0]
    before = deepcopy(supabase.storage)

    result = review_admin_tournament_setup_impact(
        supabase,
        club_id="club",
        tournament_id="t1",
        days=template["days"],
        event_options=template["event_options"],
        expected_state_fingerprint=detail["state_fingerprint"],
    )

    assert result["dry_run"] is True
    assert result["write_count"] == 0
    assert len(result["impact_fingerprint"]) == 64
    assert supabase.storage == before


def test_setup_impact_rows_without_ids_are_deterministic_and_publishable(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    supabase = SetupFakeSupabase()
    detail = get_admin_tournament_setup_detail(supabase, club_id="club", tournament_id="t1")
    days = [{"label": "Day 1", "event_date": "2026-10-01", "enabled": True, "sort_order": 1}]
    events = [
        {
            "registration_day_id": "day_1",
            "event_family_label": "Mixed Doubles",
            "division_name": "Mixed Open",
            "event_type": "MIXED_DOUBLES",
            "gender_restriction": "MIXED",
            "skill_label": "Open",
            "skill_mode": "OPEN",
            "enabled": True,
            "sort_order": 1,
        }
    ]
    before = deepcopy(supabase.storage)

    first = review_admin_tournament_setup_impact(
        supabase,
        club_id="club",
        tournament_id="t1",
        days=days,
        event_options=events,
        expected_state_fingerprint=detail["state_fingerprint"],
    )
    second = review_admin_tournament_setup_impact(
        supabase,
        club_id="club",
        tournament_id="t1",
        days=days,
        event_options=events,
        expected_state_fingerprint=detail["state_fingerprint"],
    )
    publish_preflight = publish_admin_tournament_setup(
        supabase,
        club_id="club",
        tournament_id="t1",
        days=days,
        event_options=events,
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="PUBLISH SETUP",
        expected_state_fingerprint=detail["state_fingerprint"],
        reviewed_impact_fingerprint=first["impact_fingerprint"],
        dry_run=True,
    )

    assert first["impact_fingerprint"] == second["impact_fingerprint"]
    assert first["publish_impact"]["draft_days"][0]["id"].startswith("day_")
    assert first["publish_impact"]["draft_event_options"][0]["id"].startswith("event_")
    assert publish_preflight["dry_run"] is True
    assert supabase.storage == before


def test_setup_publish_db_path_refuses_cross_tournament_row_ids_before_write(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    supabase = SetupFakeSupabase()
    supabase.storage["tournament_registration_days"].append(
        {"id": "foreign-day", "tournament_id": "other-tournament", "label": "Foreign", "sort_order": 1}
    )
    before = deepcopy(supabase.storage)

    with pytest.raises(ValueError, match="belongs to another tournament"):
        publish_registration_configuration(
            supabase,
            tournament_id="t1",
            days=[{"id": "foreign-day", "tournament_id": "t1", "label": "Day 1", "sort_order": 1}],
            event_options=[
                {
                    "id": "new-event",
                    "tournament_id": "t1",
                    "registration_day_id": "foreign-day",
                    "label": "Open",
                    "event_type": "SINGLES",
                    "gender_restriction": "ANY",
                    "sort_order": 1,
                }
            ],
        )

    assert supabase.storage == before


def test_private_operation_schema_and_ui_recovery_contracts() -> None:
    migration = Path("supabase/migrations/20260719203000_tournament_admin_operations.sql").read_text().lower()
    setup_panel = Path("apps/web/app/admin/tournament-setup/TournamentSetupPanel.tsx").read_text()
    registration_panel = Path("apps/web/app/admin/tournaments/registrations/RegistrationManagementPanel.tsx").read_text()
    api_main = Path("services/api/main.py").read_text()

    assert "enable row level security" in migration
    assert "revoke all on table public.tournament_admin_operations from public, anon, authenticated" in migration
    assert "grant select, insert, update, delete on table public.tournament_admin_operations to service_role" in migration
    assert "lock_scope text not null" in migration
    assert "idx_tournament_admin_operations_active_lock" in migration
    assert "where status in ('intent', 'mutated', 'recovery_required')" in migration
    assert "function public.admin_delete_empty_tournament_draft_cas" in migration
    assert "for update" in migration
    assert "jupr_tournament_not_empty" in migration
    assert "grant execute on function public.admin_delete_empty_tournament_draft_cas" in migration
    assert 'allow_methods=["GET", "POST", "PUT", "PATCH", "OPTIONS"]' in api_main
    assert "Review publish impact (dry run)" in setup_panel
    assert "expected_state_fingerprint" in setup_panel
    assert "direct_import_available" in registration_panel
    assert "Registration Admin cannot bypass" in registration_panel
