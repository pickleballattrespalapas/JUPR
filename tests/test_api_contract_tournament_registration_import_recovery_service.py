from __future__ import annotations

import pytest

from jupr_app.services.admin_tournament_guarded_operation import (
    StaleTournamentAdminStateError,
    TOURNAMENT_ADMIN_STALE_CAS_NO_WRITE_PREFIX,
    TournamentAdminRecoveryRequiredError,
    run_tournament_admin_guarded_operation,
)
from jupr_app.domain.tournament_admin_operations import (
    build_tournament_admin_operation_request,
)
from jupr_app.services.admin_tournament_registration_import_recovery_service import (
    REGISTRATION_IMPORT_RECONCILE_CONFIRMATION,
    reconcile_admin_tournament_registration_import_operation,
)
from tests.test_admin_match_log_service import FakeSupabase


CLIENT_KEY = "11111111-1111-4111-8111-111111111111"


def _retained_request() -> dict:
    return {
        "import_mode": "REPLACE",
        "idempotency_key": CLIENT_KEY,
        "expected_state_fingerprint": "reviewed-state",
        "expected_draw_updated_at": "2026-08-15T10:00:00+00:00",
        "confirmation_text": "IMPORT REGISTRATIONS",
        "source": "next_tournament_ops_import_registrations",
    }


OPERATION_REQUEST = build_tournament_admin_operation_request(
    club_id="club",
    surface="operations",
    action="ops_registration_import",
    entity_type="tournament_event_draw",
    entity_id="draw_1",
    lock_scope="tour_1",
    expected_state="reviewed-state",
    payload={
        "draw_id": "draw_1",
        "import_mode": "REPLACE",
        "expected_draw_updated_at": "2026-08-15T10:00:00+00:00",
    },
    idempotency_key=CLIENT_KEY,
)
OPERATION_KEY = OPERATION_REQUEST["operation_key"]


def _enable_staging(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test-key")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS", "1")


def _operation(
    *,
    status: str = "recovery_required",
    result=None,
    error_text: str | None = None,
):
    return {
        "operation_key": OPERATION_KEY,
        "request_fingerprint": OPERATION_REQUEST["request_fingerprint"],
        "client_idempotency_key": CLIENT_KEY,
        "club_id": "club",
        "surface": "operations",
        "action": "ops_registration_import",
        "entity_type": "tournament_event_draw",
        "entity_id": "draw_1",
        "lock_scope": "tour_1",
        "expected_state": "reviewed-state",
        "status": status,
        "request_json": dict(OPERATION_REQUEST),
        "result_json": result or {},
        "attempt_count": 1,
        "error_text": error_text
        or "Atomic tournament team write failed; no team set was committed.",
    }


def _reconcile(supabase, *, reference: str = CLIENT_KEY):
    return reconcile_admin_tournament_registration_import_operation(
        supabase,
        club_id="club",
        tournament_id="tour_1",
        draw_id="draw_1",
        operation_reference=reference,
        retained_request=_retained_request(),
        confirmation_text=REGISTRATION_IMPORT_RECONCILE_CONFIRMATION,
        actor_email="admin@example.com",
        actor_role="club_owner",
    )


def test_empty_normal_recovery_never_closes_from_readback_or_error_text(
    monkeypatch,
) -> None:
    _enable_staging(monkeypatch)
    tables = {
        "tournament_admin_operations": [_operation()],
        "admin_activity_log": [],
    }
    supabase = FakeSupabase(tables)
    with pytest.raises(TournamentAdminRecoveryRequiredError, match="cannot prove"):
        _reconcile(supabase)

    assert tables["tournament_admin_operations"][0]["status"] == "recovery_required"
    assert tables["admin_activity_log"] == []


def test_absent_browser_uuid_is_reserved_before_it_is_closed_not_applied(
    monkeypatch,
) -> None:
    _enable_staging(monkeypatch)
    tables = {"tournament_admin_operations": [], "admin_activity_log": []}
    supabase = FakeSupabase(tables)

    result = _reconcile(supabase)

    assert result["recovery_disposition"] == "not_applied"
    assert result["client_idempotency_key"] == CLIENT_KEY
    assert result["recovery_evidence"]["recovery_tombstone_reserved"] is True
    assert tables["tournament_admin_operations"][0]["status"] == "failed"
    assert [row["action_type"] for row in tables["admin_activity_log"]] == [
        "ops_registration_import_recovery_not_applied"
    ]
    assert tables["admin_activity_log"][0]["after_json"]["audit_marker"][
        "operation_status"
    ] == "not_applied"


def test_recovery_tombstone_blocks_a_late_original_request(monkeypatch) -> None:
    _enable_staging(monkeypatch)
    tables = {"tournament_admin_operations": [], "admin_activity_log": []}
    supabase = FakeSupabase(tables)

    result = _reconcile(supabase)
    mutated = False

    def mutate() -> dict:
        nonlocal mutated
        mutated = True
        return {"ok": True}

    with pytest.raises(TournamentAdminRecoveryRequiredError, match="prior attempt"):
        run_tournament_admin_guarded_operation(
            supabase,
            club_id="club",
            surface="operations",
            action="ops_registration_import",
            entity_type="tournament_event_draw",
            entity_id="draw_1",
            lock_scope="tour_1",
            expected_state="reviewed-state",
            current_state=lambda: "reviewed-state",
            payload=dict(OPERATION_REQUEST["payload"]),
            actor_email="admin@example.com",
            actor_role="club_owner",
            source="next_tournament_ops_import_registrations",
            mutate=mutate,
            idempotency_key=CLIENT_KEY,
        )

    assert result["recovery_disposition"] == "not_applied"
    assert mutated is False


def test_mismatched_reference_is_rejected_before_any_tombstone_write(
    monkeypatch,
) -> None:
    _enable_staging(monkeypatch)
    tables = {"tournament_admin_operations": [], "admin_activity_log": []}
    supabase = FakeSupabase(tables)

    with pytest.raises(ValueError, match="does not match the retained request"):
        _reconcile(
            supabase,
            reference="22222222-2222-4222-8222-222222222222",
        )

    assert tables["tournament_admin_operations"] == []
    assert tables["admin_activity_log"] == []


def test_original_intent_winning_tombstone_race_stays_uncertain(
    monkeypatch,
) -> None:
    _enable_staging(monkeypatch)
    tables = {"tournament_admin_operations": [], "admin_activity_log": []}
    supabase = FakeSupabase(tables)

    def lose_insert(_supabase, _payload):
        tables["tournament_admin_operations"].append(_operation(status="intent"))
        raise StaleTournamentAdminStateError("duplicate key")

    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_guarded_operation._insert_operation",
        lose_insert,
    )
    with pytest.raises(TournamentAdminRecoveryRequiredError, match="may still be in flight"):
        _reconcile(supabase)

    assert tables["tournament_admin_operations"][0]["status"] == "intent"
    assert tables["admin_activity_log"] == []


def test_stored_result_reconciles_as_completed_without_fingerprint_guess(
    monkeypatch,
) -> None:
    _enable_staging(monkeypatch)
    stored_result = {
        "ok": True,
        "mode": "tournament_registration_team_import",
        "updated_count": 1,
        "teams": [{"player1_id": 1, "player2_id": 2}],
    }
    tables = {
        "tournament_admin_operations": [
            _operation(result=stored_result),
        ],
        "admin_activity_log": [],
    }
    supabase = FakeSupabase(tables)
    result = _reconcile(supabase)

    assert result["recovery_disposition"] == "completed"
    assert result["updated_count"] == 1
    assert tables["tournament_admin_operations"][0]["status"] == "completed"
    assert tables["admin_activity_log"][0]["action_type"] == (
        "ops_registration_import_reconciliation"
    )


def test_duplicate_race_winner_later_failed_closes_as_audited_not_applied(
    monkeypatch,
) -> None:
    _enable_staging(monkeypatch)
    tables = {
        "tournament_admin_operations": [
            _operation(
                status="failed",
                error_text=(
                    f"{TOURNAMENT_ADMIN_STALE_CAS_NO_WRITE_PREFIX}"
                    "exact child snapshot changed under lock"
                ),
            )
        ],
        "admin_activity_log": [],
    }
    result = _reconcile(FakeSupabase(tables))

    assert result["recovery_disposition"] == "not_applied"
    assert tables["tournament_admin_operations"][0]["status"] == "failed"
    assert tables["tournament_admin_operations"][0]["error_text"] == (
        "authoritative recovery verified no domain effect"
    )
    assert [row["action_type"] for row in tables["admin_activity_log"]] == [
        "ops_registration_import_recovery_not_applied"
    ]
    assert tables["admin_activity_log"][0]["after_json"]["audit_marker"][
        "operation_status"
    ] == "not_applied"


def test_proven_failed_not_applied_reconciliation_retry_is_idempotent(
    monkeypatch,
) -> None:
    _enable_staging(monkeypatch)
    tables = {
        "tournament_admin_operations": [
            _operation(
                status="failed",
                error_text=(
                    f"{TOURNAMENT_ADMIN_STALE_CAS_NO_WRITE_PREFIX}"
                    "exact child snapshot changed under lock"
                ),
            )
        ],
        "admin_activity_log": [],
    }
    supabase = FakeSupabase(tables)

    first = _reconcile(supabase)
    second = _reconcile(supabase)

    assert first["recovery_disposition"] == "not_applied"
    assert second["recovery_disposition"] == "not_applied"
    assert len(tables["admin_activity_log"]) == 1


def test_arbitrary_failed_registration_import_remains_closed_without_new_audit(
    monkeypatch,
) -> None:
    _enable_staging(monkeypatch)
    original_error = "legacy failure without a runner no-write marker"
    tables = {
        "tournament_admin_operations": [
            _operation(status="failed", error_text=original_error)
        ],
        "admin_activity_log": [],
    }

    with pytest.raises(ValueError, match="already closed as not applied or failed"):
        _reconcile(FakeSupabase(tables))

    assert tables["tournament_admin_operations"][0]["error_text"] == original_error
    assert tables["admin_activity_log"] == []


def test_reconciliation_rejects_cross_tournament_reference(monkeypatch) -> None:
    _enable_staging(monkeypatch)
    operation = _operation()
    operation["lock_scope"] = "another_tournament"
    supabase = FakeSupabase(
        {"tournament_admin_operations": [operation], "admin_activity_log": []}
    )

    with pytest.raises(ValueError, match="does not belong to this tournament"):
        _reconcile(supabase)


def test_reconciliation_requires_exact_phrase(monkeypatch) -> None:
    _enable_staging(monkeypatch)
    supabase = FakeSupabase(
        {"tournament_admin_operations": [_operation()], "admin_activity_log": []}
    )

    with pytest.raises(ValueError, match=REGISTRATION_IMPORT_RECONCILE_CONFIRMATION):
        reconcile_admin_tournament_registration_import_operation(
            supabase,
            club_id="club",
            tournament_id="tour_1",
            draw_id="draw_1",
            operation_reference=CLIENT_KEY,
            retained_request=_retained_request(),
            confirmation_text="RECONCILE",
            actor_email="admin@example.com",
            actor_role="club_owner",
        )
