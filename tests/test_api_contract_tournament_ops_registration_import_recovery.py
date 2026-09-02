from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_registration_import_retains_one_exact_tab_scoped_request() -> None:
    source = (
        ROOT / "apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx"
    ).read_text(encoding="utf-8")

    assert "globalThis.sessionStorage?.getItem(importRecoveryStorageKey)" in source
    assert "globalThis.sessionStorage?.setItem(importRecoveryStorageKey" in source
    assert "globalThis.localStorage" not in source
    assert "const [registrationImportRecoveryLoaded, setRegistrationImportRecoveryLoaded]" in source
    assert "!registrationImportRecoveryLoaded || registrationImportBlocksWrites" in source
    assert "const registrationImportBlocksWrites = registrationImportRecovery !== null" in source
    assert "if (registrationImportRecovery)" in source
    assert "const idempotencyKey = globalThis.crypto.randomUUID()" in source
    assert source.index("persistRegistrationImportRecovery(recovery)") < source.index(
        "return executeRegistrationImport(recovery, generation)"
    )


def test_registration_import_uncertainty_and_reconcile_contract_is_explicit() -> None:
    source = (
        ROOT / "apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx"
    ).read_text(encoding="utf-8")

    assert 'detailRecord?.kind === "failed"' in source
    assert "detailRecord?.recovery_required !== true" in source
    assert "!explicitlyFailed && (response.status >= 500" in source
    assert "[408, 425, 429].includes(response.status)" in source
    assert "response.status === 409 || response.status >= 500" not in source
    assert "actionUncertain(" in source
    assert "retained_request: recovery.body" in source
    assert "RECONCILE REGISTRATION IMPORT" in source
    assert "import-registrations/operations/${encodeURIComponent(recovery.operationReference)}/reconcile" in source


def test_definite_registration_import_failure_clears_the_retained_request() -> None:
    source = (
        ROOT / "apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx"
    ).read_text(encoding="utf-8")

    definite_branch = source.index("if (!registrationImportErrorIsUncertain(error)) {")
    clear_recovery = source.index(
        "persistRegistrationImportRecovery(null);", definite_branch
    )
    throw_error = source.index("throw error;", definite_branch)
    assert definite_branch < clear_recovery < throw_error


def test_backend_reconcile_requires_retained_request_and_never_verifies_intent() -> None:
    routes = (ROOT / "services/api/admin_tournament_routes.py").read_text(
        encoding="utf-8"
    )
    guarded = (
        ROOT / "jupr_app/services/admin_tournament_guarded_operation.py"
    ).read_text(encoding="utf-8")
    recovery = (
        ROOT
        / "jupr_app/services/admin_tournament_registration_import_recovery_service.py"
    ).read_text(encoding="utf-8")

    assert "retained_request: AdminTournamentRegistrationImportRequest" in routes
    assert 'retained_request=payload.retained_request.model_dump(mode="json")' in routes
    assert "reserve_tournament_admin_recovery_tombstone(" in recovery
    assert "TOURNAMENT_ADMIN_RECOVERY_TOMBSTONE_ERROR" in recovery
    assert 'elif status == "intent":' in recovery
    intent_guard = guarded.index('if status == "intent":')
    verifier = guarded.index("verification = dict(verify_outcome")
    assert intent_guard < verifier
