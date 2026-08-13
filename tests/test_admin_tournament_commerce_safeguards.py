from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException

from jupr_app.domain.tournament_commerce import (
    TournamentCommerceValidationError,
)
from jupr_app.services import admin_tournament_commerce_service as service
from services.api import admin_tournament_commerce_routes as routes


UUID = "00000000-0000-4000-8000-000000000001"


class _Chain:
    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self


class _Supabase:
    def table(self, _name):
        return _Chain()


def _route(app: FastAPI, *, path: str, method: str):
    return next(
        row.endpoint
        for row in app.routes
        if getattr(row, "path", "") == path
        and method in getattr(row, "methods", set())
    )


def test_cancel_requires_literal_cancel_before_any_runtime_or_rpc():
    with pytest.raises(TournamentCommerceValidationError, match="Type CANCEL"):
        service.cancel_admin_tournament_commerce_order(
            object(),
            club_id="club",
            tournament_id=UUID,
            registration_id="reg_test",
            expected_order_updated_at="2026-07-01T00:00:00Z",
            reason="Registrant withdrew",
            confirmation_text="yes",
            idempotency_key=UUID,
            actor_email="manager@example.com",
            actor_role="manager",
        )


def test_payment_and_fulfillment_require_action_specific_phrases():
    with pytest.raises(TournamentCommerceValidationError, match="SAVE PAYMENT STATUS"):
        service.update_admin_tournament_commerce_payment(
            object(),
            club_id="club",
            tournament_id=UUID,
            registration_id="reg_test",
            payment_status="PAID",
            expected_order_updated_at="2026-07-01T00:00:00Z",
            idempotency_key=UUID,
            confirmation_text="SAVE",
            actor_email="manager@example.com",
            actor_role="manager",
        )
    with pytest.raises(TournamentCommerceValidationError, match="SAVE FULFILLMENT STATUS"):
        service.update_admin_tournament_commerce_fulfillment(
            object(),
            club_id="club",
            tournament_id=UUID,
            fulfillment_id=UUID,
            status="READY",
            notes="",
            expected_updated_at="2026-07-01T00:00:00Z",
            idempotency_key=UUID,
            confirmation_text="SAVE",
            actor_email="manager@example.com",
            actor_role="manager",
        )


def test_fulfilled_correction_requires_meaningful_note(monkeypatch):
    monkeypatch.setattr(
        service,
        "_execute",
        lambda *_args, **_kwargs: [{"id": UUID, "status": "FULFILLED"}],
    )

    with pytest.raises(
        TournamentCommerceValidationError, match="at least 8 characters"
    ):
        service.update_admin_tournament_commerce_fulfillment(
            _Supabase(),
            club_id="club",
            tournament_id=UUID,
            fulfillment_id=UUID,
            status="READY",
            notes="short",
            expected_updated_at="2026-07-01T00:00:00Z",
            idempotency_key=UUID,
            confirmation_text="SAVE FULFILLMENT STATUS",
            actor_email="manager@example.com",
            actor_role="manager",
        )


def test_completed_replay_does_not_duplicate_shared_admin_audit(monkeypatch):
    monkeypatch.setattr(
        service, "_shared_admin_audit_present", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        service,
        "write_admin_activity_log",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("duplicate shared audit")
        ),
    )

    service._audit_admin_result(
        object(),
        club_id="club",
        tournament_id=UUID,
        actor_email="manager@example.com",
        actor_role="manager",
        action="catalog_replace",
        result={"operation_id": UUID, "idempotent_replay": True},
        source="test",
    )


def test_recovery_inspector_distinguishes_completed_missing_shared_audit(
    monkeypatch,
):
    rows = iter(
        [
            [{"id": UUID, "status": "COMPLETED"}],
            [{"id": 1, "action": "order_apply"}],
        ]
    )
    monkeypatch.setattr(
        service, "_execute", lambda *_args, **_kwargs: next(rows)
    )
    monkeypatch.setattr(
        service, "_shared_admin_audit_present", lambda *_args, **_kwargs: False
    )

    result = service.inspect_admin_tournament_commerce_operation(
        _Supabase(),
        club_id="club",
        tournament_id=UUID,
        operation_id=UUID,
    )

    assert result["authoritative_mutation_complete"] is True
    assert result["recovery_state"] == "shared_audit_retry"
    assert result["safe_retry"] is True
    assert result["retry_mode"] == "same_idempotency_key"


def test_admin_routes_require_tournament_manager_permission_and_denial_audit():
    source = Path(
        "services/api/admin_tournament_commerce_routes.py"
    ).read_text(encoding="utf-8")

    assert "PERMISSION_MANAGE_TOURNAMENTS" in source
    assert "admin_tournament_commerce_denied" in source
    assert "_resolve_role_or_403(" in source
    assert "build_admin_tournament_commerce_status()" in source
    assert "build_admin_tournament_commerce_status(\n            supabase" not in source


def test_disabled_admin_status_does_not_construct_database_client(monkeypatch):
    calls: list[str] = []
    app = FastAPI()
    monkeypatch.setattr(routes, "is_tournament_commerce_enabled", lambda: False)
    monkeypatch.setattr(
        routes,
        "build_admin_tournament_commerce_status",
        lambda: {"available": False},
    )
    routes.install_admin_tournament_commerce_routes(
        app,
        get_supabase_client=lambda: calls.append("database"),
    )
    endpoint = _route(
        app,
        path="/admin/clubs/{club_id}/tournaments/commerce/status",
        method="GET",
    )

    assert endpoint("club") == {"available": False}
    assert calls == []


def test_closed_admin_mutation_runtime_rejects_before_database_access(
    monkeypatch,
):
    calls: list[str] = []
    app = FastAPI()
    monkeypatch.setattr(routes, "is_tournament_commerce_enabled", lambda: True)

    def reject_runtime(*, actor_type: str) -> None:
        assert actor_type == "ADMIN"
        raise PermissionError("Tournament commerce mutations are staging-only.")

    monkeypatch.setattr(
        routes,
        "require_tournament_commerce_mutation_runtime",
        reject_runtime,
    )
    routes.install_admin_tournament_commerce_routes(
        app,
        get_supabase_client=lambda: calls.append("database"),
    )

    cases = [
        (
            "PUT",
            (
                "/admin/clubs/{club_id}/tournaments/commerce/tournaments/"
                "{tournament_id}/catalog"
            ),
            (
                "club",
                UUID,
                routes.AdminTournamentCommerceCatalogSaveRequest(
                    expected_catalog_fingerprint="",
                    catalog={},
                    confirmation_text="SAVE",
                    idempotency_key=UUID,
                ),
            ),
        ),
        (
            "PATCH",
            (
                "/admin/clubs/{club_id}/tournaments/commerce/tournaments/"
                "{tournament_id}/orders/{registration_id}/payment"
            ),
            (
                "club",
                UUID,
                "reg_test",
                routes.AdminTournamentCommercePaymentRequest(
                    payment_status="PAID",
                    expected_order_updated_at="2026-07-01T00:00:00Z",
                    idempotency_key=UUID,
                    confirmation_text="SAVE PAYMENT STATUS",
                ),
            ),
        ),
        (
            "POST",
            (
                "/admin/clubs/{club_id}/tournaments/commerce/tournaments/"
                "{tournament_id}/orders/{registration_id}/cancel"
            ),
            (
                "club",
                UUID,
                "reg_test",
                routes.AdminTournamentCommerceOrderCancelRequest(
                    expected_order_updated_at="2026-07-01T00:00:00Z",
                    reason="Registrant withdrew",
                    confirmation_text="CANCEL",
                    idempotency_key=UUID,
                ),
            ),
        ),
        (
            "PATCH",
            (
                "/admin/clubs/{club_id}/tournaments/commerce/tournaments/"
                "{tournament_id}/fulfillment/{fulfillment_id}"
            ),
            (
                "club",
                UUID,
                UUID,
                routes.AdminTournamentCommerceFulfillmentRequest(
                    status="READY",
                    expected_updated_at="2026-07-01T00:00:00Z",
                    idempotency_key=UUID,
                    confirmation_text="SAVE FULFILLMENT STATUS",
                ),
            ),
        ),
    ]

    for method, path, args in cases:
        endpoint = _route(app, path=path, method=method)
        with pytest.raises(HTTPException) as error:
            endpoint(*args)
        assert error.value.status_code == 403

    assert calls == []
