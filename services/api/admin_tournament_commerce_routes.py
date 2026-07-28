from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query, Response
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import (
    PERMISSION_MANAGE_TOURNAMENTS,
    has_permission,
    resolve_admin_role,
)
from jupr_app.domain.admin_activity_log import (
    build_activity_payload,
    write_admin_activity_log,
)
from jupr_app.domain.tournament_commerce import (
    TournamentCommerceValidationError,
    tournament_commerce_catalog_payload,
)
from jupr_app.services.admin_tournament_commerce_service import (
    TournamentCommerceRecoveryRequiredError,
    build_admin_tournament_commerce_fulfillment_export,
    build_admin_tournament_commerce_status,
    cancel_admin_tournament_commerce_order,
    get_admin_tournament_commerce_detail,
    inspect_admin_tournament_commerce_operation,
    list_admin_tournament_commerce_tournaments,
    replace_admin_tournament_commerce_catalog,
    update_admin_tournament_commerce_fulfillment,
    update_admin_tournament_commerce_payment,
)
from jupr_app.services.public_tournament_commerce_service import (
    TournamentCommerceConflictError,
    TournamentCommerceUnavailableError,
    is_tournament_commerce_enabled,
    require_tournament_commerce_mutation_runtime,
)
from services.api.auth import authenticate_bearer, auth_header


class AdminTournamentCommerceCatalogPreviewRequest(BaseModel):
    catalog: dict[str, Any]


class AdminTournamentCommerceCatalogSaveRequest(BaseModel):
    expected_catalog_fingerprint: str = Field(default="", max_length=128)
    catalog: dict[str, Any]
    confirmation_text: str = Field(min_length=1, max_length=40)
    idempotency_key: str = Field(min_length=36, max_length=80)
    source: str = Field(
        default="next_tournament_commerce_admin", max_length=160
    )


class AdminTournamentCommercePaymentRequest(BaseModel):
    payment_status: str = Field(min_length=1, max_length=40)
    expected_order_updated_at: str = Field(min_length=1, max_length=80)
    idempotency_key: str = Field(min_length=36, max_length=80)
    source: str = Field(
        default="next_tournament_commerce_admin", max_length=160
    )


class AdminTournamentCommerceOrderCancelRequest(BaseModel):
    expected_order_updated_at: str = Field(min_length=1, max_length=80)
    reason: str = Field(min_length=1, max_length=500)
    confirmation_text: str = Field(min_length=1, max_length=40)
    idempotency_key: str = Field(min_length=36, max_length=80)
    source: str = Field(
        default="next_tournament_commerce_admin", max_length=160
    )


class AdminTournamentCommerceFulfillmentRequest(BaseModel):
    status: str = Field(min_length=1, max_length=40)
    notes: str = Field(default="", max_length=2000)
    expected_updated_at: str = Field(min_length=1, max_length=80)
    idempotency_key: str = Field(min_length=36, max_length=80)
    source: str = Field(
        default="next_tournament_commerce_admin", max_length=160
    )


def _resolve_role_or_403(
    *,
    supabase: Any,
    club_id: str,
    authorization: str | None,
    source: str,
) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    resolution = resolve_admin_role(
        supabase=supabase,
        club_id=str(club_id),
        email=user.email,
        user_id=user.user_id,
        allowlist=set(),
    )
    if not has_permission(
        resolution.role,
        PERMISSION_MANAGE_TOURNAMENTS,
    ):
        write_admin_activity_log(
            supabase,
            build_activity_payload(
                club_id=str(club_id),
                actor_email=user.email,
                actor_role=resolution.role,
                action_type="admin_tournament_commerce_denied",
                entity_type="tournament_commerce",
                entity_id="tournament_commerce",
                after_json={
                    "source_client": "fastapi/nextjs",
                    "reason": "insufficient_permission",
                },
                source_page=source,
                flagged_for_review=True,
            ),
        )
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, resolution.role


def _require_feature() -> None:
    if not is_tournament_commerce_enabled():
        raise HTTPException(
            status_code=403,
            detail="Tournament commerce is disabled.",
        )


def _require_admin_mutation_runtime() -> None:
    """Reject closed or production write surfaces before any database access."""

    try:
        require_tournament_commerce_mutation_runtime(actor_type="ADMIN")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except TournamentCommerceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _handle(exc: Exception) -> None:
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(
        exc,
        (
            TournamentCommerceConflictError,
            TournamentCommerceRecoveryRequiredError,
        ),
    ):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, TournamentCommerceValidationError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, TournamentCommerceUnavailableError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise exc


def install_admin_tournament_commerce_routes(
    app, *, get_supabase_client
) -> None:
    """Install manager-only commerce, fulfillment, and recovery routes."""

    @app.get("/admin/clubs/{club_id}/tournaments/commerce/status")
    def get_tournament_commerce_status(
        club_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        # Match the other admin status surfaces: when disabled, project only
        # local configuration and do not construct a database client.
        if is_tournament_commerce_enabled():
            supabase = get_supabase_client()
            _resolve_role_or_403(
                supabase=supabase,
                club_id=club_id,
                authorization=authorization,
                source="next_tournament_commerce_status",
            )
        # Status projects configuration only; it never probes commerce rows.
        return build_admin_tournament_commerce_status()

    @app.get("/admin/clubs/{club_id}/tournaments/commerce/tournaments")
    def get_tournament_commerce_tournaments(
        club_id: str,
        include_archived: bool = Query(default=True),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_feature()
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=club_id,
            authorization=authorization,
            source="next_tournament_commerce_list",
        )
        try:
            return list_admin_tournament_commerce_tournaments(
                supabase,
                club_id=club_id,
                include_archived=include_archived,
            )
        except Exception as exc:
            _handle(exc)

    @app.get(
        "/admin/clubs/{club_id}/tournaments/commerce/tournaments/"
        "{tournament_id}"
    )
    def get_tournament_commerce_detail(
        club_id: str,
        tournament_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_feature()
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=club_id,
            authorization=authorization,
            source="next_tournament_commerce_detail",
        )
        try:
            return get_admin_tournament_commerce_detail(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/commerce/tournaments/"
        "{tournament_id}/catalog/preview"
    )
    def preview_tournament_commerce_catalog(
        club_id: str,
        tournament_id: str,
        payload: AdminTournamentCommerceCatalogPreviewRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_feature()
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=club_id,
            authorization=authorization,
            source="next_tournament_commerce_preview",
        )
        try:
            catalog = tournament_commerce_catalog_payload(payload.catalog)
        except Exception as exc:
            _handle(exc)
        return {
            "ok": True,
            "mode": "tournament_commerce_catalog_preview",
            "tournament_id": tournament_id,
            "catalog": catalog,
        }

    @app.put(
        "/admin/clubs/{club_id}/tournaments/commerce/tournaments/"
        "{tournament_id}/catalog"
    )
    def put_tournament_commerce_catalog(
        club_id: str,
        tournament_id: str,
        payload: AdminTournamentCommerceCatalogSaveRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_feature()
        _require_admin_mutation_runtime()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=club_id,
            authorization=authorization,
            source=payload.source,
        )
        try:
            return replace_admin_tournament_commerce_catalog(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                expected_catalog_fingerprint=(
                    payload.expected_catalog_fingerprint
                ),
                catalog=payload.catalog,
                idempotency_key=payload.idempotency_key,
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.patch(
        "/admin/clubs/{club_id}/tournaments/commerce/tournaments/"
        "{tournament_id}/orders/{registration_id}/payment"
    )
    def patch_tournament_commerce_payment(
        club_id: str,
        tournament_id: str,
        registration_id: str,
        payload: AdminTournamentCommercePaymentRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_feature()
        _require_admin_mutation_runtime()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=club_id,
            authorization=authorization,
            source=payload.source,
        )
        try:
            return update_admin_tournament_commerce_payment(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                registration_id=registration_id,
                payment_status=payload.payment_status,
                expected_order_updated_at=(
                    payload.expected_order_updated_at
                ),
                idempotency_key=payload.idempotency_key,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.post(
        "/admin/clubs/{club_id}/tournaments/commerce/tournaments/"
        "{tournament_id}/orders/{registration_id}/cancel"
    )
    def post_tournament_commerce_order_cancel(
        club_id: str,
        tournament_id: str,
        registration_id: str,
        payload: AdminTournamentCommerceOrderCancelRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_feature()
        _require_admin_mutation_runtime()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=club_id,
            authorization=authorization,
            source=payload.source,
        )
        try:
            return cancel_admin_tournament_commerce_order(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                registration_id=registration_id,
                expected_order_updated_at=(
                    payload.expected_order_updated_at
                ),
                reason=payload.reason,
                confirmation_text=payload.confirmation_text,
                idempotency_key=payload.idempotency_key,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.patch(
        "/admin/clubs/{club_id}/tournaments/commerce/tournaments/"
        "{tournament_id}/fulfillment/{fulfillment_id}"
    )
    def patch_tournament_commerce_fulfillment(
        club_id: str,
        tournament_id: str,
        fulfillment_id: str,
        payload: AdminTournamentCommerceFulfillmentRequest,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_feature()
        _require_admin_mutation_runtime()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(
            supabase=supabase,
            club_id=club_id,
            authorization=authorization,
            source=payload.source,
        )
        try:
            return update_admin_tournament_commerce_fulfillment(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                fulfillment_id=fulfillment_id,
                status=payload.status,
                notes=payload.notes,
                expected_updated_at=payload.expected_updated_at,
                idempotency_key=payload.idempotency_key,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)

    @app.get(
        "/admin/clubs/{club_id}/tournaments/commerce/tournaments/"
        "{tournament_id}/fulfillment/export"
    )
    def get_tournament_commerce_fulfillment_export(
        club_id: str,
        tournament_id: str,
        authorization: str | None = auth_header(),
    ) -> Response:
        _require_feature()
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=club_id,
            authorization=authorization,
            source="next_tournament_commerce_fulfillment_export",
        )
        try:
            content, filename = (
                build_admin_tournament_commerce_fulfillment_export(
                    supabase,
                    club_id=club_id,
                    tournament_id=tournament_id,
                )
            )
        except Exception as exc:
            _handle(exc)
        return Response(
            content=content,
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )

    @app.get(
        "/admin/clubs/{club_id}/tournaments/commerce/tournaments/"
        "{tournament_id}/operations/{operation_id}"
    )
    def get_tournament_commerce_operation(
        club_id: str,
        tournament_id: str,
        operation_id: str,
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        _require_feature()
        supabase = get_supabase_client()
        _resolve_role_or_403(
            supabase=supabase,
            club_id=club_id,
            authorization=authorization,
            source="next_tournament_commerce_recovery",
        )
        try:
            return inspect_admin_tournament_commerce_operation(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                operation_id=operation_id,
            )
        except Exception as exc:
            _handle(exc)
