from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import PERMISSION_ENTER_SCORES, has_permission, resolve_admin_role
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_moneyball_service import (
    CONFIRM as MONEYBALL_CONFIRM,
    MONEYBALL_WRITE_FLAG,
    build_admin_moneyball_status,
    build_moneyball_preview,
    build_moneyball_settlement_preview,
    is_admin_moneyball_enabled,
    submit_admin_moneyball,
)
from jupr_app.services.admin_live_ladder_operation_service import (
    LiveLadderConflictError,
    LiveLadderPersistenceError,
    LiveLadderUncertainError,
    deterministic_match_context_id,
    deterministic_operation_key,
    get_durable_admin_operation,
    operation_recovery_handoff,
    reconcile_durable_admin_operation,
    replay_durable_admin_operation_if_present,
    require_staging_write_gate,
    run_durable_admin_operation,
)
from services.api.auth import authenticate_bearer, auth_header


class MoneyballPreviewRequest(BaseModel):
    player_ids: list[int] = Field(default_factory=list)
    rating_context: str = "OVERALL"
    win_rate: float = 5.0
    point_rate: float = 2.0


class MoneyballSubmitRequest(MoneyballPreviewRequest):
    scores: list[dict[str, Any]] = Field(default_factory=list)
    league_name: str = "Moneyball"
    week_tag: str = "Moneyball"
    match_type: str = "Moneyball RR"
    confirmation_text: str = ""
    settlement_fingerprint: str = ""
    expected_version: str = ""
    idempotency_key: str = ""
    source: str = "next_moneyball_admin"


class MoneyballSettlementRequest(MoneyballPreviewRequest):
    scores: list[dict[str, Any]] = Field(default_factory=list)


class MoneyballReconcileRequest(BaseModel):
    confirmation_text: str = ""
    source: str = "next_moneyball_reconcile"


CONFIRM_RECONCILE_MONEYBALL = "RECONCILE MONEYBALL"


def _resolve_role_or_403(*, supabase: Any, club_id: str, authorization: str | None, source: str) -> tuple[str, str]:
    user = authenticate_bearer(authorization)
    role_resolution = resolve_admin_role(supabase=supabase, club_id=str(club_id), email=user.email, user_id=user.user_id, allowlist=set())
    if not has_permission(role_resolution.role, PERMISSION_ENTER_SCORES):
        denied = build_activity_payload(
            club_id=str(club_id), actor_email=user.email, actor_role=role_resolution.role, action_type="admin_moneyball_denied", entity_type="moneyball", entity_id="moneyball",
            after_json={"source_client": "fastapi/nextjs", "reason": "insufficient_permission"}, source_page=source, flagged_for_review=True,
        )
        write_admin_activity_log(supabase, denied)
        raise HTTPException(status_code=403, detail="insufficient permission")
    return user.email, role_resolution.role


def _handle(exc: Exception) -> None:
    if isinstance(exc, LiveLadderConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, LiveLadderUncertainError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, LiveLadderPersistenceError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, RuntimeError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def _require_service_role() -> None:
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="Moneyball writes/recovery require SUPABASE_SERVICE_ROLE_KEY on FastAPI; browser and anonymous keys are not accepted.",
        )


def _require_confirmation(actual: str, expected: str) -> None:
    if str(actual or "").strip().upper() != str(expected).upper():
        raise HTTPException(status_code=400, detail=f"Type {expected} to continue.")


def _require_staging_recovery() -> None:
    if os.getenv("JUPR_ENV", "").strip().lower() != "staging":
        raise HTTPException(status_code=403, detail="Moneyball operation recovery is staging-only.")


def _model_payload(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()


def install_admin_moneyball_routes(app, *, get_supabase_client) -> None:
    """Register guarded Moneyball admin routes."""

    @app.get("/admin/clubs/{club_id}/moneyball/status")
    def get_admin_moneyball_status(club_id: str) -> dict[str, Any]:
        supabase = get_supabase_client() if is_admin_moneyball_enabled() else None
        return build_admin_moneyball_status(supabase, club_id=str(club_id))

    @app.post("/admin/clubs/{club_id}/moneyball/preview")
    def post_admin_moneyball_preview(club_id: str, payload: MoneyballPreviewRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_moneyball_enabled():
            raise HTTPException(status_code=403, detail="Next Moneyball is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_moneyball_preview")
        try:
            return build_moneyball_preview(supabase, club_id=str(club_id), player_ids=payload.player_ids, rating_context=payload.rating_context, win_rate=payload.win_rate, point_rate=payload.point_rate)
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/moneyball/settlement")
    def post_admin_moneyball_settlement(club_id: str, payload: MoneyballSettlementRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_moneyball_enabled():
            raise HTTPException(status_code=403, detail="Next Moneyball is disabled.")
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_moneyball_settlement_preview")
        try:
            return build_moneyball_settlement_preview(
                supabase,
                club_id=str(club_id),
                player_ids=payload.player_ids,
                scores=payload.scores,
                rating_context=payload.rating_context,
                win_rate=payload.win_rate,
                point_rate=payload.point_rate,
            )
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/moneyball/submit")
    def post_admin_moneyball_submit(club_id: str, payload: MoneyballSubmitRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        if not is_admin_moneyball_enabled():
            raise HTTPException(status_code=403, detail="Next Moneyball is disabled.")
        authenticate_bearer(authorization)
        _require_service_role()
        try:
            require_staging_write_gate(surface_label="Moneyball", flag_name=MONEYBALL_WRITE_FLAG)
        except Exception as exc:
            _handle(exc)
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        _require_confirmation(payload.confirmation_text, MONEYBALL_CONFIRM)
        try:
            replay = replay_durable_admin_operation_if_present(
                supabase,
                club_id=str(club_id),
                surface="moneyball",
                operation_type="official_publish",
                entity_id=str(payload.week_tag or "moneyball"),
                idempotency_key=payload.idempotency_key,
                request_payload=_model_payload(payload),
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
            )
            if replay is not None:
                return replay
            settlement = build_moneyball_settlement_preview(
                supabase,
                club_id=str(club_id),
                player_ids=payload.player_ids,
                scores=payload.scores,
                rating_context=payload.rating_context,
                win_rate=payload.win_rate,
                point_rate=payload.point_rate,
            )
            current_version = str(settlement.get("settlement_fingerprint") or "")
            if not payload.settlement_fingerprint or payload.settlement_fingerprint != current_version:
                raise LiveLadderConflictError("Moneyball settlement changed. Review the Python settlement again before official publish.")
            if int(settlement.get("would_publish_count") or 0) <= 0:
                raise ValueError("No valid non-tied scored Moneyball matches to save.")
            operation_key = deterministic_operation_key(
                club_id=str(club_id),
                surface="moneyball",
                operation_type="official_publish",
                entity_id=str(payload.week_tag or "moneyball"),
                idempotency_key=payload.idempotency_key,
            )
            match_context_ids = [
                deterministic_match_context_id(
                    operation_key=operation_key,
                    slot=int(row.get("match_index") or 0),
                )
                for row in settlement.get("preview", {}).get("matches", [])
                if any(
                    str(score.get("row_id") or "") == str(row.get("row_id") or "")
                    and int(score.get("score_t1") or 0) != int(score.get("score_t2") or 0)
                    and (int(score.get("score_t1") or 0) + int(score.get("score_t2") or 0)) > 0
                    for score in payload.scores
                )
            ]
            recovery = operation_recovery_handoff(
                surface="moneyball",
                entity_id=str(payload.week_tag or "moneyball"),
                match_context_ids=match_context_ids,
            )
            return run_durable_admin_operation(
                supabase,
                club_id=str(club_id),
                surface="moneyball",
                operation_type="official_publish",
                entity_id=str(payload.week_tag or "moneyball"),
                idempotency_key=payload.idempotency_key,
                expected_version=payload.expected_version,
                current_version=current_version,
                request_payload=_model_payload(payload),
                recovery=recovery,
                actor_email=actor_email,
                actor_role=actor_role,
                source=payload.source,
                mutate=lambda: submit_admin_moneyball(
                    supabase,
                    club_id=str(club_id),
                    player_ids=payload.player_ids,
                    scores=payload.scores,
                    rating_context=payload.rating_context,
                    league_name=payload.league_name,
                    week_tag=payload.week_tag,
                    match_type=payload.match_type,
                    win_rate=payload.win_rate,
                    point_rate=payload.point_rate,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    confirmation_text=payload.confirmation_text,
                    expected_settlement_fingerprint=payload.settlement_fingerprint,
                    publish_context_prefix=operation_key,
                    source=payload.source,
                ),
                current_version_resolver=lambda: str(
                    build_moneyball_settlement_preview(
                        supabase,
                        club_id=str(club_id),
                        player_ids=payload.player_ids,
                        scores=payload.scores,
                        rating_context=payload.rating_context,
                        win_rate=payload.win_rate,
                        point_rate=payload.point_rate,
                    ).get("settlement_fingerprint")
                    or ""
                ),
            )
        except Exception as exc:
            _handle(exc)

    @app.get("/admin/clubs/{club_id}/moneyball/operations/{operation_key}")
    def get_admin_moneyball_operation(club_id: str, operation_key: str, authorization: str | None = auth_header()) -> dict[str, Any]:
        authenticate_bearer(authorization)
        _require_service_role()
        _require_staging_recovery()
        supabase = get_supabase_client()
        _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source="next_moneyball_operation_status")
        try:
            return get_durable_admin_operation(supabase, club_id=str(club_id), operation_key=str(operation_key), surface="moneyball")
        except Exception as exc:
            _handle(exc)

    @app.post("/admin/clubs/{club_id}/moneyball/operations/{operation_key}/reconcile")
    def post_admin_moneyball_reconcile(club_id: str, operation_key: str, payload: MoneyballReconcileRequest, authorization: str | None = auth_header()) -> dict[str, Any]:
        authenticate_bearer(authorization)
        _require_service_role()
        _require_staging_recovery()
        supabase = get_supabase_client()
        actor_email, actor_role = _resolve_role_or_403(supabase=supabase, club_id=str(club_id), authorization=authorization, source=payload.source)
        try:
            return reconcile_durable_admin_operation(
                supabase,
                club_id=str(club_id),
                operation_key=str(operation_key),
                surface="moneyball",
                actor_email=actor_email,
                actor_role=actor_role,
                confirmation_text=payload.confirmation_text,
                expected_confirmation=CONFIRM_RECONCILE_MONEYBALL,
                source=payload.source,
            )
        except Exception as exc:
            _handle(exc)
