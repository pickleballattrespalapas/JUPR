from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

from jupr_app.data.load import load_data
from jupr_app.domain.match_canonical_audit import build_match_canonical_audit
from jupr_app.domain.match_canonical_migration import normalize_legacy_matches_for_canonical
from jupr_app.services.admin_guarded_write_service import (
    GuardedWriteRecoveryRequired,
    begin_guarded_operation,
    canonical_fingerprint,
    get_guarded_operation,
    operation_result,
    require_staging_service_role_write,
    update_guarded_operation,
)
from jupr_app.services.staging_write_guard import (
    staging_match_canonical_normalize_writes_enabled,
)

TRUTHY = {"1", "true", "yes", "y", "on"}
CONFIRM_APPLY = "APPLY NORMALIZE"
WORKFLOW = "match_canonical_normalize"


def is_admin_match_canonical_audit_enabled() -> bool:
    return os.getenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_CANONICAL_AUDIT", "").strip().lower() in TRUTHY


def build_admin_match_canonical_audit_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    enabled = is_admin_match_canonical_audit_enabled()
    return {
        "ok": True,
        "enabled": enabled,
        "status": "ready" if enabled and supabase is not None else "disabled",
        "club_id": str(club_id),
        "required_permissions": {"audit": "view_audit_log", "apply": "manage_matches"},
        "write_environment": "staging_only",
        "service_role_required": True,
        "normalize_writes_enabled": staging_match_canonical_normalize_writes_enabled(),
        "options_endpoint": f"/admin/clubs/{club_id}/match-canonical-audit/options" if enabled else None,
        "audit_endpoint": f"/admin/clubs/{club_id}/match-canonical-audit/run" if enabled else None,
        "normalize_endpoint": f"/admin/clubs/{club_id}/match-canonical-audit/normalize" if enabled else None,
        "confirmation_text": CONFIRM_APPLY,
        "recovery_routes": {
            "operation_status": f"/admin/clubs/{club_id}/match-canonical-audit/operations/{{operation_key}}",
            "match_log": "/admin/match-log",
            "replay_history": "/admin/replay-history",
            "streamlit_fallback": "match_canonical_audit",
        },
    }


def _load_ctx(supabase: Any, *, club_id: str, match_limit: int = 5000) -> Any:
    (
        df_players_all,
        df_players_active,
        df_leagues,
        df_matches,
        df_meta,
        df_badges,
        df_player_badges,
        name_to_id,
        id_to_name,
        schema_degraded,
        schema_degraded_reason,
    ) = load_data(supabase, str(club_id), match_limit=int(match_limit))
    return SimpleNamespace(
        supabase=supabase,
        club_id=str(club_id),
        df_players_all=df_players_all,
        df_players_active=df_players_active,
        df_leagues=df_leagues,
        df_matches=df_matches,
        df_meta=df_meta,
        df_badges=df_badges,
        df_player_badges=df_player_badges,
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        public_mode=False,
        admin_logged_in=True,
        schema_degraded=schema_degraded,
        schema_degraded_reason=schema_degraded_reason,
    )


def build_admin_match_canonical_options(supabase: Any, *, club_id: str) -> dict[str, Any]:
    if not is_admin_match_canonical_audit_enabled():
        raise PermissionError("Next Match Canonical Audit is disabled.")
    ctx = _load_ctx(supabase, club_id=str(club_id), match_limit=5000)
    players = []
    df_players = ctx.df_players_all
    if df_players is not None and not df_players.empty and "id" in df_players.columns:
        for _, row in df_players.sort_values("name" if "name" in df_players.columns else "id").iterrows():
            try:
                pid = int(row.get("id"))
            except Exception:
                continue
            players.append({"player_id": pid, "player_name": str(row.get("name") or f"Player {pid}")})
    leagues = []
    df_matches = ctx.df_matches
    if df_matches is not None and not df_matches.empty and "league" in df_matches.columns:
        leagues = sorted(
            str(item).strip()
            for item in df_matches["league"].dropna().astype(str).tolist()
            if str(item).strip()
        )
        leagues = sorted(set(leagues))
    return {"ok": True, "players": players, "leagues": leagues, "schema_degraded": bool(ctx.schema_degraded), "schema_degraded_reason": ctx.schema_degraded_reason}


def run_admin_match_canonical_audit(
    supabase: Any,
    *,
    club_id: str,
    player_id: int,
    league_id: str | None = None,
    limit: int = 1200,
) -> dict[str, Any]:
    if not is_admin_match_canonical_audit_enabled():
        raise PermissionError("Next Match Canonical Audit is disabled.")
    ctx = _load_ctx(supabase, club_id=str(club_id), match_limit=max(int(limit or 1200), 5000))
    report = build_match_canonical_audit(
        ctx,
        club_id=str(club_id),
        player_id=int(player_id),
        league_id=str(league_id).strip() if league_id else None,
        limit=max(100, min(int(limit or 1200), 5000)),
    )
    return {"ok": True, "mode": "match_canonical_audit", "report": report, "schema_degraded": bool(ctx.schema_degraded), "schema_degraded_reason": ctx.schema_degraded_reason}


def run_admin_match_canonical_normalize(
    supabase: Any,
    *,
    club_id: str,
    player_id: int,
    match_ids: list[int] | None,
    dry_run: bool,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    preview_fingerprint: str = "",
    operation_key: str = "",
    source: str = "next_match_canonical_audit",
) -> dict[str, Any]:
    if not is_admin_match_canonical_audit_enabled():
        raise PermissionError("Next Match Canonical Audit is disabled.")
    if not bool(dry_run) and str(confirmation_text or "").strip().upper() != CONFIRM_APPLY:
        raise ValueError(f"Type {CONFIRM_APPLY} to apply canonical normalization updates.")
    ctx = _load_ctx(supabase, club_id=str(club_id), match_limit=5000)
    result = normalize_legacy_matches_for_canonical(
        supabase,
        ctx=ctx,
        club_id=str(club_id),
        player_id=int(player_id),
        match_ids=[int(mid) for mid in (match_ids or [])],
        dry_run=True,
    )
    if bool(dry_run):
        return {
            "ok": True,
            "mode": "match_canonical_normalize_preview",
            "read_only": True,
            "result": result,
            "preview_fingerprint": result.get("preview_fingerprint"),
            "confirmation_text": CONFIRM_APPLY,
        }

    selected_ids = sorted({int(mid) for mid in (match_ids or [])})
    if not selected_ids:
        raise ValueError("Select the exact reviewed match IDs before applying canonical normalization.")
    current_fingerprint = str(result.get("preview_fingerprint") or "")
    if not preview_fingerprint or str(preview_fingerprint).strip() != current_fingerprint:
        raise ValueError("Canonical normalization preview is stale. Run and review a new dry run before applying.")
    proposals_by_id = {
        int(row.get("match_id")): row
        for row in (result.get("proposals") or [])
        if row.get("match_id") is not None
    }
    if set(selected_ids) != set(proposals_by_id):
        missing = sorted(set(selected_ids) - set(proposals_by_id))
        extra = sorted(set(proposals_by_id) - set(selected_ids))
        raise ValueError(
            "Apply IDs must exactly match the current reviewed dry-run proposal set. "
            f"Missing proposals: {missing or 'none'}; unselected proposals: {extra or 'none'}."
        )
    if result.get("manual_review_needed"):
        raise ValueError("Canonical dry run still contains manual-review rows. Resolve or narrow the selection before applying.")

    rpc_patches = [
        {
            "id": match_id,
            "expected": proposals_by_id[match_id].get("expected") or {},
            "changes": proposals_by_id[match_id].get("patch") or {},
        }
        for match_id in selected_ids
    ]
    request_payload = {
        "player_id": int(player_id),
        "match_ids": selected_ids,
        "preview_fingerprint": current_fingerprint,
        "patches": rpc_patches,
    }
    require_staging_service_role_write(
        supabase,
        workflow="Match Canonical Audit",
        required_tables=("matches", "match_edit_operations"),
    )
    operation, idempotent = begin_guarded_operation(
        supabase,
        club_id=str(club_id),
        workflow=WORKFLOW,
        action="match_canonical_normalize",
        operation_key=operation_key,
        request_payload=request_payload,
        actor_email=actor_email,
        actor_role=actor_role,
        source=source,
        before_json={"preview_fingerprint": current_fingerprint, "patches": rpc_patches},
    )
    if idempotent:
        return operation_result(operation)

    fingerprint = canonical_fingerprint(request_payload)
    try:
        response = supabase.rpc(
            "apply_match_canonical_patches_atomic",
            {
                "p_club_id": str(club_id),
                "p_operation_key": str(operation_key).strip(),
                "p_request_fingerprint": fingerprint,
                "p_patches": rpc_patches,
            },
        ).execute()
    except Exception as exc:
        current = get_guarded_operation(
            supabase,
            club_id=str(club_id),
            workflow=WORKFLOW,
            operation_key=str(operation_key).strip(),
        )
        if current and str(current.get("status")) == "completed":
            return operation_result(current)
        if current and str(current.get("status")) == "intent_recorded":
            update_guarded_operation(
                supabase,
                operation_id=current.get("id"),
                operation_key=operation_key,
                status="failed",
                error_text=str(exc),
            )
            raise RuntimeError("Atomic canonical apply failed before commit; no match rows were changed.") from exc
        raise GuardedWriteRecoveryRequired(
            str(operation_key),
            "Canonical apply outcome is uncertain. Stop, inspect the operation, Match Log, and Replay History before retrying.",
        ) from exc

    payload = getattr(response, "data", None)
    rpc_result = dict(payload[0] if isinstance(payload, list) and payload else payload or {})
    if sorted(int(value) for value in (rpc_result.get("updated_ids") or [])) != selected_ids:
        update_guarded_operation(
            supabase,
            operation_id=operation.get("id"),
            operation_key=operation_key,
            status="recovery_required",
            result_json=rpc_result,
            error_text="RPC readback IDs did not match reviewed selection.",
        )
        raise GuardedWriteRecoveryRequired(
            str(operation_key),
            "Canonical apply readback disagreed with the reviewed selection. Stop and recover through Match Log/Replay History.",
        )

    try:
        readback_rows = getattr(
            supabase.table("matches")
            .select("id,league,match_type,context_type,context_id,t1_p1,t1_p2,t2_p1,t2_p2,score_t1,score_t2")
            .eq("club_id", str(club_id))
            .in_("id", selected_ids)
            .execute(),
            "data",
            None,
        ) or []
    except Exception as exc:
        update_guarded_operation(
            supabase,
            operation_id=operation.get("id"),
            operation_key=operation_key,
            status="recovery_required",
            result_json=rpc_result,
            error_text=str(exc),
        )
        raise GuardedWriteRecoveryRequired(
            str(operation_key),
            "Canonical apply committed but post-commit readback was unavailable. Stop and recover through Match Log/Replay History.",
        ) from exc
    readback_by_id = {int(row.get("id")): dict(row) for row in readback_rows}
    mismatches: list[int] = []
    for patch in rpc_patches:
        row = readback_by_id.get(int(patch["id"]))
        if row is None or any(row.get(key) != value for key, value in patch["changes"].items()):
            mismatches.append(int(patch["id"]))
    if mismatches:
        # The SQL transaction committed and already wrote its completion audit.
        # Preserve truthful uncertainty instead of ever issuing the patches again.
        update_guarded_operation(
            supabase,
            operation_id=operation.get("id"),
            operation_key=operation_key,
            status="recovery_required",
            result_json=rpc_result,
            error_text=f"Post-commit readback mismatch for IDs {mismatches}",
        )
        raise GuardedWriteRecoveryRequired(
            str(operation_key),
            f"Canonical apply committed but readback failed for {mismatches}. Stop and recover through Match Log/Replay History.",
        )
    return {**rpc_result, "readback_verified": True}


def get_admin_match_canonical_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_key: str,
) -> dict[str, Any]:
    operation = get_guarded_operation(
        supabase,
        club_id=str(club_id),
        workflow=WORKFLOW,
        operation_key=str(operation_key).strip(),
    )
    if operation is None:
        raise ValueError("Canonical operation was not found for this club.")
    return {
        "ok": True,
        "operation_key": operation.get("operation_key"),
        "status": operation.get("status"),
        "result": operation.get("result_json") or {},
        "error": operation.get("error_text"),
        "updated_at": operation.get("updated_at"),
        "recovery": {"match_log": "/admin/match-log", "replay_history": "/admin/replay-history"},
    }
