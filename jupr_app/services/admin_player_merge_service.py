from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any
import os
from uuid import UUID, uuid4

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_player_editor_service import is_admin_player_editor_enabled

CONFIRM_MERGE = "MERGE"
CONFIRM_COMPENSATE = "COMPENSATE MERGE"
CONFIRM_REPLAY_EVIDENCE = "CONFIRM REPLAY RECOVERY"
TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
PLAYER_COLUMNS = ("t1_p1", "t1_p2", "t2_p1", "t2_p2")


class PlayerMergeConflictError(ValueError):
    """The database changed after the operator reviewed the merge preview."""


class PlayerMergeSetupError(RuntimeError):
    """The server-only transactional merge contract is not installed/configured."""


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_int(value: Any, *, field: str = "value") -> int:
    try:
        return int(float(value))
    except Exception as exc:
        raise ValueError(f"{field} must be a whole number.") from exc


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _clean_uuid(value: Any, *, field: str) -> str:
    try:
        return str(UUID(str(value or "").strip()))
    except Exception as exc:
        raise ValueError(f"{field} must be a UUID.") from exc


def _fetch_player(supabase: Any, *, club_id: str, player_id: int) -> dict[str, Any] | None:
    rows = _safe_rows(
        supabase.table("players")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("id", int(player_id))
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def _fetch_league_rows(supabase: Any, *, club_id: str, player_id: int) -> list[dict[str, Any]]:
    return _safe_rows(
        supabase.table("league_ratings")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("player_id", int(player_id))
        .execute()
    )


def _match_reference_ids(supabase: Any, *, club_id: str, player_id: int) -> dict[str, list[int]]:
    references: dict[str, list[int]] = {}
    for column in PLAYER_COLUMNS:
        rows = _safe_rows(
            supabase.table("matches")
            .select("id")
            .eq("club_id", str(club_id))
            .eq(column, int(player_id))
            .execute()
        )
        references[column] = sorted(
            {
                _safe_int(row.get("id"), field="match_id")
                for row in rows
                if row.get("id") not in (None, "")
            }
        )
    return references


def _match_reference_counts(reference_ids: dict[str, list[int]]) -> dict[str, int]:
    counts = {column: len(reference_ids.get(column) or []) for column in PLAYER_COLUMNS}
    counts["total"] = sum(counts.values())
    return counts


def _match_collision_ids(
    supabase: Any,
    *,
    club_id: str,
    source_player_id: int,
    target_player_id: int,
) -> list[int]:
    source_refs = _match_reference_ids(supabase, club_id=str(club_id), player_id=int(source_player_id))
    source_ids = {match_id for values in source_refs.values() for match_id in values}
    if not source_ids:
        return []
    target_refs = _match_reference_ids(supabase, club_id=str(club_id), player_id=int(target_player_id))
    target_ids = {match_id for values in target_refs.values() for match_id in values}
    return sorted(source_ids & target_ids)


def _official_tournament_match_ids(
    supabase: Any,
    *,
    club_id: str,
    player_id: int,
) -> list[int]:
    """Return immutable official tournament projections for one player.

    Those rows are derived from tournament teams/games. Rewriting only the
    Match projection during a generic player merge would desynchronize the
    authoritative tournament source, so the merge must stop before its RPC.
    """

    match_ids: set[int] = set()
    for column in PLAYER_COLUMNS:
        rows = _safe_rows(
            supabase.table("matches")
            .select("id,tournament_game_id")
            .eq("club_id", str(club_id))
            .eq(column, int(player_id))
            .execute()
        )
        match_ids.update(
            _safe_int(row.get("id"), field="match_id")
            for row in rows
            if row.get("id") not in (None, "")
            and row.get("tournament_game_id") not in (None, "")
        )
    return sorted(match_ids)


def _social_identity_ids(supabase: Any, *, club_id: str, player_id: int) -> list[str]:
    rows = _safe_rows(
        supabase.table("club_people")
        .select("id")
        .eq("club_id", str(club_id))
        .eq("linked_player_id", int(player_id))
        .execute()
    )
    return sorted({str(row.get("id") or "") for row in rows if str(row.get("id") or "").strip()})


def _social_identity_counts(*, source_ids: list[str], target_ids: list[str]) -> dict[str, int]:
    return {"source_linked": len(source_ids), "target_linked": len(target_ids)}


def _merge_expected_state(
    *,
    source: dict[str, Any],
    target: dict[str, Any],
    match_reference_ids: dict[str, list[int]],
    source_league_rows: list[dict[str, Any]],
    move_ids: list[int],
    delete_ids: list[int],
    source_social_ids: list[str],
    target_social_ids: list[str],
) -> dict[str, Any]:
    return {
        "source_player": {
            "id": _safe_int(source.get("id"), field="source_player_id"),
            "name": source.get("name"),
            "active": source.get("active") is not False,
            "inactive_at": source.get("inactive_at"),
        },
        "target_player": {
            "id": _safe_int(target.get("id"), field="target_player_id"),
            "name": target.get("name"),
            "active": target.get("active") is not False,
            "inactive_at": target.get("inactive_at"),
        },
        "match_reference_ids": {
            column: [int(match_id) for match_id in sorted(match_reference_ids.get(column) or [])]
            for column in PLAYER_COLUMNS
        },
        "source_league_rows": sorted(
            (dict(row) for row in source_league_rows),
            key=lambda row: _safe_int(row.get("id"), field="league_rating_id"),
        ),
        "league_rating_plan": {
            "move_ids": [int(row_id) for row_id in sorted(move_ids)],
            "delete_ids": [int(row_id) for row_id in sorted(delete_ids)],
        },
        "source_social_ids": sorted(source_social_ids),
        "target_social_ids": sorted(target_social_ids),
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _recovery_payload(*, operation_id: str, status: str = "merged_pending_replay") -> dict[str, Any]:
    streamlit_fallback_url = (
        os.getenv("JUPR_STREAMLIT_FALLBACK_URL", "https://juprtrespalapas.streamlit.app").strip()
        or "https://juprtrespalapas.streamlit.app"
    )
    return {
        "operation_id": str(operation_id),
        "status": str(status),
        "replay_required": status == "merged_pending_replay",
        "required_replay_scope": "ALL (Full System Reset)",
        "replay_route": f"/admin/replay-history?target=ALL&merge_operation={operation_id}",
        "tracked_replay_fallback_url": streamlit_fallback_url,
        "compensation_endpoint": f"/admin/clubs/{{club_id}}/players/editor/merge/{operation_id}/compensate",
        "replay_evidence_endpoint": f"/admin/clubs/{{club_id}}/players/editor/merge/{operation_id}/replay-evidence",
        "operator_rule": "Run a tracked full Replay History job, then attach its succeeded job ID. If Next does not display a job ID, use the Streamlit Admin Tools replay fallback. Compensate only before replay and only if no newer edits exist.",
    }


def build_admin_player_merge_preview(supabase: Any, *, club_id: str, source_player_id: Any, target_player_id: Any) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    src_id = _safe_int(source_player_id, field="source_player_id")
    dst_id = _safe_int(target_player_id, field="target_player_id")
    if src_id == dst_id:
        raise ValueError("source and target players must be different")
    source = _fetch_player(supabase, club_id=str(club_id), player_id=src_id)
    target = _fetch_player(supabase, club_id=str(club_id), player_id=dst_id)
    if not source:
        raise ValueError("source player not found")
    if not target:
        raise ValueError("target player not found")
    source_leagues = _fetch_league_rows(supabase, club_id=str(club_id), player_id=src_id)
    target_leagues = _fetch_league_rows(supabase, club_id=str(club_id), player_id=dst_id)
    target_names = {str(row.get("league_name") or "") for row in target_leagues}
    move_ids: list[int] = []
    delete_ids: list[int] = []
    conflicts: list[str] = []
    for row in source_leagues:
        league_name = str(row.get("league_name") or "")
        rid = _safe_int(row.get("id"), field="league_rating_id")
        if league_name in target_names:
            conflicts.append(league_name)
            delete_ids.append(rid)
        else:
            move_ids.append(rid)
    match_reference_ids = _match_reference_ids(
        supabase,
        club_id=str(club_id),
        player_id=src_id,
    )
    collision_match_ids = _match_collision_ids(
        supabase,
        club_id=str(club_id),
        source_player_id=src_id,
        target_player_id=dst_id,
    )
    official_tournament_match_ids = _official_tournament_match_ids(
        supabase,
        club_id=str(club_id),
        player_id=src_id,
    )
    source_social_ids = _social_identity_ids(
        supabase,
        club_id=str(club_id),
        player_id=src_id,
    )
    target_social_ids = _social_identity_ids(
        supabase,
        club_id=str(club_id),
        player_id=dst_id,
    )
    expected_state = _merge_expected_state(
        source=source,
        target=target,
        match_reference_ids=match_reference_ids,
        source_league_rows=source_leagues,
        move_ids=move_ids,
        delete_ids=delete_ids,
        source_social_ids=source_social_ids,
        target_social_ids=target_social_ids,
    )
    can_merge = (
        source.get("active") is not False
        and not source.get("inactive_at")
        and target.get("active") is not False
        and not target.get("inactive_at")
        and not collision_match_ids
        and not official_tournament_match_ids
    )
    warnings = [
        "After executing a merge, run a tracked Replay History ALL job and attach its succeeded job ID.",
        "The preview fingerprint is single-use evidence. Refresh the preview after any concurrent player, match, league, or social-link change.",
    ]
    if collision_match_ids:
        warnings.insert(
            0,
            "Merge blocked: source and target already appear in the same match. Correct those matches before merging.",
        )
    if official_tournament_match_ids:
        warnings.insert(
            0,
            "Merge blocked: the source player appears in immutable official tournament matches. Correct the authoritative tournament participant first; a generic player merge cannot rewrite only the rating projection.",
        )
    return {
        "ok": True,
        "mode": "player_merge_preview",
        "source_player": {"id": src_id, "name": _clean_text(source.get("name"), limit=160)},
        "target_player": {"id": dst_id, "name": _clean_text(target.get("name"), limit=160)},
        "can_merge": can_merge,
        "preview_fingerprint": _fingerprint(expected_state),
        "match_reference_counts": _match_reference_counts(match_reference_ids),
        "collision_match_ids": collision_match_ids,
        "official_tournament_match_ids": official_tournament_match_ids,
        "league_rating_plan": {
            "source_rows": source_leagues,
            "target_rows": target_leagues,
            "move_ids": move_ids,
            "delete_ids": delete_ids,
            "conflicts": sorted(set(conflicts)),
        },
        "social_identity_counts": _social_identity_counts(
            source_ids=source_social_ids,
            target_ids=target_social_ids,
        ),
        "warnings": warnings,
    }


def _expected_state_from_preview_inputs(
    supabase: Any,
    *,
    club_id: str,
    source_player_id: int,
    target_player_id: int,
) -> dict[str, Any]:
    source = _fetch_player(supabase, club_id=str(club_id), player_id=source_player_id)
    target = _fetch_player(supabase, club_id=str(club_id), player_id=target_player_id)
    if not source or not target:
        raise PlayerMergeConflictError("The player records changed after preview. Refresh and review the merge again.")
    source_leagues = _fetch_league_rows(supabase, club_id=str(club_id), player_id=source_player_id)
    target_leagues = _fetch_league_rows(supabase, club_id=str(club_id), player_id=target_player_id)
    target_names = {str(row.get("league_name") or "") for row in target_leagues}
    move_ids = [
        _safe_int(row.get("id"), field="league_rating_id")
        for row in source_leagues
        if str(row.get("league_name") or "") not in target_names
    ]
    delete_ids = [
        _safe_int(row.get("id"), field="league_rating_id")
        for row in source_leagues
        if str(row.get("league_name") or "") in target_names
    ]
    return _merge_expected_state(
        source=source,
        target=target,
        match_reference_ids=_match_reference_ids(
            supabase,
            club_id=str(club_id),
            player_id=source_player_id,
        ),
        source_league_rows=source_leagues,
        move_ids=move_ids,
        delete_ids=delete_ids,
        source_social_ids=_social_identity_ids(
            supabase,
            club_id=str(club_id),
            player_id=source_player_id,
        ),
        target_social_ids=_social_identity_ids(
            supabase,
            club_id=str(club_id),
            player_id=target_player_id,
        ),
    )


def _rpc_payload(response: Any) -> dict[str, Any]:
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        return dict(data[0])
    raise PlayerMergeSetupError("Transactional Player Editor RPC returned an invalid response.")


def _raise_rpc_failure(payload: dict[str, Any]) -> None:
    if payload.get("ok") is not False:
        return
    code = str(payload.get("code") or "PLAYER_MERGE_FAILED")
    if code in {"PLAYER_MERGE_STALE_PREVIEW", "PLAYER_MERGE_COMPENSATION_STALE"}:
        raise PlayerMergeConflictError(
            "Player, match, league, or social-link state changed after review. Refresh before taking another action."
        )
    messages = {
        "SOURCE_PLAYER_NOT_FOUND": "source player not found",
        "TARGET_PLAYER_NOT_FOUND": "target player not found",
        "SOURCE_PLAYER_INACTIVE": "source player is inactive",
        "TARGET_PLAYER_INACTIVE": "target player is inactive",
        "PLAYER_MERGE_MATCH_COLLISION": "source and target already appear in the same match",
        "PLAYER_MERGE_REPLAY_IN_PROGRESS": "player merge is blocked while a club replay job is pending or running",
        "PLAYER_MERGE_OPERATION_NOT_FOUND": "player merge operation not found",
        "PLAYER_MERGE_COMPENSATION_NOT_ALLOWED": "merge compensation is no longer allowed",
        "PLAYER_MERGE_COMPENSATION_REPLAY_STARTED": "merge compensation is blocked because a full replay job has already started",
        "PLAYER_MERGE_REPLAY_ALREADY_RESOLVED": "merge replay recovery is already resolved",
        "REPLAY_JOB_NOT_FOUND": "replay job not found",
        "REPLAY_JOB_NOT_VALID_RECOVERY_EVIDENCE": "replay job is not a succeeded full-system replay created after this merge",
    }
    raise ValueError(messages.get(code, code.replace("_", " ").lower()))


def _rpc_available(supabase: Any) -> bool:
    return callable(getattr(supabase, "rpc", None))


def _server_runtime() -> bool:
    return any(
        str(os.getenv(name, "")).strip().lower() in {"staging", "production", "prod"}
        for name in ("JUPR_ENV", "ENVIRONMENT", "VERCEL_ENV")
    )


def _existing_merge_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_id: str,
) -> dict[str, Any] | None:
    rows = _safe_rows(
        supabase.table("admin_player_merge_operations")
        .select("id,club_id,source_player_id,target_player_id,status,preview_fingerprint,result_json")
        .eq("club_id", str(club_id))
        .eq("id", str(operation_id))
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def _execute_local_merge_for_tests(
    supabase: Any,
    *,
    club_id: str,
    preview: dict[str, Any],
    actor_email: str,
    actor_role: str,
    source: str,
    operation_id: str,
) -> dict[str, Any]:
    """Compatibility path for in-memory/local adapters; deployed writes require the RPC."""
    src_id = int(preview["source_player"]["id"])
    dst_id = int(preview["target_player"]["id"])
    src_name = str(preview["source_player"].get("name") or f"#{src_id}")
    dst_name = str(preview["target_player"].get("name") or f"#{dst_id}")
    match_updates: dict[str, int] = {}
    for column in PLAYER_COLUMNS:
        rows = _safe_rows(
            supabase.table("matches")
            .update({column: dst_id})
            .eq("club_id", str(club_id))
            .eq(column, src_id)
            .execute()
        )
        match_updates[column] = len(rows)
    plan = preview["league_rating_plan"]
    deleted_league_rows: list[dict[str, Any]] = []
    for rid in plan.get("delete_ids") or []:
        deleted_league_rows.extend(
            _safe_rows(
                supabase.table("league_ratings")
                .delete()
                .eq("club_id", str(club_id))
                .eq("id", int(rid))
                .execute()
            )
        )
    moved_league_rows: list[dict[str, Any]] = []
    for rid in plan.get("move_ids") or []:
        moved_league_rows.extend(
            _safe_rows(
                supabase.table("league_ratings")
                .update({"player_id": dst_id})
                .eq("club_id", str(club_id))
                .eq("id", int(rid))
                .execute()
            )
        )
    source_social_rows = _safe_rows(
        supabase.table("club_people")
        .select("id")
        .eq("club_id", str(club_id))
        .eq("linked_player_id", src_id)
        .execute()
    )
    target_social_rows = _safe_rows(
        supabase.table("club_people")
        .select("id")
        .eq("club_id", str(club_id))
        .eq("linked_player_id", dst_id)
        .execute()
    )
    social_rows: list[dict[str, Any]] = []
    for row in source_social_rows:
        social_rows.extend(
            _safe_rows(
                supabase.table("club_people")
                .update({"linked_player_id": None if target_social_rows else dst_id})
                .eq("club_id", str(club_id))
                .eq("id", str(row.get("id")))
                .execute()
            )
        )
    source_player_rows = _safe_rows(
        supabase.table("players")
        .update(
            {
                "active": False,
                "inactive_at": _now_iso(),
                "name": f"{src_name} (MERGED into {dst_name} #{dst_id})"[:160],
            }
        )
        .eq("club_id", str(club_id))
        .eq("id", src_id)
        .execute()
    )
    result = {
        "ok": True,
        "mode": "player_merge_execute",
        "transaction_mode": "local_test_adapter",
        "operation_id": operation_id,
        "operation_status": "merged_pending_replay",
        "source_player_id": src_id,
        "target_player_id": dst_id,
        "preview_fingerprint": preview["preview_fingerprint"],
        "match_updates": match_updates,
        "league_rating_plan": plan,
        "moved_league_rating_count": len(moved_league_rows),
        "deleted_conflicting_league_rating_count": len(deleted_league_rows),
        "social_identity_rows_updated": len(social_rows),
        "source_player": source_player_rows[0] if source_player_rows else None,
        "requires_replay": True,
    }
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="merge_player_editor_players_admin",
        entity_type="players",
        entity_id=f"{src_id}->{dst_id}",
        before_json=preview,
        after_json={**result, "source_client": "fastapi/nextjs"},
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings = ["Merge completed. Run Replay History ALL and attach the succeeded replay job ID."]
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return {**result, "warnings": warnings, "recovery": _recovery_payload(operation_id=operation_id)}


def execute_admin_player_merge(
    supabase: Any,
    *,
    club_id: str,
    source_player_id: Any,
    target_player_id: Any,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    preview_fingerprint: str,
    operation_id: str | None = None,
    source: str = "next_player_editor_merge",
) -> dict[str, Any]:
    if str(confirmation_text or "").strip().upper() != CONFIRM_MERGE:
        raise ValueError(f"Type {CONFIRM_MERGE} to merge player records.")
    supplied_fingerprint = str(preview_fingerprint or "").strip().lower()
    if len(supplied_fingerprint) != 64 or any(character not in "0123456789abcdef" for character in supplied_fingerprint):
        raise ValueError("Refresh and review the merge preview before executing it.")
    requested_src_id = _safe_int(source_player_id, field="source_player_id")
    requested_dst_id = _safe_int(target_player_id, field="target_player_id")
    operation_uuid = _clean_uuid(operation_id or uuid4(), field="operation_id")
    if operation_id and _rpc_available(supabase):
        existing = _existing_merge_operation(
            supabase,
            club_id=str(club_id),
            operation_id=operation_uuid,
        )
        if existing:
            if (
                _safe_int(existing.get("source_player_id"), field="source_player_id") != requested_src_id
                or _safe_int(existing.get("target_player_id"), field="target_player_id") != requested_dst_id
                or str(existing.get("preview_fingerprint") or "") != supplied_fingerprint
            ):
                raise ValueError("operation_id already belongs to a different reviewed merge")
            stored_result = existing.get("result_json")
            result = dict(stored_result) if isinstance(stored_result, dict) else {}
            status = str(existing.get("status") or result.get("operation_status") or "unknown")
            return {
                **result,
                "ok": bool(result.get("ok", True)),
                "mode": str(result.get("mode") or "player_merge_execute"),
                "operation_id": operation_uuid,
                "operation_status": status,
                "idempotent_replay": True,
                "requires_replay": status == "merged_pending_replay",
                "warnings": ["Existing merge operation returned; no player rows were written again."],
                "recovery": _recovery_payload(operation_id=operation_uuid, status=status),
            }
    preview = build_admin_player_merge_preview(
        supabase,
        club_id=str(club_id),
        source_player_id=requested_src_id,
        target_player_id=requested_dst_id,
    )
    if supplied_fingerprint != preview["preview_fingerprint"]:
        raise PlayerMergeConflictError("The merge preview is stale. Refresh and review it again.")
    if not preview.get("can_merge"):
        raise ValueError("The merge preview is blocked. Resolve inactive players or match collisions first.")
    src_id = int(preview["source_player"]["id"])
    dst_id = int(preview["target_player"]["id"])
    if not _rpc_available(supabase):
        if _server_runtime():
            raise PlayerMergeSetupError(
                "Transactional Player Editor merge RPC is unavailable; apply the merge migration and configure the server-only service-role client."
            )
        return _execute_local_merge_for_tests(
            supabase,
            club_id=str(club_id),
            preview=preview,
            actor_email=actor_email,
            actor_role=actor_role,
            source=source,
            operation_id=operation_uuid,
        )
    expected_state = _expected_state_from_preview_inputs(
        supabase,
        club_id=str(club_id),
        source_player_id=src_id,
        target_player_id=dst_id,
    )
    if _fingerprint(expected_state) != supplied_fingerprint:
        raise PlayerMergeConflictError("The merge preview became stale before execution. Refresh and review it again.")
    try:
        response = supabase.rpc(
            "server_merge_player_accounts",
            {
                "p_operation_id": operation_uuid,
                "p_club_id": str(club_id),
                "p_source_player_id": src_id,
                "p_target_player_id": dst_id,
                "p_preview_fingerprint": supplied_fingerprint,
                "p_expected_state": expected_state,
                "p_actor_email": str(actor_email or ""),
                "p_actor_role": str(actor_role or ""),
                "p_source_page": str(source or "next_player_editor_merge"),
            },
        ).execute()
    except Exception as exc:
        raise PlayerMergeSetupError(
            "Transactional Player Editor merge failed before a committed result was returned. Check the operation ID before retrying."
        ) from exc
    result = _rpc_payload(response)
    _raise_rpc_failure(result)
    return {
        **result,
        "league_rating_plan": preview["league_rating_plan"],
        "warnings": ["Merge committed atomically. Run Replay History ALL and attach its succeeded replay job ID."],
        "recovery": _recovery_payload(operation_id=operation_uuid),
    }


def get_admin_player_merge_operation(supabase: Any, *, club_id: str, operation_id: Any) -> dict[str, Any]:
    operation_uuid = _clean_uuid(operation_id, field="operation_id")
    rows = _safe_rows(
        supabase.table("admin_player_merge_operations")
        .select("id,club_id,source_player_id,target_player_id,status,preview_fingerprint,result_json,replay_job_id,replay_verified_at,compensated_at,created_at,updated_at")
        .eq("club_id", str(club_id))
        .eq("id", operation_uuid)
        .limit(1)
        .execute()
    )
    if not rows:
        raise ValueError("player merge operation not found")
    row = rows[0]
    return {
        "ok": True,
        "mode": "player_merge_operation",
        "operation": row,
        "recovery": _recovery_payload(
            operation_id=operation_uuid,
            status=str(row.get("status") or "unknown"),
        ),
    }


def compensate_admin_player_merge(
    supabase: Any,
    *,
    club_id: str,
    operation_id: Any,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_player_editor_merge_compensation",
) -> dict[str, Any]:
    if str(confirmation_text or "").strip().upper() != CONFIRM_COMPENSATE:
        raise ValueError(f"Type {CONFIRM_COMPENSATE} to compensate this merge.")
    operation_uuid = _clean_uuid(operation_id, field="operation_id")
    if not _rpc_available(supabase):
        raise PlayerMergeSetupError("Merge compensation requires the server-only transactional RPC.")
    try:
        response = supabase.rpc(
            "server_compensate_player_merge",
            {
                "p_operation_id": operation_uuid,
                "p_club_id": str(club_id),
                "p_actor_email": str(actor_email or ""),
                "p_actor_role": str(actor_role or ""),
                "p_source_page": str(source or "next_player_editor_merge_compensation"),
            },
        ).execute()
    except Exception as exc:
        raise PlayerMergeSetupError("Merge compensation RPC did not return a confirmed result.") from exc
    result = _rpc_payload(response)
    _raise_rpc_failure(result)
    return {
        **result,
        "recovery": _recovery_payload(operation_id=operation_uuid, status="compensated"),
        "warnings": ["Pre-merge player, match, league business fields, and social links were restored; trigger-maintained timestamps may advance. Do not attach replay evidence for this operation."],
    }


def verify_admin_player_merge_replay(
    supabase: Any,
    *,
    club_id: str,
    operation_id: Any,
    replay_job_id: Any,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_player_editor_merge_replay_evidence",
) -> dict[str, Any]:
    if str(confirmation_text or "").strip().upper() != CONFIRM_REPLAY_EVIDENCE:
        raise ValueError(f"Type {CONFIRM_REPLAY_EVIDENCE} to attach replay evidence.")
    operation_uuid = _clean_uuid(operation_id, field="operation_id")
    replay_uuid = _clean_uuid(replay_job_id, field="replay_job_id")
    if not _rpc_available(supabase):
        raise PlayerMergeSetupError("Replay recovery evidence requires the server-only transactional RPC.")
    try:
        response = supabase.rpc(
            "server_verify_player_merge_replay",
            {
                "p_operation_id": operation_uuid,
                "p_club_id": str(club_id),
                "p_replay_job_id": replay_uuid,
                "p_actor_email": str(actor_email or ""),
                "p_actor_role": str(actor_role or ""),
                "p_source_page": str(source or "next_player_editor_merge_replay_evidence"),
            },
        ).execute()
    except Exception as exc:
        raise PlayerMergeSetupError("Replay recovery evidence RPC did not return a confirmed result.") from exc
    result = _rpc_payload(response)
    _raise_rpc_failure(result)
    return {
        **result,
        "recovery": _recovery_payload(operation_id=operation_uuid, status="replay_verified"),
        "warnings": ["Succeeded full-system replay evidence is attached; merge recovery is complete."],
    }
