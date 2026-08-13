from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_guarded_write_service import (
    GuardedWriteRecoveryRequired,
    begin_guarded_operation,
    canonical_fingerprint,
    get_guarded_operation,
    operation_result,
    update_guarded_operation,
)

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
DEFAULT_NEW_PLAYER_JUPR = 3.5


class PlayerEditorConflictError(RuntimeError):
    """A reviewed Player Editor row changed before compare-and-swap."""

    def __init__(self, message: str, *, operation_key: str = ""):
        self.operation_key = str(operation_key or "")
        super().__init__(message)


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def is_admin_player_editor_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR")


def is_api_audit_log_required() -> bool:
    return _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG")


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _jupr_to_elo(value: Any, *, field_name: str) -> float:
    rating = _safe_float(value)
    if rating is None:
        raise ValueError(f"{field_name} is required.")
    if rating < 1.0 or rating > 7.0:
        raise ValueError(f"{field_name} must be between 1.0 and 7.0.")
    return float(rating) * 400.0


def _elo_to_jupr(value: Any) -> float | None:
    rating = _safe_float(value)
    return None if rating is None else float(rating) / 400.0


def _player_payload(row: dict[str, Any]) -> dict[str, Any]:
    pid = _safe_int(row.get("id"))
    payload = {
        "id": int(pid or 0),
        "club_id": str(row.get("club_id") or ""),
        "name": _clean_text(row.get("name"), limit=160),
        "rating": row.get("rating"),
        "rating_jupr": _elo_to_jupr(row.get("rating")),
        "starting_rating": row.get("starting_rating"),
        "starting_jupr": _elo_to_jupr(row.get("starting_rating")),
        "wins": row.get("wins"),
        "losses": row.get("losses"),
        "matches_played": row.get("matches_played"),
        "active": bool(row.get("active", row.get("is_active", True))) and not bool(row.get("inactive_at")),
        "inactive_at": row.get("inactive_at"),
        "last_game_at": row.get("last_game_at"),
    }
    return {**payload, "state_fingerprint": canonical_fingerprint(payload)}


def _league_rating_payload(row: dict[str, Any]) -> dict[str, Any]:
    rid = _safe_int(row.get("id"))
    payload = {
        "id": int(rid or 0),
        "league_name": _clean_text(row.get("league_name"), limit=120),
        "rating": row.get("rating"),
        "rating_jupr": _elo_to_jupr(row.get("rating")),
        "starting_rating": row.get("starting_rating"),
        "starting_jupr": _elo_to_jupr(row.get("starting_rating")),
        "wins": row.get("wins"),
        "losses": row.get("losses"),
        "matches_played": row.get("matches_played"),
        "is_active": bool(row.get("is_active", True)),
        "inactive_at": row.get("inactive_at"),
    }
    return {**payload, "state_fingerprint": canonical_fingerprint(payload)}


def _fetch_players(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    rows = _safe_rows(
        supabase.table("players")
        .select("id,club_id,name,rating,starting_rating,wins,losses,matches_played,active,inactive_at,last_game_at")
        .eq("club_id", str(club_id))
        .order("name", desc=False)
        .execute()
    )
    return [_player_payload(row) for row in rows if _safe_int(row.get("id")) is not None]


def _fetch_player_row(supabase: Any, *, club_id: str, player_id: int) -> dict[str, Any] | None:
    rows = _safe_rows(
        supabase.table("players")
        .select("id,club_id,name,rating,starting_rating,wins,losses,matches_played,active,inactive_at,last_game_at")
        .eq("club_id", str(club_id))
        .eq("id", int(player_id))
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def _fetch_player(supabase: Any, *, club_id: str, player_id: int) -> dict[str, Any] | None:
    row = _fetch_player_row(
        supabase,
        club_id=str(club_id),
        player_id=int(player_id),
    )
    return _player_payload(row) if row else None


def _fetch_player_rows_by_name(
    supabase: Any,
    *,
    club_id: str,
    name: str,
) -> list[dict[str, Any]]:
    return _safe_rows(
        supabase.table("players")
        .select("id,club_id,name,rating,starting_rating,wins,losses,matches_played,active,inactive_at,last_game_at")
        .eq("club_id", str(club_id))
        .eq("name", str(name))
        .execute()
    )


def _mark_recovery_required(
    supabase: Any,
    *,
    operation: dict[str, Any],
    operation_key: str,
    error_text: str,
    result_json: Any = None,
) -> None:
    """Best-effort mark for an ambiguous domain call; caller always raises recovery."""
    try:
        update_guarded_operation(
            supabase,
            operation_id=operation.get("id"),
            operation_key=operation_key,
            status="recovery_required",
            result_json=result_json,
            error_text=error_text,
        )
    except Exception:
        # The durable intent is already present. A stale ledger must not turn an
        # ambiguous domain outcome into a generic retryable 500.
        pass


def _find_player_editor_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_key: str,
) -> dict[str, Any] | None:
    for workflow in (
        "player_editor_create",
        "player_editor_update",
        "player_editor_league_rating_update",
    ):
        operation = get_guarded_operation(
            supabase,
            club_id=str(club_id),
            workflow=workflow,
            operation_key=str(operation_key),
        )
        if operation is not None:
            return operation
    return None


def _payload_matches_patch(current: dict[str, Any], patch: dict[str, Any]) -> bool:
    for field, expected in patch.items():
        if field == "inactive_at":
            continue
        actual = current.get(field)
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            try:
                if abs(float(actual) - float(expected)) > 1e-9:
                    return False
            except (TypeError, ValueError):
                return False
        elif actual != expected:
            return False
    return True


def reconcile_admin_player_editor_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_key: str,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    source: str = "next_player_editor_operation_reconcile",
) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != "RECONCILE PLAYER OPERATION":
        raise ValueError("Type RECONCILE PLAYER OPERATION to reconcile this exact operation.")
    operation = _find_player_editor_operation(
        supabase,
        club_id=str(club_id),
        operation_key=str(operation_key),
    )
    if operation is None:
        raise ValueError("Player Editor operation was not found.")
    status = str(operation.get("status") or "")
    if status == "completed":
        return operation_result(operation)
    if status not in {"intent_recorded", "recovery_required"}:
        raise ValueError(f"Player Editor operation is {status or 'unknown'} and cannot be reconciled.")
    workflow = str(operation.get("workflow") or "")
    evidence = operation.get("result_json") or {}
    before_json = operation.get("before_json") or {}
    authoritative: dict[str, Any] | None = None
    expected_after = None
    expected_before = None
    mode = ""
    action_type = ""
    entity_type = ""
    entity_id = ""

    if workflow == "player_editor_create":
        planned = evidence.get("planned") if isinstance(evidence, dict) else None
        expected_insert = planned.get("player") if isinstance(planned, dict) else None
        preexisting_ids = evidence.get("preexisting_player_ids") if isinstance(evidence, dict) else None
        if not isinstance(expected_insert, dict) or not isinstance(preexisting_ids, list):
            raise GuardedWriteRecoveryRequired(
                str(operation_key),
                "Create-operation evidence is insufficient for automatic reconciliation.",
            )
        expected_after = dict(expected_insert)
        rows = _fetch_player_rows_by_name(
            supabase,
            club_id=str(club_id),
            name=str(expected_after.get("name") or ""),
        )
        new_rows = [row for row in rows if row.get("id") not in preexisting_ids]
        authoritative = _player_payload(new_rows[0]) if len(new_rows) == 1 else None
        proven = authoritative is not None and _payload_matches_patch(authoritative, expected_after)
        create_absence_proven = (
            not new_rows
            and {row.get("id") for row in rows} == set(preexisting_ids)
        )
        mode = "player_editor_create"
        action_type = "reconcile_player_editor_create"
        entity_type = "player"
        entity_id = str((authoritative or {}).get("id") or "")
        expected_before = None
    elif workflow == "player_editor_update":
        planned = evidence.get("planned") if isinstance(evidence, dict) else None
        expected_after = planned.get("player") if isinstance(planned, dict) else None
        before = before_json.get("player") if isinstance(before_json, dict) else None
        if not isinstance(expected_after, dict) or not isinstance(before, dict):
            raise GuardedWriteRecoveryRequired(
                str(operation_key),
                "Player-edit evidence is insufficient for automatic reconciliation.",
            )
        row = _fetch_player_row(
            supabase,
            club_id=str(club_id),
            player_id=int(expected_after.get("id") or before.get("id") or 0),
        )
        authoritative = _player_payload(row) if row else None
        proven = authoritative is not None and authoritative.get("state_fingerprint") == expected_after.get("state_fingerprint")
        mode = "player_editor_update"
        action_type = "reconcile_player_editor_update"
        entity_type = "player"
        entity_id = str(expected_after.get("id") or before.get("id") or "")
        expected_before = before
    elif workflow == "player_editor_league_rating_update":
        planned = evidence.get("planned") if isinstance(evidence, dict) else None
        expected_after = planned.get("league_rating") if isinstance(planned, dict) else None
        before = before_json.get("league_rating") if isinstance(before_json, dict) else None
        if not isinstance(expected_after, dict) or not isinstance(before, dict):
            raise GuardedWriteRecoveryRequired(
                str(operation_key),
                "League-rating evidence is insufficient for automatic reconciliation.",
            )
        rows = _safe_rows(
            supabase.table("league_ratings")
            .select("id,club_id,player_id,league_name,rating,starting_rating,wins,losses,matches_played,is_active,inactive_at")
            .eq("club_id", str(club_id))
            .eq("id", int(expected_after.get("id") or before.get("id") or 0))
            .limit(1)
            .execute()
        )
        row = rows[0] if len(rows) == 1 else None
        authoritative = _league_rating_payload(row) if row else None
        proven = authoritative is not None and authoritative.get("state_fingerprint") == expected_after.get("state_fingerprint")
        mode = "player_editor_league_rating_update"
        action_type = "reconcile_player_editor_league_rating"
        entity_type = "league_rating"
        entity_id = str(expected_after.get("id") or before.get("id") or "")
        expected_before = before
    else:
        raise ValueError("Unsupported Player Editor reconciliation workflow.")

    if not proven:
        if workflow == "player_editor_create" and create_absence_proven:
            update_guarded_operation(
                supabase,
                operation_id=operation.get("id"),
                operation_key=str(operation_key),
                status="failed",
                result_json={
                    "reconciled": True,
                    "proof": "created_player_absent",
                    "preexisting_player_ids": preexisting_ids,
                },
                error_text="Authoritative readback proves the player create did not commit.",
            )
            return {
                "ok": False,
                "mode": "player_editor_create_reconciled_failed",
                "operation_key": str(operation_key),
                "status": "failed",
                "recovery_required": False,
            }
        if expected_before and authoritative and authoritative.get("state_fingerprint") == expected_before.get("state_fingerprint"):
            update_guarded_operation(
                supabase,
                operation_id=operation.get("id"),
                operation_key=str(operation_key),
                status="failed",
                result_json={"reconciled": True, "proof": "authoritative_original_state", "authoritative": authoritative},
                error_text="Authoritative readback proves the requested mutation did not commit.",
            )
            return {
                "ok": False,
                "mode": f"{mode}_reconciled_failed",
                "operation_key": str(operation_key),
                "status": "failed",
                "recovery_required": False,
                "authoritative": authoritative,
            }
        raise GuardedWriteRecoveryRequired(
            str(operation_key),
            "Authoritative state does not exactly prove the intended or original Player Editor state. Keep this operation blocked.",
        )

    audit_write = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=str(actor_email or ""),
            actor_role=str(actor_role or ""),
            action_type=action_type,
            entity_type=entity_type,
            entity_id=entity_id,
            before_json=before_json,
            after_json={
                "source_client": "fastapi/nextjs",
                "source_page": source,
                "operation_key": str(operation_key),
                "proof": "authoritative_state_fingerprint",
                "authoritative": authoritative,
            },
            source_page=source,
            flagged_for_review=True,
        ),
    )
    if not audit_write.ok:
        _mark_recovery_required(
            supabase,
            operation=operation,
            operation_key=str(operation_key),
            result_json={**evidence, "reconcile_audit_failed": True},
            error_text="Authoritative proof succeeded, but reconciliation audit did not persist.",
        )
        raise GuardedWriteRecoveryRequired(
            str(operation_key),
            "Player Editor proof succeeded, but its reconciliation audit is unavailable.",
        )
    result: dict[str, Any] = {
        "ok": True,
        "mode": mode,
        "operation_key": str(operation_key),
        "idempotent_replay": True,
        "reconciled": True,
        "warnings": [],
    }
    if workflow == "player_editor_league_rating_update":
        result["league_rating"] = authoritative
        result["league_ratings"] = []
    else:
        result["player"] = authoritative
    update_guarded_operation(
        supabase,
        operation_id=operation.get("id"),
        operation_key=str(operation_key),
        status="completed",
        after_json={"authoritative": authoritative, "reconciled": True},
        result_json=result,
    )
    return result


def _fetch_league_ratings(supabase: Any, *, club_id: str, player_id: int) -> list[dict[str, Any]]:
    rows = _safe_rows(
        supabase.table("league_ratings")
        .select("id,league_name,rating,starting_rating,wins,losses,matches_played,is_active,inactive_at")
        .eq("club_id", str(club_id))
        .eq("player_id", int(player_id))
        .order("league_name", desc=False)
        .execute()
    )
    return [_league_rating_payload(row) for row in rows if _safe_int(row.get("id")) is not None]


def _match_reference_counts(supabase: Any, *, club_id: str, player_id: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
        try:
            rows = _safe_rows(
                supabase.table("matches")
                .select("id")
                .eq("club_id", str(club_id))
                .eq(column, int(player_id))
                .execute()
            )
        except Exception:
            rows = []
        counts[column] = len(rows)
    counts["total"] = sum(counts.values())
    return counts


def build_admin_player_editor_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "players_endpoint": None,
            "player_detail_endpoint": None,
            "social_identities_endpoint": None,
            "player_merge_endpoint": None,
            "merge_operation_endpoint": None,
            "transactional_merge_ready": False,
            "warnings": ["Next Player Editor is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR on FastAPI for a closed-club pilot."],
        }
    player_count = None
    transactional_merge_ready = False
    if supabase is not None:
        try:
            player_count = len(_fetch_players(supabase, club_id=str(club_id)))
        except Exception:
            player_count = None
        if os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
            try:
                _safe_rows(
                    supabase.table("admin_player_merge_operations")
                    .select("id")
                    .eq("club_id", str(club_id))
                    .limit(1)
                    .execute()
                )
                transactional_merge_ready = True
            except Exception:
                transactional_merge_ready = False
    return {
        "enabled": True,
        "status": "ready_for_transactional_player_editor_pilot",
        "players_endpoint": "/admin/clubs/{club_id}/players/editor/players",
        "player_detail_endpoint": "/admin/clubs/{club_id}/players/editor/players/{player_id}",
        "social_identities_endpoint": "/admin/clubs/{club_id}/players/editor/social-identities",
        "player_merge_endpoint": "/admin/clubs/{club_id}/players/editor/merge",
        "merge_operation_endpoint": "/admin/clubs/{club_id}/players/editor/merge/{operation_id}",
        "transactional_merge_ready": transactional_merge_ready,
        "player_count": player_count,
        "warnings": [
            "Player create/update, league-rating edits, social identity linking, and stale-guarded atomic merge are enabled. Every merge remains pending until succeeded full-replay evidence is attached or pre-replay compensation completes.",
            *([] if transactional_merge_ready else ["Transactional merge is not write-ready until FastAPI has SUPABASE_SERVICE_ROLE_KEY and the merge migration."]),
        ],
    }


def list_admin_player_editor_players(supabase: Any, *, club_id: str) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    players = _fetch_players(supabase, club_id=str(club_id))
    return {"ok": True, "mode": "player_editor_list", "players": players, "count": len(players)}


def get_admin_player_editor_detail(supabase: Any, *, club_id: str, player_id: int) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    player = _fetch_player(supabase, club_id=str(club_id), player_id=int(player_id))
    if player is None:
        raise ValueError("player not found")
    league_ratings = _fetch_league_ratings(supabase, club_id=str(club_id), player_id=int(player_id))
    return {
        "ok": True,
        "mode": "player_editor_detail",
        "player": player,
        "league_ratings": league_ratings,
        "match_reference_counts": _match_reference_counts(supabase, club_id=str(club_id), player_id=int(player_id)),
    }


def create_admin_player_editor_player(
    supabase: Any,
    *,
    club_id: str,
    name: str,
    starting_jupr: Any = DEFAULT_NEW_PLAYER_JUPR,
    actor_email: str,
    actor_role: str,
    idempotency_key: str,
    source: str = "next_player_editor",
) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    clean_name = _clean_text(name, limit=160)
    if not clean_name:
        raise ValueError("Player name is required.")
    rating = _safe_float(starting_jupr)
    if rating is None:
        rating = DEFAULT_NEW_PLAYER_JUPR
    if rating < 1.0 or rating > 7.0:
        raise ValueError("Starting JUPR must be between 1.0 and 7.0.")
    request_payload = {"name": clean_name, "starting_jupr": float(rating)}
    operation, idempotent = begin_guarded_operation(
        supabase,
        club_id=str(club_id),
        workflow="player_editor_create",
        action="create_player_editor_player",
        operation_key=idempotency_key,
        request_payload=request_payload,
        actor_email=actor_email,
        actor_role=actor_role,
        source=source,
        before_json=None,
    )
    if idempotent:
        return operation_result(operation)
    insert_payload = {
        "club_id": str(club_id),
        "name": clean_name,
        "rating": float(rating) * 400.0,
        "starting_rating": float(rating) * 400.0,
        "wins": 0,
        "losses": 0,
        "matches_played": 0,
        "active": True,
        "last_game_at": None,
        "inactive_at": None,
    }
    try:
        preexisting_rows = _fetch_player_rows_by_name(
            supabase,
            club_id=str(club_id),
            name=clean_name,
        )
    except Exception as exc:
        update_guarded_operation(
            supabase,
            operation_id=operation.get("id"),
            operation_key=idempotency_key,
            status="failed",
            error_text="Player create preflight failed before mutation.",
        )
        raise RuntimeError("Player create was not started because preflight could not be verified.") from exc
    update_guarded_operation(
        supabase,
        operation_id=operation.get("id"),
        operation_key=idempotency_key,
        status="intent_recorded",
        result_json={
            "phase": "preflight",
            "planned": {"player": insert_payload},
            "preexisting_player_ids": [row.get("id") for row in preexisting_rows],
        },
    )
    try:
        inserted_rows = _safe_rows(
            supabase.table("players").insert(insert_payload).execute()
        )
    except Exception as exc:
        readback: list[dict[str, Any]] | None = None
        try:
            readback = _fetch_player_rows_by_name(
                supabase,
                club_id=str(club_id),
                name=clean_name,
            )
        except Exception:
            readback = None
        _mark_recovery_required(
            supabase,
            operation=operation,
            operation_key=idempotency_key,
            result_json={
                "planned": {"player": insert_payload},
                "preexisting_player_ids": [row.get("id") for row in preexisting_rows],
                "readback_verified": readback is not None,
                "players": [
                    _player_payload(row)
                    for row in (readback or [])
                ],
            },
            error_text="Player insert returned an ambiguous transport result.",
        )
        raise GuardedWriteRecoveryRequired(
            idempotency_key,
            "The player may have been created. Inspect this exact operation before retrying; do not use a new key.",
        ) from exc
    if len(inserted_rows) != 1:
        _mark_recovery_required(
            supabase,
            operation=operation,
            operation_key=idempotency_key,
            result_json={
                "planned": {"player": insert_payload},
                "preexisting_player_ids": [row.get("id") for row in preexisting_rows],
                "inserted_count": len(inserted_rows),
            },
            error_text="Player insert response did not contain exactly one row.",
        )
        raise GuardedWriteRecoveryRequired(
            idempotency_key,
            "The player create result could not be verified. Inspect this exact operation before retrying.",
        )
    player = _player_payload(inserted_rows[0])
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="create_player_editor_player",
        entity_type="player",
        entity_id=str(player.get("id") if player else clean_name),
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "player": player or {"name": clean_name}},
        source_page=source,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and is_api_audit_log_required():
        _mark_recovery_required(
            supabase,
            operation=operation,
            operation_key=idempotency_key,
            result_json={
                "planned": {"player": insert_payload},
                "preexisting_player_ids": [row.get("id") for row in preexisting_rows],
                "player": player,
            },
            error_text="Required completion audit did not persist.",
        )
        raise GuardedWriteRecoveryRequired(
            idempotency_key,
            "The player may be created, but its required audit is unavailable. Inspect Player Editor before retrying.",
        )
    result = {
        "ok": True,
        "mode": "player_editor_create",
        "player": player,
        "operation_key": idempotency_key,
        "idempotent_replay": False,
        "recovery": {
            "operation_status": f"/admin/clubs/{{club_id}}/players/editor/operations/{idempotency_key}",
            "operator_rule": "Retry the exact unchanged request with the same idempotency key after an interrupted response.",
        },
        "warnings": warnings,
    }
    update_guarded_operation(
        supabase,
        operation_id=operation.get("id"),
        operation_key=idempotency_key,
        status="completed",
        after_json={"player": player},
        result_json=result,
    )
    return result


def update_admin_player_editor_player(
    supabase: Any,
    *,
    club_id: str,
    player_id: int,
    patch: dict[str, Any],
    actor_email: str,
    actor_role: str,
    expected_state_fingerprint: str,
    idempotency_key: str,
    source: str = "next_player_editor",
) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    update_payload: dict[str, Any] = {}
    request_patch: dict[str, Any] = {}
    if "name" in patch:
        name = _clean_text(patch.get("name"), limit=160)
        if not name:
            raise ValueError("Player name is required.")
        update_payload["name"] = name
        request_patch["name"] = name
    if "rating_jupr" in patch:
        update_payload["rating"] = _jupr_to_elo(patch.get("rating_jupr"), field_name="Overall JUPR")
        request_patch["rating"] = update_payload["rating"]
    if "starting_jupr" in patch:
        update_payload["starting_rating"] = _jupr_to_elo(patch.get("starting_jupr"), field_name="Starting JUPR")
        request_patch["starting_rating"] = update_payload["starting_rating"]
    if "active" in patch:
        next_active = bool(patch.get("active"))
        update_payload["active"] = next_active
        request_patch["active"] = next_active
    if not update_payload:
        raise ValueError("No supported player fields were provided.")
    request_payload = {
        "player_id": int(player_id),
        "expected_state_fingerprint": str(expected_state_fingerprint or "").strip().lower(),
        "patch": request_patch,
    }
    existing_operation = get_guarded_operation(
        supabase,
        club_id=str(club_id),
        workflow="player_editor_update",
        operation_key=idempotency_key,
    )
    if existing_operation is not None:
        if str(existing_operation.get("request_fingerprint") or "") != canonical_fingerprint(request_payload):
            raise ValueError("operation_key was already used for a different request.")
        if str(existing_operation.get("status") or "") == "completed":
            return operation_result(existing_operation)
        raise GuardedWriteRecoveryRequired(
            idempotency_key,
            "The Player Editor operation is incomplete. Inspect its recovery status before retrying.",
        )
    before_row = _fetch_player_row(
        supabase,
        club_id=str(club_id),
        player_id=int(player_id),
    )
    if before_row is None:
        raise ValueError("player not found")
    before = _player_payload(before_row)
    if str(expected_state_fingerprint or "").strip().lower() != str(before.get("state_fingerprint") or ""):
        raise PlayerEditorConflictError(
            "Player changed after it was loaded. Reload Player Editor and review the edit.",
            operation_key=idempotency_key,
        )
    if "active" in request_patch:
        update_payload["inactive_at"] = (
            None
            if bool(request_patch["active"])
            else (before_row.get("inactive_at") or datetime.now(timezone.utc).isoformat())
        )
    operation, idempotent = begin_guarded_operation(
        supabase,
        club_id=str(club_id),
        workflow="player_editor_update",
        action="update_player_editor_player",
        operation_key=idempotency_key,
        request_payload=request_payload,
        actor_email=actor_email,
        actor_role=actor_role,
        source=source,
        before_json={"player": before},
    )
    if idempotent:
        return operation_result(operation)
    planned_after = _player_payload({**before_row, **update_payload})
    update_guarded_operation(
        supabase,
        operation_id=operation.get("id"),
        operation_key=idempotency_key,
        status="intent_recorded",
        result_json={"phase": "preflight", "planned": {"player": planned_after}},
    )
    update_query = (
        supabase.table("players")
        .update(update_payload)
        .eq("club_id", str(club_id))
        .eq("id", int(player_id))
        .eq("name", before_row.get("name"))
        .eq("rating", before_row.get("rating"))
        .eq("starting_rating", before_row.get("starting_rating"))
        .eq("wins", before_row.get("wins"))
        .eq("losses", before_row.get("losses"))
        .eq("matches_played", before_row.get("matches_played"))
        .eq("active", before_row.get("active", before_row.get("is_active", True)))
    )
    if before_row.get("inactive_at") is None:
        update_query = update_query.is_("inactive_at", None)
    else:
        update_query = update_query.eq("inactive_at", before_row.get("inactive_at"))
    if before_row.get("last_game_at") is None:
        update_query = update_query.is_("last_game_at", None)
    else:
        update_query = update_query.eq("last_game_at", before_row.get("last_game_at"))
    try:
        updated_rows = _safe_rows(update_query.execute())
    except Exception as exc:
        readback: dict[str, Any] | None = None
        try:
            readback = _fetch_player_row(
                supabase,
                club_id=str(club_id),
                player_id=int(player_id),
            )
        except Exception:
            readback = None
        _mark_recovery_required(
            supabase,
            operation=operation,
            operation_key=idempotency_key,
            result_json={
                "planned": {"player": planned_after},
                "readback_verified": readback is not None,
                "player": _player_payload(readback) if readback else None,
            },
            error_text="Player compare-and-swap update returned an ambiguous transport result.",
        )
        raise GuardedWriteRecoveryRequired(
            idempotency_key,
            "The player edit may have committed. Inspect this exact operation before retrying; do not use a new key.",
        ) from exc
    if not updated_rows:
        update_guarded_operation(
            supabase,
            operation_id=operation.get("id"),
            operation_key=idempotency_key,
            status="failed",
            error_text="Player state changed before compare-and-swap update.",
        )
        raise PlayerEditorConflictError(
            "Player changed after it was loaded. Reload Player Editor and review the edit.",
            operation_key=idempotency_key,
        )
    after = _player_payload(updated_rows[0]) if updated_rows else _fetch_player(supabase, club_id=str(club_id), player_id=int(player_id))
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="update_player_editor_player",
        entity_type="player",
        entity_id=str(int(player_id)),
        before_json={"player": before},
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "patch": update_payload, "player": after},
        source_page=source,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and is_api_audit_log_required():
        _mark_recovery_required(
            supabase,
            operation=operation,
            operation_key=idempotency_key,
            result_json={"planned": {"player": planned_after}, "player": after},
            error_text="Required completion audit did not persist.",
        )
        raise GuardedWriteRecoveryRequired(
            idempotency_key,
            "The player edit may be committed, but its required audit is unavailable. Inspect Player Editor before retrying.",
        )
    result = {
        "ok": True,
        "mode": "player_editor_update",
        "player": after,
        "operation_key": idempotency_key,
        "idempotent_replay": False,
        "recovery": {
            "operation_status": f"/admin/clubs/{{club_id}}/players/editor/operations/{idempotency_key}",
            "operator_rule": "Retry the exact unchanged request with the same idempotency key after an interrupted response.",
        },
        "warnings": warnings,
    }
    update_guarded_operation(
        supabase,
        operation_id=operation.get("id"),
        operation_key=idempotency_key,
        status="completed",
        after_json={"player": after},
        result_json=result,
    )
    return result
