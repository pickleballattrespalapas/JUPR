"""Canonical match write pipeline for production multi-tenant clubs.

This module is the *only* supported write interface for match mutations.

Invariants enforced by every public function:
- `club_id` is required and always used as a tenant boundary for reads/writes.
- All write operations are executed via one retry wrapper (`_run_write`).
- After every mutation, dependent projections are rebuilt deterministically:
  1) match rating snapshots
  2) player overall ratings
  3) league ratings
  4) player activity fields
  5) badge evaluation queue events

TODO: optimize full-projection rebuilds into server-side RPC transactions once the
schema/API contract is finalized.
"""

from __future__ import annotations

# Match writes must go through match_pipeline.

from typing import Any, Mapping

import pandas as pd

from jupr_app.data.retry import sb_retry
from jupr_app.data.sb_write import sb_delete, sb_insert, sb_update
from jupr_app.domain.audit_logger import log_event
from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.player_merge import merge_player_into as merge_player_into_domain
from jupr_app.domain.replay_history import FULL_RESET_LABEL, replay_history

_ALLOWED_MATCH_KEYS = {
    "date",
    "league",
    "match_type",
    "week_tag",
    "tournament_id",
    "tournament_game_id",
    "context_type",
    "context_id",
    "idempotency_key",
    "t1_p1",
    "t1_p2",
    "t2_p1",
    "t2_p2",
    "score_t1",
    "score_t2",
}
_SLOT_KEYS = ("t1_p1", "t1_p2", "t2_p1", "t2_p2")


class MatchPipelineError(RuntimeError):
    """Raised when a match pipeline mutation fails and cannot complete safely."""


def require_club_scope(club_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Require a payload/filter to carry a matching club tenant scope."""
    scoped_club_id = _require_club_id(club_id)
    scoped_payload = dict(payload or {})
    payload_club_id = str(scoped_payload.get("club_id") or "").strip()
    if not payload_club_id:
        raise MatchPipelineError("club_id scope is required for match_pipeline writes")
    if payload_club_id != scoped_club_id:
        raise MatchPipelineError("club_id scope mismatch for match_pipeline write")
    return scoped_payload


def _pipeline_result(
    *,
    success: bool,
    match_id: int | None,
    warnings: list[str] | None = None,
    error: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": bool(success),
        "match_id": _as_int(match_id),
        "warnings": list(warnings or []),
        "error": error,
    }
    payload.update(extra)
    return payload


def _run_write(fn):
    """Execute a write callable through the module's single retry wrapper."""
    return sb_retry(fn)


def _scoped_filters(club_id: str, filters: Mapping[str, Any]) -> dict[str, Any]:
    return require_club_scope(club_id, filters)


def _scoped_payload(club_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return require_club_scope(club_id, payload)


def _require_club_id(club_id: str) -> str:
    normalized = str(club_id or "").strip()
    if not normalized:
        raise ValueError("club_id is required")
    return normalized


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_match_payload(club_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    row = {k: v for k, v in dict(payload).items() if k in _ALLOWED_MATCH_KEYS}
    row["club_id"] = club_id
    for slot in _SLOT_KEYS:
        if slot in row:
            row[slot] = _as_int(row.get(slot))
    if "score_t1" in row:
        row["score_t1"] = int(row.get("score_t1") or 0)
    if "score_t2" in row:
        row["score_t2"] = int(row.get("score_t2") or 0)
    return row


def _require_idempotency_key(payload: Mapping[str, Any]) -> str:
    key = str(payload.get("idempotency_key") or "").strip()
    if not key:
        raise ValueError("idempotency_key is required")
    return key


def _find_existing_match_by_idempotency_key(*, supabase: Any, club_id: str, idempotency_key: str) -> dict[str, Any] | None:
    existing_rows = (
        supabase.table("matches")
        .select("*")
        .eq("club_id", club_id)
        .eq("idempotency_key", idempotency_key)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not existing_rows:
        return None
    return dict(existing_rows[0])


def _rebuild_state(*, supabase: Any, club_id: str) -> dict[str, int]:
    """Recalculate snapshots and derived ratings after match mutations."""
    replay = replay_history(
        supabase=supabase,
        club_id=club_id,
        df_meta=None,
        target_reset=FULL_RESET_LABEL,
    )
    return {
        "matches_processed": int(replay.get("matches_rewritten") or 0),
        "players_updated": int(replay.get("players_updated") or 0),
        "league_rows_upserted": int(replay.get("league_ratings_rows") or 0),
    }


def recalculate_state(*, supabase: Any, club_id: str) -> dict[str, int]:
    """Public wrapper for deterministic post-mutation state rebuilds."""
    return _rebuild_state(supabase=supabase, club_id=club_id)


def _build_processing_context(*, supabase: Any, club_id: str) -> dict[str, Any]:
    players_rows = (
        supabase.table("players")
        .select("id,name,rating,wins,losses,matches_played,last_game_at")
        .eq("club_id", club_id)
        .execute()
        .data
        or []
    )
    leagues_rows = (
        supabase.table("league_ratings")
        .select("player_id,league_name,rating,wins,losses,matches_played,starting_rating")
        .eq("club_id", club_id)
        .execute()
        .data
        or []
    )
    meta_rows = (
        supabase.table("leagues_metadata")
        .select("league_name,k_factor")
        .eq("club_id", club_id)
        .execute()
        .data
        or []
    )
    name_to_id = {
        str(row.get("name") or "").strip(): int(row["id"])
        for row in players_rows
        if str(row.get("name") or "").strip() and _as_int(row.get("id")) is not None
    }
    return {
        "name_to_id": name_to_id,
        "df_players_all": pd.DataFrame(players_rows),
        "df_leagues": pd.DataFrame(leagues_rows),
        "df_meta": pd.DataFrame(meta_rows),
    }


def record_match(*, supabase: Any, club_id: str, match_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create one match for `club_id` and rebuild all dependent projections.

    Invariant: callers must not write matches directly; this function is the
    canonical creation path and ensures snapshots, ratings, activity, and badge
    queue side-effects stay in sync.
    """
    scoped_club_id = _require_club_id(club_id)
    scoped_idempotency_key = _require_idempotency_key(match_payload)

    # Migration expectation: enforce DB uniqueness with
    #   UNIQUE (club_id, idempotency_key)
    # on matches to guarantee race-safe, cross-process idempotency.
    existing_match = _find_existing_match_by_idempotency_key(
        supabase=supabase,
        club_id=scoped_club_id,
        idempotency_key=scoped_idempotency_key,
    )
    if existing_match:
        existing_match_id = _as_int(existing_match.get("id"))
        log_event(
            supabase=supabase,
            club_id=scoped_club_id,
            actor=str(match_payload.get("actor") or "match_pipeline"),
            action_type="record_match",
            payload={
                "match_id": existing_match_id,
                "success": True,
                "idempotent_hit": True,
                "match_payload": dict(match_payload),
            },
        )
        return _pipeline_result(
            success=True,
            match_id=existing_match_id,
            warnings=[],
            existing=existing_match,
            idempotent_hit=True,
        )

    payload = _coerce_match_payload(scoped_club_id, match_payload)
    inserted_rows: list[dict[str, Any]] = []
    inserted_match_id: int | None = None
    warnings: list[str] = []
    operation_success = False

    try:
        processing_ctx = _build_processing_context(supabase=supabase, club_id=scoped_club_id)

        def _write_match(match_row: dict[str, Any], context_id: str | None, context_type: str, idempotency_key: str):
            nonlocal inserted_match_id
            insert_payload = dict(match_row)
            insert_payload["context_id"] = context_id
            insert_payload["context_type"] = context_type
            insert_payload["idempotency_key"] = idempotency_key
            inserted = _run_write(lambda: sb_insert(supabase, "matches", _scoped_payload(scoped_club_id, insert_payload)))
            rows = getattr(inserted, "data", None) or []
            inserted_rows.extend(rows)
            if rows and inserted_match_id is None:
                inserted_match_id = _as_int((rows[0] or {}).get("id"))
            return inserted

        process_matches(
            [payload],
            supabase_admin=supabase,
            supabase=supabase,
            club_id=scoped_club_id,
            name_to_id=processing_ctx["name_to_id"],
            df_players_all=processing_ctx["df_players_all"],
            df_leagues=processing_ctx["df_leagues"],
            df_meta=processing_ctx["df_meta"],
            sb_retry=_run_write,
            match_writer=_write_match,
        )
        inserted_match_id = inserted_match_id or (_as_int((inserted_rows[0] or {}).get("id")) if inserted_rows else None)
        operation_success = True
        return _pipeline_result(
            success=True,
            match_id=inserted_match_id,
            warnings=warnings,
            inserted=inserted_rows,
        )
    except Exception as exc:
        if inserted_match_id is not None:
            _run_write(lambda: sb_delete(
                supabase,
                "matches",
                filters=_scoped_filters(scoped_club_id, {"club_id": scoped_club_id, "id": int(inserted_match_id)}),
            ))
            warnings.append("rolled_back_inserted_match")
        err = MatchPipelineError(f"record_match failed: {exc}")
        return _pipeline_result(
            success=False,
            match_id=inserted_match_id,
            warnings=warnings,
            error=str(err),
        )
    finally:
        log_event(
            supabase=supabase,
            club_id=scoped_club_id,
            actor=str(match_payload.get("actor") or "match_pipeline"),
            action_type="record_match",
            payload={
                "match_id": inserted_match_id,
                "success": operation_success,
                "match_payload": dict(match_payload),
            },
        )


def update_match(
    *,
    supabase: Any,
    club_id: str,
    match_id: int,
    patch: Mapping[str, Any],
    rebuild_state: bool = True,
) -> dict[str, Any]:
    """Update one match row for `club_id` and rebuild all dependent projections.

    Invariant: any mutable match-field change must flow through this function so
    downstream snapshots/ratings/activity remain deterministic.
    """
    scoped_club_id = _require_club_id(club_id)
    target_match_id = int(match_id)
    safe_patch = _coerce_match_payload(scoped_club_id, patch)
    safe_patch.pop("club_id", None)
    match_before_rows = (
        supabase.table("matches")
        .select("*")
        .eq("club_id", scoped_club_id)
        .eq("id", target_match_id)
        .limit(1)
        .execute()
        .data
        or []
    )
    match_before = dict(match_before_rows[0]) if match_before_rows else None
    warnings: list[str] = []
    operation_success = False

    try:
        updated = _run_write(lambda: sb_update(
            supabase,
            "matches",
            safe_patch,
            filters=_scoped_filters(scoped_club_id, {"club_id": scoped_club_id, "id": target_match_id}),
        ))
        updated_rows = getattr(updated, "data", None) or []
        rebuild = _rebuild_state(supabase=supabase, club_id=scoped_club_id) if rebuild_state else None
        operation_success = True
        return _pipeline_result(
            success=True,
            match_id=target_match_id,
            warnings=warnings,
            updated=updated_rows,
            rebuild=rebuild or {},
        )
    except Exception as exc:
        if match_before:
            restore_patch = {k: v for k, v in match_before.items() if k in _ALLOWED_MATCH_KEYS}
            _run_write(lambda restore_patch=restore_patch: sb_update(
                supabase,
                "matches",
                restore_patch,
                filters=_scoped_filters(scoped_club_id, {"club_id": scoped_club_id, "id": target_match_id}),
            ))
            warnings.append("rolled_back_match_patch")
        err = MatchPipelineError(f"update_match failed: {exc}")
        return _pipeline_result(
            success=False,
            match_id=target_match_id,
            warnings=warnings,
            error=str(err),
        )
    finally:
        log_event(
            supabase=supabase,
            club_id=scoped_club_id,
            actor=str(patch.get("actor") or "match_pipeline"),
            action_type="update_match",
            payload={
                "match_id": target_match_id,
                "success": operation_success,
                "patch": dict(patch),
            },
        )


def delete_match(*, supabase: Any, club_id: str, match_id: int) -> dict[str, Any]:
    """Delete one match row for `club_id` and rebuild all dependent projections.

    Invariant: deletions must be followed by a deterministic projection rebuild
    to avoid stale ratings, snapshots, player activity, or badge queue drift.
    """
    scoped_club_id = _require_club_id(club_id)
    target_match_id = int(match_id)
    operation_success = False
    try:
        deleted = _run_write(lambda: sb_delete(
            supabase,
            "matches",
            filters=_scoped_filters(scoped_club_id, {"club_id": scoped_club_id, "id": target_match_id}),
        ))
        rebuild = _rebuild_state(supabase=supabase, club_id=scoped_club_id)
        operation_success = True
        return _pipeline_result(
            success=True,
            match_id=target_match_id,
            warnings=[],
            deleted=getattr(deleted, "data", None) or [],
            rebuild=rebuild,
        )
    except Exception as exc:
        err = MatchPipelineError(f"delete_match failed: {exc}")
        return _pipeline_result(success=False, match_id=target_match_id, warnings=[], error=str(err))
    finally:
        log_event(
            supabase=supabase,
            club_id=scoped_club_id,
            actor="match_pipeline",
            action_type="delete_match",
            payload={"match_id": target_match_id, "success": operation_success},
        )


def delete_matches(*, supabase: Any, club_id: str, match_ids: list[int]) -> dict[str, Any]:
    """Delete many match rows for `club_id` and rebuild dependent projections once."""
    scoped_club_id = _require_club_id(club_id)
    ids = sorted({int(mid) for mid in (match_ids or [])})
    deleted_rows: list[dict[str, Any]] = []
    try:
        for match_id in ids:
            deleted = _run_write(lambda match_id=match_id: sb_delete(
                supabase,
                "matches",
                filters=_scoped_filters(scoped_club_id, {"club_id": scoped_club_id, "id": int(match_id)}),
            ))
            deleted_rows.extend(getattr(deleted, "data", None) or [])

        rebuild = _rebuild_state(supabase=supabase, club_id=scoped_club_id)
        return _pipeline_result(success=True, match_id=None, warnings=[], deleted=deleted_rows, rebuild=rebuild)
    except Exception as exc:
        err = MatchPipelineError(f"delete_matches failed: {exc}")
        return _pipeline_result(success=False, match_id=None, warnings=[], error=str(err))


def merge_player_into(*, supabase: Any, club_id: str, source_player_id: int, target_player_id: int) -> dict[str, Any]:
    """Backward-compatible wrapper around the dedicated domain merge function."""
    return merge_player_into_domain(
        supabase=supabase,
        club_id=club_id,
        source_player_id=source_player_id,
        destination_player_id=target_player_id,
        actor="match_pipeline",
    )


def reassign_match_players(
    *,
    supabase: Any,
    club_id: str,
    match_id: int,
    reassignments: Mapping[str, int],
) -> dict[str, Any]:
    """Reassign one match's player slots and rebuild all dependent projections.

    Invariant: slot-level reassignment must always remain scoped by `club_id`
    and trigger full projection recalculation.
    """
    scoped_club_id = _require_club_id(club_id)
    target_match_id = int(match_id)
    patch = {
        slot: int(pid)
        for slot, pid in dict(reassignments).items()
        if slot in _SLOT_KEYS and pid is not None
    }
    if not patch:
        raise ValueError("reassignments must include at least one valid match slot")

    try:
        updated = _run_write(lambda: sb_update(
            supabase,
            "matches",
            patch,
            filters=_scoped_filters(scoped_club_id, {"club_id": scoped_club_id, "id": target_match_id}),
        ))
        rebuild = _rebuild_state(supabase=supabase, club_id=scoped_club_id)
        return _pipeline_result(
            success=True,
            match_id=target_match_id,
            warnings=[],
            updated=getattr(updated, "data", None) or [],
            rebuild=rebuild,
        )
    except Exception as exc:
        err = MatchPipelineError(f"reassign_match_players failed: {exc}")
        return _pipeline_result(success=False, match_id=target_match_id, warnings=[], error=str(err))


__all__ = [
    "MatchPipelineError",
    "require_club_scope",
    "record_match",
    "update_match",
    "recalculate_state",
    "delete_match",
    "delete_matches",
    "merge_player_into",
    "reassign_match_players",
]
