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

from datetime import datetime, timezone
from typing import Any, Mapping

from jupr_app.data.retry import sb_retry
from jupr_app.data.sb_write import sb_delete, sb_insert, sb_update, sb_upsert
from jupr_app.domain.audit_logger import log_event
from jupr_app.domain.constants import CAP_LOSER_GAIN_ELO, DEFAULT_K_FACTOR, MIN_WIN_DELTA_ELO
from jupr_app.domain.gamification.badge_queue import enqueue_badge_eval
from jupr_app.domain.player_activity import build_player_activity_update, coerce_utc_datetime, max_activity_time
from jupr_app.domain.ratings import calculate_hybrid_elo

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


def _snapshot_ratings_state(*, supabase: Any, club_id: str) -> dict[str, list[dict[str, Any]]]:
    players_rows = (
        supabase.table("players")
        .select("id,rating,wins,losses,matches_played,last_game_at,is_active,inactive_at")
        .eq("club_id", club_id)
        .execute()
        .data
        or []
    )
    league_rows = (
        supabase.table("league_ratings")
        .select("club_id,player_id,league_name,rating,wins,losses,matches_played,starting_rating,is_active,inactive_at")
        .eq("club_id", club_id)
        .execute()
        .data
        or []
    )
    return {
        "players": [dict(r) for r in players_rows],
        "league_ratings": [dict(r) for r in league_rows],
    }


def _restore_ratings_state(*, supabase: Any, club_id: str, snapshot: Mapping[str, Any]) -> None:
    for row in snapshot.get("players", []):
        pid = _as_int(row.get("id"))
        if pid is None:
            continue
        payload = {
            "rating": row.get("rating"),
            "wins": row.get("wins"),
            "losses": row.get("losses"),
            "matches_played": row.get("matches_played"),
            "last_game_at": row.get("last_game_at"),
            "is_active": row.get("is_active"),
            "inactive_at": row.get("inactive_at"),
        }
        _run_write(lambda pid=pid, payload=payload: sb_update(
            supabase,
            "players",
            payload,
            filters={"club_id": club_id, "id": int(pid)},
        ))

    _run_write(lambda: sb_delete(
        supabase,
        "league_ratings",
        filters={"club_id": club_id},
    ))
    for row in snapshot.get("league_ratings", []):
        payload = {
            "club_id": club_id,
            "player_id": int(row["player_id"]),
            "league_name": str(row.get("league_name") or ""),
            "rating": row.get("rating"),
            "wins": row.get("wins"),
            "losses": row.get("losses"),
            "matches_played": row.get("matches_played"),
            "starting_rating": row.get("starting_rating"),
            "is_active": row.get("is_active"),
            "inactive_at": row.get("inactive_at"),
        }
        _run_write(lambda payload=payload: sb_upsert(
            supabase,
            "league_ratings",
            payload,
            conflict="club_id,player_id,league_name",
        ))


def _run_write(fn):
    """Execute a write callable through the module's single retry wrapper."""
    return sb_retry(fn)


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


def _rebuild_state(*, supabase: Any, club_id: str) -> dict[str, int]:
    """Rebuild all match-derived projections for a single club deterministically."""
    players_rows = (
        supabase.table("players")
        .select("id,rating,starting_rating,created_at,last_game_at")
        .eq("club_id", club_id)
        .execute()
        .data
        or []
    )
    matches = (
        supabase.table("matches")
        .select("id,date,league,match_type,score_t1,score_t2,t1_p1,t1_p2,t2_p1,t2_p2")
        .eq("club_id", club_id)
        .order("date", desc=False)
        .order("id", desc=False)
        .execute()
        .data
        or []
    )

    player_state: dict[int, dict[str, Any]] = {}
    for row in players_rows:
        pid = _as_int(row.get("id"))
        if pid is None:
            continue
        base_rating = row.get("starting_rating")
        if base_rating is None:
            base_rating = row.get("rating", 1200.0)
        player_state[pid] = {
            "r": float(base_rating or 1200.0),
            "base_rating": float(base_rating or 1200.0),
            "w": 0,
            "l": 0,
            "mp": 0,
            "created_at": row.get("created_at"),
            "existing_last_game_at": row.get("last_game_at"),
        }

    league_state: dict[tuple[int, str], dict[str, Any]] = {}
    match_snapshot_updates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []

    def ensure_player(pid: int) -> None:
        if pid in player_state:
            return
        player_state[pid] = {
            "r": 1200.0,
            "base_rating": 1200.0,
            "w": 0,
            "l": 0,
            "mp": 0,
            "created_at": None,
            "existing_last_game_at": None,
        }

    last_game_updates: dict[int, datetime] = {}

    for row in matches:
        mid = _as_int(row.get("id"))
        if mid is None:
            continue

        slots = [_as_int(row.get(k)) for k in _SLOT_KEYS]
        if any(pid is None for pid in slots):
            continue
        p1, p2, p3, p4 = (int(slots[0]), int(slots[1]), int(slots[2]), int(slots[3]))

        s1 = int(row.get("score_t1") or 0)
        s2 = int(row.get("score_t2") or 0)
        if (s1 + s2) <= 0:
            continue

        for pid in (p1, p2, p3, p4):
            ensure_player(pid)

        sr1 = float(player_state[p1]["r"])
        sr2 = float(player_state[p2]["r"])
        sr3 = float(player_state[p3]["r"])
        sr4 = float(player_state[p4]["r"])

        do1, do2 = calculate_hybrid_elo(
            (sr1 + sr2) / 2.0,
            (sr3 + sr4) / 2.0,
            s1,
            s2,
            k_factor=float(DEFAULT_K_FACTOR),
            min_win_delta=float(MIN_WIN_DELTA_ELO),
            cap_loser_gain=float(CAP_LOSER_GAIN_ELO),
        )

        t1_won = s1 > s2
        t2_won = s2 > s1

        for pid, delta, won in ((p1, do1, t1_won), (p2, do1, t1_won), (p3, do2, t2_won), (p4, do2, t2_won)):
            player_state[pid]["r"] = float(player_state[pid]["r"]) + float(delta)
            player_state[pid]["mp"] = int(player_state[pid]["mp"]) + 1
            if s1 != s2:
                if won:
                    player_state[pid]["w"] = int(player_state[pid]["w"]) + 1
                else:
                    player_state[pid]["l"] = int(player_state[pid]["l"]) + 1

        er1 = float(player_state[p1]["r"])
        er2 = float(player_state[p2]["r"])
        er3 = float(player_state[p3]["r"])
        er4 = float(player_state[p4]["r"])

        league_name = str(row.get("league") or "").strip()
        is_popup = str(row.get("match_type") or "") == "PopUp"
        if league_name and not is_popup:
            for pid in (p1, p2, p3, p4):
                key = (pid, league_name)
                if key not in league_state:
                    league_state[key] = {"r": float(player_state[pid]["r"]), "w": 0, "l": 0, "mp": 0}
            li1, li2 = calculate_hybrid_elo(
                (float(league_state[(p1, league_name)]["r"]) + float(league_state[(p2, league_name)]["r"])) / 2.0,
                (float(league_state[(p3, league_name)]["r"]) + float(league_state[(p4, league_name)]["r"])) / 2.0,
                s1,
                s2,
                k_factor=float(DEFAULT_K_FACTOR),
                min_win_delta=float(MIN_WIN_DELTA_ELO),
                cap_loser_gain=float(CAP_LOSER_GAIN_ELO),
            )
            for pid, delta, won in ((p1, li1, t1_won), (p2, li1, t1_won), (p3, li2, t2_won), (p4, li2, t2_won)):
                key = (pid, league_name)
                league_state[key]["r"] = float(league_state[key]["r"]) + float(delta)
                league_state[key]["mp"] = int(league_state[key]["mp"]) + 1
                if s1 != s2:
                    if won:
                        league_state[key]["w"] = int(league_state[key]["w"]) + 1
                    else:
                        league_state[key]["l"] = int(league_state[key]["l"]) + 1

        match_dt = coerce_utc_datetime(row.get("date")) or datetime.fromtimestamp(0, timezone.utc)
        for pid in (p1, p2, p3, p4):
            last_game_updates[pid] = max_activity_time(last_game_updates.get(pid), match_dt) or match_dt

        snapshot = {
            "elo_delta": float(abs(do1) if t1_won else abs(do2)),
            "t1_p1_r": sr1,
            "t1_p2_r": sr2,
            "t2_p1_r": sr3,
            "t2_p2_r": sr4,
            "t1_p1_r_end": er1,
            "t1_p2_r_end": er2,
            "t2_p1_r_end": er3,
            "t2_p2_r_end": er4,
        }
        badge_payload = {
            "match_id": str(mid),
            "score_t1": s1,
            "score_t2": s2,
            "t1_p1": p1,
            "t1_p2": p2,
            "t2_p1": p3,
            "t2_p2": p4,
            "t1_p1_r": sr1,
            "t1_p2_r": sr2,
            "t2_p1_r": sr3,
            "t2_p2_r": sr4,
        }
        match_snapshot_updates.append((mid, snapshot, badge_payload))

    for match_id, snapshot, badge_payload in match_snapshot_updates:
        _run_write(lambda match_id=match_id, snapshot=snapshot: sb_update(
            supabase,
            "matches",
            snapshot,
            filters={"club_id": club_id, "id": int(match_id)},
        ))
        _run_write(lambda badge_payload=badge_payload: enqueue_badge_eval(
            supabase,
            club_id=club_id,
            event_type="match_recorded",
            player_ids=[
                int(badge_payload["t1_p1"]),
                int(badge_payload["t1_p2"]),
                int(badge_payload["t2_p1"]),
                int(badge_payload["t2_p2"]),
            ],
            context_id="overall",
            match_id=str(badge_payload["match_id"]),
            payload=badge_payload,
        ))

    for pid, state in player_state.items():
        latest_match = last_game_updates.get(pid)
        activity_update = build_player_activity_update(state.get("existing_last_game_at"), latest_match)
        payload = {
            "rating": float(state["r"]),
            "wins": int(state["w"]),
            "losses": int(state["l"]),
            "matches_played": int(state["mp"]),
        }
        payload.update(activity_update)
        _run_write(lambda pid=pid, payload=payload: sb_update(
            supabase,
            "players",
            payload,
            filters={"club_id": club_id, "id": int(pid)},
        ))

    _run_write(lambda: sb_delete(
        supabase,
        "league_ratings",
        filters={"club_id": club_id},
    ))
    for (pid, league_name), state in league_state.items():
        payload = {
            "club_id": club_id,
            "player_id": int(pid),
            "league_name": str(league_name),
            "rating": float(state["r"]),
            "wins": int(state["w"]),
            "losses": int(state["l"]),
            "matches_played": int(state["mp"]),
            "starting_rating": float(player_state.get(pid, {}).get("base_rating", 1200.0)),
            "is_active": True,
            "inactive_at": None,
        }
        _run_write(lambda payload=payload: sb_upsert(
            supabase,
            "league_ratings",
            payload,
            conflict="club_id,player_id,league_name",
        ))

    return {
        "matches_processed": len(match_snapshot_updates),
        "players_updated": len(player_state),
        "league_rows_upserted": len(league_state),
    }


def record_match(*, supabase: Any, club_id: str, match_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create one match for `club_id` and rebuild all dependent projections.

    Invariant: callers must not write matches directly; this function is the
    canonical creation path and ensures snapshots, ratings, activity, and badge
    queue side-effects stay in sync.
    """
    scoped_club_id = _require_club_id(club_id)
    payload = _coerce_match_payload(scoped_club_id, match_payload)
    snapshot = _snapshot_ratings_state(supabase=supabase, club_id=scoped_club_id)
    inserted_rows: list[dict[str, Any]] = []
    inserted_match_id: int | None = None
    warnings: list[str] = []
    operation_success = False

    try:
        inserted = _run_write(lambda: sb_insert(supabase, "matches", payload))
        inserted_rows = getattr(inserted, "data", None) or []
        inserted_match_id = _as_int((inserted_rows[0] or {}).get("id")) if inserted_rows else None
        rebuild = _rebuild_state(supabase=supabase, club_id=scoped_club_id)
        operation_success = True
        return _pipeline_result(
            success=True,
            match_id=inserted_match_id,
            warnings=warnings,
            inserted=inserted_rows,
            rebuild=rebuild,
        )
    except Exception as exc:
        if inserted_match_id is not None:
            _run_write(lambda: sb_delete(
                supabase,
                "matches",
                filters={"club_id": scoped_club_id, "id": int(inserted_match_id)},
            ))
            warnings.append("rolled_back_inserted_match")
        _restore_ratings_state(supabase=supabase, club_id=scoped_club_id, snapshot=snapshot)
        warnings.append("ratings_restored_from_snapshot")
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


def update_match(*, supabase: Any, club_id: str, match_id: int, patch: Mapping[str, Any]) -> dict[str, Any]:
    """Update one match row for `club_id` and rebuild all dependent projections.

    Invariant: any mutable match-field change must flow through this function so
    downstream snapshots/ratings/activity remain deterministic.
    """
    scoped_club_id = _require_club_id(club_id)
    target_match_id = int(match_id)
    safe_patch = _coerce_match_payload(scoped_club_id, patch)
    safe_patch.pop("club_id", None)
    snapshot = _snapshot_ratings_state(supabase=supabase, club_id=scoped_club_id)
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
            filters={"club_id": scoped_club_id, "id": target_match_id},
        ))
        updated_rows = getattr(updated, "data", None) or []
        rebuild = _rebuild_state(supabase=supabase, club_id=scoped_club_id)
        operation_success = True
        return _pipeline_result(
            success=True,
            match_id=target_match_id,
            warnings=warnings,
            updated=updated_rows,
            rebuild=rebuild,
        )
    except Exception as exc:
        if match_before:
            restore_patch = {k: v for k, v in match_before.items() if k in _ALLOWED_MATCH_KEYS}
            _run_write(lambda restore_patch=restore_patch: sb_update(
                supabase,
                "matches",
                restore_patch,
                filters={"club_id": scoped_club_id, "id": target_match_id},
            ))
            warnings.append("rolled_back_match_patch")
        _restore_ratings_state(supabase=supabase, club_id=scoped_club_id, snapshot=snapshot)
        warnings.append("ratings_restored_from_snapshot")
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
            filters={"club_id": scoped_club_id, "id": target_match_id},
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
                filters={"club_id": scoped_club_id, "id": int(match_id)},
            ))
            deleted_rows.extend(getattr(deleted, "data", None) or [])

        rebuild = _rebuild_state(supabase=supabase, club_id=scoped_club_id)
        return _pipeline_result(success=True, match_id=None, warnings=[], deleted=deleted_rows, rebuild=rebuild)
    except Exception as exc:
        err = MatchPipelineError(f"delete_matches failed: {exc}")
        return _pipeline_result(success=False, match_id=None, warnings=[], error=str(err))


def merge_player_into(*, supabase: Any, club_id: str, source_player_id: int, target_player_id: int) -> dict[str, Any]:
    """Replace all source-player match slots with target-player slots for `club_id`.

    Invariant: player merge operations must atomically rewrite match references
    through this module and trigger full projection rebuilds.
    """
    scoped_club_id = _require_club_id(club_id)
    src = int(source_player_id)
    dst = int(target_player_id)
    if src == dst:
        raise ValueError("source_player_id and target_player_id must differ")

    operation_success = False
    rewritten = 0
    try:
        matches = (
            supabase.table("matches")
            .select("id,t1_p1,t1_p2,t2_p1,t2_p2")
            .eq("club_id", scoped_club_id)
            .or_(f"t1_p1.eq.{src},t1_p2.eq.{src},t2_p1.eq.{src},t2_p2.eq.{src}")
            .execute()
            .data
            or []
        )

        for row in matches:
            match_id = int(row["id"])
            patch: dict[str, int] = {}
            for slot in _SLOT_KEYS:
                if _as_int(row.get(slot)) == src:
                    patch[slot] = dst
            if not patch:
                continue
            _run_write(lambda match_id=match_id, patch=patch: sb_update(
                supabase,
                "matches",
                patch,
                filters={"club_id": scoped_club_id, "id": match_id},
            ))
            rewritten += 1

        rebuild = _rebuild_state(supabase=supabase, club_id=scoped_club_id)
        operation_success = True
        return _pipeline_result(success=True, match_id=None, warnings=[], matches_rewritten=rewritten, rebuild=rebuild)
    except Exception as exc:
        err = MatchPipelineError(f"merge_player_into failed: {exc}")
        return _pipeline_result(success=False, match_id=None, warnings=[], error=str(err))
    finally:
        log_event(
            supabase=supabase,
            club_id=scoped_club_id,
            actor="match_pipeline",
            action_type="merge_player",
            payload={"source_player_id": src, "target_player_id": dst, "matches_rewritten": rewritten, "success": operation_success},
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
            filters={"club_id": scoped_club_id, "id": target_match_id},
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
    "record_match",
    "update_match",
    "delete_match",
    "delete_matches",
    "merge_player_into",
    "reassign_match_players",
]
