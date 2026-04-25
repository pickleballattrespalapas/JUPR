from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import json
import logging

import pandas as pd

from jupr_app.domain.gamification.badge_queue import enqueue_badge_eval
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue
from jupr_app.domain.gamification.live_awards import run_live_badge_awards

logger = logging.getLogger(__name__)

def compute_recompute_scope(patches: List[Dict[str, Any]]) -> Dict[str, bool]:
    """
    Conservative scope flags:
      - standings: week_tag/league/date/is_active changes can affect week views & league views
      - ratings: league/date/match_type/is_active changes can affect ratings history
    """
    standings = False
    ratings = False

    for p in patches:
        keys = set(p.keys())
        # id is always present
        keys.discard("id")

        if keys.intersection({"week_tag", "league", "league_id", "date", "is_active", "t1_p1", "t1_p2", "t2_p1", "t2_p2", "score_t1", "score_t2"}):
            standings = True

        if keys.intersection({"league", "league_id", "date", "match_type", "is_active", "t1_p1", "t1_p2", "t2_p1", "t2_p2", "score_t1", "score_t2"}):
            ratings = True

    return {"standings": standings, "ratings": ratings}


def _iso_utc(value: Any) -> Optional[str]:
    """Convert a datetime-ish to ISO8601 with timezone (+00:00) suitable for Postgres timestamptz."""
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return ts.isoformat()


def _normalize_blank_to_none(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


def _safe_insert_audit_event(
    supabase,
    payload: Dict[str, Any],
) -> None:
    """
    Best-effort: if the table doesn't exist, do not fail the admin operation.
    Recommended table shape (optional):
      admin_audit_events(created_at timestamptz default now(), club_id text, actor text, action_type text, payload_json jsonb)
    """
    try:
        supabase.table("admin_audit_events").insert(
            {
                "club_id": payload.get("club_id"),
                "actor": payload.get("actor"),
                "action_type": payload.get("action_type", "bulk_match_edit"),
                "payload_json": payload,
            }
        ).execute()
    except Exception:
        # Intentionally swallow errors (table may not exist in some deployments)
        return


def apply_bulk_match_edits(
    supabase,
    club_id: str,
    patches: List[Dict[str, Any]],
    actor: str,
    source: str = "match_log.bulk_match_editor",
    correction_note: str | None = None,
) -> Dict[str, Any]:
    """
    Apply per-match patches safely.

    Safety invariants:
      - If league/date changes AND week_tag is not explicitly set in that patch, auto-null week_tag.
      - If notes/week_tag are blank strings, normalize to None.
      - Recompute last_game_at for players involved in edited matches (since date/active changes matter).

    Returns:
      {
        updated_count, updated_ids, affected_leagues, recompute_scope, warnings
      }
    """
    if not patches:
        raise ValueError("No patches provided.")

    allowed_patch_fields = {
        "id", "league", "date", "week_tag", "match_type", "notes", "is_active",
        "t1_p1", "t1_p2", "t2_p1", "t2_p2", "score_t1", "score_t2",
    }

    # Ensure ids are ints
    ids = [int(p["id"]) for p in patches if "id" in p]
    ids = sorted(set(ids))
    if not ids:
        raise ValueError("No match ids found in patches.")

    # We need before-values for:
    #  - auto-null week_tag decision (league/date change)
    #  - affected leagues
    #  - player ids for last_game_at recompute
    select_cols = [
        "id",
        "league",
        "date",
        "week_tag",
        "match_type",
        "t1_p1",
        "t1_p2",
        "t2_p1",
        "t2_p2",
    ]

    # If any patch includes optional fields, fetch them too.
    wants_notes = any("notes" in p for p in patches)
    wants_is_active = any("is_active" in p for p in patches)
    if wants_notes:
        select_cols.append("notes")
    if wants_is_active:
        select_cols.append("is_active")

    # Fetch before rows
    resp = supabase.table("matches").select(",".join(select_cols)).eq("club_id", club_id).in_("id", ids).execute()
    rows = resp.data or []
    before_by_id = {int(r["id"]): r for r in rows}

    missing = [mid for mid in ids if mid not in before_by_id]
    if missing:
        raise ValueError(f"Some match IDs were not found for this club_id: {missing[:10]}")

    affected_leagues: Set[str] = set()
    affected_players: Set[int] = set()
    badge_eligible_players: Set[int] = set()

    # For audit
    audit_before: List[Dict[str, Any]] = []
    audit_after: List[Dict[str, Any]] = []
    applied: List[Dict[str, Any]] = []

    warnings: List[str] = []
    patch_uses_player_slots = any(set(p.keys()).intersection({"t1_p1", "t1_p2", "t2_p1", "t2_p2"}) for p in patches)
    valid_player_ids: Set[int] = set()
    if patch_uses_player_slots:
        p_rows = supabase.table("players").select("id").eq("club_id", club_id).execute().data or []
        valid_player_ids = {
            int(r["id"])
            for r in p_rows
            if r.get("id") is not None
        }

    # Apply sequentially (safe correctness > speed).
    # If you want to optimize later, we can group identical updates or use an RPC for true transactions.
    updated_ids: List[int] = []

    for p in patches:
        unknown_fields = set(p.keys()) - allowed_patch_fields
        if unknown_fields:
            raise ValueError(f"Unsupported patch fields: {sorted(unknown_fields)}")
        mid = int(p["id"])
        before = before_by_id[mid]

        old_league = _normalize_blank_to_none(before.get("league"))
        old_date = before.get("date")

        # Track affected leagues
        if old_league is not None:
            affected_leagues.add(str(old_league))

        # Track affected players for last_game_at recompute
        for col in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            v = before.get(col)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                try:
                    pid = int(v)
                    affected_players.add(pid)
                    is_popup = bool(before.get("is_popup", False)) or str(before.get("match_type") or "") == "PopUp"
                    if not is_popup:
                        badge_eligible_players.add(pid)
                except Exception:
                    pass

        # Build update dict from patch (excluding id)
        update: Dict[str, Any] = {}
        for k, v in p.items():
            if k == "id":
                continue
            if k in ("week_tag", "notes"):
                update[k] = _normalize_blank_to_none(v)
            elif k == "date":
                # allow datetime or iso string
                update[k] = _iso_utc(v) if not (isinstance(v, str) and v.strip()) else v
            elif k in ("score_t1", "score_t2"):
                if v is None or (isinstance(v, str) and not v.strip()):
                    raise ValueError(f"Match {mid}: {k} cannot be blank.")
                try:
                    score_val = int(v)
                except Exception as exc:
                    raise ValueError(f"Match {mid}: {k} must be an integer.") from exc
                if score_val < 0:
                    raise ValueError(f"Match {mid}: {k} cannot be negative.")
                update[k] = score_val
            elif k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
                if v is None or (isinstance(v, str) and not str(v).strip()):
                    raise ValueError(f"Match {mid}: {k} cannot be blank for rated doubles.")
                try:
                    pid = int(v)
                except Exception as exc:
                    raise ValueError(f"Match {mid}: {k} must be a valid player ID.") from exc
                if valid_player_ids and pid not in valid_player_ids:
                    raise ValueError(f"Match {mid}: player {pid} for {k} is not in this club.")
                update[k] = pid
            else:
                update[k] = v

        # Safety: auto-null week_tag if league/date changed and week_tag wasn't explicitly set
        league_changed = False
        date_changed = False

        if "league" in update:
            new_league = _normalize_blank_to_none(update["league"])
            if new_league != old_league:
                league_changed = True
                if new_league is not None:
                    affected_leagues.add(str(new_league))

        if "date" in update:
            new_date = update["date"]
            # compare as ISO if possible
            old_iso = _iso_utc(old_date)
            new_iso = _iso_utc(new_date) if not (isinstance(new_date, str) and new_date.strip()) else new_date
            if new_iso != old_iso:
                date_changed = True
                update["date"] = new_iso  # normalize

        if (league_changed or date_changed) and ("week_tag" not in update):
            update["week_tag"] = None
            warnings.append(f"Match {mid}: week_tag auto-cleared because league/date changed.")

        candidate_players = {
            "t1_p1": int(update.get("t1_p1", before.get("t1_p1"))),
            "t1_p2": int(update.get("t1_p2", before.get("t1_p2"))),
            "t2_p1": int(update.get("t2_p1", before.get("t2_p1"))),
            "t2_p2": int(update.get("t2_p2", before.get("t2_p2"))),
        }
        for pid in candidate_players.values():
            affected_players.add(int(pid))
        match_type_after = str(update.get("match_type", before.get("match_type")) or "")
        is_popup_after = bool(before.get("is_popup", False)) or (match_type_after == "PopUp")
        if not is_popup_after:
            for pid in candidate_players.values():
                badge_eligible_players.add(int(pid))
        if len(set(candidate_players.values())) != 4:
            raise ValueError(f"Match {mid}: duplicate player detected in one rated doubles match.")

        if "score_t1" in update or "score_t2" in update:
            s1 = int(update.get("score_t1", before.get("score_t1", 0) or 0))
            s2 = int(update.get("score_t2", before.get("score_t2", 0) or 0))
            if s1 < 0 or s2 < 0:
                raise ValueError(f"Match {mid}: scores must be non-negative integers.")

        if not update:
            continue

        # audit snapshots (only changed fields + id)
        b_snap = {"id": mid}
        a_snap = {"id": mid}
        for k, newv in update.items():
            b_snap[k] = before.get(k)
            a_snap[k] = newv

        # Apply update
        now_iso = pd.Timestamp.utcnow().isoformat()
        update["updated_at"] = now_iso
        update["updated_by"] = actor
        if correction_note is not None:
            update["correction_note"] = _normalize_blank_to_none(correction_note)
        supabase.table("matches").update(update).eq("club_id", club_id).eq("id", mid).execute()

        updated_ids.append(mid)
        applied.append({"id": mid, **update})
        audit_before.append(b_snap)
        audit_after.append(a_snap)

    recompute_scope = compute_recompute_scope(patches)

    # Audit event (best effort)
    _safe_insert_audit_event(
        supabase,
        {
            "club_id": club_id,
            "actor": actor,
            "action_type": "bulk_match_edit",
            "source": source,
            "updated_count": len(updated_ids),
            "updated_ids": updated_ids,
            "affected_leagues": sorted(affected_leagues),
            "recompute_scope": recompute_scope,
            "before": audit_before,
            "after": audit_after,
            "patches": applied,
            "warnings": warnings,
        },
    )

    # Recompute last_game_at for affected players (best effort)
    try:
        from jupr_app.domain.player_activity import recompute_last_game_at_for_players

        if affected_players:
            recompute_last_game_at_for_players(
                supabase=supabase,
                club_id=str(club_id),
                player_ids=affected_players,
            )
    except Exception:
        # don't fail admin operation if recompute fails
        warnings.append("Unable to recompute last_game_at for players automatically. (Non-fatal)")

    badge_summary: Dict[str, Any] = {"mode": "skipped", "awarded_count": 0, "candidate_count": 0, "badge_ids": []}
    if updated_ids and supabase is not None and badge_eligible_players:
        for match_id in updated_ids:
            enqueue_result = enqueue_badge_eval(
                supabase,
                club_id=str(club_id),
                event_type="match_updated",
                player_ids=sorted(badge_eligible_players),
                match_id=str(match_id),
                payload={"updated_ids": updated_ids[:50]},
            )
            should_fallback = not bool(enqueue_result.get("queued"))
            worker_result = None
            if not should_fallback:
                try:
                    worker_result = process_badge_eval_queue(supabase, max_jobs=1, time_budget_seconds=2)
                    badge_summary = {"mode": "queue", **worker_result}
                    should_fallback = bool(worker_result.get("errored")) or (
                        int(worker_result.get("processed") or 0) == 0 and int(worker_result.get("errored") or 0) > 0
                    )
                except Exception as exc:  # noqa: BLE001
                    should_fallback = True
                    logger.warning("Badge queue worker failed during bulk match edit: %s", exc)
            if should_fallback:
                try:
                    badge_summary = run_live_badge_awards(
                        supabase,
                        club_id=str(club_id),
                        player_ids=sorted(badge_eligible_players),
                        event_type="match_updated",
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Inline live badge fallback failed during bulk edit: %s", exc)
                    badge_summary = {"mode": "inline_error", "error": str(exc)}

    return {
        "updated_count": len(updated_ids),
        "updated_ids": updated_ids,
        "affected_leagues": sorted(affected_leagues),
        "recompute_scope": recompute_scope,
        "warnings": warnings,
        "badge_summary": badge_summary,
    }
