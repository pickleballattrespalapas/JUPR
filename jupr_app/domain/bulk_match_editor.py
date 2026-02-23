from __future__ import annotations

# Match writes must go through match_pipeline.

from typing import Any, Dict, List, Optional, Set, Tuple
import json

import pandas as pd

from jupr_app.domain.gamification.badge_queue import enqueue_badge_eval
from jupr_app.domain.audit_logger import log_event
from jupr_app.domain.match_pipeline import recalculate_state, update_match

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

        if keys.intersection({"week_tag", "league", "league_id", "date", "is_active"}):
            standings = True

        if keys.intersection({"league", "league_id", "date", "match_type", "is_active"}):
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




def apply_bulk_match_edits(
    supabase,
    club_id: str,
    patches: List[Dict[str, Any]],
    actor: str,
    source: str = "match_log.bulk_match_editor",
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

    # For audit
    applied: List[Dict[str, Any]] = []

    warnings: List[str] = []

    # Apply sequentially (safe correctness > speed).
    # If you want to optimize later, we can group identical updates or use an RPC for true transactions.
    updated_ids: List[int] = []

    for p in patches:
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
                    affected_players.add(int(v))
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

        if not update:
            continue

        # Apply update via pipeline
        update_match(
            supabase=supabase,
            club_id=str(club_id),
            match_id=int(mid),
            patch=update,
            rebuild_state=False,
        )

        updated_ids.append(mid)
        applied.append({"id": mid, **update})

    recompute_scope = compute_recompute_scope(patches)

    if updated_ids:
        recalculate_state(supabase=supabase, club_id=str(club_id))

    # Audit event (best effort)
    log_event(
        supabase=supabase,
        club_id=str(club_id),
        actor=actor,
        action_type="bulk_match_edit",
        payload={
            "club_id": club_id,
            "actor": actor,
            "source": source,
            "updated_count": len(updated_ids),
            "updated_ids": updated_ids,
            "affected_leagues": sorted(affected_leagues),
            "recompute_scope": recompute_scope,
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

    if updated_ids and supabase is not None:
        for match_id in updated_ids:
            enqueue_badge_eval(
                supabase,
                club_id=str(club_id),
                event_type="match_updated",
                player_ids=sorted(affected_players),
                match_id=str(match_id),
                payload={"updated_ids": updated_ids[:50]},
            )

    return {
        "updated_count": len(updated_ids),
        "updated_ids": updated_ids,
        "affected_leagues": sorted(affected_leagues),
        "recompute_scope": recompute_scope,
        "warnings": warnings,
    }
