"""Canonical match submission pipeline (infrastructure stub only).

This module provides a single future-facing entry point (`submit_match`) for
consolidating match write logic across league, ladder, tournament, and admin
contexts.

Constraints for this initial infrastructure PR:
- Additive only: no integration with existing flows.
- No behavior changes to current application paths beyond canonical match insert.
- Non-insert helpers remain explicit stubs.
"""

from __future__ import annotations

from jupr_app.data.sb_write import sb_insert, sb_update, sb_upsert

import os
from typing import Any, Dict, Literal, Optional

from jupr_app.data.client import make_supabase

_ALLOWED_CONTEXT_TYPES = {"league", "ladder", "tournament", "round_robin", "moneyball", "admin"}


def get_supabase_client() -> Any:
    """Build a Supabase client from environment credentials."""
    supabase_url = os.getenv("SUPABASE_URL", "")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "") or os.getenv(
        "SUPABASE_KEY", ""
    )

    if not supabase_url or not supabase_key:
        raise ValueError(
            "Supabase credentials are missing. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY)."
        )

    return make_supabase(supabase_url, supabase_key)


def submit_match(
    club_id: str,
    context_type: Literal["league", "ladder", "tournament", "round_robin", "moneyball", "admin"],
    context_id: Optional[str],
    match_payload: Dict[str, Any],
    idempotency_key: Optional[str] = None,
    run_context_hooks: bool = True,
) -> Dict[str, Any]:
    """Validate input and route to canonical pipeline steps.

    The canonical insert step writes to Supabase and enforces idempotency when
    an idempotency key is provided. Other downstream stages remain no-op stubs.
    """
    if not club_id or not isinstance(club_id, str):
        raise ValueError("club_id must be a non-empty string")

    if context_type not in _ALLOWED_CONTEXT_TYPES:
        raise ValueError(
            "context_type must be one of: league, ladder, tournament, round_robin, moneyball, admin"
        )

    if not isinstance(match_payload, dict):
        raise ValueError("match_payload must be a dict")

    insert_result = _insert_match_record(
        club_id=club_id,
        context_type=context_type,
        context_id=context_id,
        match_payload=match_payload,
        idempotency_key=idempotency_key,
    )
    rating_result = _apply_rating_engine_stub(match_payload=match_payload)
    hooks_result: Dict[str, Any]
    if run_context_hooks:
        hooks_result = _run_context_hooks(
            club_id=club_id,
            context_type=context_type,
            context_id=context_id,
            match_payload=match_payload,
            rating_result=rating_result,
        )
    else:
        hooks_result = {
            "stub": True,
            "name": "run_context_hooks",
            "action": "skipped_by_flag",
        }

    return {
        "ok": True,
        "status": "inserted_with_stubbed_post_steps",
        "message": "Canonical submit_match pipeline executed. Insert is live; downstream steps are currently stubs.",
        "club_id": club_id,
        "context_type": context_type,
        "context_id": context_id,
        "idempotency_key": idempotency_key,
        "insert": insert_result,
        "rating": rating_result,
        "hooks": hooks_result,
    }


def _insert_match_record(
    club_id: str,
    context_type: str,
    context_id: Optional[str],
    match_payload: Dict[str, Any],
    idempotency_key: Optional[str],
) -> Dict[str, Any]:
    """Insert a match row, returning the existing row when idempotency matches.

    This function checks for an existing `(club_id, idempotency_key)` match
    before insert. If one exists, it is returned directly and no insert is
    performed.
    """
    supabase = get_supabase_client()

    if idempotency_key:
        existing_response = (
            supabase.table("matches")
            .select("*")
            .eq("club_id", club_id)
            .eq("idempotency_key", idempotency_key)
            .limit(1)
            .execute()
        )
        existing_rows = getattr(existing_response, "data", None) or []
        if existing_rows:
            return dict(existing_rows[0])

    payload: Dict[str, Any] = dict(match_payload)
    payload["club_id"] = club_id
    payload["context_type"] = context_type
    payload["context_id"] = context_id
    if idempotency_key:
        payload["idempotency_key"] = idempotency_key

    matches_table = supabase.table("matches")
    insert_response = matches_table.insert(payload).execute()
    inserted_rows = getattr(insert_response, "data", None) or []
    if not inserted_rows:
        raise RuntimeError("Supabase insert returned no row data for matches insert")

    return dict(inserted_rows[0])


def _apply_rating_engine_stub(match_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Stub: placeholder for future rating engine application (currently no-op)."""
    participants: list[dict[str, Any]] = []
    for slot in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
        pid = match_payload.get(slot)
        if pid is None:
            continue
        end_key = f"{slot}_r_end"
        start_key = f"{slot}_r"
        participants.append(
            {
                "player_id": int(pid),
                "new_rating": match_payload.get(end_key),
                "starting_rating": match_payload.get(start_key),
            }
        )
    return {
        "stub": True,
        "name": "apply_rating_engine",
        "action": "noop",
        "details": "No rating changes applied.",
        "participants": participants,
    }


def _run_context_hooks(
    club_id: str,
    context_type: str,
    context_id: Optional[str],
    match_payload: Dict[str, Any],
    rating_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Run context-specific hooks after insert + rating stages."""
    if context_type == "league":
        return _league_hook(
            club_id=club_id,
            league_id=str(context_id or match_payload.get("league") or "").strip(),
            match_payload=match_payload,
            rating_result=rating_result,
        )

    _ = (context_id, match_payload, rating_result)
    return {
        "stub": True,
        "name": "run_context_hooks",
        "context_type": context_type,
        "action": "noop",
        "details": "No context hook executed.",
    }


def _league_hook(
    club_id: str,
    league_id: str,
    match_payload: Dict[str, Any],
    rating_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Update league standings rows from pipeline-produced rating_result."""
    if not league_id:
        return {
            "stub": True,
            "name": "league_hook",
            "action": "skipped",
            "details": "League context_id was empty.",
        }

    participants = rating_result.get("participants") or []
    if not participants:
        return {
            "stub": True,
            "name": "league_hook",
            "action": "skipped",
            "details": "No participants in rating_result.",
        }

    score_t1 = int(match_payload.get("score_t1", 0) or 0)
    score_t2 = int(match_payload.get("score_t2", 0) or 0)
    winner_team = None
    if score_t1 != score_t2:
        winner_team = 1 if score_t1 > score_t2 else 2

    team_for_slot = {"t1_p1": 1, "t1_p2": 1, "t2_p1": 2, "t2_p2": 2}
    supabase = get_supabase_client()
    updated_rows = 0

    for slot, team in team_for_slot.items():
        pid = match_payload.get(slot)
        if pid is None:
            continue
        participant = next((p for p in participants if int(p.get("player_id", -1)) == int(pid)), None)
        if not participant or participant.get("new_rating") is None:
            continue

        existing = (
            supabase.table("league_ratings")
            .select("id,wins,losses,matches_played,starting_rating")
            .eq("club_id", club_id)
            .eq("league_name", league_id)
            .eq("player_id", int(pid))
            .limit(1)
            .execute()
        )
        existing_row = ((getattr(existing, "data", None) or [None])[0])

        wins_add = int(winner_team == team)
        losses_add = int(winner_team is not None and winner_team != team)

        if existing_row:
            payload = {
                "rating": float(participant["new_rating"]),
                "wins": int(existing_row.get("wins", 0) or 0) + wins_add,
                "losses": int(existing_row.get("losses", 0) or 0) + losses_add,
                "matches_played": int(existing_row.get("matches_played", 0) or 0) + 1,
                "is_active": True,
                "inactive_at": None,
            }
            sb_update(supabase, "league_ratings", payload, filters={"id": int(existing_row["id"]), "club_id": club_id})
            updated_rows += 1
            continue

        start_rating = participant.get("starting_rating")
        if start_rating is None:
            start_rating = participant["new_rating"]
        payload = {
            "club_id": club_id,
            "league_name": league_id,
            "player_id": int(pid),
            "rating": float(participant["new_rating"]),
            "starting_rating": float(start_rating),
            "wins": wins_add,
            "losses": losses_add,
            "matches_played": 1,
            "is_active": True,
            "inactive_at": None,
        }
        sb_insert(supabase, "league_ratings", payload)
        updated_rows += 1

    return {
        "stub": False,
        "name": "league_hook",
        "action": "updated",
        "league_id": league_id,
        "rows_updated": int(updated_rows),
    }
