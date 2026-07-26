from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from jupr_app.data.load import load_data
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.build_challenge_notice_message import build_challenge_notice_message
from jupr_app.domain.challenge_ladder import (
    TIER_ORDER,
    ladder_bucket_challenge,
    ladder_can_initiate_challenge,
    ladder_can_receive_challenge,
    ladder_compute_status_map,
    month_key_utc,
    normalize_tier_id,
)
from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.matches.side_effects import queue_player_updates, run_badge_side_effects
from jupr_app.domain.tier_movement import compute_out_of_tier_streak
from jupr_app.services.admin_live_ladder_operation_service import (
    build_match_log_recovery_url,
    deterministic_match_context_id,
    is_staging_write_gate_enabled,
    stable_request_fingerprint,
)
from jupr_app.services.public_challenge_ladder_service import build_public_challenge_ladder

TRUTHY = {"1", "true", "yes", "y", "on"}
FINAL_STATUSES = {"CANCELLED", "CANCELED", "FORFEITED", "COMPLETED", "EXPIRED_ACCEPTANCE"}
OPEN_STATUSES = {"PENDING_ACCEPTANCE", "ACCEPTED_SCHEDULING", "ACCEPTED", "IN_PROGRESS", "AWAITING_VERIFICATION", "OVERDUE_PLAY"}
CONFIRM = "SAVE LADDER"
CONFIRM_CREATE = "CREATE LADDER CHALLENGE"
CONFIRM_RESULT = "PUBLISH LADDER RESULT"
CONFIRM_FORFEIT = "RECORD LADDER FORFEIT"
CONFIRM_CLOCK = "START LADDER CLOCK"
CONFIRM_ACCEPT = "ACCEPT LADDER CHALLENGE"
CONFIRM_PASS = "RECORD LADDER PASS"
CONFIRM_ROSTER_ADD = "ADD LADDER PLAYER"
CONFIRM_ROSTER_MOVE = "MOVE LADDER PLAYER"
CONFIRM_ROSTER_REPLACE = "REPLACE LADDER TIER"
CONFIRM_OVERRIDES = "SAVE LADDER OVERRIDES"
CHALLENGE_LADDER_WRITE_FLAG = "JUPR_ENABLE_STAGING_NEXT_ADMIN_CHALLENGE_LADDER_WRITES"
CHALLENGE_LADDER_RESULT_FINALIZE_RPC = "admin_finalize_challenge_ladder_result_v1"
CHALLENGE_LADDER_RESULT_ATOMIC_RPC = "admin_apply_challenge_ladder_result_atomic_v1"
CHALLENGE_LADDER_FORFEIT_FINALIZE_RPC = "admin_finalize_challenge_ladder_forfeit_v1"
OFFICIAL_CHALLENGE_MATCH_SELECT = (
    "id,club_id,context_type,context_id,deleted_at,t1_p1,t1_p2,t2_p1,t2_p2,"
    "score_t1,score_t2"
)


def is_admin_challenge_ladder_enabled() -> bool:
    return os.getenv("JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER", "").strip().lower() in TRUTHY


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _clean(value: Any, *, limit: int = 240) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().isoformat()


def _player_names(supabase: Any, *, club_id: str) -> dict[int, str]:
    try:
        rows = _safe_rows(supabase.table("players").select("id,name").eq("club_id", str(club_id)).execute())
    except Exception:
        rows = []
    names: dict[int, str] = {}
    for row in rows:
        pid = _safe_int(row.get("id"))
        if pid is not None:
            names[int(pid)] = _clean(row.get("name"), limit=160) or f"Player {pid}"
    return names


def _challenge_row(row: dict[str, Any], names: dict[int, str]) -> dict[str, Any]:
    challenger = _safe_int(row.get("challenger_id"))
    defender = _safe_int(row.get("defender_id"))
    winner = _safe_int(row.get("winner_id"))
    return {
        "id": _safe_int(row.get("id")),
        "tier_id": normalize_tier_id(str(row.get("tier_id") or "")),
        "status": str(row.get("status") or ""),
        "bucket": ladder_bucket_challenge(row),
        "challenger_id": challenger,
        "challenger_name": names.get(int(challenger), f"Player {challenger}") if challenger is not None else "—",
        "defender_id": defender,
        "defender_name": names.get(int(defender), f"Player {defender}") if defender is not None else "—",
        "created_at": row.get("created_at"),
        "accept_by": row.get("accept_by"),
        "accepted_at": row.get("accepted_at"),
        "play_by": row.get("play_by"),
        "completed_at": row.get("completed_at"),
        "winner_id": winner,
        "winner_name": names.get(int(winner), f"Player {winner}") if winner is not None else None,
        "resolution_notes": row.get("resolution_notes") or row.get("admin_note"),
        "version": str(row.get("updated_at") or row.get("created_at") or ""),
    }


def _settings(supabase: Any, *, club_id: str) -> dict[str, Any]:
    defaults = {"challenge_range": 7, "accept_window_hours": 48, "play_window_days": 7, "cooldown_hours": 72, "protected_hours": 72, "pass_hold_hours": 72}
    try:
        row = _first(supabase.table("ladder_settings").select("*").eq("club_id", str(club_id)).limit(1).execute()) or {}
    except Exception:
        row = {}
    return {**defaults, **row}


def _roster_rows(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(supabase.table("ladder_roster").select("*").eq("club_id", str(club_id)).execute())
    except Exception:
        return []


def _active_roster_by_player(supabase: Any, *, club_id: str) -> dict[int, dict[str, Any]]:
    rows = _roster_rows(supabase, club_id=str(club_id))
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        pid = _safe_int(row.get("player_id"))
        if pid is None:
            continue
        if row.get("is_active") is False:
            continue
        result[int(pid)] = dict(row)
    return result


def _ladder_status_map(
    supabase: Any,
    *,
    club_id: str,
    settings: dict[str, Any],
    names: dict[int, str],
) -> dict[int, dict[str, Any]]:
    try:
        roster_rows = _safe_rows(supabase.table("ladder_roster").select("*").eq("club_id", str(club_id)).execute())
        flag_rows = _safe_rows(supabase.table("ladder_player_flags").select("*").eq("club_id", str(club_id)).execute())
        challenge_rows = _safe_rows(supabase.table("ladder_challenges").select("*").eq("club_id", str(club_id)).execute())
        pass_rows = _safe_rows(supabase.table("ladder_pass_usage").select("*").eq("club_id", str(club_id)).execute())
    except Exception as exc:
        raise RuntimeError("Unable to verify ladder player eligibility.") from exc
    roster_frame = pd.DataFrame(roster_rows)
    flag_frame = pd.DataFrame(flag_rows)
    challenge_frame = pd.DataFrame(challenge_rows)
    pass_frame = pd.DataFrame(pass_rows)
    for column in ("vacation_until", "reinstate_required", "reinstate_notes"):
        if column not in flag_frame.columns:
            flag_frame[column] = None
    for column in ("created_at", "accept_by", "accepted_at", "play_by", "completed_at", "winner_id", "status", "challenger_id", "defender_id"):
        if column not in challenge_frame.columns:
            challenge_frame[column] = None
    for column in ("player_id", "used_at"):
        if column not in pass_frame.columns:
            pass_frame[column] = None
    return ladder_compute_status_map(
        roster_frame,
        flag_frame,
        challenge_frame,
        pass_frame,
        settings,
        names,
        now_utc=_now(),
    )


def _admin_roster_row(row: dict[str, Any], names: dict[int, str]) -> dict[str, Any]:
    player_id = _safe_int(row.get("player_id"))
    return {
        "id": row.get("id"),
        "player_id": player_id,
        "player_name": names.get(int(player_id), f"Player {player_id}") if player_id is not None else "—",
        "tier_id": normalize_tier_id(str(row.get("tier_id") or "")),
        "rank": _safe_int(row.get("rank")),
        "is_active": row.get("is_active") is not False,
        "joined_at": row.get("joined_at"),
        "left_at": row.get("left_at"),
        "notes": row.get("notes"),
        "updated_at": row.get("updated_at"),
    }


def _admin_player_flag_row(row: dict[str, Any], names: dict[int, str]) -> dict[str, Any]:
    player_id = _safe_int(row.get("player_id"))
    return {
        "player_id": player_id,
        "player_name": names.get(int(player_id), f"Player {player_id}") if player_id is not None else "—",
        "vacation_until": row.get("vacation_until"),
        "reinstate_required": bool(row.get("reinstate_required", False)),
        "reinstate_notes": row.get("reinstate_notes"),
        "tier_move_flag": bool(row.get("tier_move_flag", False)),
        "tier_move_dest_tier": normalize_tier_id(str(row.get("tier_move_dest_tier") or "")) if row.get("tier_move_dest_tier") else None,
        "tier_move_count": int(_safe_int(row.get("tier_move_count")) or 0),
    }


def _vacation_until_iso(value: Any) -> str | None:
    cleaned = _clean(value, limit=80)
    if not cleaned:
        return None
    try:
        parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Vacation until must be a valid ISO date-time.") from exc
    if parsed.tzinfo is None:
        raise ValueError("Vacation until must include a UTC offset or Z suffix.")
    return parsed.astimezone(timezone.utc).isoformat()


def _challenge(supabase: Any, *, club_id: str, challenge_id: int) -> dict[str, Any]:
    row = _first(supabase.table("ladder_challenges").select("*").eq("club_id", str(club_id)).eq("id", int(challenge_id)).limit(1).execute())
    if row is None:
        raise ValueError("challenge not found")
    return row


def build_admin_challenge_ladder_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        return {"enabled": False, "status": "guarded_off", "warnings": ["Enable JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER to use Challenge Ladder Admin in Next."]}
    summary = {"active_player_count": 0, "active_challenge_count": 0, "tier_count": len(TIER_ORDER)}
    if supabase is not None:
        try:
            summary = build_public_challenge_ladder(supabase, club_id=str(club_id)).get("summary", summary)
        except Exception:
            pass
    writes_enabled = is_staging_write_gate_enabled(CHALLENGE_LADDER_WRITE_FLAG)
    return {
        "enabled": True,
        "authority": "python_fastapi",
        "writes_enabled": writes_enabled,
        "status": "ready_for_challenge_ladder_admin" if writes_enabled else "read_only_streamlit_fallback",
        "summary": summary,
        "warnings": [] if writes_enabled else [
            f"Next Challenge Ladder writes require JUPR_ENV=staging and {CHALLENGE_LADDER_WRITE_FLAG}=1 on FastAPI. Use Streamlit Challenge Ladder Admin otherwise."
        ],
        "confirmation_text": {
            "create": CONFIRM_CREATE,
            "update": CONFIRM,
            "result": CONFIRM_RESULT,
            "forfeit": CONFIRM_FORFEIT,
            "clock": CONFIRM_CLOCK,
            "accept": CONFIRM_ACCEPT,
            "pass": CONFIRM_PASS,
            "roster_add": CONFIRM_ROSTER_ADD,
            "roster_move": CONFIRM_ROSTER_MOVE,
            "roster_replace": CONFIRM_ROSTER_REPLACE,
            "overrides": CONFIRM_OVERRIDES,
        },
        "streamlit_fallback": "challenge_ladder_admin",
        "recovery": {"match_log_url": "/admin/match-log", "replay_history_url": "/admin/replay-history"},
    }


def get_admin_challenge_ladder_dashboard(supabase: Any, *, club_id: str) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    public_payload = build_public_challenge_ladder(supabase, club_id=str(club_id))
    names = _player_names(supabase, club_id=str(club_id))
    try:
        challenge_rows = _safe_rows(supabase.table("ladder_challenges").select("*").eq("club_id", str(club_id)).order("created_at", desc=True).limit(500).execute())
    except Exception:
        challenge_rows = []
    bucket_counts: dict[str, int] = {}
    for row in challenge_rows:
        bucket = ladder_bucket_challenge(row)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    challenges = [_challenge_row(row, names) for row in challenge_rows[:100]]
    try:
        settings = _safe_rows(supabase.table("ladder_settings").select("*").eq("club_id", str(club_id)).limit(1).execute())
    except Exception:
        settings = []
    player_options = [
        {"player_id": player_id, "player_name": player_name}
        for player_id, player_name in sorted(names.items(), key=lambda item: (item[1].casefold(), item[0]))
    ]
    roster_rows = sorted(
        (_admin_roster_row(row, names) for row in _roster_rows(supabase, club_id=str(club_id))),
        key=lambda row: (
            not bool(row.get("is_active")),
            TIER_ORDER.index(str(row.get("tier_id"))) if row.get("tier_id") in TIER_ORDER else len(TIER_ORDER),
            int(row.get("rank") or 999999),
            str(row.get("player_name") or "").casefold(),
        ),
    )
    try:
        flag_rows = _safe_rows(supabase.table("ladder_player_flags").select("*").eq("club_id", str(club_id)).execute())
    except Exception:
        flag_rows = []
    player_flags = sorted(
        (_admin_player_flag_row(row, names) for row in flag_rows),
        key=lambda row: (str(row.get("player_name") or "").casefold(), int(row.get("player_id") or 0)),
    )
    payload = {
        "ok": True,
        "mode": "challenge_ladder_admin_dashboard",
        **public_payload,
        "bucket_counts": bucket_counts,
        "challenges": challenges,
        "settings_row": settings[0] if settings else {},
        "player_options": player_options,
        "roster_rows": roster_rows,
        "player_flags": player_flags,
    }
    payload["state_version"] = stable_request_fingerprint(
        {
            "club_id": str(club_id),
            "settings": payload.get("settings_row") or {},
            "challenges": challenges,
            "roster_rows": roster_rows,
            "player_flags": player_flags,
        }
    )
    payload["authority"] = "python_fastapi"
    return payload


def get_admin_challenge_ladder_tier_movement_review(supabase: Any, *, club_id: str) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    names = _player_names(supabase, club_id=str(club_id))
    roster_rows = [row for row in _roster_rows(supabase, club_id=str(club_id)) if row.get("is_active") is not False]
    try:
        raw_matches = _safe_rows(
            supabase.table("matches")
            .select("*")
            .eq("club_id", str(club_id))
            .order("id", desc=True)
            .limit(5000)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Unable to load matches for tier-movement review.") from exc
    match_rows = [row for row in raw_matches if not row.get("deleted_at")]
    df_matches = pd.DataFrame(match_rows)
    triggers: list[dict[str, Any]] = []
    evaluated_player_count = 0
    for roster_row in roster_rows:
        player_id = _safe_int(roster_row.get("player_id"))
        if player_id is None:
            continue
        evaluated_player_count += 1
        current_tier = normalize_tier_id(str(roster_row.get("tier_id") or ""))
        joined_at = pd.to_datetime(roster_row.get("joined_at"), utc=True, errors="coerce")
        joined_datetime = joined_at.to_pydatetime() if pd.notna(joined_at) else None
        streak = compute_out_of_tier_streak(
            pid=int(player_id),
            joined_at_utc=joined_datetime,
            current_tier_id=current_tier,
            df_matches=df_matches,
        )
        destination = normalize_tier_id(str(streak.get("dest_tier") or "")) if streak.get("dest_tier") else None
        count = int(_safe_int(streak.get("count")) or 0)
        if destination is None or destination == current_tier or count < 10:
            continue
        latest = streak.get("latest_match_at")
        triggers.append(
            {
                "player_id": int(player_id),
                "player_name": names.get(int(player_id), f"Player {player_id}"),
                "current_tier": current_tier,
                "destination_tier": destination,
                "consecutive_match_count": count,
                "latest_match_at": latest.isoformat() if isinstance(latest, datetime) else None,
            }
        )
    triggers.sort(
        key=lambda row: (
            -int(row.get("consecutive_match_count") or 0),
            -pd.Timestamp(row["latest_match_at"]).timestamp() if row.get("latest_match_at") else float("inf"),
            str(row.get("player_name") or "").casefold(),
        )
    )
    payload = {
        "ok": True,
        "mode": "challenge_ladder_tier_movement_review",
        "summary": {
            "evaluated_player_count": evaluated_player_count,
            "match_count": len(match_rows),
            "trigger_count": len(triggers),
            "required_consecutive_matches": 10,
        },
        "triggers": triggers,
    }
    return payload


def _write_ladder_audit(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action_type: str,
    entity_id: str,
    before: Any,
    after: Any,
    source: str,
    note: str | None = None,
    entity_type: str = "ladder_challenge",
) -> str | None:
    write = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type=action_type,
            entity_type=entity_type,
            entity_id=str(entity_id),
            before_json=before,
            after_json={"source_client": "fastapi/nextjs", **(after if isinstance(after, dict) else {"value": after})},
            note=note,
            source_page=source,
            flagged_for_review=True,
        ),
    )
    if not write.ok and os.getenv("JUPR_REQUIRE_API_AUDIT_LOG", "").strip().lower() in TRUTHY:
        raise RuntimeError("audit log write required but unavailable")
    return write.warning


def create_admin_challenge_ladder_challenge(
    supabase: Any,
    *,
    club_id: str,
    challenger_id: int,
    defender_id: int,
    tier_id: str,
    ledger_ref: str | None,
    override: bool,
    start_clock: bool,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    challenger_contact: str | None = None,
    source: str = "next_challenge_ladder_admin_create",
) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_CREATE:
        raise ValueError(f"Type {CONFIRM_CREATE} to create a ladder challenge.")
    challenger = int(challenger_id)
    defender = int(defender_id)
    if challenger == defender:
        raise ValueError("Challenger and defender must be different players.")
    tier = normalize_tier_id(tier_id)
    if tier not in TIER_ORDER:
        raise ValueError("unsupported tier")
    settings = _settings(supabase, club_id=str(club_id))
    roster = _active_roster_by_player(supabase, club_id=str(club_id))
    chal_row = roster.get(challenger)
    def_row = roster.get(defender)
    if chal_row is None or def_row is None:
        raise ValueError("Both challenger and defender must be active ladder players.")
    if normalize_tier_id(str(chal_row.get("tier_id") or "")) != tier or normalize_tier_id(str(def_row.get("tier_id") or "")) != tier:
        raise ValueError("Both players must be active in the selected tier.")
    chal_rank = int(_safe_int(chal_row.get("rank"),) or 999999)
    def_rank = int(_safe_int(def_row.get("rank"),) or 999999)
    errors = []
    if def_rank >= chal_rank:
        errors.append("Defender must be ranked above challenger.")
    if (chal_rank - def_rank) > int(settings.get("challenge_range") or 7):
        errors.append("Rank gap exceeds challenge range.")
    names = _player_names(supabase, club_id=str(club_id))
    if not override:
        status_map = _ladder_status_map(
            supabase,
            club_id=str(club_id),
            settings=settings,
            names=names,
        )
        challenger_status = str(status_map.get(challenger, {}).get("status") or "Unknown")
        defender_status = str(status_map.get(defender, {}).get("status") or "Unknown")
        if not ladder_can_initiate_challenge(challenger_status):
            errors.append(f"Challenger is not eligible to initiate (status: {challenger_status}).")
        if not ladder_can_receive_challenge(defender_status):
            errors.append(f"Defender is not eligible to be challenged (status: {defender_status}).")
    if errors and not override:
        raise ValueError("Cannot create challenge: " + "; ".join(errors))
    now = _now()
    payload: dict[str, Any] = {
        "club_id": str(club_id),
        "challenger_id": challenger,
        "defender_id": defender,
        "challenger_rank_at_create": chal_rank,
        "defender_rank_at_create": def_rank,
        "status": "PENDING_ACCEPTANCE",
        "created_by": actor_email,
        "ledger_ref": _clean(ledger_ref, limit=500) or None,
        "tier_id": tier,
        "created_at": now.isoformat(),
        "accept_by": (now + timedelta(hours=int(settings.get("accept_window_hours") or 48))).isoformat() if start_clock else None,
    }
    created = _first(supabase.table("ladder_challenges").insert(payload).execute()) or payload
    warning = _write_ladder_audit(supabase, club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, action_type="challenge_create", entity_id=str(created.get("id") or "new"), before=None, after=created, source=source)
    notice = build_challenge_notice_message(
        challenge_id=_safe_int(created.get("id")),
        tier_id=tier,
        challenger_name=names.get(challenger, f"Player {challenger}"),
        defender_name=names.get(defender, f"Player {defender}"),
        challenger_contact=_clean(challenger_contact, limit=240),
        admin_name=_clean(os.getenv("LADDER_ADMIN_NAME", "Ladder Admin"), limit=160),
        admin_contact=_clean(os.getenv("LADDER_ADMIN_CONTACT", ""), limit=240),
        ledger_ref=_clean(ledger_ref, limit=500) or None,
    )
    payload = {
        "ok": True,
        "mode": "challenge_ladder_create",
        "challenge": _challenge_row(created, names),
        "notice": notice,
        "warnings": [warning] if warning else [],
    }
    return payload


def start_admin_challenge_ladder_clock(supabase: Any, *, club_id: str, challenge_id: int, actor_email: str, actor_role: str, confirmation_text: str, source: str = "next_challenge_ladder_admin_clock") -> dict[str, Any]:
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_CLOCK:
        raise ValueError(f"Type {CONFIRM_CLOCK} to start the challenge clock.")
    before = _challenge(supabase, club_id=str(club_id), challenge_id=int(challenge_id))
    if str(before.get("status") or "") != "PENDING_ACCEPTANCE":
        raise ValueError("Only pending challenges can start the accept clock.")
    settings = _settings(supabase, club_id=str(club_id))
    patch = {"accept_by": (_now() + timedelta(hours=int(settings.get("accept_window_hours") or 48))).isoformat(), "updated_at": _now_iso()}
    updated = _first(supabase.table("ladder_challenges").update(patch).eq("club_id", str(club_id)).eq("id", int(challenge_id)).execute()) or {**before, **patch}
    names = _player_names(supabase, club_id=str(club_id))
    warning = _write_ladder_audit(supabase, club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, action_type="challenge_start_clock", entity_id=str(challenge_id), before=before, after=updated, source=source)
    return {"ok": True, "mode": "challenge_ladder_start_clock", "challenge": _challenge_row(updated, names), "warnings": [warning] if warning else []}


def accept_admin_challenge_ladder_challenge(supabase: Any, *, club_id: str, challenge_id: int, actor_email: str, actor_role: str, confirmation_text: str, source: str = "next_challenge_ladder_admin_accept") -> dict[str, Any]:
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_ACCEPT:
        raise ValueError(f"Type {CONFIRM_ACCEPT} to accept the challenge.")
    before = _challenge(supabase, club_id=str(club_id), challenge_id=int(challenge_id))
    if str(before.get("status") or "") != "PENDING_ACCEPTANCE":
        raise ValueError("Only pending challenges can be accepted.")
    settings = _settings(supabase, club_id=str(club_id))
    now = _now()
    patch = {"status": "ACCEPTED_SCHEDULING", "accepted_at": now.isoformat(), "play_by": (now + timedelta(days=int(settings.get("play_window_days") or 7))).isoformat(), "updated_at": now.isoformat()}
    updated = _first(supabase.table("ladder_challenges").update(patch).eq("club_id", str(club_id)).eq("id", int(challenge_id)).execute()) or {**before, **patch}
    names = _player_names(supabase, club_id=str(club_id))
    warning = _write_ladder_audit(supabase, club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, action_type="challenge_accept", entity_id=str(challenge_id), before=before, after=updated, source=source)
    return {"ok": True, "mode": "challenge_ladder_accept", "challenge": _challenge_row(updated, names), "warnings": [warning] if warning else []}


def record_admin_challenge_ladder_pass(
    supabase: Any,
    *,
    club_id: str,
    challenge_id: int,
    player_id: int,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_challenge_ladder_pass",
) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_PASS:
        raise ValueError(f"Type {CONFIRM_PASS} to record the monthly pass.")
    before = _challenge(supabase, club_id=str(club_id), challenge_id=int(challenge_id))
    status = _clean(before.get("status"), limit=40).upper()
    if status not in OPEN_STATUSES or before.get("completed_at") or before.get("winner_id"):
        raise ValueError("Only an open, unresolved challenge can use a monthly pass.")
    safe_player_id = int(player_id)
    participants = {_safe_int(before.get("challenger_id")), _safe_int(before.get("defender_id"))}
    if safe_player_id not in participants:
        raise ValueError("Pass user must be the challenger or defender.")

    now = _now()
    month_key = month_key_utc(now)
    existing = _first(
        supabase.table("ladder_pass_usage")
        .select("id,challenge_id,used_at")
        .eq("club_id", str(club_id))
        .eq("player_id", safe_player_id)
        .eq("month_key", month_key)
        .limit(1)
        .execute()
    )
    if existing is not None:
        raise ValueError(f"This player already used a ladder pass in {month_key}.")

    usage_payload = {
        "club_id": str(club_id),
        "player_id": safe_player_id,
        "month_key": month_key,
        "used_at": now.isoformat(),
        "challenge_id": int(challenge_id),
    }
    usage = _first(supabase.table("ladder_pass_usage").insert(usage_payload).execute()) or usage_payload
    patch = {
        "status": "CANCELED",
        "pass_used_by": safe_player_id,
        "pass_used_at": now.isoformat(),
        "resolution_notes": "Pass used",
        "completed_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }
    updated = _first(
        supabase.table("ladder_challenges")
        .update(patch)
        .eq("club_id", str(club_id))
        .eq("id", int(challenge_id))
        .execute()
    ) or {**before, **patch}
    names = _player_names(supabase, club_id=str(club_id))
    warning = _write_ladder_audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="challenge_pass_used",
        entity_id=str(challenge_id),
        before=before,
        after={**updated, "pass_usage_id": usage.get("id"), "month_key": month_key},
        source=source,
    )
    return {
        "ok": True,
        "mode": "challenge_ladder_pass",
        "challenge": _challenge_row(updated, names),
        "pass_usage": usage,
        "warnings": [warning] if warning else [],
    }


def add_admin_challenge_ladder_roster_player(
    supabase: Any,
    *,
    club_id: str,
    player_id: int,
    tier_id: str,
    admin_note: str | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_challenge_ladder_roster_add",
) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_ROSTER_ADD:
        raise ValueError(f"Type {CONFIRM_ROSTER_ADD} to add the ladder player.")
    tier = normalize_tier_id(tier_id)
    if tier not in TIER_ORDER:
        raise ValueError("unsupported tier")
    safe_player_id = int(player_id)
    names = _player_names(supabase, club_id=str(club_id))
    if safe_player_id not in names:
        raise ValueError("Player must belong to this club.")

    rows = _roster_rows(supabase, club_id=str(club_id))
    existing_rows = [row for row in rows if _safe_int(row.get("player_id")) == safe_player_id]
    active = next((row for row in existing_rows if row.get("is_active") is not False), None)
    if active is not None:
        raise ValueError(
            f"Player is already active in {normalize_tier_id(str(active.get('tier_id') or ''))} at rank {_safe_int(active.get('rank')) or '—'}."
        )
    next_rank = 1 + max(
        [
            int(_safe_int(row.get("rank")) or 0)
            for row in rows
            if row.get("is_active") is not False and normalize_tier_id(str(row.get("tier_id") or "")) == tier
        ]
        or [0]
    )
    now_iso = _now_iso()
    before = existing_rows[0] if existing_rows else None
    patch = {
        "is_active": True,
        "tier_id": tier,
        "rank": next_rank,
        "left_at": None,
        "joined_at": now_iso,
        "updated_at": now_iso,
    }
    if before is not None:
        saved = _first(
            supabase.table("ladder_roster")
            .update(patch)
            .eq("club_id", str(club_id))
            .eq("player_id", safe_player_id)
            .execute()
        ) or {**before, **patch}
        action_type = "roster_reactivate_append"
    else:
        insert_payload = {
            "club_id": str(club_id),
            "player_id": safe_player_id,
            **patch,
        }
        saved = _first(supabase.table("ladder_roster").insert(insert_payload).execute()) or insert_payload
        action_type = "roster_append"
    warning = _write_ladder_audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type=action_type,
        entity_type="ladder_roster",
        entity_id=f"{club_id}:{safe_player_id}",
        before=before,
        after=saved,
        source=source,
        note=_clean(admin_note, limit=1000) or None,
    )
    return {
        "ok": True,
        "mode": "challenge_ladder_roster_add",
        "roster": _admin_roster_row(saved, names),
        "reactivated": before is not None,
        "warnings": [warning] if warning else [],
    }


def _build_admin_challenge_ladder_tier_roster_preview(
    supabase: Any,
    *,
    club_id: str,
    tier_id: str,
    ranked_player_ids: list[int],
) -> dict[str, Any]:
    tier = normalize_tier_id(tier_id)
    if tier not in TIER_ORDER:
        raise ValueError("unsupported tier")
    player_ids = [int(player_id) for player_id in ranked_player_ids]
    if not player_ids:
        raise ValueError("Provide at least one player for the replacement roster.")
    if len(player_ids) > 200:
        raise ValueError("A tier replacement is limited to 200 players.")
    if len(set(player_ids)) != len(player_ids):
        raise ValueError("The replacement roster cannot contain duplicate players.")

    names = _player_names(supabase, club_id=str(club_id))
    missing_player_ids = [player_id for player_id in player_ids if player_id not in names]
    if missing_player_ids:
        raise ValueError("Every replacement player must belong to this club.")
    try:
        roster_rows = _safe_rows(
            supabase.table("ladder_roster").select("*").eq("club_id", str(club_id)).execute()
        )
        challenge_rows = _safe_rows(
            supabase.table("ladder_challenges").select("*").eq("club_id", str(club_id)).execute()
        )
    except Exception as exc:
        raise RuntimeError("Unable to build a safe tier-roster replacement preview.") from exc

    active_rows = [row for row in roster_rows if row.get("is_active") is not False]
    active_by_player: dict[int, dict[str, Any]] = {}
    for row in active_rows:
        player_id = _safe_int(row.get("player_id"))
        if player_id is not None:
            active_by_player[int(player_id)] = dict(row)
    existing_player_ids = {
        int(player_id)
        for player_id in (_safe_int(row.get("player_id")) for row in roster_rows)
        if player_id is not None
    }
    current_rows = sorted(
        (
            row
            for row in active_rows
            if normalize_tier_id(str(row.get("tier_id") or "")) == tier
        ),
        key=lambda row: (
            int(_safe_int(row.get("rank")) or 999999),
            int(_safe_int(row.get("player_id")) or 999999),
        ),
    )
    current_player_ids = [
        int(player_id)
        for player_id in (_safe_int(row.get("player_id")) for row in current_rows)
        if player_id is not None
    ]
    proposed_id_set = set(player_ids)
    removed_player_ids = [player_id for player_id in current_player_ids if player_id not in proposed_id_set]

    proposed_roster: list[dict[str, Any]] = []
    moved_from_other_tiers: list[dict[str, Any]] = []
    reordered_count = 0
    retained_count = 0
    added_count = 0
    reactivated_count = 0
    affected_player_ids = set(removed_player_ids)
    moved_source_tiers: set[str] = set()
    for new_rank, player_id in enumerate(player_ids, start=1):
        active = active_by_player.get(player_id)
        previous_tier = normalize_tier_id(str(active.get("tier_id") or "")) if active else None
        previous_rank = _safe_int(active.get("rank")) if active else None
        if active is None:
            change = "reactivated" if player_id in existing_player_ids else "added"
            if change == "reactivated":
                reactivated_count += 1
            else:
                added_count += 1
            affected_player_ids.add(player_id)
        elif previous_tier != tier:
            change = "moved"
            moved_source_tiers.add(str(previous_tier))
            affected_player_ids.add(player_id)
        elif previous_rank != new_rank:
            change = "reordered"
            reordered_count += 1
            affected_player_ids.add(player_id)
        else:
            change = "retained"
            retained_count += 1
        proposed = {
            "rank": new_rank,
            "player_id": player_id,
            "player_name": names[player_id],
            "previous_tier": previous_tier,
            "previous_rank": previous_rank,
            "change": change,
        }
        proposed_roster.append(proposed)
        if change == "moved":
            moved_from_other_tiers.append(dict(proposed))

    source_tier_recompressions: list[dict[str, Any]] = []
    for source_tier in sorted(moved_source_tiers):
        remaining_rows = sorted(
            (
                row
                for row in active_rows
                if normalize_tier_id(str(row.get("tier_id") or "")) == source_tier
                and int(_safe_int(row.get("player_id")) or -1) not in proposed_id_set
            ),
            key=lambda row: (
                int(_safe_int(row.get("rank")) or 999999),
                int(_safe_int(row.get("player_id")) or 999999),
            ),
        )
        for expected_rank, row in enumerate(remaining_rows, start=1):
            player_id = _safe_int(row.get("player_id"))
            old_rank = _safe_int(row.get("rank"))
            if player_id is None or old_rank == expected_rank:
                continue
            affected_player_ids.add(int(player_id))
            source_tier_recompressions.append(
                {
                    "tier_id": source_tier,
                    "player_id": int(player_id),
                    "player_name": names.get(int(player_id), f"Player {player_id}"),
                    "old_rank": old_rank,
                    "new_rank": expected_rank,
                }
            )

    open_challenge_blockers: list[dict[str, Any]] = []
    for challenge in challenge_rows:
        if str(challenge.get("status") or "").upper() not in OPEN_STATUSES:
            continue
        participants = {
            int(player_id)
            for player_id in (
                _safe_int(challenge.get("challenger_id")),
                _safe_int(challenge.get("defender_id")),
            )
            if player_id is not None
        }
        blocked_ids = sorted(participants.intersection(affected_player_ids))
        if not blocked_ids:
            continue
        open_challenge_blockers.append(
            {
                "challenge_id": _safe_int(challenge.get("id")),
                "status": str(challenge.get("status") or ""),
                "affected_player_ids": blocked_ids,
                "affected_player_names": [names.get(player_id, f"Player {player_id}") for player_id in blocked_ids],
            }
        )

    active_snapshot = sorted(
        (
            {
                "player_id": int(player_id),
                "tier_id": normalize_tier_id(str(row.get("tier_id") or "")),
                "rank": _safe_int(row.get("rank")),
            }
            for row in active_rows
            for player_id in [_safe_int(row.get("player_id"))]
            if player_id is not None
        ),
        key=lambda row: (int(row["player_id"]), str(row["tier_id"]), int(row["rank"] or 999999)),
    )
    fingerprint_payload = {
        "club_id": str(club_id),
        "tier_id": tier,
        "ranked_player_ids": player_ids,
        "active_roster": active_snapshot,
    }
    preview_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    warnings: list[str] = []
    if moved_from_other_tiers:
        warnings.append("Players moved from other tiers will be removed from those tiers, and their former tiers will be recompressed.")
    if open_challenge_blockers:
        warnings.append("Resolve the listed open challenges before replacing this tier roster.")
    return {
        "ok": True,
        "mode": "challenge_ladder_roster_replace_preview",
        "tier_id": tier,
        "can_apply": not open_challenge_blockers,
        "preview_fingerprint": preview_fingerprint,
        "summary": {
            "current_count": len(current_player_ids),
            "proposed_count": len(player_ids),
            "retained_count": retained_count,
            "reordered_count": reordered_count,
            "added_count": added_count,
            "reactivated_count": reactivated_count,
            "removed_count": len(removed_player_ids),
            "moved_from_other_tier_count": len(moved_from_other_tiers),
            "source_tier_recompression_count": len(source_tier_recompressions),
        },
        "current_roster": [
            {
                "rank": _safe_int(row.get("rank")),
                "player_id": _safe_int(row.get("player_id")),
                "player_name": names.get(int(_safe_int(row.get("player_id")) or -1), "Unknown player"),
            }
            for row in current_rows
        ],
        "proposed_roster": proposed_roster,
        "removed_players": [
            {
                "rank": _safe_int(row.get("rank")),
                "player_id": _safe_int(row.get("player_id")),
                "player_name": names.get(int(_safe_int(row.get("player_id")) or -1), "Unknown player"),
            }
            for row in current_rows
            if int(_safe_int(row.get("player_id")) or -1) in set(removed_player_ids)
        ],
        "moved_from_other_tiers": moved_from_other_tiers,
        "source_tier_recompressions": source_tier_recompressions,
        "open_challenge_blockers": open_challenge_blockers,
        "warnings": warnings,
    }


def preview_admin_challenge_ladder_tier_roster_replacement(
    supabase: Any,
    *,
    club_id: str,
    tier_id: str,
    ranked_names: list[str],
) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    cleaned_names = [_clean(name, limit=160) for name in ranked_names]
    cleaned_names = [name for name in cleaned_names if name]
    if not cleaned_names:
        raise ValueError("Paste at least one player name.")
    if len(cleaned_names) > 200:
        raise ValueError("A tier replacement is limited to 200 players.")
    duplicates = sorted({name for name in cleaned_names if cleaned_names.count(name) > 1})
    if duplicates:
        raise ValueError("Duplicate names are not allowed: " + ", ".join(duplicates))
    names = _player_names(supabase, club_id=str(club_id))
    player_ids_by_name: dict[str, list[int]] = {}
    for player_id, player_name in names.items():
        player_ids_by_name.setdefault(player_name, []).append(int(player_id))
    missing = [name for name in cleaned_names if name not in player_ids_by_name]
    if missing:
        raise ValueError("Create these club players before replacing the tier: " + ", ".join(missing))
    ambiguous = [name for name in cleaned_names if len(player_ids_by_name[name]) != 1]
    if ambiguous:
        raise ValueError("These names are ambiguous in the club player list: " + ", ".join(ambiguous))
    return _build_admin_challenge_ladder_tier_roster_preview(
        supabase,
        club_id=str(club_id),
        tier_id=tier_id,
        ranked_player_ids=[player_ids_by_name[name][0] for name in cleaned_names],
    )


def replace_admin_challenge_ladder_tier_roster(
    supabase: Any,
    *,
    club_id: str,
    tier_id: str,
    ranked_player_ids: list[int],
    preview_fingerprint: str,
    admin_note: str | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_challenge_ladder_roster_replace",
) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_ROSTER_REPLACE:
        raise ValueError(f"Type {CONFIRM_ROSTER_REPLACE} to replace the tier roster.")
    preview = _build_admin_challenge_ladder_tier_roster_preview(
        supabase,
        club_id=str(club_id),
        tier_id=tier_id,
        ranked_player_ids=ranked_player_ids,
    )
    if _clean(preview_fingerprint, limit=128) != str(preview["preview_fingerprint"]):
        raise ValueError("The tier roster changed after preview. Preview the replacement again before applying it.")
    if not preview["can_apply"]:
        raise ValueError("Resolve open challenges involving affected players before replacing the tier roster.")

    tier = str(preview["tier_id"])
    proposed_ids = [int(row["player_id"]) for row in preview["proposed_roster"]]
    affected_ids = set(proposed_ids)
    affected_ids.update(int(row["player_id"]) for row in preview["removed_players"])
    affected_ids.update(int(row["player_id"]) for row in preview["source_tier_recompressions"])
    before_rows = [
        row
        for row in _roster_rows(supabase, club_id=str(club_id))
        if int(_safe_int(row.get("player_id")) or -1) in affected_ids
    ]
    now_iso = _now_iso()
    upsert_rows = [
        {
            "club_id": str(club_id),
            "player_id": int(row["player_id"]),
            "tier_id": tier,
            "rank": int(row["rank"]),
            "is_active": True,
            "joined_at": now_iso,
            "left_at": None,
            "updated_at": now_iso,
        }
        for row in preview["proposed_roster"]
    ]
    supabase.table("ladder_roster").upsert(upsert_rows, on_conflict="club_id,player_id").execute()
    for removed in preview["removed_players"]:
        supabase.table("ladder_roster").update(
            {"is_active": False, "left_at": now_iso, "updated_at": now_iso}
        ).eq("club_id", str(club_id)).eq("player_id", int(removed["player_id"])).execute()
    recompressed_player_ids: list[int] = []
    for recompression in preview["source_tier_recompressions"]:
        player_id = int(recompression["player_id"])
        supabase.table("ladder_roster").update(
            {"rank": int(recompression["new_rank"]), "updated_at": now_iso}
        ).eq("club_id", str(club_id)).eq("player_id", player_id).execute()
        recompressed_player_ids.append(player_id)

    final_rows = _roster_rows(supabase, club_id=str(club_id))
    final_target = sorted(
        (
            row
            for row in final_rows
            if row.get("is_active") is not False
            and normalize_tier_id(str(row.get("tier_id") or "")) == tier
        ),
        key=lambda row: int(_safe_int(row.get("rank")) or 999999),
    )
    final_signature = [
        (int(_safe_int(row.get("player_id")) or -1), int(_safe_int(row.get("rank")) or -1))
        for row in final_target
    ]
    expected_signature = list(zip(proposed_ids, range(1, len(proposed_ids) + 1)))
    if final_signature != expected_signature:
        raise RuntimeError("Tier roster replacement did not persist the reviewed player order. Review ladder activity before retrying.")

    names = _player_names(supabase, club_id=str(club_id))
    after_rows = [
        row
        for row in final_rows
        if int(_safe_int(row.get("player_id")) or -1) in affected_ids
    ]
    warning = _write_ladder_audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="roster_replace_tier",
        entity_type="ladder_roster",
        entity_id=f"{club_id}:{tier}",
        before={"tier_id": tier, "rows": before_rows},
        after={
            "tier_id": tier,
            "rows": after_rows,
            "summary": preview["summary"],
            "preview_fingerprint": preview["preview_fingerprint"],
            "recompressed_player_ids": recompressed_player_ids,
        },
        source=source,
        note=_clean(admin_note, limit=1000) or None,
    )
    return {
        "ok": True,
        "mode": "challenge_ladder_roster_replace",
        "tier_id": tier,
        "roster": [_admin_roster_row(row, names) for row in final_target],
        "summary": preview["summary"],
        "recompressed_player_ids": recompressed_player_ids,
        "warnings": [warning] if warning else [],
    }


def move_admin_challenge_ladder_roster_player(
    supabase: Any,
    *,
    club_id: str,
    player_id: int,
    destination_tier: str,
    recompress_old: bool,
    admin_note: str | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_challenge_ladder_roster_move",
) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_ROSTER_MOVE:
        raise ValueError(f"Type {CONFIRM_ROSTER_MOVE} to move the ladder player.")
    destination = normalize_tier_id(destination_tier)
    if destination not in TIER_ORDER:
        raise ValueError("unsupported destination tier")
    safe_player_id = int(player_id)
    rows = _roster_rows(supabase, club_id=str(club_id))
    before = next(
        (
            row
            for row in rows
            if _safe_int(row.get("player_id")) == safe_player_id and row.get("is_active") is not False
        ),
        None,
    )
    if before is None:
        raise ValueError("Player must be active on this club's ladder.")
    previous_tier = normalize_tier_id(str(before.get("tier_id") or ""))
    if destination == previous_tier:
        raise ValueError("Destination tier must differ from the current tier.")
    next_rank = 1 + max(
        [
            int(_safe_int(row.get("rank")) or 0)
            for row in rows
            if row.get("is_active") is not False and normalize_tier_id(str(row.get("tier_id") or "")) == destination
        ]
        or [0]
    )
    now_iso = _now_iso()
    patch = {"tier_id": destination, "rank": next_rank, "updated_at": now_iso}
    saved = _first(
        supabase.table("ladder_roster")
        .update(patch)
        .eq("club_id", str(club_id))
        .eq("player_id", safe_player_id)
        .execute()
    ) or {**before, **patch}

    recompressed_player_ids: list[int] = []
    if recompress_old:
        old_tier_rows = sorted(
            (
                row
                for row in _roster_rows(supabase, club_id=str(club_id))
                if row.get("is_active") is not False
                and _safe_int(row.get("player_id")) != safe_player_id
                and normalize_tier_id(str(row.get("tier_id") or "")) == previous_tier
            ),
            key=lambda row: (int(_safe_int(row.get("rank")) or 999999), int(_safe_int(row.get("player_id")) or 999999)),
        )
        for expected_rank, row in enumerate(old_tier_rows, start=1):
            roster_player_id = _safe_int(row.get("player_id"))
            if roster_player_id is None or _safe_int(row.get("rank")) == expected_rank:
                continue
            supabase.table("ladder_roster").update({"rank": expected_rank, "updated_at": now_iso}).eq(
                "club_id", str(club_id)
            ).eq("player_id", int(roster_player_id)).execute()
            recompressed_player_ids.append(int(roster_player_id))

    names = _player_names(supabase, club_id=str(club_id))
    audit_after = {
        **saved,
        "previous_tier": previous_tier,
        "recompressed_player_ids": recompressed_player_ids,
    }
    warning = _write_ladder_audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="roster_move_tier",
        entity_type="ladder_roster",
        entity_id=f"{club_id}:{safe_player_id}",
        before=before,
        after=audit_after,
        source=source,
        note=_clean(admin_note, limit=1000) or None,
    )
    return {
        "ok": True,
        "mode": "challenge_ladder_roster_move",
        "roster": _admin_roster_row(saved, names),
        "previous_tier": previous_tier,
        "recompressed_count": len(recompressed_player_ids),
        "warnings": [warning] if warning else [],
    }


def save_admin_challenge_ladder_player_overrides(
    supabase: Any,
    *,
    club_id: str,
    player_id: int,
    vacation_until: str | None,
    reinstate_required: bool,
    reinstate_notes: str | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_challenge_ladder_player_overrides",
) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_OVERRIDES:
        raise ValueError(f"Type {CONFIRM_OVERRIDES} to save ladder overrides.")
    safe_player_id = int(player_id)
    active_roster = _active_roster_by_player(supabase, club_id=str(club_id))
    if safe_player_id not in active_roster:
        raise ValueError("Player must be active on this club's ladder.")
    names = _player_names(supabase, club_id=str(club_id))
    if safe_player_id not in names:
        raise ValueError("Player must belong to this club.")

    existing = _first(
        supabase.table("ladder_player_flags")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("player_id", safe_player_id)
        .limit(1)
        .execute()
    )
    mutable = {
        "vacation_until": _vacation_until_iso(vacation_until),
        "reinstate_required": bool(reinstate_required),
        "reinstate_notes": _clean(reinstate_notes, limit=1000) or None,
    }
    if existing is not None:
        saved = _first(
            supabase.table("ladder_player_flags")
            .update(mutable)
            .eq("club_id", str(club_id))
            .eq("player_id", safe_player_id)
            .execute()
        ) or {**existing, **mutable}
    else:
        payload = {"club_id": str(club_id), "player_id": safe_player_id, **mutable}
        saved = _first(supabase.table("ladder_player_flags").insert(payload).execute()) or payload
    warning = _write_ladder_audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="flags_save",
        entity_type="ladder_player_flags",
        entity_id=f"{club_id}:{safe_player_id}",
        before=existing,
        after=saved,
        source=source,
    )
    return {
        "ok": True,
        "mode": "challenge_ladder_player_overrides",
        "player_flags": _admin_player_flag_row(saved, names),
        "warnings": [warning] if warning else [],
    }


def update_admin_challenge_ladder_challenge(
    supabase: Any,
    *,
    club_id: str,
    challenge_id: int,
    status: str,
    admin_note: str | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_challenge_ladder_admin",
) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    if _clean(confirmation_text, limit=80).upper() != CONFIRM:
        raise ValueError(f"Type {CONFIRM} to update the challenge ladder item.")
    safe_id = _safe_int(challenge_id)
    if safe_id is None:
        raise ValueError("challenge_id is required")
    before = _challenge(supabase, club_id=str(club_id), challenge_id=int(safe_id))
    clean_status = _clean(status, limit=40).upper()
    allowed = {"PENDING_ACCEPTANCE", "ACCEPTED_SCHEDULING", "IN_PROGRESS", "AWAITING_VERIFICATION", "OVERDUE_PLAY", "CANCELLED", "CANCELED", "FORFEITED", "COMPLETED", "EXPIRED_ACCEPTANCE"}
    if clean_status not in allowed:
        raise ValueError("unsupported challenge status")
    patch: dict[str, Any] = {"status": clean_status, "updated_at": _now_iso()}
    if clean_status in FINAL_STATUSES and not before.get("completed_at"):
        patch["completed_at"] = _now_iso()
    if admin_note is not None:
        patch["admin_note"] = _clean(admin_note, limit=1000) or None
    updated = _first(supabase.table("ladder_challenges").update(patch).eq("club_id", str(club_id)).eq("id", int(safe_id)).execute()) or {**before, **patch}
    names = _player_names(supabase, club_id=str(club_id))
    warning = _write_ladder_audit(supabase, club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, action_type="update_challenge_ladder_admin", entity_id=str(safe_id), before={"status": before.get("status")}, after={"status": updated.get("status")}, source=source)
    return {"ok": True, "mode": "challenge_ladder_admin_update", "challenge": _challenge_row(updated, names), "warnings": [warning] if warning else []}


def _validate_games(games: list[list[int]] | list[tuple[int, int]], *, label: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for idx, item in enumerate(games or [], start=1):
        if len(item) != 2:
            raise ValueError(f"{label} game {idx} must contain two scores")
        a, b = int(item[0]), int(item[1])
        if a == 0 and b == 0:
            continue
        if a == b:
            raise ValueError(f"{label} game {idx} cannot be tied")
        if a < 0 or b < 0 or (a + b) <= 0:
            raise ValueError(f"{label} game {idx} is invalid")
        out.append((a, b))
    if len(out) < 2:
        raise ValueError(f"{label} requires at least two entered games")
    return out


def _side_winner(games: list[tuple[int, int]]) -> str:
    chal_games = sum(1 for a, b in games if a > b)
    def_games = sum(1 for a, b in games if b > a)
    if chal_games > def_games:
        return "challenger"
    if def_games > chal_games:
        return "defender"
    point_diff = sum(a - b for a, b in games)
    if point_diff > 0:
        return "challenger"
    if point_diff < 0:
        return "defender"
    return "defender"


def compute_ladder_result_winner(match_a_games: list[tuple[int, int]], match_b_games: list[tuple[int, int]]) -> dict[str, Any]:
    a_winner = _side_winner(match_a_games)
    b_winner = _side_winner(match_b_games)
    chal_match_wins = int(a_winner == "challenger") + int(b_winner == "challenger")
    def_match_wins = int(a_winner == "defender") + int(b_winner == "defender")
    games_chal = sum(1 for a, b in [*match_a_games, *match_b_games] if a > b)
    games_def = sum(1 for a, b in [*match_a_games, *match_b_games] if b > a)
    points_diff = sum(a - b for a, b in [*match_a_games, *match_b_games])
    if chal_match_wins > def_match_wins:
        side = "challenger"
    elif def_match_wins > chal_match_wins:
        side = "defender"
    elif games_chal > games_def:
        side = "challenger"
    elif games_def > games_chal:
        side = "defender"
    elif points_diff > 0:
        side = "challenger"
    elif points_diff < 0:
        side = "defender"
    else:
        side = "defender"
    return {"computed_winner_side": side, "matchA_winner_side": a_winner, "matchB_winner_side": b_winner, "games_won_chal": games_chal, "games_won_def": games_def, "points_diff": points_diff, "match_wins_chal": chal_match_wins, "match_wins_def": def_match_wins}


def _sum_scores(games: list[tuple[int, int]]) -> tuple[int, int]:
    return sum(a for a, _ in games), sum(b for _, b in games)


def preview_admin_challenge_ladder_result(
    *,
    challenger_id: int,
    defender_id: int,
    partner_a_challenger_id: int,
    partner_a_defender_id: int,
    partner_b_challenger_id: int,
    partner_b_defender_id: int,
    match_a_games: list[list[int]],
    match_b_games: list[list[int]],
    winner_override: str = "computed",
) -> dict[str, Any]:
    locked = {int(challenger_id), int(defender_id)}
    partners = [int(partner_a_challenger_id), int(partner_a_defender_id), int(partner_b_challenger_id), int(partner_b_defender_id)]
    if any(pid in locked for pid in partners):
        raise ValueError("Partners cannot be challenger or defender")
    if int(partner_a_challenger_id) == int(partner_a_defender_id) or int(partner_b_challenger_id) == int(partner_b_defender_id):
        raise ValueError("Match partners must be different people")
    if (
        int(partner_a_challenger_id) != int(partner_b_defender_id)
        or int(partner_a_defender_id) != int(partner_b_challenger_id)
    ):
        raise ValueError(
            "The same two swing partners must switch ranked-player sides between Match A and Match B"
        )
    a_games = _validate_games(match_a_games, label="Match A")
    b_games = _validate_games(match_b_games, label="Match B")
    winner = compute_ladder_result_winner(a_games, b_games)
    override = str(winner_override or "computed").strip().lower()
    if override in {"challenger", "defender"}:
        final_side = override
    else:
        final_side = str(winner["computed_winner_side"])
    final_winner_id = int(challenger_id) if final_side == "challenger" else int(defender_id)
    a_s1, a_s2 = _sum_scores(a_games)
    b_s1, b_s2 = _sum_scores(b_games)
    return {"ok": True, "winner_summary": winner, "final_winner_side": final_side, "final_winner_id": final_winner_id, "scores": {"match_a": {"score_t1": a_s1, "score_t2": a_s2, "games": a_games}, "match_b": {"score_t1": b_s1, "score_t2": b_s2, "games": b_games}}}


def _prepare_admin_challenge_ladder_result(
    supabase: Any,
    *,
    club_id: str,
    challenge_id: int,
    partner_a_challenger_id: int,
    partner_a_defender_id: int,
    partner_b_challenger_id: int,
    partner_b_defender_id: int,
    match_a_games: list[list[int]],
    match_b_games: list[list[int]],
    winner_override: str,
) -> tuple[dict[str, Any], dict[int, str], dict[str, int], dict[str, Any]]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    challenge = _challenge(supabase, club_id=str(club_id), challenge_id=int(challenge_id))
    if str(challenge.get("status") or "") not in {"ACCEPTED_SCHEDULING", "ACCEPTED", "IN_PROGRESS", "AWAITING_VERIFICATION", "OVERDUE_PLAY"}:
        raise ValueError("Challenge must be accepted/in progress before previewing a result.")
    if challenge.get("completed_at") or challenge.get("winner_id"):
        raise ValueError("Challenge already has a recorded result.")

    partners = {
        "a_chal": int(partner_a_challenger_id),
        "a_def": int(partner_a_defender_id),
        "b_chal": int(partner_b_challenger_id),
        "b_def": int(partner_b_defender_id),
    }
    names = _player_names(supabase, club_id=str(club_id))
    missing_partner_ids = sorted({player_id for player_id in partners.values() if player_id not in names})
    if missing_partner_ids:
        raise ValueError("All partners must be players in this club.")

    preview = preview_admin_challenge_ladder_result(
        challenger_id=int(challenge["challenger_id"]),
        defender_id=int(challenge["defender_id"]),
        partner_a_challenger_id=partners["a_chal"],
        partner_a_defender_id=partners["a_def"],
        partner_b_challenger_id=partners["b_chal"],
        partner_b_defender_id=partners["b_def"],
        match_a_games=match_a_games,
        match_b_games=match_b_games,
        winner_override=winner_override,
    )
    return challenge, names, partners, preview


def preview_admin_challenge_ladder_result_for_challenge(
    supabase: Any,
    *,
    club_id: str,
    challenge_id: int,
    partner_a_challenger_id: int,
    partner_a_defender_id: int,
    partner_b_challenger_id: int,
    partner_b_defender_id: int,
    match_a_games: list[list[int]],
    match_b_games: list[list[int]],
    match_date: str,
    winner_override: str,
    publish_official_matches: bool,
) -> dict[str, Any]:
    """Validate and preview a played result without writing matches, ranks, or audit rows."""

    challenge, names, partners, preview = _prepare_admin_challenge_ladder_result(
        supabase,
        club_id=str(club_id),
        challenge_id=int(challenge_id),
        partner_a_challenger_id=partner_a_challenger_id,
        partner_a_defender_id=partner_a_defender_id,
        partner_b_challenger_id=partner_b_challenger_id,
        partner_b_defender_id=partner_b_defender_id,
        match_a_games=match_a_games,
        match_b_games=match_b_games,
        winner_override=winner_override,
    )
    final_winner_id = int(preview["final_winner_id"])
    payload = {
        "ok": True,
        "mode": "challenge_ladder_result_preview",
        "challenge": _challenge_row(challenge, names),
        "preview": preview,
        "partner_names": {key: names[player_id] for key, player_id in partners.items()},
        "match_date": _clean(match_date, limit=80) or _now_iso(),
        "would_publish_official_matches": bool(publish_official_matches),
        "rank_result": {
            "would_swap": final_winner_id == int(challenge["challenger_id"]),
            "reason": "challenger win" if final_winner_id == int(challenge["challenger_id"]) else "defender held",
        },
    }
    payload["preview_fingerprint"] = stable_request_fingerprint(
        {
            "club_id": str(club_id),
            "challenge_version": payload["challenge"].get("version"),
            "challenge_id": int(challenge_id),
            "preview": preview,
            "partner_names": payload["partner_names"],
            "match_date": payload["match_date"],
            "would_publish_official_matches": payload["would_publish_official_matches"],
            "rank_result": payload["rank_result"],
        }
    )
    payload["authority"] = "python_fastapi"
    return payload


def prepare_admin_challenge_ladder_result_atomic_plan(
    supabase: Any,
    *,
    club_id: str,
    challenge_id: int,
    partner_a_challenger_id: int,
    partner_a_defender_id: int,
    partner_b_challenger_id: int,
    partner_b_defender_id: int,
    match_a_games: list[list[int]],
    match_b_games: list[list[int]],
    match_date: str,
    winner_override: str,
    publish_official_matches: bool,
    expected_preview_fingerprint: str,
    operation_key: str,
) -> dict[str, Any]:
    """Build the immutable Python write plan persisted with the durable intent."""

    preview_response = preview_admin_challenge_ladder_result_for_challenge(
        supabase,
        club_id=str(club_id),
        challenge_id=int(challenge_id),
        partner_a_challenger_id=partner_a_challenger_id,
        partner_a_defender_id=partner_a_defender_id,
        partner_b_challenger_id=partner_b_challenger_id,
        partner_b_defender_id=partner_b_defender_id,
        match_a_games=match_a_games,
        match_b_games=match_b_games,
        match_date=match_date,
        winner_override=winner_override,
        publish_official_matches=publish_official_matches,
    )
    if (
        not expected_preview_fingerprint
        or str(expected_preview_fingerprint)
        != str(preview_response.get("preview_fingerprint") or "")
    ):
        raise ValueError(
            "Ladder result preview is stale. Review the Python result again before official publish."
        )
    clean_operation_key = _clean(operation_key, limit=80)
    if not clean_operation_key:
        raise ValueError("A durable operation key is required to build the result plan.")

    challenge = _challenge(
        supabase,
        club_id=str(club_id),
        challenge_id=int(challenge_id),
    )
    partners = {
        "a_chal": int(partner_a_challenger_id),
        "a_def": int(partner_a_defender_id),
        "b_chal": int(partner_b_challenger_id),
        "b_def": int(partner_b_defender_id),
    }
    preview = dict(preview_response.get("preview") or {})
    completed_at = _now_iso()
    final_winner_id = int(preview["final_winner_id"])
    resolution_notes = (
        f"Next result publish. Winner: {final_winner_id}. "
        f"Summary: {preview['winner_summary']}"
    )
    payloads: list[dict[str, Any]] = []
    contexts: list[str] = []
    write_plan: dict[str, Any] = {
        "match_rows": [],
        "player_updates": [],
        "league_rating_updates": [],
        "league_metadata_expectations": [],
    }
    side_effect_context: dict[str, Any] = {
        "affected_player_ids": [],
        "successful_match_dates": [],
        "has_badge_eligible_match": False,
        "match_payloads": [],
    }
    if publish_official_matches:
        payloads = _official_payloads(
            challenge=challenge,
            preview=preview,
            partners=partners,
            match_date=str(preview_response.get("match_date") or match_date or completed_at),
            publish_context_prefix=clean_operation_key,
        )
        contexts = [str(payload["context_id"]) for payload in payloads]
        if len(contexts) != 2 or len(set(contexts)) != 2:
            raise RuntimeError("Challenge result plan requires two distinct deterministic contexts.")
        df_players_all, df_leagues, df_meta, name_to_id = _load_match_context(
            supabase,
            str(club_id),
        )
        process_result = process_matches(
            payloads,
            supabase=supabase,
            club_id=str(club_id),
            name_to_id=name_to_id,
            df_players_all=df_players_all,
            df_leagues=df_leagues,
            df_meta=df_meta,
            build_write_plan_only=True,
        )
        if (
            int(process_result.get("inserted") or 0) != 2
            or int(process_result.get("skipped_incomplete") or 0) != 0
            or int(process_result.get("skipped_empty") or 0) != 0
            or int(process_result.get("skipped_unrated") or 0) != 0
        ):
            raise RuntimeError(
                "Challenge result write-plan generation did not produce exactly two rated matches."
            )
        candidate_plan = process_result.get("write_plan")
        if not isinstance(candidate_plan, dict):
            raise RuntimeError("Challenge result write-plan generation returned no plan.")
        write_plan = {
            key: list(candidate_plan.get(key) or [])
            for key in write_plan
        }
        if (
            len(write_plan["match_rows"]) != 2
            or len(write_plan["player_updates"]) != 4
        ):
            raise RuntimeError(
                "Challenge result atomic plan requires two matches and four player updates."
            )
        candidate_side_effects = process_result.get("side_effect_context")
        if isinstance(candidate_side_effects, dict):
            side_effect_context = dict(candidate_side_effects)

    challenge_expected = {
        key: challenge.get(key)
        for key in (
            "id",
            "club_id",
            "challenger_id",
            "defender_id",
            "tier_id",
            "status",
            "updated_at",
            "winner_id",
            "completed_at",
            "public_result_json",
        )
    }
    challenge_tier = str(challenge.get("tier_id") or "")
    tier_roster = [
        {
            key: row.get(key)
            for key in (
                "id",
                "player_id",
                "tier_id",
                "rank",
                "is_active",
                "updated_at",
            )
        }
        for row in _roster_rows(supabase, club_id=str(club_id))
        if str(row.get("tier_id") or "") == challenge_tier
    ]
    tier_roster.sort(key=lambda row: int(_safe_int(row.get("id")) or 0))
    atomic_core = {
        "version": 1,
        "challenge_expected": challenge_expected,
        "tier_roster_expected": tier_roster,
        "winner_id": final_winner_id,
        "completed_at": completed_at,
        "resolution_notes": resolution_notes,
        "publish_official_matches": bool(publish_official_matches),
        "match_context_ids": contexts,
        "match_payloads": payloads,
        "write_plan": write_plan,
        "side_effect_context": side_effect_context,
        "preview": preview,
        "preview_fingerprint": str(expected_preview_fingerprint),
    }
    atomic_core["plan_fingerprint"] = stable_request_fingerprint(atomic_core)
    return {
        "preview_response": preview_response,
        "atomic_core": atomic_core,
    }


def _load_match_context(supabase: Any, club_id: str) -> tuple[Any, Any, Any, Any]:
    df_players_all, _df_players_active, df_leagues, _df_matches, df_meta, _df_badges, _df_player_badges, name_to_id, _id_to_name, _schema_degraded, _schema_degraded_reason = load_data(supabase, str(club_id), match_limit=5000)
    return df_players_all, df_leagues, df_meta, name_to_id


def _official_payloads(
    *,
    challenge: dict[str, Any],
    preview: dict[str, Any],
    partners: dict[str, int],
    match_date: str,
    publish_context_prefix: str | None = None,
) -> list[dict[str, Any]]:
    chal = int(challenge["challenger_id"])
    defender = int(challenge["defender_id"])
    score_a = preview["scores"]["match_a"]
    score_b = preview["scores"]["match_b"]
    base = {"date": match_date, "league": "OVERALL", "match_type": "ChallengeLadder", "is_popup": False, "context_type": "challenge_ladder"}
    context_prefix = _clean(publish_context_prefix, limit=80)
    return [
        {**base, "context_id": deterministic_match_context_id(operation_key=context_prefix, slot="a") if context_prefix else int(challenge["id"]), "t1_p1": chal, "t1_p2": int(partners["a_chal"]), "t2_p1": defender, "t2_p2": int(partners["a_def"]), "s1": int(score_a["score_t1"]), "s2": int(score_a["score_t2"])},
        {**base, "context_id": deterministic_match_context_id(operation_key=context_prefix, slot="b") if context_prefix else int(challenge["id"]), "t1_p1": chal, "t1_p2": int(partners["b_chal"]), "t2_p1": defender, "t2_p2": int(partners["b_def"]), "s1": int(score_b["score_t1"]), "s2": int(score_b["score_t2"])},
    ]


def _published_match_relation(
    supabase: Any,
    *,
    club_id: str,
    challenge: dict[str, Any],
    payloads: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Resolve the two exact operation-scoped match IDs after official publish."""

    if len(payloads) != 2:
        raise RuntimeError("Challenge result publish requires exactly two official match payloads.")
    challenger_id = _safe_int(challenge.get("challenger_id"))
    defender_id = _safe_int(challenge.get("defender_id"))
    partner_a_challenger = _safe_int(payloads[0].get("t1_p2"))
    partner_a_defender = _safe_int(payloads[0].get("t2_p2"))
    partner_b_challenger = _safe_int(payloads[1].get("t1_p2"))
    partner_b_defender = _safe_int(payloads[1].get("t2_p2"))
    if (
        challenger_id is None
        or defender_id is None
        or partner_a_challenger is None
        or partner_a_defender is None
        or partner_b_challenger is None
        or partner_b_defender is None
        or partner_a_challenger in {challenger_id, defender_id}
        or partner_a_defender in {challenger_id, defender_id}
        or partner_a_challenger == partner_a_defender
        or partner_a_challenger != partner_b_defender
        or partner_a_defender != partner_b_challenger
    ):
        raise RuntimeError(
            "Official challenge match payloads do not satisfy the ranked-player and swing-partner format."
        )
    relation_ids: dict[str, int] = {}
    contexts: list[str] = []
    for slot, payload in zip(("a", "b"), payloads, strict=True):
        context_id = str(payload.get("context_id") or "").strip()
        if not context_id:
            raise RuntimeError(f"Official challenge match {slot.upper()} is missing its durable context.")
        rows = _safe_rows(
            supabase.table("matches")
            .select(OFFICIAL_CHALLENGE_MATCH_SELECT)
            .eq("club_id", str(club_id))
            .eq("context_type", "challenge_ladder")
            .eq("context_id", context_id)
            .limit(2)
            .execute()
        )
        rows = [row for row in rows if not row.get("deleted_at")]
        if len(rows) != 1:
            raise RuntimeError(
                f"Official challenge match {slot.upper()} could not be resolved to exactly one active row."
            )
        row = rows[0]
        match_id = _safe_int(row.get("id"))
        if match_id is None or match_id <= 0:
            raise RuntimeError(f"Official challenge match {slot.upper()} has no public match ID.")
        expected = {
            "t1_p1": _safe_int(challenge.get("challenger_id")),
            "t1_p2": _safe_int(payload.get("t1_p2")),
            "t2_p1": _safe_int(challenge.get("defender_id")),
            "t2_p2": _safe_int(payload.get("t2_p2")),
            "score_t1": _safe_int(payload.get("s1")),
            "score_t2": _safe_int(payload.get("s2")),
        }
        observed = {key: _safe_int(row.get(key)) for key in expected}
        if observed != expected:
            raise RuntimeError(
                f"Official challenge match {slot.upper()} does not match the reviewed participants and score."
            )
        relation_ids[slot] = int(match_id)
        contexts.append(context_id)
    if relation_ids["a"] == relation_ids["b"]:
        raise RuntimeError("Challenge result publish requires two different official matches.")
    return {"version": 1, "match_ids": relation_ids}, contexts


def _rpc_object(response: Any) -> dict[str, Any]:
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return dict(data[0])
    return {}


def _rank_result_payload(rank_change: dict[str, Any]) -> dict[str, Any]:
    challenger = (
        rank_change.get("challenger")
        if isinstance(rank_change.get("challenger"), dict)
        else {}
    )
    defender = (
        rank_change.get("defender")
        if isinstance(rank_change.get("defender"), dict)
        else {}
    )
    return {
        "swapped": bool(rank_change.get("swapped")),
        "reason": "challenger win" if rank_change.get("swapped") else "defender held",
        "challenger_old_rank": _safe_int(challenger.get("before")),
        "challenger_new_rank": _safe_int(challenger.get("after")),
        "defender_old_rank": _safe_int(defender.get("before")),
        "defender_new_rank": _safe_int(defender.get("after")),
    }


def _finalize_played_result(
    supabase: Any,
    *,
    club_id: str,
    challenge_id: int,
    operation_key: str,
    winner_id: int,
    completed_at: str,
    resolution_notes: str,
    public_result_json: dict[str, Any] | None,
    match_contexts: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    if len(match_contexts) not in {0, 2}:
        raise RuntimeError("Challenge result finalization requires zero or two official match contexts.")
    params = {
        "p_club_id": str(club_id),
        "p_challenge_id": int(challenge_id),
        "p_operation_key": str(operation_key),
        "p_winner_id": int(winner_id),
        "p_completed_at": str(completed_at),
        "p_resolution_notes": str(resolution_notes),
        "p_public_result_json": public_result_json,
        "p_match_context_a": match_contexts[0] if match_contexts else None,
        "p_match_context_b": match_contexts[1] if match_contexts else None,
    }
    try:
        payload = _rpc_object(
            supabase.rpc(CHALLENGE_LADDER_RESULT_FINALIZE_RPC, params).execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Challenge result finalization failed closed. Apply the canonical Challenge Ladder public-result "
            "migration before enabling result publish."
        ) from exc
    updated = payload.get("challenge")
    rank_change = payload.get("rank_result")
    final_public_result = payload.get("public_result_json")
    if not isinstance(updated, dict) or not isinstance(rank_change, dict):
        raise RuntimeError("Challenge result finalization returned an incomplete authoritative response.")
    rank_result = _rank_result_payload(rank_change)
    return (
        dict(updated),
        rank_result,
        dict(final_public_result) if isinstance(final_public_result, dict) else None,
    )


def _apply_played_result_atomic(
    supabase: Any,
    *,
    club_id: str,
    challenge_id: int,
    operation_key: str,
    winner_id: int,
    completed_at: str,
    resolution_notes: str,
    atomic_core: dict[str, Any],
    plan_fingerprint: str,
    write_plan: dict[str, Any],
    match_contexts: list[str],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if len(match_contexts) != 2:
        raise RuntimeError("Atomic Challenge Ladder publish requires two exact match contexts.")
    if not isinstance(atomic_core, dict) or not str(plan_fingerprint or "").strip():
        raise RuntimeError("Atomic Challenge Ladder publish requires its persisted plan binding.")
    params = {
        "p_club_id": str(club_id),
        "p_challenge_id": int(challenge_id),
        "p_operation_key": str(operation_key),
        "p_atomic_core": atomic_core,
        "p_plan_fingerprint": str(plan_fingerprint),
        "p_winner_id": int(winner_id),
        "p_completed_at": str(completed_at),
        "p_resolution_notes": str(resolution_notes),
        "p_match_rows": list(write_plan.get("match_rows") or []),
        "p_player_updates": list(write_plan.get("player_updates") or []),
        "p_league_rating_updates": list(
            write_plan.get("league_rating_updates") or []
        ),
        "p_league_metadata_expectations": list(
            write_plan.get("league_metadata_expectations") or []
        ),
        "p_match_context_a": match_contexts[0],
        "p_match_context_b": match_contexts[1],
    }
    try:
        payload = _rpc_object(
            supabase.rpc(CHALLENGE_LADDER_RESULT_ATOMIC_RPC, params).execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Challenge result atomic match/rating finalization failed closed. "
            "No partial plan may be retried blindly."
        ) from exc
    updated = payload.get("challenge")
    rank_change = payload.get("rank_result")
    public_result = payload.get("public_result_json")
    official_matches = payload.get("official_matches")
    if (
        not isinstance(updated, dict)
        or not isinstance(rank_change, dict)
        or not isinstance(public_result, dict)
        or not isinstance(official_matches, dict)
        or int(official_matches.get("inserted") or 0) != 2
    ):
        raise RuntimeError(
            "Challenge result atomic finalizer returned incomplete completion evidence."
        )
    return (
        dict(updated),
        _rank_result_payload(rank_change),
        dict(public_result),
        dict(official_matches),
    )


def _run_atomic_match_side_effects(
    supabase: Any,
    *,
    club_id: str,
    operation_key: str,
    write_plan: dict[str, Any],
    side_effect_context: dict[str, Any],
) -> dict[str, Any]:
    clean_operation_key = str(operation_key or "").strip()
    if not clean_operation_key:
        raise RuntimeError(
            "Challenge result post-processing requires its durable operation key."
        )
    db_matches = [
        dict(row)
        for row in (write_plan.get("match_rows") or [])
        if isinstance(row, dict)
    ]
    affected_players = {
        int(value) for value in (side_effect_context.get("affected_player_ids") or [])
    }
    successful_dates = [
        str(value)
        for value in (side_effect_context.get("successful_match_dates") or [])
    ]
    match_payloads = [
        dict(value)
        for value in (side_effect_context.get("match_payloads") or [])
        if isinstance(value, dict)
    ]
    badge_summary = run_badge_side_effects(
        supabase=supabase,
        club_id=str(club_id),
        has_badge_eligible_match=bool(
            side_effect_context.get("has_badge_eligible_match")
        ),
        affected_players=affected_players,
        db_matches=db_matches,
        match_payloads=match_payloads,
        dedupe_match_id=clean_operation_key,
    )
    player_update_queue = queue_player_updates(
        supabase=supabase,
        club_id=str(club_id),
        db_matches=db_matches,
        affected_players=affected_players,
        successful_match_dates=successful_dates,
    )
    if str(badge_summary.get("mode") or "") == "inline_error":
        raise RuntimeError(
            "Challenge result badge processing did not complete; retry durable recovery."
        )
    if (
        str(player_update_queue.get("mode") or "") == "error"
        or int(player_update_queue.get("failed") or 0) > 0
    ):
        raise RuntimeError(
            "Challenge result player-update queueing did not complete; retry durable recovery."
        )
    return {
        "badge_summary": badge_summary,
        "player_update_queue": player_update_queue,
    }


def _normalized_recovery_rank_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("Challenge Ladder core receipt is missing its rank result.")
    if isinstance(value.get("challenger"), dict) and isinstance(
        value.get("defender"), dict
    ):
        normalized = _rank_result_payload(value)
    else:
        normalized = {
            "swapped": bool(value.get("swapped")),
            "reason": str(value.get("reason") or ""),
            "challenger_old_rank": _safe_int(value.get("challenger_old_rank")),
            "challenger_new_rank": _safe_int(value.get("challenger_new_rank")),
            "defender_old_rank": _safe_int(value.get("defender_old_rank")),
            "defender_new_rank": _safe_int(value.get("defender_new_rank")),
        }
    if any(
        normalized.get(key) is None
        for key in (
            "challenger_old_rank",
            "challenger_new_rank",
            "defender_old_rank",
            "defender_new_rank",
        )
    ):
        raise RuntimeError("Challenge Ladder core receipt has incomplete rank evidence.")
    expected_reason = (
        "challenger win" if bool(normalized.get("swapped")) else "defender held"
    )
    if normalized.get("reason") not in {"", expected_reason}:
        raise RuntimeError("Challenge Ladder core receipt has inconsistent rank evidence.")
    normalized["reason"] = expected_reason
    return normalized


def _same_completed_at(left: Any, right: Any) -> bool:
    try:
        left_at = datetime.fromisoformat(str(left).replace("Z", "+00:00"))
        right_at = datetime.fromisoformat(str(right).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return str(left or "") == str(right or "")
    if left_at.tzinfo is None or right_at.tzinfo is None:
        return str(left or "") == str(right or "")
    return left_at.astimezone(timezone.utc) == right_at.astimezone(timezone.utc)


def _verify_recovered_challenge(
    supabase: Any,
    *,
    club_id: str,
    operation: dict[str, Any],
    receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_challenge = receipt.get("challenge")
    if not isinstance(receipt_challenge, dict):
        raise RuntimeError("Challenge Ladder core receipt is missing its challenge snapshot.")
    challenge_id = _safe_int(operation.get("entity_id"))
    if challenge_id is None or challenge_id <= 0:
        raise RuntimeError("Challenge Ladder operation has an invalid challenge identity.")
    current = _challenge(
        supabase,
        club_id=str(club_id),
        challenge_id=int(challenge_id),
    )
    for key in ("id", "challenger_id", "defender_id", "winner_id"):
        if _safe_int(current.get(key)) != _safe_int(receipt_challenge.get(key)):
            raise RuntimeError(
                f"Challenge Ladder recovery found a changed {key.replace('_', ' ')}."
            )
    if _safe_int(current.get("forfeit_by")) != _safe_int(
        receipt_challenge.get("forfeit_by")
    ):
        raise RuntimeError("Challenge Ladder recovery found a changed forfeit player.")
    for key in ("club_id", "tier_id", "status"):
        if str(current.get(key) or "") != str(receipt_challenge.get(key) or ""):
            raise RuntimeError(
                f"Challenge Ladder recovery found a changed {key.replace('_', ' ')}."
            )
    for key in ("forfeit_reason", "resolution_notes"):
        if current.get(key) != receipt_challenge.get(key):
            raise RuntimeError(
                f"Challenge Ladder recovery found changed {key.replace('_', ' ')}."
            )
    if str(current.get("status") or "") not in {"COMPLETED", "FORFEITED"}:
        raise RuntimeError("Challenge Ladder core receipt is not reflected by a final challenge.")
    if not _same_completed_at(
        current.get("completed_at"), receipt_challenge.get("completed_at")
    ):
        raise RuntimeError("Challenge Ladder recovery found a changed completion time.")
    if current.get("public_result_json") != receipt_challenge.get(
        "public_result_json"
    ):
        raise RuntimeError(
            "Challenge Ladder recovery found changed public result evidence."
        )
    return current, receipt_challenge


def _verify_recovered_ranks(
    supabase: Any,
    *,
    club_id: str,
    challenge: dict[str, Any],
    rank_result: dict[str, Any],
) -> None:
    challenger_id = _safe_int(challenge.get("challenger_id"))
    defender_id = _safe_int(challenge.get("defender_id"))
    if challenger_id is None or defender_id is None:
        raise RuntimeError("Challenge Ladder recovery has no ranked participants.")
    roster = _active_roster_by_player(supabase, club_id=str(club_id))
    challenger = roster.get(int(challenger_id))
    defender = roster.get(int(defender_id))
    if challenger is None or defender is None:
        raise RuntimeError("Challenge Ladder recovery could not resolve both active ranks.")
    if (
        str(challenger.get("tier_id") or "") != str(challenge.get("tier_id") or "")
        or str(defender.get("tier_id") or "")
        != str(challenge.get("tier_id") or "")
        or _safe_int(challenger.get("rank"))
        != _safe_int(rank_result.get("challenger_new_rank"))
        or _safe_int(defender.get("rank"))
        != _safe_int(rank_result.get("defender_new_rank"))
    ):
        raise RuntimeError(
            "Challenge Ladder recovery rank readback does not match the committed receipt."
        )
    winner_id = _safe_int(challenge.get("winner_id"))
    expected_swap = winner_id == challenger_id
    if bool(rank_result.get("swapped")) != expected_swap:
        raise RuntimeError(
            "Challenge Ladder recovery winner and rank movement do not agree."
        )


def _verified_recovery_plan(
    operation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_json = operation.get("request_json")
    if not isinstance(request_json, dict):
        raise RuntimeError("Challenge Ladder operation has no persisted request.")
    client_request = request_json.get("client_request")
    atomic_core = request_json.get("atomic_core")
    if not isinstance(client_request, dict) or not isinstance(atomic_core, dict):
        raise RuntimeError(
            "Challenge Ladder result operation has no persisted atomic plan."
        )
    if stable_request_fingerprint(client_request) != str(
        operation.get("request_fingerprint") or ""
    ):
        raise RuntimeError(
            "Challenge Ladder persisted request does not match its durable fingerprint."
        )
    plan_fingerprint = str(atomic_core.get("plan_fingerprint") or "")
    unsigned_core = {
        key: value for key, value in atomic_core.items() if key != "plan_fingerprint"
    }
    if (
        not plan_fingerprint
        or stable_request_fingerprint(unsigned_core) != plan_fingerprint
    ):
        raise RuntimeError(
            "Challenge Ladder persisted atomic plan fingerprint is invalid."
        )
    return client_request, atomic_core


def recover_admin_challenge_ladder_operation_result(
    supabase: Any,
    *,
    operation: dict[str, Any],
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any] | None:
    """Finish an operation whose database core committed but API response was lost."""

    del actor_email, actor_role, source  # The durable-operation completion writes the recovery audit.
    if str(operation.get("surface") or "") != "challenge_ladder":
        return None
    operation_type = str(operation.get("operation_type") or "")
    if operation_type not in {"publish_result", "record_forfeit"}:
        return None
    receipt = operation.get("result_json")
    if not isinstance(receipt, dict) or receipt.get("core_committed") is not True:
        return None
    club_id = str(operation.get("club_id") or "")
    if not club_id:
        raise RuntimeError("Challenge Ladder recovery receipt has no club identity.")
    expected_mode = (
        "challenge_ladder_result_core"
        if operation_type == "publish_result"
        else "challenge_ladder_forfeit_core"
    )
    if str(receipt.get("mode") or "") != expected_mode:
        raise RuntimeError("Challenge Ladder core receipt has the wrong operation mode.")
    current, _receipt_challenge = _verify_recovered_challenge(
        supabase,
        club_id=club_id,
        operation=operation,
        receipt=receipt,
    )
    rank_result = _normalized_recovery_rank_result(receipt.get("rank_result"))
    _verify_recovered_ranks(
        supabase,
        club_id=club_id,
        challenge=current,
        rank_result=rank_result,
    )
    names = _player_names(supabase, club_id=club_id)

    if operation_type == "record_forfeit":
        if (
            str(current.get("status") or "") != "FORFEITED"
            or current.get("public_result_json") is not None
        ):
            raise RuntimeError(
                "Challenge Ladder forfeit receipt does not match the final challenge."
            )
        post_processors = receipt.get("post_processors")
        if not isinstance(post_processors, dict) or str(
            post_processors.get("status") or ""
        ) != "complete":
            raise RuntimeError(
                "Challenge Ladder forfeit receipt has incomplete core processing."
            )
        return {
            "ok": True,
            "mode": "challenge_ladder_forfeit",
            "challenge": _challenge_row(current, names),
            "rank_result": rank_result,
            "warnings": [],
        }

    _client_request, atomic_core = _verified_recovery_plan(operation)
    if str(receipt.get("plan_fingerprint") or "") != str(
        atomic_core.get("plan_fingerprint") or ""
    ):
        raise RuntimeError(
            "Challenge Ladder core receipt does not match its persisted plan."
        )
    challenge_expected = atomic_core.get("challenge_expected")
    preview = atomic_core.get("preview")
    write_plan = atomic_core.get("write_plan")
    side_effect_context = atomic_core.get("side_effect_context")
    contexts = [
        str(value) for value in (atomic_core.get("match_context_ids") or [])
    ]
    payloads = [
        dict(row)
        for row in (atomic_core.get("match_payloads") or [])
        if isinstance(row, dict)
    ]
    if (
        not isinstance(challenge_expected, dict)
        or not isinstance(preview, dict)
        or not isinstance(write_plan, dict)
        or not isinstance(side_effect_context, dict)
        or receipt.get("side_effect_context") != side_effect_context
        or _safe_int(challenge_expected.get("id"))
        != _safe_int(operation.get("entity_id"))
        or str(challenge_expected.get("club_id") or "") != club_id
        or _safe_int(atomic_core.get("winner_id"))
        != _safe_int(current.get("winner_id"))
        or not _same_completed_at(
            atomic_core.get("completed_at"), current.get("completed_at")
        )
    ):
        raise RuntimeError(
            "Challenge Ladder committed result does not match its persisted plan."
        )

    publish_official = bool(atomic_core.get("publish_official_matches"))
    official_matches = receipt.get("official_matches")
    post_processors = receipt.get("post_processors")
    if not isinstance(official_matches, dict) or not isinstance(
        post_processors, dict
    ):
        raise RuntimeError(
            "Challenge Ladder result receipt is missing completion evidence."
        )
    public_result_json = receipt.get("public_result_json")
    if public_result_json != current.get("public_result_json"):
        raise RuntimeError(
            "Challenge Ladder public result changed after its core receipt."
        )

    if publish_official:
        if len(contexts) != 2 or len(payloads) != 2:
            raise RuntimeError(
                "Challenge Ladder recovery plan requires two official matches."
            )
        observed_relation, observed_contexts = _published_match_relation(
            supabase,
            club_id=club_id,
            challenge=current,
            payloads=payloads,
        )
        if observed_contexts != contexts:
            raise RuntimeError(
                "Challenge Ladder official match contexts changed after commit."
            )
        if (
            int(official_matches.get("inserted") or 0) != 2
            or official_matches.get("atomic") is not True
            or official_matches.get("skipped") is not False
            or official_matches.get("match_ids")
            != observed_relation.get("match_ids")
            or official_matches.get("match_context_ids") != contexts
            or not isinstance(public_result_json, dict)
            or {
                "version": public_result_json.get("version"),
                "match_ids": public_result_json.get("match_ids"),
            }
            != observed_relation
            or _normalized_recovery_rank_result(
                public_result_json.get("rank_change")
            )
            != rank_result
        ):
            raise RuntimeError(
                "Challenge Ladder official relation does not match its core receipt."
            )
        if str(post_processors.get("status") or "") not in {
            "pending",
            "complete",
        }:
            raise RuntimeError(
                "Challenge Ladder result receipt has an invalid post-processor state."
            )
        side_effects = _run_atomic_match_side_effects(
            supabase,
            club_id=club_id,
            operation_key=str(operation.get("operation_key") or ""),
            write_plan=write_plan,
            side_effect_context=side_effect_context,
        )
        official_result = {**official_matches, **side_effects}
    else:
        if (
            contexts
            or payloads
            or int(official_matches.get("inserted") or 0) != 0
            or official_matches.get("atomic") is not True
            or official_matches.get("skipped") is not True
            or official_matches.get("match_ids") != {}
            or official_matches.get("match_context_ids") != []
            or public_result_json is not None
            or str(post_processors.get("status") or "") != "complete"
        ):
            raise RuntimeError(
                "Challenge Ladder non-official result receipt is inconsistent."
            )
        official_result = {"inserted": 0, "skipped": True}

    return {
        "ok": True,
        "mode": "challenge_ladder_result",
        "challenge": _challenge_row(current, names),
        "preview": preview,
        "official_matches": official_result,
        "match_context_ids": contexts,
        "rank_result": rank_result,
        "public_result_json": public_result_json,
        "correction": {
            "match_log_url": build_match_log_recovery_url(
                context_type="challenge_ladder",
                context_ids=contexts,
                fallback_context_id=str(operation.get("entity_id") or ""),
            ),
            "replay_history_url": "/admin/replay-history",
            "instructions": (
                "Correct official ladder matches in Match Log, then run and verify "
                "Replay History before changing ladder state again."
            ),
        },
        "warnings": [],
    }


def _finalize_forfeit(
    supabase: Any,
    *,
    club_id: str,
    challenge_id: int,
    operation_key: str,
    forfeited_by_id: int,
    completed_at: str,
    forfeit_reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = {
        "p_club_id": str(club_id),
        "p_challenge_id": int(challenge_id),
        "p_operation_key": str(operation_key),
        "p_forfeited_by_id": int(forfeited_by_id),
        "p_completed_at": str(completed_at),
        "p_forfeit_reason": str(forfeit_reason),
    }
    try:
        payload = _rpc_object(
            supabase.rpc(CHALLENGE_LADDER_FORFEIT_FINALIZE_RPC, params).execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Challenge forfeit finalization failed closed. Apply the canonical Challenge Ladder public-result "
            "migration before enabling forfeits."
        ) from exc
    updated = payload.get("challenge")
    rank_change = payload.get("rank_result")
    if not isinstance(updated, dict) or not isinstance(rank_change, dict):
        raise RuntimeError("Challenge forfeit finalization returned an incomplete authoritative response.")
    return dict(updated), _rank_result_payload(rank_change)


def record_admin_challenge_ladder_result(
    supabase: Any,
    *,
    club_id: str,
    challenge_id: int,
    partner_a_challenger_id: int,
    partner_a_defender_id: int,
    partner_b_challenger_id: int,
    partner_b_defender_id: int,
    match_a_games: list[list[int]],
    match_b_games: list[list[int]],
    match_date: str,
    winner_override: str,
    publish_official_matches: bool,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    expected_preview_fingerprint: str | None = None,
    publish_context_prefix: str | None = None,
    atomic_core: dict[str, Any] | None = None,
    source: str = "next_challenge_ladder_result",
) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_RESULT:
        raise ValueError(f"Type {CONFIRM_RESULT} to publish a ladder result.")
    operation_key = _clean(publish_context_prefix, limit=80)
    if not operation_key:
        raise ValueError("A durable operation key is required to publish a ladder result.")
    if not isinstance(atomic_core, dict):
        raise RuntimeError(
            "Challenge result publish requires the operation-bound atomic write plan."
        )
    plan_fingerprint = str(atomic_core.get("plan_fingerprint") or "")
    unsigned_core = {
        key: value for key, value in atomic_core.items() if key != "plan_fingerprint"
    }
    if (
        not plan_fingerprint
        or stable_request_fingerprint(unsigned_core) != plan_fingerprint
    ):
        raise RuntimeError("Challenge result atomic plan fingerprint is invalid.")
    challenge_expected = atomic_core.get("challenge_expected")
    preview = atomic_core.get("preview")
    write_plan = atomic_core.get("write_plan")
    side_effect_context = atomic_core.get("side_effect_context")
    if (
        not isinstance(challenge_expected, dict)
        or not isinstance(preview, dict)
        or not isinstance(write_plan, dict)
        or not isinstance(side_effect_context, dict)
        or _safe_int(challenge_expected.get("id")) != int(challenge_id)
        or str(challenge_expected.get("club_id") or "") != str(club_id)
        or bool(atomic_core.get("publish_official_matches"))
        != bool(publish_official_matches)
    ):
        raise RuntimeError("Challenge result atomic plan does not match this request.")
    challenge = dict(challenge_expected)
    payloads = [
        dict(row)
        for row in (atomic_core.get("match_payloads") or [])
        if isinstance(row, dict)
    ]
    contexts = [
        str(value) for value in (atomic_core.get("match_context_ids") or [])
    ]
    if (
        not expected_preview_fingerprint
        or str(atomic_core.get("preview_fingerprint") or "")
        != str(expected_preview_fingerprint)
    ):
        raise RuntimeError("Challenge result preview fingerprint is not bound to the plan.")
    official_result: dict[str, Any] = {"inserted": 0, "skipped": True}
    public_result_json: dict[str, Any] | None = None
    completed_at = str(atomic_core.get("completed_at") or "")
    resolution_notes = str(atomic_core.get("resolution_notes") or "")
    final_winner_id = int(atomic_core.get("winner_id") or 0)
    if (
        not completed_at
        or not resolution_notes
        or final_winner_id != int(preview.get("final_winner_id") or -1)
    ):
        raise RuntimeError("Challenge result atomic outcome metadata is invalid.")
    if publish_official_matches:
        updated, rank_result, public_result_json, official_result = (
            _apply_played_result_atomic(
                supabase,
                club_id=str(club_id),
                challenge_id=int(challenge_id),
                operation_key=operation_key,
                winner_id=final_winner_id,
                completed_at=completed_at,
                resolution_notes=resolution_notes,
                atomic_core=atomic_core,
                plan_fingerprint=plan_fingerprint,
                write_plan=write_plan,
                match_contexts=contexts,
            )
        )
        official_result.update(
            _run_atomic_match_side_effects(
                supabase,
                club_id=str(club_id),
                operation_key=operation_key,
                write_plan=write_plan,
                side_effect_context=side_effect_context,
            )
        )
    else:
        updated, rank_result, public_result_json = _finalize_played_result(
            supabase,
            club_id=str(club_id),
            challenge_id=int(challenge_id),
            operation_key=operation_key,
            winner_id=final_winner_id,
            completed_at=completed_at,
            resolution_notes=resolution_notes,
            public_result_json=None,
            match_contexts=[],
        )
    names = _player_names(supabase, club_id=str(club_id))
    warning = _write_ladder_audit(supabase, club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, action_type="challenge_result_publish", entity_id=str(challenge_id), before=challenge, after={"challenge": updated, "preview": preview, "official_matches": official_result, "rank_result": rank_result, "public_result_json": public_result_json, "payloads": payloads}, source=source)
    return {
        "ok": True,
        "mode": "challenge_ladder_result",
        "challenge": _challenge_row(updated, names),
        "preview": preview,
        "official_matches": official_result,
        "match_context_ids": contexts,
        "rank_result": rank_result,
        "public_result_json": public_result_json,
        "correction": {
            "match_log_url": build_match_log_recovery_url(
                context_type="challenge_ladder",
                context_ids=contexts,
                fallback_context_id=str(challenge_id),
            ),
            "replay_history_url": "/admin/replay-history",
            "instructions": "Correct official ladder matches in Match Log, then run and verify Replay History before changing ladder state again.",
        },
        "warnings": [warning] if warning else [],
    }


def record_admin_challenge_ladder_forfeit(
    supabase: Any,
    *,
    club_id: str,
    challenge_id: int,
    forfeited_by_id: int,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    admin_note: str | None = None,
    operation_key: str | None = None,
    source: str = "next_challenge_ladder_forfeit",
) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_FORFEIT:
        raise ValueError(f"Type {CONFIRM_FORFEIT} to record a ladder forfeit.")
    durable_operation_key = _clean(operation_key, limit=80)
    if not durable_operation_key:
        raise ValueError("A durable operation key is required to record a ladder forfeit.")
    challenge = _challenge(supabase, club_id=str(club_id), challenge_id=int(challenge_id))
    forfeited = int(forfeited_by_id)
    chal = int(challenge["challenger_id"])
    defender = int(challenge["defender_id"])
    if forfeited not in {chal, defender}:
        raise ValueError("forfeited_by_id must be challenger or defender")
    completed_at = _now_iso()
    forfeit_reason = _clean(admin_note, limit=500) or "Forfeit"
    updated, rank_result = _finalize_forfeit(
        supabase,
        club_id=str(club_id),
        challenge_id=int(challenge_id),
        operation_key=durable_operation_key,
        forfeited_by_id=forfeited,
        completed_at=completed_at,
        forfeit_reason=forfeit_reason,
    )
    names = _player_names(supabase, club_id=str(club_id))
    warning = _write_ladder_audit(supabase, club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, action_type="challenge_forfeit", entity_id=str(challenge_id), before=challenge, after={"challenge": updated, "rank_result": rank_result}, source=source)
    return {"ok": True, "mode": "challenge_ladder_forfeit", "challenge": _challenge_row(updated, names), "rank_result": rank_result, "warnings": [warning] if warning else []}
