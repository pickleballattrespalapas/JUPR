from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from jupr_app.data.load import load_data
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.build_challenge_notice_message import build_challenge_notice_message
from jupr_app.domain.challenge_ladder import TIER_ORDER, ladder_bucket_challenge, ladder_compute_status_map, month_key_utc, normalize_tier_id
from jupr_app.domain.tier_movement import compute_out_of_tier_streak
from jupr_app.services.context import ServiceContext
from jupr_app.services.match_service import submit_match_batch
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
CONFIRM_OVERRIDES = "SAVE LADDER OVERRIDES"


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
    return {
        "enabled": True,
        "status": "ready_for_challenge_ladder_admin",
        "summary": summary,
        "warnings": [],
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
            "overrides": CONFIRM_OVERRIDES,
        },
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
    return {
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
    return {
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
        if challenger_status != "Ready to Defend":
            errors.append(f"Challenger is not eligible to initiate (status: {challenger_status}).")
        if defender_status not in {"Ready to Defend", "Cooldown"}:
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
    return {
        "ok": True,
        "mode": "challenge_ladder_create",
        "challenge": _challenge_row(created, names),
        "notice": notice,
        "warnings": [warning] if warning else [],
    }


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
    if int(partner_a_challenger_id) == int(partner_b_challenger_id) or int(partner_a_defender_id) == int(partner_b_defender_id):
        raise ValueError("Each side must swap partners between Match A and Match B")
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
    return {
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


def _load_match_context(supabase: Any, club_id: str) -> tuple[Any, Any, Any, Any]:
    df_players_all, _df_players_active, df_leagues, _df_matches, df_meta, _df_badges, _df_player_badges, name_to_id, _id_to_name, _schema_degraded, _schema_degraded_reason = load_data(supabase, str(club_id), match_limit=5000)
    return df_players_all, df_leagues, df_meta, name_to_id


def _official_payloads(*, challenge: dict[str, Any], preview: dict[str, Any], partners: dict[str, int], match_date: str) -> list[dict[str, Any]]:
    chal = int(challenge["challenger_id"])
    defender = int(challenge["defender_id"])
    score_a = preview["scores"]["match_a"]
    score_b = preview["scores"]["match_b"]
    base = {"date": match_date, "league": "OVERALL", "match_type": "ChallengeLadder", "is_popup": False, "context_type": "challenge_ladder", "context_id": int(challenge["id"])}
    return [
        {**base, "t1_p1": chal, "t1_p2": int(partners["a_chal"]), "t2_p1": defender, "t2_p2": int(partners["a_def"]), "s1": int(score_a["score_t1"]), "s2": int(score_a["score_t2"])},
        {**base, "t1_p1": chal, "t1_p2": int(partners["b_chal"]), "t2_p1": defender, "t2_p2": int(partners["b_def"]), "s1": int(score_b["score_t1"]), "s2": int(score_b["score_t2"])},
    ]


def _swap_ranks(supabase: Any, *, club_id: str, challenger_id: int, defender_id: int) -> dict[str, Any]:
    rows = _roster_rows(supabase, club_id=str(club_id))
    chal = next((row for row in rows if _safe_int(row.get("player_id")) == int(challenger_id) and row.get("is_active") is not False), None)
    defender = next((row for row in rows if _safe_int(row.get("player_id")) == int(defender_id) and row.get("is_active") is not False), None)
    if chal is None or defender is None:
        return {"swapped": False, "reason": "active roster rows not found"}
    chal_rank = _safe_int(chal.get("rank"))
    def_rank = _safe_int(defender.get("rank"))
    if chal_rank is None or def_rank is None:
        return {"swapped": False, "reason": "rank missing"}
    if chal.get("id") is not None:
        supabase.table("ladder_roster").update({"rank": def_rank, "updated_at": _now_iso()}).eq("club_id", str(club_id)).eq("id", chal.get("id")).execute()
    else:
        supabase.table("ladder_roster").update({"rank": def_rank, "updated_at": _now_iso()}).eq("club_id", str(club_id)).eq("player_id", int(challenger_id)).execute()
    if defender.get("id") is not None:
        supabase.table("ladder_roster").update({"rank": chal_rank, "updated_at": _now_iso()}).eq("club_id", str(club_id)).eq("id", defender.get("id")).execute()
    else:
        supabase.table("ladder_roster").update({"rank": chal_rank, "updated_at": _now_iso()}).eq("club_id", str(club_id)).eq("player_id", int(defender_id)).execute()
    return {"swapped": True, "challenger_old_rank": chal_rank, "challenger_new_rank": def_rank, "defender_old_rank": def_rank, "defender_new_rank": chal_rank}


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
    source: str = "next_challenge_ladder_result",
) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_RESULT:
        raise ValueError(f"Type {CONFIRM_RESULT} to publish a ladder result.")
    challenge, _names, partners, preview = _prepare_admin_challenge_ladder_result(
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
    official_result: dict[str, Any] = {"inserted": 0, "skipped": True}
    payloads: list[dict[str, Any]] = []
    if publish_official_matches:
        payloads = _official_payloads(challenge=challenge, preview=preview, partners=partners, match_date=match_date or _now_iso())
        df_players_all, df_leagues, df_meta, name_to_id = _load_match_context(supabase, str(club_id))
        service_ctx = ServiceContext(supabase=supabase, club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, source="challenge_ladder_admin")
        result = submit_match_batch(service_ctx, payloads, name_to_id=name_to_id, df_players_all=df_players_all, df_leagues=df_leagues, df_meta=df_meta)
        if not result.ok:
            raise ValueError("; ".join(result.errors) or "Could not process challenge ladder matches")
        official_result = result.data if isinstance(result.data, dict) else {"result": result.data}
    final_winner_id = int(preview["final_winner_id"])
    patch = {"status": "COMPLETED", "winner_id": final_winner_id, "completed_at": _now_iso(), "resolution_notes": f"Next result publish. Winner: {final_winner_id}. Summary: {preview['winner_summary']}", "updated_at": _now_iso()}
    updated = _first(supabase.table("ladder_challenges").update(patch).eq("club_id", str(club_id)).eq("id", int(challenge_id)).execute()) or {**challenge, **patch}
    rank_result = {"swapped": False, "reason": "defender held"}
    if final_winner_id == int(challenge["challenger_id"]):
        rank_result = _swap_ranks(supabase, club_id=str(club_id), challenger_id=int(challenge["challenger_id"]), defender_id=int(challenge["defender_id"]))
    names = _player_names(supabase, club_id=str(club_id))
    warning = _write_ladder_audit(supabase, club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, action_type="challenge_result_publish", entity_id=str(challenge_id), before=challenge, after={"challenge": updated, "preview": preview, "official_matches": official_result, "rank_result": rank_result, "payloads": payloads}, source=source)
    return {"ok": True, "mode": "challenge_ladder_result", "challenge": _challenge_row(updated, names), "preview": preview, "official_matches": official_result, "rank_result": rank_result, "warnings": [warning] if warning else []}


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
    source: str = "next_challenge_ladder_forfeit",
) -> dict[str, Any]:
    if not is_admin_challenge_ladder_enabled():
        raise PermissionError("Next Challenge Ladder Admin is disabled.")
    if _clean(confirmation_text, limit=80).upper() != CONFIRM_FORFEIT:
        raise ValueError(f"Type {CONFIRM_FORFEIT} to record a ladder forfeit.")
    challenge = _challenge(supabase, club_id=str(club_id), challenge_id=int(challenge_id))
    forfeited = int(forfeited_by_id)
    chal = int(challenge["challenger_id"])
    defender = int(challenge["defender_id"])
    if forfeited not in {chal, defender}:
        raise ValueError("forfeited_by_id must be challenger or defender")
    winner = defender if forfeited == chal else chal
    patch = {"status": "FORFEITED", "forfeit_by": forfeited, "winner_id": winner, "completed_at": _now_iso(), "forfeit_reason": _clean(admin_note, limit=500) or "Forfeit", "updated_at": _now_iso()}
    updated = _first(supabase.table("ladder_challenges").update(patch).eq("club_id", str(club_id)).eq("id", int(challenge_id)).execute()) or {**challenge, **patch}
    rank_result = {"swapped": False, "reason": "defender held"}
    if winner == chal:
        rank_result = _swap_ranks(supabase, club_id=str(club_id), challenger_id=chal, defender_id=defender)
    names = _player_names(supabase, club_id=str(club_id))
    warning = _write_ladder_audit(supabase, club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, action_type="challenge_forfeit", entity_id=str(challenge_id), before=challenge, after={"challenge": updated, "rank_result": rank_result}, source=source)
    return {"ok": True, "mode": "challenge_ladder_forfeit", "challenge": _challenge_row(updated, names), "rank_result": rank_result, "warnings": [warning] if warning else []}
