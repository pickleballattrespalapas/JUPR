from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_league_manager_service import is_admin_league_manager_enabled

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
CONFIRM_CREATE_SESSION = "CREATE LIVE SESSION"
CONFIRM_SAVE_SESSION = "SAVE SESSION"
CONFIRM_SAVE_ROUND = "SAVE ROUND"
SESSION_STATUSES = {"setup", "active", "paused", "complete", "archived"}
ROUND_STATUSES = {"draft", "generated", "submitted", "voided"}


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _first_row(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _clean_text(value: Any, *, limit: int = 240) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _session_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "club_id": str(row.get("club_id") or ""),
        "league_name": str(row.get("league_name") or ""),
        "week_tag": str(row.get("week_tag") or ""),
        "status": str(row.get("status") or "setup"),
        "total_rounds": _safe_int(row.get("total_rounds"), 1) or 1,
        "current_round": _safe_int(row.get("current_round"), 1) or 1,
        "roster_json": _as_list(row.get("roster_json")),
        "current_court_state_json": _as_list(row.get("current_court_state_json")),
        "notes": row.get("notes"),
        "created_by": row.get("created_by"),
        "updated_by": row.get("updated_by"),
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _round_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "club_id": str(row.get("club_id") or ""),
        "session_id": str(row.get("session_id") or ""),
        "round_number": _safe_int(row.get("round_number"), 1) or 1,
        "round_label": row.get("round_label"),
        "status": str(row.get("status") or "draft"),
        "match_date": row.get("match_date"),
        "preview_json": _as_dict(row.get("preview_json")),
        "matches_json": _as_list(row.get("matches_json")),
        "movement_json": _as_dict(row.get("movement_json")),
        "submitted_match_count": _safe_int(row.get("submitted_match_count"), 0) or 0,
        "submitted_match_ids": _as_list(row.get("submitted_match_ids")),
        "submitted_at": row.get("submitted_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _court_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "club_id": str(row.get("club_id") or ""),
        "session_id": str(row.get("session_id") or ""),
        "round_number": _safe_int(row.get("round_number"), 1) or 1,
        "court_number": _safe_int(row.get("court_number"), 1) or 1,
        "format_type": str(row.get("format_type") or "4-player"),
        "player_names": _as_list(row.get("player_names")),
        "players_json": _as_list(row.get("players_json")),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _normalize_courts(courts: list[dict[str, Any]], *, default_round: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, raw in enumerate(courts or [], start=1):
        if not isinstance(raw, dict):
            continue
        court_number = _safe_int(raw.get("court_number", raw.get("court")), index) or index
        if court_number in seen:
            continue
        seen.add(court_number)
        player_names = raw.get("player_names")
        if isinstance(player_names, str):
            names = [name.strip() for name in player_names.replace(",", "\n").split("\n") if name.strip()]
        else:
            names = [str(name or "").strip() for name in _as_list(player_names) if str(name or "").strip()]
        normalized.append(
            {
                "round_number": _safe_int(raw.get("round_number"), default_round) or default_round,
                "court_number": int(court_number),
                "format_type": _clean_text(raw.get("format_type", raw.get("formatType") or "4-player"), limit=80) or "4-player",
                "player_names": names,
                "players_json": _as_list(raw.get("players_json", raw.get("players"))),
            }
        )
    normalized.sort(key=lambda row: int(row.get("court_number") or 0))
    return normalized


def _fetch_session_row(supabase: Any, *, club_id: str, session_id: str) -> dict[str, Any] | None:
    try:
        return _first_row(
            supabase.table("league_live_sessions")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("id", str(session_id))
            .limit(1)
            .execute()
        )
    except Exception:
        return None


def _fetch_rounds(supabase: Any, *, club_id: str, session_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("league_live_rounds")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("session_id", str(session_id))
            .execute()
        )
    except Exception:
        rows = []
    return sorted((_round_payload(row) for row in rows), key=lambda row: int(row.get("round_number") or 0))


def _fetch_courts(supabase: Any, *, club_id: str, session_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("league_live_courts")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("session_id", str(session_id))
            .execute()
        )
    except Exception:
        rows = []
    return sorted(
        (_court_payload(row) for row in rows),
        key=lambda row: (int(row.get("round_number") or 0), int(row.get("court_number") or 0)),
    )


def _replace_court_snapshots(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    round_number: int,
    courts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized = _normalize_courts(courts, default_round=int(round_number))
    try:
        (
            supabase.table("league_live_courts")
            .delete()
            .eq("club_id", str(club_id))
            .eq("session_id", str(session_id))
            .eq("round_number", int(round_number))
            .execute()
        )
    except Exception:
        pass
    if not normalized:
        return []
    payloads = [
        {
            "id": str(uuid4()),
            "club_id": str(club_id),
            "session_id": str(session_id),
            "round_number": int(round_number),
            "court_number": int(row["court_number"]),
            "format_type": str(row.get("format_type") or "4-player"),
            "player_names": row.get("player_names") or [],
            "players_json": row.get("players_json") or [],
            "updated_at": _now_iso(),
        }
        for row in normalized
    ]
    try:
        inserted = _safe_rows(supabase.table("league_live_courts").insert(payloads).execute())
    except Exception:
        inserted = payloads
    return [_court_payload(row) for row in inserted]


def _audit(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action_type: str,
    entity_id: str,
    before_json: dict[str, Any] | None = None,
    after_json: dict[str, Any] | None = None,
    source: str,
) -> list[str]:
    payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type=action_type,
        entity_type="league_live_session",
        entity_id=str(entity_id or ""),
        before_json=before_json or {},
        after_json={"source_client": "fastapi/nextjs", "source_page": source, **(after_json or {})},
        source_page=source,
        flagged_for_review=True,
    )
    write = write_admin_activity_log(supabase, payload)
    if not write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return [write.warning] if write.warning else []


def build_admin_league_live_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "sessions_endpoint": None,
            "warnings": ["Next League Manager is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER on FastAPI."],
        }
    count = 0
    if supabase is not None:
        try:
            count = len(_safe_rows(supabase.table("league_live_sessions").select("id").eq("club_id", str(club_id)).execute()))
        except Exception:
            count = 0
    return {
        "enabled": True,
        "status": "ready_for_persistent_league_live_sessions",
        "sessions_endpoint": "/admin/clubs/{club_id}/league-manager/live-sessions",
        "session_count": count,
        "warnings": [],
    }


def list_admin_league_live_sessions(
    supabase: Any,
    *,
    club_id: str,
    status: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    query = supabase.table("league_live_sessions").select("*").eq("club_id", str(club_id))
    clean_status = _clean_text(status, limit=40).lower()
    if clean_status:
        query = query.eq("status", clean_status)
    try:
        rows = _safe_rows(query.order("updated_at", desc=True).limit(max(1, min(int(limit or 50), 200))).execute())
    except Exception:
        rows = _safe_rows(query.execute())
    sessions = [_session_payload(row) for row in rows]
    return {"ok": True, "mode": "league_live_sessions_list", "sessions": sessions, "count": len(sessions)}


def get_admin_league_live_session(supabase: Any, *, club_id: str, session_id: str) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    row = _fetch_session_row(supabase, club_id=str(club_id), session_id=str(session_id))
    if row is None:
        raise ValueError("league live session not found")
    session = _session_payload(row)
    rounds = _fetch_rounds(supabase, club_id=str(club_id), session_id=str(session_id))
    courts = _fetch_courts(supabase, club_id=str(club_id), session_id=str(session_id))
    return {"ok": True, "mode": "league_live_session_detail", "session": session, "rounds": rounds, "courts": courts}


def create_admin_league_live_session(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    week_tag: str,
    total_rounds: int,
    current_round: int = 1,
    roster: list[dict[str, Any]] | None = None,
    courts: list[dict[str, Any]] | None = None,
    notes: str | None = None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_league_live_session_create",
) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_CREATE_SESSION:
        raise ValueError(f"Type {CONFIRM_CREATE_SESSION} to create a persisted league live session.")
    clean_league = _clean_text(league_name, limit=120)
    if not clean_league:
        raise ValueError("league_name is required")
    safe_total_rounds = max(1, min(_safe_int(total_rounds, 1) or 1, 50))
    safe_current_round = max(1, min(_safe_int(current_round, 1) or 1, safe_total_rounds))
    session_id = str(uuid4())
    now = _now_iso()
    normalized_courts = _normalize_courts(courts or [], default_round=safe_current_round)
    payload = {
        "id": session_id,
        "club_id": str(club_id),
        "league_name": clean_league,
        "week_tag": _clean_text(week_tag, limit=80),
        "status": "active",
        "total_rounds": safe_total_rounds,
        "current_round": safe_current_round,
        "roster_json": _as_list(roster),
        "current_court_state_json": normalized_courts,
        "notes": _clean_text(notes, limit=1000) or None,
        "created_by": str(actor_email or ""),
        "updated_by": str(actor_email or ""),
        "started_at": now,
        "created_at": now,
        "updated_at": now,
    }
    inserted = _first_row(supabase.table("league_live_sessions").insert(payload).execute()) or payload
    court_rows = _replace_court_snapshots(supabase, club_id=str(club_id), session_id=session_id, round_number=safe_current_round, courts=normalized_courts)
    warnings = _audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="create_league_live_session_admin",
        entity_id=session_id,
        after_json={"session": _session_payload(inserted), "court_count": len(court_rows)},
        source=source,
    )
    return {"ok": True, "mode": "league_live_session_create", "session": _session_payload(inserted), "courts": court_rows, "warnings": warnings}


def update_admin_league_live_session_snapshot(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    patch: dict[str, Any],
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_league_live_session_snapshot",
) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_SAVE_SESSION:
        raise ValueError(f"Type {CONFIRM_SAVE_SESSION} to save the league live session snapshot.")
    before = _fetch_session_row(supabase, club_id=str(club_id), session_id=str(session_id))
    if before is None:
        raise ValueError("league live session not found")
    current_round = _safe_int(patch.get("current_round"), _safe_int(before.get("current_round"), 1) or 1) or 1
    total_rounds = max(1, min(_safe_int(patch.get("total_rounds"), _safe_int(before.get("total_rounds"), 1) or 1) or 1, 50))
    current_round = max(1, min(current_round, total_rounds))
    status = _clean_text(patch.get("status") or before.get("status") or "active", limit=40).lower()
    if status not in SESSION_STATUSES:
        raise ValueError("unsupported session status")
    normalized_courts = _normalize_courts(patch.get("courts") or patch.get("current_court_state_json") or [], default_round=current_round)
    payload: dict[str, Any] = {
        "status": status,
        "total_rounds": total_rounds,
        "current_round": current_round,
        "updated_by": str(actor_email or ""),
        "updated_at": _now_iso(),
    }
    if "week_tag" in patch:
        payload["week_tag"] = _clean_text(patch.get("week_tag"), limit=80)
    if "notes" in patch:
        payload["notes"] = _clean_text(patch.get("notes"), limit=1000) or None
    if "roster" in patch or "roster_json" in patch:
        payload["roster_json"] = _as_list(patch.get("roster", patch.get("roster_json")))
    if normalized_courts:
        payload["current_court_state_json"] = normalized_courts
    if status == "complete" and not before.get("completed_at"):
        payload["completed_at"] = _now_iso()
    updated = _first_row(
        supabase.table("league_live_sessions")
        .update(payload)
        .eq("club_id", str(club_id))
        .eq("id", str(session_id))
        .execute()
    ) or {**before, **payload}
    court_rows = []
    if normalized_courts:
        court_rows = _replace_court_snapshots(supabase, club_id=str(club_id), session_id=str(session_id), round_number=current_round, courts=normalized_courts)
    warnings = _audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="update_league_live_session_snapshot_admin",
        entity_id=str(session_id),
        before_json={"session": _session_payload(before)},
        after_json={"session": _session_payload(updated), "court_count": len(court_rows)},
        source=source,
    )
    return {"ok": True, "mode": "league_live_session_snapshot_update", "session": _session_payload(updated), "courts": court_rows, "warnings": warnings}


def save_admin_league_live_round(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    round_number: int,
    round_label: str | None = None,
    match_date: str | None = None,
    preview: dict[str, Any] | None = None,
    matches: list[dict[str, Any]] | None = None,
    movement: dict[str, Any] | None = None,
    submitted_match_count: int | None = None,
    submitted_match_ids: list[Any] | None = None,
    courts: list[dict[str, Any]] | None = None,
    advance_after_save: bool = True,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_league_live_round_save",
) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_SAVE_ROUND:
        raise ValueError(f"Type {CONFIRM_SAVE_ROUND} to save the league live round state.")
    session_row = _fetch_session_row(supabase, club_id=str(club_id), session_id=str(session_id))
    if session_row is None:
        raise ValueError("league live session not found")
    safe_round = max(1, _safe_int(round_number, 1) or 1)
    matches_list = _as_list(matches)
    count = submitted_match_count if submitted_match_count is not None else len(matches_list)
    count = max(0, _safe_int(count, 0) or 0)
    status = "submitted" if count > 0 else "generated"
    now = _now_iso()
    existing = _first_row(
        supabase.table("league_live_rounds")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("session_id", str(session_id))
        .eq("round_number", safe_round)
        .limit(1)
        .execute()
    )
    payload = {
        "club_id": str(club_id),
        "session_id": str(session_id),
        "round_number": safe_round,
        "round_label": _clean_text(round_label, limit=80) or f"Round {safe_round}",
        "status": status,
        "match_date": _clean_text(match_date, limit=20) or None,
        "preview_json": _as_dict(preview),
        "matches_json": matches_list,
        "movement_json": _as_dict(movement),
        "submitted_match_count": count,
        "submitted_match_ids": _as_list(submitted_match_ids),
        "submitted_at": now if count > 0 else None,
        "updated_at": now,
    }
    if existing:
        saved = _first_row(
            supabase.table("league_live_rounds")
            .update(payload)
            .eq("id", str(existing.get("id")))
            .execute()
        ) or {**existing, **payload}
    else:
        insert_payload = {"id": str(uuid4()), "created_at": now, **payload}
        saved = _first_row(supabase.table("league_live_rounds").insert(insert_payload).execute()) or insert_payload
    court_rows = _replace_court_snapshots(supabase, club_id=str(club_id), session_id=str(session_id), round_number=safe_round, courts=courts or [])
    next_round = min(max(safe_round + 1, _safe_int(session_row.get("current_round"), 1) or 1), _safe_int(session_row.get("total_rounds"), safe_round) or safe_round)
    session_patch: dict[str, Any] = {
        "current_court_state_json": _normalize_courts(courts or [], default_round=next_round) or _as_list(session_row.get("current_court_state_json")),
        "updated_by": str(actor_email or ""),
        "updated_at": now,
    }
    if advance_after_save and next_round > (_safe_int(session_row.get("current_round"), 1) or 1):
        session_patch["current_round"] = next_round
    if count > 0:
        session_patch["status"] = "active"
    updated_session = _first_row(
        supabase.table("league_live_sessions")
        .update(session_patch)
        .eq("club_id", str(club_id))
        .eq("id", str(session_id))
        .execute()
    ) or {**session_row, **session_patch}
    warnings = _audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="save_league_live_round_admin",
        entity_id=str(session_id),
        before_json={"session": _session_payload(session_row), "round": _round_payload(existing or {}) if existing else None},
        after_json={"session": _session_payload(updated_session), "round": _round_payload(saved), "court_count": len(court_rows)},
        source=source,
    )
    return {
        "ok": True,
        "mode": "league_live_round_save",
        "session": _session_payload(updated_session),
        "round": _round_payload(saved),
        "courts": court_rows,
        "warnings": warnings,
    }
