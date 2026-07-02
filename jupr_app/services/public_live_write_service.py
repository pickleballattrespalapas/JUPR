from __future__ import annotations

from datetime import datetime, timedelta, timezone
from secrets import token_urlsafe
from typing import Any
from uuid import uuid4

from jupr_app.domain.live_beta_engine import create_round_robin_event, update_round_robin_score
from jupr_app.services.public_live_service import public_live_session_detail

PUBLIC_LIVE_SESSION_TTL_HOURS = 24
MIN_PUBLIC_RR_PLAYERS = 4
MAX_PUBLIC_RR_PLAYERS = 20


class PublicLiveSessionError(ValueError):
    """Raised for user-correctable public JUPR Live session errors."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expires_at_iso() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=PUBLIC_LIVE_SESSION_TTL_HOURS)).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _normalize_name(value: Any) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split()).strip()


def normalize_public_participant_names(names: list[Any]) -> list[str]:
    clean: list[str] = []
    seen: set[str] = set()
    for raw_name in names or []:
        name = _normalize_name(raw_name)
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        clean.append(name[:80])
        seen.add(key)
    return clean


def _state_from_event(
    *,
    club_id: str,
    session_key: str,
    edit_token: str,
    event: dict[str, Any],
    participant_names: list[str],
) -> dict[str, Any]:
    event_name = str(event.get("name") or "JUPR Live Round Robin")
    return {
        "version": 1,
        "mode": "public_quick_session",
        "source": "public_web",
        "club_id": str(club_id),
        "session_key": str(session_key),
        "event_name": event_name,
        "event_type": str(event.get("type") or "round_robin"),
        "private": {
            "edit_token": str(edit_token),
        },
        "page_state": {
            "event": event,
            "event_name": event_name,
            "type_label": "Round Robin",
            "participant_count": len(participant_names),
            "participant_text": "\n".join(participant_names),
            "rating_mode": "Unrated",
            "live_session_key": str(session_key),
        },
        "widget_state": {},
    }


def _event_from_row(row: dict[str, Any]) -> dict[str, Any]:
    state = row.get("state")
    if not isinstance(state, dict):
        return {}
    page_state = state.get("page_state")
    if not isinstance(page_state, dict):
        return {}
    event = page_state.get("event")
    return event if isinstance(event, dict) else {}


def _edit_token_from_row(row: dict[str, Any]) -> str:
    state = row.get("state")
    if not isinstance(state, dict):
        return ""
    private = state.get("private")
    if not isinstance(private, dict):
        return ""
    return str(private.get("edit_token") or "")


def _upsert_live_session_row(
    supabase,
    *,
    club_id: str,
    session_key: str,
    title: str,
    state: dict[str, Any],
    expires_at: str,
) -> dict[str, Any]:
    now = _now_iso()
    payload = {
        "club_id": str(club_id),
        "session_key": str(session_key),
        "title": str(title or "JUPR Live Round Robin"),
        "status": "active",
        "state": state,
        "source": "public_web",
        "updated_at": now,
        "last_seen_at": now,
        "expires_at": expires_at,
    }
    resp = (
        supabase.table("live_sessions")
        .upsert(payload, on_conflict="club_id,session_key")
        .execute()
    )
    row = _safe_first(resp)
    if row:
        return row
    resp = (
        supabase.table("live_sessions")
        .select("club_id,session_key,title,status,state,created_at,updated_at,last_seen_at,expires_at")
        .eq("club_id", str(club_id))
        .eq("session_key", str(session_key))
        .limit(1)
        .execute()
    )
    return _safe_first(resp) or payload


def get_public_live_session_row(
    supabase,
    *,
    club_id: str,
    session_key: str,
) -> dict[str, Any] | None:
    resp = (
        supabase.table("live_sessions")
        .select("club_id,session_key,title,status,state,created_at,updated_at,last_seen_at,expires_at")
        .eq("club_id", str(club_id))
        .eq("session_key", str(session_key))
        .limit(1)
        .execute()
    )
    return _safe_first(resp)


def create_public_round_robin_session(
    supabase,
    *,
    club_id: str,
    event_name: str,
    participant_names: list[Any],
) -> dict[str, Any]:
    names = normalize_public_participant_names(participant_names)
    if len(names) < MIN_PUBLIC_RR_PLAYERS:
        raise PublicLiveSessionError("Enter at least 4 unique player names.")
    if len(names) > MAX_PUBLIC_RR_PLAYERS:
        raise PublicLiveSessionError("Public JUPR Live supports up to 20 players for this first web version.")

    session_key = uuid4().hex
    edit_token = token_urlsafe(24)
    title = _normalize_name(event_name) or "JUPR Live Round Robin"
    event = create_round_robin_event(name=title, participant_names=names)
    state = _state_from_event(
        club_id=str(club_id),
        session_key=session_key,
        edit_token=edit_token,
        event=event,
        participant_names=names,
    )
    row = _upsert_live_session_row(
        supabase,
        club_id=str(club_id),
        session_key=session_key,
        title=title,
        state=state,
        expires_at=_expires_at_iso(),
    )
    return {
        "edit_token": edit_token,
        "session": public_live_session_detail(row),
    }


def update_public_round_robin_scores(
    supabase,
    *,
    club_id: str,
    session_key: str,
    edit_token: str,
    scores: list[dict[str, Any]],
) -> dict[str, Any]:
    row = get_public_live_session_row(
        supabase,
        club_id=str(club_id),
        session_key=str(session_key),
    )
    if not row:
        raise PublicLiveSessionError("Live session not found.")
    expected_token = _edit_token_from_row(row)
    if not expected_token or str(edit_token or "") != expected_token:
        raise PermissionError("Invalid edit token for this public live session.")

    event = _event_from_row(row)
    if str(event.get("type") or "") != "round_robin":
        raise PublicLiveSessionError("Only public round-robin score updates are supported right now.")

    for score in scores or []:
        match_id = str(score.get("match_id") or "").strip()
        if not match_id:
            continue
        raw_a = score.get("score_a")
        raw_b = score.get("score_b")
        score_a = None if raw_a in (None, "") else int(raw_a)
        score_b = None if raw_b in (None, "") else int(raw_b)
        if score_a is not None and (score_a < 0 or score_a > 99):
            raise PublicLiveSessionError("Scores must be between 0 and 99.")
        if score_b is not None and (score_b < 0 or score_b > 99):
            raise PublicLiveSessionError("Scores must be between 0 and 99.")
        update_round_robin_score(event, match_id, score_a, score_b)

    state = row.get("state") if isinstance(row.get("state"), dict) else {}
    page_state = state.setdefault("page_state", {})
    if not isinstance(page_state, dict):
        page_state = {}
        state["page_state"] = page_state
    page_state["event"] = event
    state["event_name"] = str(event.get("name") or row.get("title") or "JUPR Live Round Robin")
    state["event_type"] = str(event.get("type") or "round_robin")

    updated = _upsert_live_session_row(
        supabase,
        club_id=str(club_id),
        session_key=str(session_key),
        title=str(row.get("title") or event.get("name") or "JUPR Live Round Robin"),
        state=state,
        expires_at=str(row.get("expires_at") or _expires_at_iso()),
    )
    return {"session": public_live_session_detail(updated)}
