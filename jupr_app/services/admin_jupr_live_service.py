from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from jupr_app.data.load import load_data
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.live_beta_engine import (
    create_league_event,
    create_round_robin_event,
    find_match_by_id,
    set_match_score,
    start_next_league_round,
)
from jupr_app.services.admin_live_ladder_operation_service import (
    deterministic_match_context_id,
    is_staging_write_gate_enabled,
)
from jupr_app.services.context import ServiceContext
from jupr_app.services.match_service import submit_match_batch

TRUTHY = {"1", "true", "yes", "y", "on"}
SESSION_STATUSES = {"active", "completed", "abandoned", "archived"}
CONFIRM_CREATE = "CREATE LIVE SESSION"
CONFIRM_STATUS = "SAVE LIVE SESSION"
CONFIRM_SCORES = "SAVE LIVE SCORES"
CONFIRM_PUBLISH = "PUBLISH LIVE MATCHES"
CONFIRM_ADVANCE = "ADVANCE LIVE ROUND"
SUPPORTED_ADMIN_EVENT_TYPES = {"round_robin", "league_ladder", "league", "ladder"}
JUPR_LIVE_WRITE_FLAG = "JUPR_ENABLE_STAGING_NEXT_ADMIN_JUPR_LIVE_WRITES"


def is_admin_jupr_live_enabled() -> bool:
    return os.getenv("JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE", "").strip().lower() in TRUTHY


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _clean_text(value: Any, *, limit: int = 240) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _state(row: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(row.get("state"))


def _event_from_state(state: dict[str, Any]) -> dict[str, Any]:
    page_state = _as_dict(state.get("page_state"))
    event = page_state.get("event")
    return dict(event) if isinstance(event, dict) else {}


def _put_event(state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    next_state = dict(state or {})
    page_state = _as_dict(next_state.get("page_state"))
    page_state["event"] = dict(event or {})
    page_state["event_name"] = str(event.get("name") or next_state.get("event_name") or "JUPR Live")
    page_state["event_type"] = str(event.get("type") or next_state.get("event_type") or "round_robin")
    page_state["current_round_number"] = event.get("currentRoundNumber")
    next_state["page_state"] = page_state
    next_state["event_name"] = page_state["event_name"]
    next_state["event_type"] = page_state["event_type"]
    return next_state


def _session_payload(row: dict[str, Any]) -> dict[str, Any]:
    state = _state(row)
    event = _event_from_state(state)
    return {
        "id": str(row.get("id") or ""),
        "club_id": str(row.get("club_id") or ""),
        "session_key": str(row.get("session_key") or ""),
        "title": _clean_text(row.get("title"), limit=160),
        "status": str(row.get("status") or "active"),
        "source": str(row.get("source") or "jupr_live_admin"),
        "created_by_email": row.get("created_by_email"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "version": str(row.get("updated_at") or row.get("created_at") or ""),
        "last_seen_at": row.get("last_seen_at"),
        "expires_at": row.get("expires_at"),
        "event_type": event.get("type") or state.get("event_type") or state.get("eventType"),
        "current_round_number": event.get("currentRoundNumber"),
        "total_rounds": event.get("totalRounds"),
        "state": state,
        # club_id may be opaque/UUID-shaped; only a caller with a resolved club
        # slug may construct the public route.
        "public_url_path": None,
    }


def _player_rows_by_id(supabase: Any, *, club_id: str, player_ids: list[int]) -> dict[int, dict[str, Any]]:
    wanted = {int(pid) for pid in player_ids if pid is not None}
    if not wanted:
        return {}
    rows = _safe_rows(supabase.table("players").select("id,name,club_id").eq("club_id", str(club_id)).execute())
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            pid = int(row.get("id"))
        except Exception:
            continue
        if pid in wanted:
            result[pid] = row
    return result


def _resolve_names_and_ids(supabase: Any, *, club_id: str, participant_names: list[str], player_ids: list[int] | None) -> tuple[list[str], dict[str, int], list[int]]:
    names = [_clean_text(name, limit=160) for name in (participant_names or []) if _clean_text(name, limit=160)]
    ids = [int(pid) for pid in (player_ids or []) if pid is not None]
    resolved_ids: dict[str, int] = {}
    if ids:
        players = _player_rows_by_id(supabase, club_id=str(club_id), player_ids=ids)
        if len(ids) != len(names):
            names = [str(players.get(pid, {}).get("name") or f"Player {pid}") for pid in ids]
        if len(ids) != len(names):
            raise ValueError("player_ids and participant_names must have the same length when both are supplied.")
        for idx, pid in enumerate(ids):
            if pid not in players:
                raise ValueError(f"Player id {pid} was not found in this club.")
            resolved_ids[names[idx]] = pid
    return names, resolved_ids, ids


def _base_live_state(*, club_id: str, source: str, event_type: str, title: str, names: list[str], ids: list[int], event: dict[str, Any], type_label: str) -> dict[str, Any]:
    return {
        "version": 1,
        "mode": "admin_official_staging",
        "source": source,
        "club_id": str(club_id),
        "event_type": event_type,
        "participant_names": names,
        "participant_player_ids": ids,
        "rating_mode": "official_publish_required",
        "page_state": {
            "event": event,
            "event_name": title,
            "type_label": type_label,
            "participant_count": len(names),
            "participant_text": "\n".join(names),
            "rating_mode": "Official on publish",
            "current_round_number": event.get("currentRoundNumber"),
        },
        "official_publish": {
            "published_live_match_ids": [],
            "published_at": None,
            "publish_result": None,
        },
    }


def _build_round_robin_state(supabase: Any, *, club_id: str, title: str, participant_names: list[str], player_ids: list[int] | None, source: str) -> dict[str, Any]:
    names, resolved_ids, ids = _resolve_names_and_ids(supabase, club_id=str(club_id), participant_names=participant_names, player_ids=player_ids)
    event: dict[str, Any] = {}
    if len(names) >= 4:
        event = create_round_robin_event(name=title, participant_names=names, resolved_ids=resolved_ids)
    return _base_live_state(club_id=str(club_id), source=source, event_type="round_robin", title=title, names=names, ids=ids, event=event, type_label="Round Robin")


def _build_league_ladder_state(
    supabase: Any,
    *,
    club_id: str,
    title: str,
    participant_names: list[str],
    player_ids: list[int] | None,
    total_rounds: int,
    court_sizes: list[int] | None,
    source: str,
) -> dict[str, Any]:
    names, resolved_ids, ids = _resolve_names_and_ids(supabase, club_id=str(club_id), participant_names=participant_names, player_ids=player_ids)
    event: dict[str, Any] = {}
    if len(names) >= 4:
        event = create_league_event(name=title, participant_names=names, total_rounds=max(1, int(total_rounds or 3)), resolved_ids=resolved_ids, court_sizes=court_sizes)
    return _base_live_state(club_id=str(club_id), source=source, event_type="league", title=title, names=names, ids=ids, event=event, type_label="League / Ladder")


def build_admin_jupr_live_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_jupr_live_enabled():
        return {"enabled": False, "status": "guarded_off", "warnings": ["Enable JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE to manage JUPR Live sessions in Next."]}
    counts = {"active": 0, "completed": 0, "abandoned": 0, "archived": 0}
    if supabase is not None:
        try:
            rows = _safe_rows(supabase.table("live_sessions").select("status").eq("club_id", str(club_id)).execute())
            for row in rows:
                status = str(row.get("status") or "").lower()
                if status in counts:
                    counts[status] += 1
        except Exception:
            pass
    writes_enabled = is_staging_write_gate_enabled(JUPR_LIVE_WRITE_FLAG)
    return {
        "enabled": True,
        "writes_enabled": writes_enabled,
        "status": "ready_for_jupr_live_admin" if writes_enabled else "read_only_streamlit_fallback",
        "counts": counts,
        "warnings": [] if writes_enabled else [
            f"Next JUPR Live Admin writes require JUPR_ENV=staging and {JUPR_LIVE_WRITE_FLAG}=1 on FastAPI. Use Streamlit JUPR Live Admin otherwise."
        ],
        "confirmation_text": {"create": CONFIRM_CREATE, "status": CONFIRM_STATUS, "scores": CONFIRM_SCORES, "publish": CONFIRM_PUBLISH, "advance": CONFIRM_ADVANCE},
        "streamlit_fallback": "jupr_live_admin",
        "recovery": {"match_log_url": "/admin/match-log", "replay_history_url": "/admin/replay-history"},
        "authority": "python_fastapi",
    }


def list_admin_jupr_live_sessions(supabase: Any, *, club_id: str, status: str | None = None, limit: int = 100) -> dict[str, Any]:
    if not is_admin_jupr_live_enabled():
        raise PermissionError("Next JUPR Live Admin is disabled.")
    query = supabase.table("live_sessions").select("*").eq("club_id", str(club_id))
    clean_status = _clean_text(status, limit=40).lower()
    if clean_status:
        query = query.eq("status", clean_status)
    try:
        rows = _safe_rows(query.order("updated_at", desc=True).limit(max(1, min(int(limit or 100), 300))).execute())
    except Exception:
        rows = _safe_rows(query.execute())
    sessions = [_session_payload(row) for row in rows]
    return {"ok": True, "mode": "jupr_live_admin_sessions", "sessions": sessions, "count": len(sessions)}


def get_admin_jupr_live_session(supabase: Any, *, club_id: str, session_key: str) -> dict[str, Any]:
    if not is_admin_jupr_live_enabled():
        raise PermissionError("Next JUPR Live Admin is disabled.")
    row = _safe_first(supabase.table("live_sessions").select("*").eq("club_id", str(club_id)).eq("session_key", str(session_key)).limit(1).execute())
    if row is None:
        raise ValueError("live session not found")
    return {"ok": True, "mode": "jupr_live_admin_session_detail", "session": _session_payload(row)}


def create_admin_jupr_live_session(
    supabase: Any,
    *,
    club_id: str,
    title: str,
    event_type: str,
    participant_names: list[str] | None,
    player_ids: list[int] | None = None,
    total_rounds: int = 3,
    court_sizes: list[int] | None = None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_jupr_live_admin_create",
) -> dict[str, Any]:
    if not is_admin_jupr_live_enabled():
        raise PermissionError("Next JUPR Live Admin is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_CREATE:
        raise ValueError(f"Type {CONFIRM_CREATE} to create a durable JUPR Live session.")
    clean_event_type = _clean_text(event_type or "round_robin", limit=60).lower().replace(" ", "_").replace("/", "_")
    if clean_event_type not in SUPPORTED_ADMIN_EVENT_TYPES:
        raise ValueError("unsupported JUPR Live event type; use Tournament Live for brackets/draws")
    if clean_event_type in {"league", "ladder"}:
        clean_event_type = "league_ladder"
    session_key = uuid4().hex
    now = _now_iso()
    clean_title = _clean_text(title, limit=160) or "JUPR Live Session"
    if clean_event_type == "round_robin":
        state = _build_round_robin_state(supabase, club_id=str(club_id), title=clean_title, participant_names=list(participant_names or []), player_ids=player_ids, source=source)
    else:
        state = _build_league_ladder_state(supabase, club_id=str(club_id), title=clean_title, participant_names=list(participant_names or []), player_ids=player_ids, total_rounds=total_rounds, court_sizes=court_sizes, source=source)
    payload = {"club_id": str(club_id), "session_key": session_key, "status": "active", "title": clean_title, "state": state, "source": "jupr_live_admin", "created_by_email": str(actor_email or "").strip().lower() or None, "created_at": now, "updated_at": now, "last_seen_at": now, "expires_at": (datetime.now(timezone.utc) + timedelta(hours=18)).isoformat()}
    inserted = _safe_first(supabase.table("live_sessions").insert(payload).execute()) or payload
    _audit(supabase, club_id=club_id, actor_email=actor_email, actor_role=actor_role, action_type="create_jupr_live_session_admin", entity_id=session_key, before_json={}, after_json={"session": _session_payload(inserted)}, source=source)
    return {"ok": True, "mode": "jupr_live_admin_session_create", "session": _session_payload(inserted)}


def _live_row(supabase: Any, *, club_id: str, session_key: str) -> dict[str, Any]:
    row = _safe_first(supabase.table("live_sessions").select("*").eq("club_id", str(club_id)).eq("session_key", str(session_key)).limit(1).execute())
    if row is None:
        raise ValueError("live session not found")
    return row


def _update_live_row(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    patch: dict[str, Any],
    expected_version: str | None = None,
) -> dict[str, Any]:
    query = supabase.table("live_sessions").update(patch).eq("club_id", str(club_id)).eq("session_key", str(session_key))
    if str(expected_version or "").strip():
        query = query.eq("updated_at", str(expected_version))
    updated = _safe_first(query.execute())
    if updated is None and str(expected_version or "").strip():
        raise ValueError("JUPR Live session changed. Reload the Python state before saving.")
    return updated or {"club_id": str(club_id), "session_key": str(session_key), **patch}


def update_admin_jupr_live_scores(supabase: Any, *, club_id: str, session_key: str, scores: list[dict[str, Any]], actor_email: str, actor_role: str, confirmation_text: str, expected_version: str | None = None, source: str = "next_jupr_live_admin_scores") -> dict[str, Any]:
    if not is_admin_jupr_live_enabled():
        raise PermissionError("Next JUPR Live Admin is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_SCORES:
        raise ValueError(f"Type {CONFIRM_SCORES} to save JUPR Live scores.")
    before = _live_row(supabase, club_id=str(club_id), session_key=str(session_key))
    state = _state(before)
    event = _event_from_state(state)
    if str(event.get("type") or "") not in {"round_robin", "league"}:
        raise ValueError("Score entry supports one-off JUPR Live round-robin and league/ladder sessions. Use Tournament Live/Ops for brackets.")
    changed = 0
    for score in scores or []:
        match_id = _clean_text(score.get("match_id"), limit=120)
        if not match_id:
            continue
        raw_a = score.get("score_a")
        raw_b = score.get("score_b")
        score_a = None if raw_a in (None, "") else int(raw_a)
        score_b = None if raw_b in (None, "") else int(raw_b)
        if score_a is not None and (score_a < 0 or score_a > 99):
            raise ValueError("Scores must be between 0 and 99.")
        if score_b is not None and (score_b < 0 or score_b > 99):
            raise ValueError("Scores must be between 0 and 99.")
        match = find_match_by_id(event, match_id)
        if match is None:
            continue
        set_match_score(match, score_a, score_b)
        changed += 1
    next_state = _put_event(state, event)
    patch = {"state": next_state, "updated_at": _now_iso(), "last_seen_at": _now_iso()}
    updated = _update_live_row(supabase, club_id=str(club_id), session_key=str(session_key), patch=patch, expected_version=expected_version)
    _audit(supabase, club_id=club_id, actor_email=actor_email, actor_role=actor_role, action_type="score_jupr_live_session_admin", entity_id=session_key, before_json={"session": _session_payload(before)}, after_json={"session": _session_payload(updated), "changed_scores": changed}, source=source)
    return {"ok": True, "mode": "jupr_live_admin_scores", "changed_scores": changed, "session": _session_payload(updated)}


def advance_admin_jupr_live_league_round(supabase: Any, *, club_id: str, session_key: str, actor_email: str, actor_role: str, confirmation_text: str, expected_version: str | None = None, source: str = "next_jupr_live_admin_advance") -> dict[str, Any]:
    if not is_admin_jupr_live_enabled():
        raise PermissionError("Next JUPR Live Admin is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_ADVANCE:
        raise ValueError(f"Type {CONFIRM_ADVANCE} to advance the JUPR Live league round.")
    before = _live_row(supabase, club_id=str(club_id), session_key=str(session_key))
    state = _state(before)
    event = _event_from_state(state)
    if str(event.get("type") or "") != "league":
        raise ValueError("Only league/ladder JUPR Live sessions can advance rounds.")
    start_next_league_round(event)
    next_state = _put_event(state, event)
    patch = {"state": next_state, "updated_at": _now_iso(), "last_seen_at": _now_iso()}
    updated = _update_live_row(supabase, club_id=str(club_id), session_key=str(session_key), patch=patch, expected_version=expected_version)
    _audit(supabase, club_id=club_id, actor_email=actor_email, actor_role=actor_role, action_type="advance_jupr_live_league_round_admin", entity_id=session_key, before_json={"session": _session_payload(before)}, after_json={"session": _session_payload(updated)}, source=source)
    return {"ok": True, "mode": "jupr_live_admin_advance", "session": _session_payload(updated)}


def _participant_map(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(p.get("id")): dict(p) for p in (event.get("participants") or []) if p.get("id")}


def _event_matches(event: dict[str, Any]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if str(event.get("type") or "") == "round_robin":
        for round_row in event.get("rounds") or []:
            for match in (round_row or {}).get("matches") or []:
                row = dict(match)
                row["round"] = round_row.get("number")
                matches.append(row)
        return matches
    if str(event.get("type") or "") == "league":
        for round_row in event.get("rounds") or []:
            for court in (round_row or {}).get("courts") or []:
                for mini_round in court.get("miniRounds") or []:
                    for match in mini_round.get("matches") or []:
                        row = dict(match)
                        row["round"] = round_row.get("number")
                        row["court"] = court.get("courtNumber")
                        row["mini_round"] = mini_round.get("number")
                        matches.append(row)
        return matches
    return []


def _team_player_ids(participants: dict[str, dict[str, Any]], team: list[Any]) -> list[int]:
    ids: list[int] = []
    for token in team or []:
        participant = participants.get(str(token))
        if not participant or participant.get("player_id") is None:
            raise ValueError("Official publish requires every JUPR Live participant to be linked to an official player id.")
        ids.append(int(participant["player_id"]))
    if len(ids) != 2:
        raise ValueError("Official JUPR Live publish currently requires doubles teams with two linked players per side.")
    return ids


def _publish_payloads_from_event(event: dict[str, Any], *, session_key: str, published_ids: set[str], match_date: str, publish_context_prefix: str | None = None) -> list[dict[str, Any]]:
    participants = _participant_map(event)
    payloads: list[dict[str, Any]] = []
    for match in _event_matches(event):
        match_id = str(match.get("id") or "")
        if not match_id or match_id in published_ids:
            continue
        if match.get("scoreA") is None or match.get("scoreB") is None:
            continue
        s1 = int(match.get("scoreA") or 0)
        s2 = int(match.get("scoreB") or 0)
        if s1 == s2:
            raise ValueError(f"Match {match_id} is tied; official matches cannot be published with tied scores.")
        if (s1 + s2) <= 0:
            continue
        team_a = _team_player_ids(participants, list(match.get("teamA") or []))
        team_b = _team_player_ids(participants, list(match.get("teamB") or []))
        context_prefix = _clean_text(publish_context_prefix, limit=80)
        payloads.append({"date": match_date, "league": "OVERALL", "match_type": "JUPR Live", "is_popup": False, "context_type": "jupr_live", "context_id": deterministic_match_context_id(operation_key=context_prefix, slot=match_id) if context_prefix else str(session_key), "live_match_id": match_id, "week_tag": f"JUPR Live {str(match_date)[:10]}", "t1_p1": team_a[0], "t1_p2": team_a[1], "t2_p1": team_b[0], "t2_p2": team_b[1], "s1": s1, "s2": s2})
    return payloads


def admin_jupr_live_publish_contexts(
    session: dict[str, Any],
    *,
    operation_key: str,
) -> list[str]:
    """Predict the exact official-match contexts for pre-mutation recovery."""
    state = _as_dict((session or {}).get("state"))
    event = _event_from_state(state)
    if str(event.get("type") or "") not in {"round_robin", "league"}:
        raise ValueError(
            "Official JUPR Live publish supports one-off round-robin and league/ladder sessions. "
            "Use Tournament Live/Ops for brackets."
        )
    official = _as_dict(state.get("official_publish"))
    published_ids = {
        str(mid)
        for mid in (official.get("published_live_match_ids") or official.get("published_match_ids") or [])
    }
    prefix = _clean_text(operation_key, limit=80)
    participants = _participant_map(event)
    contexts: list[str] = []
    for match in _event_matches(event):
        match_id = str(match.get("id") or "")
        if not match_id or match_id in published_ids:
            continue
        if match.get("scoreA") is None or match.get("scoreB") is None:
            continue
        score_a = int(match.get("scoreA") or 0)
        score_b = int(match.get("scoreB") or 0)
        if score_a == score_b:
            raise ValueError(f"Match {match_id} is tied; official matches cannot be published with tied scores.")
        if (score_a + score_b) <= 0:
            continue
        # Resolve both teams before acquiring a durable write lease. Missing
        # official player linkage is a validation failure, not an uncertain write.
        _team_player_ids(participants, list(match.get("teamA") or []))
        _team_player_ids(participants, list(match.get("teamB") or []))
        contexts.append(deterministic_match_context_id(operation_key=prefix, slot=match_id))
    return contexts


def admin_jupr_live_pending_operation_key(session: dict[str, Any]) -> str | None:
    state = _as_dict((session or {}).get("state"))
    official = _as_dict(state.get("official_publish"))
    pending = _clean_text(official.get("pending_operation_key"), limit=80)
    return pending or None


def publish_admin_jupr_live_matches(supabase: Any, *, club_id: str, session_key: str, match_date: str | None, actor_email: str, actor_role: str, confirmation_text: str, expected_version: str | None = None, publish_context_prefix: str | None = None, source: str = "next_jupr_live_admin_publish") -> dict[str, Any]:
    if not is_admin_jupr_live_enabled():
        raise PermissionError("Next JUPR Live Admin is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_PUBLISH:
        raise ValueError(f"Type {CONFIRM_PUBLISH} to publish official JUPR Live matches.")
    before = _live_row(supabase, club_id=str(club_id), session_key=str(session_key))
    if str(expected_version or "").strip() and str(before.get("updated_at") or "") != str(expected_version):
        raise ValueError("JUPR Live session changed. Reload the Python state before publishing.")
    state = _state(before)
    event = _event_from_state(state)
    if str(event.get("type") or "") not in {"round_robin", "league"}:
        raise ValueError("Official JUPR Live publish supports one-off round-robin and league/ladder sessions. Use Tournament Live/Ops for brackets.")
    official = _as_dict(state.get("official_publish"))
    if _clean_text(official.get("pending_operation_key"), limit=80):
        raise ValueError(
            "This JUPR Live session has an interrupted official publish. Reconcile that durable operation and "
            "inspect Match Log/Replay History before any new publish."
        )
    published_ids = {str(mid) for mid in (official.get("published_live_match_ids") or official.get("published_match_ids") or [])}
    date_value = _clean_text(match_date, limit=80) or _now_iso()
    payloads = _publish_payloads_from_event(event, session_key=str(session_key), published_ids=published_ids, match_date=date_value, publish_context_prefix=publish_context_prefix)
    if not payloads:
        raise ValueError("No unpublished scored JUPR Live matches are ready to publish.")
    # Claim the session version before the external match batch.  If the batch
    # later becomes uncertain, the durable session visibly retains the pending
    # operation and a stale client cannot publish a second copy.
    reservation_version = _now_iso()
    official["pending_operation_key"] = _clean_text(publish_context_prefix, limit=80) or None
    official["pending_live_match_ids"] = [str(payload.get("live_match_id") or "") for payload in payloads]
    official["publish_started_at"] = reservation_version
    state["official_publish"] = official
    _update_live_row(
        supabase,
        club_id=str(club_id),
        session_key=str(session_key),
        patch={"state": state, "updated_at": reservation_version, "last_seen_at": reservation_version},
        expected_version=expected_version,
    )
    df_players_all, _df_players_active, df_leagues, _df_matches, df_meta, _df_badges, _df_player_badges, name_to_id, _id_to_name, _schema_degraded, _schema_degraded_reason = load_data(supabase, str(club_id), match_limit=5000)
    service_ctx = ServiceContext(supabase=supabase, club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, source="jupr_live_admin")
    result = submit_match_batch(service_ctx, payloads, name_to_id=name_to_id, df_players_all=df_players_all, df_leagues=df_leagues, df_meta=df_meta)
    if not result.ok:
        raise ValueError("; ".join(result.errors) or "Unable to publish JUPR Live matches.")
    newly_published = [str(payload.get("live_match_id")) for payload in payloads if payload.get("live_match_id")]
    official["published_live_match_ids"] = sorted(published_ids | set(newly_published))
    official["published_at"] = _now_iso()
    official["publish_result"] = result.data if isinstance(result.data, dict) else {"result": result.data}
    official.pop("pending_operation_key", None)
    official.pop("pending_live_match_ids", None)
    official.pop("publish_started_at", None)
    state["official_publish"] = official
    patch = {"state": state, "updated_at": _now_iso(), "last_seen_at": _now_iso()}
    updated = _update_live_row(supabase, club_id=str(club_id), session_key=str(session_key), patch=patch, expected_version=reservation_version)
    _audit(supabase, club_id=club_id, actor_email=actor_email, actor_role=actor_role, action_type="publish_jupr_live_matches_admin", entity_id=session_key, before_json={"session": _session_payload(before)}, after_json={"session": _session_payload(updated), "payload_count": len(payloads), "result": official["publish_result"]}, source=source)
    contexts = [str(payload.get("context_id") or "") for payload in payloads]
    return {
        "ok": True,
        "mode": "jupr_live_admin_publish",
        "published_count": len(payloads),
        "result": official["publish_result"],
        "match_context_ids": contexts,
        "session": _session_payload(updated),
        "correction": {
            "match_log_url": f"/admin/match-log?context_type=jupr_live&context_id={contexts[0] if contexts else session_key}",
            "replay_history_url": f"/admin/replay-history?context_type=jupr_live&context_id={contexts[0] if contexts else session_key}",
            "instructions": "Correct official one-off results in Match Log, then run and verify Replay History. Do not republish the session as a correction.",
        },
    }


def update_admin_jupr_live_session_status(supabase: Any, *, club_id: str, session_key: str, status: str, title: str | None, actor_email: str, actor_role: str, confirmation_text: str, expected_version: str | None = None, source: str = "next_jupr_live_admin_status") -> dict[str, Any]:
    if not is_admin_jupr_live_enabled():
        raise PermissionError("Next JUPR Live Admin is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_STATUS:
        raise ValueError(f"Type {CONFIRM_STATUS} to update the JUPR Live session.")
    clean_status = _clean_text(status, limit=40).lower()
    if clean_status not in SESSION_STATUSES:
        raise ValueError("unsupported live session status")
    before = _live_row(supabase, club_id=str(club_id), session_key=str(session_key))
    patch: dict[str, Any] = {"status": clean_status, "updated_at": _now_iso(), "last_seen_at": _now_iso()}
    if title is not None:
        patch["title"] = _clean_text(title, limit=160) or None
    updated = _update_live_row(supabase, club_id=str(club_id), session_key=str(session_key), patch=patch, expected_version=expected_version)
    _audit(supabase, club_id=club_id, actor_email=actor_email, actor_role=actor_role, action_type="update_jupr_live_session_admin", entity_id=session_key, before_json={"session": _session_payload(before)}, after_json={"session": _session_payload(updated)}, source=source)
    return {"ok": True, "mode": "jupr_live_admin_session_update", "session": _session_payload(updated)}


def _audit(supabase: Any, *, club_id: str, actor_email: str, actor_role: str, action_type: str, entity_id: str, before_json: dict[str, Any], after_json: dict[str, Any], source: str) -> None:
    payload = build_activity_payload(club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, action_type=action_type, entity_type="live_session", entity_id=str(entity_id), before_json=before_json, after_json={"source_client": "fastapi/nextjs", **after_json}, source_page=source, flagged_for_review=True)
    write = write_admin_activity_log(supabase, payload)
    if not write.ok and os.getenv("JUPR_REQUIRE_API_AUDIT_LOG", "").strip().lower() in TRUTHY:
        raise RuntimeError("audit log write required but unavailable")
