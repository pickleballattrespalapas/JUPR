from __future__ import annotations

import copy
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from jupr_app.data.load import load_data
from jupr_app.domain.adaptive_play_engine import (
    advance_generator_event,
    create_generator_preview,
    generator_event_standings,
    mark_generator_round_played,
    mutate_generator_roster,
    save_generator_round,
    schedule_export_rows,
    skip_generator_round,
    start_generator_event,
)
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_jupr_live_service import (
    JUPR_LIVE_WRITE_FLAG,
    is_admin_jupr_live_enabled,
)
from jupr_app.services.admin_live_ladder_operation_service import (
    deterministic_match_context_id,
    is_staging_write_gate_enabled,
)
from jupr_app.services.direct_match_entry_service import submit_atomic_direct_matches


GENERATOR_KINDS = {"round_robin", "ladder"}
PLAY_FORMATS = {"singles", "doubles"}


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
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _state(row: dict[str, Any]) -> dict[str, Any]:
    return _as_dict(row.get("state"))


def _event_from_state(state: dict[str, Any]) -> dict[str, Any]:
    page_state = _as_dict(state.get("page_state"))
    event = page_state.get("event")
    return copy.deepcopy(event) if isinstance(event, dict) else {}


def _put_event(state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    next_state = _as_dict(state)
    page_state = _as_dict(next_state.get("page_state"))
    page_state.update(
        {
            "event": copy.deepcopy(event),
            "event_name": str(event.get("name") or "Play session"),
            "event_type": str(event.get("type") or ""),
            "type_label": (
                "Round-Robin Generator"
                if str(event.get("generatorKind")) == "round_robin"
                else "Ladder Generator"
            ),
            "play_format": str(event.get("playFormat") or "doubles"),
            "current_round_number": int(event.get("currentRoundNumber") or 1),
            "participant_count": len(event.get("participants") or []),
        }
    )
    next_state["page_state"] = page_state
    next_state["event_name"] = page_state["event_name"]
    next_state["event_type"] = page_state["event_type"]
    next_state["generator_kind"] = str(event.get("generatorKind") or "")
    next_state["play_format"] = str(event.get("playFormat") or "")
    return next_state


def _base_state(*, club_id: str, event: dict[str, Any], source: str) -> dict[str, Any]:
    state = {
        "version": 3,
        "mode": "admin_play_generator",
        "source": source,
        "club_id": str(club_id),
        "generator_kind": str(event.get("generatorKind") or ""),
        "play_format": str(event.get("playFormat") or ""),
        "page_state": {},
        "official_publish": {
            "published_match_ids": [],
            "published_at": None,
            "publish_result": None,
        },
    }
    return _put_event(state, event)


def _session_payload(row: dict[str, Any]) -> dict[str, Any]:
    state = _state(row)
    event = _event_from_state(state)
    return {
        "id": str(row.get("id") or ""),
        "club_id": str(row.get("club_id") or ""),
        "session_key": str(row.get("session_key") or ""),
        "title": _clean_text(row.get("title") or event.get("name"), limit=160),
        "status": str(row.get("status") or event.get("status") or "active"),
        "source": str(row.get("source") or "play_generator"),
        "created_by_email": row.get("created_by_email"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "version": str(row.get("updated_at") or row.get("created_at") or ""),
        "last_seen_at": row.get("last_seen_at"),
        "expires_at": row.get("expires_at"),
        "generator_kind": str(event.get("generatorKind") or state.get("generator_kind") or ""),
        "play_format": str(event.get("playFormat") or state.get("play_format") or ""),
        "current_round_number": int(event.get("currentRoundNumber") or 1) if event else None,
        "total_rounds": int(event.get("totalRounds") or 0) if event else None,
        "event": event,
        "schedule_rows": schedule_export_rows(event) if event else [],
        "scoring_mode": str(event.get("scoringMode") or "scored") if event else "scored",
        "standings_sort": str(event.get("standingsSort") or "wins") if event else "wins",
        "standings": generator_event_standings(event) if event else [],
        "official_publish": _as_dict(state.get("official_publish")),
    }


def _is_generator_row(row: dict[str, Any], generator_kind: str | None = None) -> bool:
    state = _state(row)
    if str(state.get("mode") or "") != "admin_play_generator":
        return False
    if generator_kind:
        return str(state.get("generator_kind") or "") == str(generator_kind)
    return True


def build_play_generator_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_jupr_live_enabled():
        return {
            "enabled": False,
            "writes_enabled": False,
            "status": "guarded_off",
            "warnings": ["Round-Robin and Ladder Generator administration is disabled."],
            "counts": {"round_robin": 0, "ladder": 0, "active": 0, "completed": 0},
        }
    counts = {"round_robin": 0, "ladder": 0, "active": 0, "completed": 0}
    if supabase is not None:
        try:
            rows = _safe_rows(
                supabase.table("live_sessions")
                .select("club_id,status,state")
                .eq("club_id", str(club_id))
                .limit(500)
                .execute()
            )
        except Exception:
            rows = []
        for row in rows:
            if not _is_generator_row(row):
                continue
            state = _state(row)
            kind = str(state.get("generator_kind") or "")
            if kind in counts:
                counts[kind] += 1
            status = str(row.get("status") or "")
            if status in counts:
                counts[status] += 1
    writes_enabled = is_staging_write_gate_enabled(JUPR_LIVE_WRITE_FLAG)
    return {
        "enabled": True,
        "writes_enabled": writes_enabled,
        "status": "ready_for_play_generators" if writes_enabled else "read_only",
        "warnings": [] if writes_enabled else ["Generator writes are disabled in this environment."],
        "counts": counts,
        "authority": "python_fastapi",
    }


def _player_rows_by_id(
    supabase: Any,
    *,
    club_id: str,
    player_ids: list[int],
) -> dict[int, dict[str, Any]]:
    wanted = {int(value) for value in player_ids if value is not None}
    if not wanted:
        return {}
    try:
        rows = _safe_rows(
            supabase.table("players")
            .select("id,name,club_id")
            .eq("club_id", str(club_id))
            .in_("id", sorted(wanted))
            .execute()
        )
    except Exception:
        rows = []
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            pid = int(row.get("id"))
        except Exception:
            continue
        if pid in wanted:
            result[pid] = row
    return result


def _resolve_names_and_ids(
    supabase: Any,
    *,
    club_id: str,
    participant_names: list[str],
    player_ids: list[int] | None,
) -> tuple[list[str], list[int]]:
    names = [_clean_text(value, limit=160) for value in participant_names or []]
    names = [value for value in names if value]
    ids = [int(value) for value in (player_ids or []) if value is not None]
    if ids and len(ids) != len(names):
        raise ValueError("Official player IDs and participant names must have the same length.")
    if not ids:
        return names, []
    players = _player_rows_by_id(supabase, club_id=str(club_id), player_ids=ids)
    missing = [pid for pid in ids if pid not in players]
    if missing:
        raise ValueError(f"Player id {missing[0]} was not found in this club.")
    return names, ids


def preview_play_generator(
    supabase: Any,
    *,
    club_id: str,
    generator_kind: str,
    play_format: str,
    title: str,
    participant_names: list[str],
    player_ids: list[int] | None,
    total_rounds: int,
    court_count: int,
    standings_sort: str = "wins",
    scoring_mode: str = "scored",
) -> dict[str, Any]:
    names, ids = _resolve_names_and_ids(
        supabase,
        club_id=str(club_id),
        participant_names=participant_names,
        player_ids=player_ids,
    )
    event = create_generator_preview(
        generator_kind=generator_kind,
        play_format=play_format,
        title=title,
        participant_names=names,
        player_ids=ids,
        total_rounds=total_rounds,
        court_count=court_count,
        standings_sort=standings_sort,
        scoring_mode=scoring_mode,
    )
    return {
        "ok": True,
        "mode": "play_generator_preview",
        "preview": event,
        "schedule_rows": schedule_export_rows(event),
    }


def _audit(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action_type: str,
    entity_id: str,
    before_json: dict[str, Any],
    after_json: dict[str, Any],
    source: str,
) -> None:
    payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type=action_type,
        entity_type="play_generator_session",
        entity_id=str(entity_id),
        before_json=before_json,
        after_json=after_json,
        source_page=source,
    )
    write_admin_activity_log(supabase, payload)


def create_play_generator_session(
    supabase: Any,
    *,
    club_id: str,
    generator_kind: str,
    play_format: str,
    title: str,
    participant_names: list[str],
    player_ids: list[int] | None,
    total_rounds: int,
    court_count: int,
    preview_fingerprint: str | None,
    actor_email: str,
    actor_role: str,
    source: str,
    standings_sort: str = "wins",
    scoring_mode: str = "scored",
) -> dict[str, Any]:
    preview = preview_play_generator(
        supabase,
        club_id=str(club_id),
        generator_kind=generator_kind,
        play_format=play_format,
        title=title,
        participant_names=participant_names,
        player_ids=player_ids,
        total_rounds=total_rounds,
        court_count=court_count,
        standings_sort=standings_sort,
        scoring_mode=scoring_mode,
    )["preview"]
    supplied = _clean_text(preview_fingerprint, limit=128)
    if supplied and supplied != str(preview.get("previewFingerprint") or ""):
        raise ValueError("The roster or settings changed after preview. Preview the schedule again before starting.")
    event = start_generator_event(preview)
    session_key = uuid4().hex
    now = _now_iso()
    clean_title = _clean_text(title, limit=160) or str(event.get("name") or "Play session")
    payload = {
        "club_id": str(club_id),
        "session_key": session_key,
        "status": "active",
        "title": clean_title,
        "state": _base_state(club_id=str(club_id), event=event, source=source),
        "source": "play_generator",
        "created_by_email": str(actor_email or "").strip().lower() or None,
        "created_at": now,
        "updated_at": now,
        "last_seen_at": now,
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
    }
    inserted = _safe_first(supabase.table("live_sessions").insert(payload).execute()) or payload
    session = _session_payload(inserted)
    _audit(
        supabase,
        club_id=club_id,
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="create_play_generator_session",
        entity_id=session_key,
        before_json={},
        after_json={"session": session},
        source=source,
    )
    return {"ok": True, "mode": "play_generator_start", "session": session}


def list_play_generator_sessions(
    supabase: Any,
    *,
    club_id: str,
    generator_kind: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    query = (
        supabase.table("live_sessions")
        .select("*")
        .eq("club_id", str(club_id))
    )
    clean_status = _clean_text(status, limit=40).lower()
    if clean_status:
        query = query.eq("status", clean_status)
    try:
        rows = _safe_rows(
            query.order("updated_at", desc=True)
            .limit(max(1, min(int(limit or 100), 300)))
            .execute()
        )
    except Exception:
        rows = _safe_rows(query.execute())
    clean_kind = _clean_text(generator_kind, limit=40).lower() or None
    sessions = [
        _session_payload(row)
        for row in rows
        if _is_generator_row(row, clean_kind)
    ]
    return {
        "ok": True,
        "mode": "play_generator_sessions",
        "sessions": sessions,
        "count": len(sessions),
    }


def _live_row(supabase: Any, *, club_id: str, session_key: str) -> dict[str, Any]:
    row = _safe_first(
        supabase.table("live_sessions")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("session_key", str(session_key))
        .limit(1)
        .execute()
    )
    if row is None or not _is_generator_row(row):
        raise ValueError("Generator session not found.")
    return row


def get_play_generator_session(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
) -> dict[str, Any]:
    return {
        "ok": True,
        "mode": "play_generator_session_detail",
        "session": _session_payload(
            _live_row(supabase, club_id=str(club_id), session_key=str(session_key))
        ),
    }


def _update_live_row(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    patch: dict[str, Any],
    expected_version: str | None,
) -> dict[str, Any]:
    query = (
        supabase.table("live_sessions")
        .update(patch)
        .eq("club_id", str(club_id))
        .eq("session_key", str(session_key))
    )
    if str(expected_version or "").strip():
        query = query.eq("updated_at", str(expected_version))
    updated = _safe_first(query.execute())
    if updated is None and str(expected_version or "").strip():
        raise ValueError("This generator session changed. Reload it before saving.")
    return updated or {
        "club_id": str(club_id),
        "session_key": str(session_key),
        **patch,
    }


def _persist_event(
    supabase: Any,
    *,
    before: dict[str, Any],
    event: dict[str, Any],
    expected_version: str,
    status: str | None = None,
) -> dict[str, Any]:
    state = _put_event(_state(before), event)
    now = _now_iso()
    patch = {
        "state": state,
        "updated_at": now,
        "last_seen_at": now,
    }
    if status:
        patch["status"] = status
    if status == "completed":
        patch["completed_at"] = now
    return _update_live_row(
        supabase,
        club_id=str(before.get("club_id") or ""),
        session_key=str(before.get("session_key") or ""),
        patch=patch,
        expected_version=expected_version,
    )


def save_play_generator_round(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    round_number: int,
    scores: list[dict[str, Any]],
    expected_version: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    before = _live_row(supabase, club_id=str(club_id), session_key=str(session_key))
    event = save_generator_round(
        _event_from_state(_state(before)),
        round_number=int(round_number),
        scores=scores,
    )
    updated = _persist_event(
        supabase,
        before=before,
        event=event,
        expected_version=expected_version,
    )
    session = _session_payload(updated)
    _audit(
        supabase,
        club_id=club_id,
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="save_play_generator_round",
        entity_id=session_key,
        before_json={"session": _session_payload(before)},
        after_json={"session": session, "round_number": int(round_number)},
        source=source,
    )
    return {"ok": True, "mode": "play_generator_round_scores", "session": session}



def mark_play_generator_round_played(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    round_number: int,
    expected_version: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    before = _live_row(supabase, club_id=str(club_id), session_key=str(session_key))
    event = mark_generator_round_played(
        _event_from_state(_state(before)),
        round_number=int(round_number),
    )
    updated = _persist_event(
        supabase,
        before=before,
        event=event,
        expected_version=expected_version,
    )
    session = _session_payload(updated)
    _audit(
        supabase,
        club_id=club_id,
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="mark_play_generator_round_played",
        entity_id=session_key,
        before_json={"session": _session_payload(before)},
        after_json={"session": session, "round_number": int(round_number)},
        source=source,
    )
    return {"ok": True, "mode": "play_generator_round_played", "session": session}


def skip_play_generator_round(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    round_number: int,
    reason: str,
    expected_version: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    before = _live_row(supabase, club_id=str(club_id), session_key=str(session_key))
    event = skip_generator_round(
        _event_from_state(_state(before)),
        round_number=int(round_number),
        reason=reason,
    )
    updated = _persist_event(
        supabase,
        before=before,
        event=event,
        expected_version=expected_version,
    )
    session = _session_payload(updated)
    _audit(
        supabase,
        club_id=club_id,
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="skip_play_generator_round",
        entity_id=session_key,
        before_json={"session": _session_payload(before)},
        after_json={"session": session, "round_number": int(round_number), "reason": reason},
        source=source,
    )
    return {"ok": True, "mode": "play_generator_round_skip", "session": session}


def advance_play_generator_session(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    expected_version: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    before = _live_row(supabase, club_id=str(club_id), session_key=str(session_key))
    event = advance_generator_event(_event_from_state(_state(before)))
    row_status = "completed" if str(event.get("status")) == "completed" else None
    updated = _persist_event(
        supabase,
        before=before,
        event=event,
        expected_version=expected_version,
        status=row_status,
    )
    session = _session_payload(updated)
    _audit(
        supabase,
        club_id=club_id,
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="advance_play_generator_session",
        entity_id=session_key,
        before_json={"session": _session_payload(before)},
        after_json={"session": session},
        source=source,
    )
    return {"ok": True, "mode": "play_generator_advance", "session": session}


def mutate_play_generator_roster(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    action: str,
    participant_id: str | None,
    name: str | None,
    player_id: int | None,
    substitute_scope: str,
    roster_order: list[str] | None,
    expected_version: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    before = _live_row(supabase, club_id=str(club_id), session_key=str(session_key))
    if player_id is not None:
        _resolve_names_and_ids(
            supabase,
            club_id=str(club_id),
            participant_names=[_clean_text(name, limit=160) or "Player"],
            player_ids=[int(player_id)],
        )
    event = mutate_generator_roster(
        _event_from_state(_state(before)),
        action=action,
        participant_id=participant_id,
        name=name,
        player_id=player_id,
        substitute_scope=substitute_scope,
        roster_order=roster_order,
    )
    updated = _persist_event(
        supabase,
        before=before,
        event=event,
        expected_version=expected_version,
    )
    session = _session_payload(updated)
    _audit(
        supabase,
        club_id=club_id,
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="mutate_play_generator_roster",
        entity_id=session_key,
        before_json={"session": _session_payload(before)},
        after_json={"session": session, "action": action},
        source=source,
    )
    return {"ok": True, "mode": "play_generator_roster", "session": session}


def complete_play_generator_session(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    expected_version: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    before = _live_row(supabase, club_id=str(club_id), session_key=str(session_key))
    event = _event_from_state(_state(before))
    current = int(event.get("currentRoundNumber") or 1)
    current_row = next(
        (row for row in event.get("rounds") or [] if int(row.get("number") or 0) == current),
        None,
    )
    if current_row and str(current_row.get("status")) not in {"saved", "played", "skipped"}:
        raise ValueError("Save scores, mark the round played, or skip it before completing the session.")
    event["status"] = "completed"
    event["completedAt"] = _now_iso()
    updated = _persist_event(
        supabase,
        before=before,
        event=event,
        expected_version=expected_version,
        status="completed",
    )
    session = _session_payload(updated)
    _audit(
        supabase,
        club_id=club_id,
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="complete_play_generator_session",
        entity_id=session_key,
        before_json={"session": _session_payload(before)},
        after_json={"session": session},
        source=source,
    )
    return {"ok": True, "mode": "play_generator_complete", "session": session}


def _saved_matches(event: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for round_row in event.get("rounds") or []:
        if str(round_row.get("status")) != "saved":
            continue
        number = int(round_row.get("number") or 0)
        matches = list(round_row.get("matches") or [])
        if not matches:
            matches = [
                match
                for court in round_row.get("courts") or []
                for match in court.get("matches") or []
            ]
        for match in matches:
            if match.get("scoreA") is None or match.get("scoreB") is None:
                continue
            rows.append((number, dict(match)))
    return rows


def _publish_payloads(
    event: dict[str, Any],
    *,
    session_key: str,
    match_date: str,
    operation_key: str,
    published_ids: set[str],
) -> list[dict[str, Any]]:
    participants = {
        str(row.get("id")): row
        for row in event.get("participants") or []
        if row.get("id")
    }
    play_format = str(event.get("playFormat") or "doubles")
    generator_kind = str(event.get("generatorKind") or "round_robin")
    context_type = (
        "round_robin_generator"
        if generator_kind == "round_robin"
        else "ladder_generator"
    )
    match_type = (
        "Round-Robin Generator"
        if generator_kind == "round_robin"
        else "Ladder Generator"
    )
    payloads: list[dict[str, Any]] = []
    for round_number, match in _saved_matches(event):
        match_id = str(match.get("id") or "")
        if not match_id or match_id in published_ids:
            continue
        side_a = [str(value) for value in match.get("sideA") or match.get("teamA") or []]
        side_b = [str(value) for value in match.get("sideB") or match.get("teamB") or []]
        expected_side = 1 if play_format == "singles" else 2
        if len(side_a) != expected_side or len(side_b) != expected_side:
            raise ValueError(f"Match {match_id} does not match the selected play format.")
        try:
            ids_a = [int(participants[pid]["player_id"]) for pid in side_a]
            ids_b = [int(participants[pid]["player_id"]) for pid in side_b]
        except Exception as exc:
            raise ValueError(
                "Official publication requires every participant to be linked to an official player ID."
            ) from exc
        score_a = int(match.get("scoreA") or 0)
        score_b = int(match.get("scoreB") or 0)
        if score_a == score_b:
            raise ValueError(f"Match {match_id} is tied.")
        base = {
            "date": match_date,
            "league": "OVERALL",
            "match_type": match_type,
            "week_tag": f"{_clean_text(event.get('name'), limit=80)} Round {round_number}",
            "is_popup": False,
            "context_type": context_type,
            "context_id": deterministic_match_context_id(
                operation_key=str(operation_key),
                slot=f"{session_key}:{match_id}",
            ),
            "live_match_id": match_id,
            "match_format": play_format,
            "s1": score_a,
            "s2": score_b,
            "score_t1": score_a,
            "score_t2": score_b,
        }
        if play_format == "singles":
            base.update({"t1_p1": ids_a[0], "t2_p1": ids_b[0]})
        else:
            base.update(
                {
                    "t1_p1": ids_a[0],
                    "t1_p2": ids_a[1],
                    "t2_p1": ids_b[0],
                    "t2_p2": ids_b[1],
                }
            )
        payloads.append(base)
    return payloads


def publish_play_generator_matches(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    match_date: str | None,
    expected_version: str,
    idempotency_key: str,
    operation_key: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    before = _live_row(supabase, club_id=str(club_id), session_key=str(session_key))
    if str(before.get("updated_at") or "") != str(expected_version or ""):
        raise ValueError("This generator session changed. Reload it before publishing.")
    state = _state(before)
    event = _event_from_state(state)
    official = _as_dict(state.get("official_publish"))
    published_ids = {
        str(value)
        for value in official.get("published_match_ids") or []
        if str(value)
    }
    date_value = _clean_text(match_date, limit=80) or _now_iso()
    payloads = _publish_payloads(
        event,
        session_key=str(session_key),
        match_date=date_value,
        operation_key=operation_key,
        published_ids=published_ids,
    )
    if not payloads:
        raise ValueError("No unpublished saved matches are ready to publish.")

    (
        df_players_all,
        _df_players_active,
        df_leagues,
        _df_matches,
        df_meta,
        _df_badges,
        _df_player_badges,
        name_to_id,
        _id_to_name,
        _schema_degraded,
        _schema_degraded_reason,
    ) = load_data(supabase, str(club_id))

    play_format = str(event.get("playFormat") or "doubles")
    result = submit_atomic_direct_matches(
        supabase,
        club_id=str(club_id),
        matches=payloads,
        match_format=play_format,
        idempotency_key=str(idempotency_key),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        source=source,
        name_to_id=name_to_id,
        df_players_all=df_players_all,
        df_leagues=df_leagues if play_format == "doubles" else None,
        df_meta=df_meta,
    )
    newly_published = [str(payload["live_match_id"]) for payload in payloads]
    official["published_match_ids"] = sorted(published_ids.union(newly_published))
    official["published_at"] = _now_iso()
    official["publish_result"] = result
    state["official_publish"] = official
    now = _now_iso()
    updated = _update_live_row(
        supabase,
        club_id=str(club_id),
        session_key=str(session_key),
        patch={"state": state, "updated_at": now, "last_seen_at": now},
        expected_version=expected_version,
    )
    session = _session_payload(updated)
    _audit(
        supabase,
        club_id=club_id,
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="publish_play_generator_matches",
        entity_id=session_key,
        before_json={"session": _session_payload(before)},
        after_json={"session": session, "published_count": len(payloads)},
        source=source,
    )
    return {
        "ok": True,
        "mode": "play_generator_publish",
        "published_count": len(payloads),
        "session": session,
        "result": result,
    }
