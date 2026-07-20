from __future__ import annotations

import base64
import copy
import csv
import hashlib
import hmac
import io
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd

from jupr_app.domain.live_beta_engine import (
    apply_round_substitution,
    apply_single_game_substitution,
    create_league_event,
    create_round_robin_event,
    find_match_by_id,
    is_league_round_complete,
    match_is_scored,
    matches_for_round,
    round_robin_matches,
    set_match_score,
    start_next_league_round,
)
from jupr_app.domain.live_social import normalize_skill_levels
from jupr_app.domain.live_social_submit import (
    _find_strong_duplicate_candidates,
    save_resolved_social_live_event,
)
from jupr_app.services.public_live_operation_service import (
    PublicLiveConflictError,
    PublicLiveRecoveryRequiredError,
    begin_public_live_operation,
    canonical_fingerprint,
    claim_public_live_completion_executor,
    completed_operation_result,
    edit_token_matches,
    hash_edit_token,
    update_public_live_operation,
)
from jupr_app.services.public_live_service import is_public_live_session_row, public_live_session_detail


PUBLIC_LIVE_SESSION_TTL_HOURS = 24
MIN_PUBLIC_RR_PLAYERS = 4
MAX_PUBLIC_RR_PLAYERS = 20
SUPPORTED_LIVE_MODES = {"quick", "club_social"}
SUPPORTED_PUBLIC_EVENT_TYPES = {"round_robin", "league_ladder"}


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


def _normalize_event_type(value: Any) -> str:
    clean = str(value or "round_robin").strip().lower().replace(" ", "_").replace("/", "_")
    if clean in {"league", "ladder", "league__ladder", "league_ladder"}:
        return "league_ladder"
    if clean in {"round_robin", "roundrobin"}:
        return "round_robin"
    raise PublicLiveSessionError("Public JUPR Live supports Round Robin or League / Ladder events.")


def _normalize_live_mode(value: Any) -> str:
    clean = str(value or "quick").strip().lower().replace(" ", "_")
    if clean not in SUPPORTED_LIVE_MODES:
        raise PublicLiveSessionError("Public JUPR Live mode must be Quick Session or Club Social.")
    return clean


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


def _token_secret(explicit: str | None = None) -> str:
    secret = str(explicit or os.getenv("JUPR_PUBLIC_LIVE_TOKEN_SECRET", "")).strip()
    if len(secret) < 32:
        raise RuntimeError(
            "JUPR_PUBLIC_LIVE_TOKEN_SECRET must be a stable server-only secret of at least 32 characters."
        )
    return secret


def _deterministic_edit_token(*, secret: str, operation_key: str) -> str:
    digest = hmac.new(
        secret.encode("utf-8"),
        f"jupr:public-live-edit:v1:{operation_key}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _state_from_event(
    *,
    club_id: str,
    session_key: str,
    event: dict[str, Any],
    participant_names: list[str],
    live_mode: str,
    host_name: str | None,
    skill_levels: list[str],
) -> dict[str, Any]:
    event_name = str(event.get("name") or "JUPR Live Session")
    return {
        "version": 2,
        "mode": "public_club_social" if live_mode == "club_social" else "public_quick_session",
        "source": "public_web",
        "club_id": str(club_id),
        "session_key": str(session_key),
        "event_name": event_name,
        "event_type": str(event.get("type") or "round_robin"),
        "social": {
            "enabled": live_mode == "club_social",
            "host_name": str(host_name or "")[:160] or None,
            "skill_levels": skill_levels,
            "submission": None,
        },
        "page_state": {
            "event": event,
            "event_name": event_name,
            "type_label": "League / Ladder" if str(event.get("type")) == "league" else "Round Robin",
            "participant_count": len(participant_names),
            "participant_text": "\n".join(participant_names),
            "rating_mode": "Unrated",
            "live_session_key": str(session_key),
        },
        "widget_state": {},
    }


def _state(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("state")
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _event_from_state(state: dict[str, Any]) -> dict[str, Any]:
    page_state = state.get("page_state")
    if not isinstance(page_state, dict):
        return {}
    event = page_state.get("event")
    return copy.deepcopy(event) if isinstance(event, dict) else {}


def _put_event(state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    next_state = copy.deepcopy(state)
    page_state = next_state.setdefault("page_state", {})
    if not isinstance(page_state, dict):
        page_state = {}
        next_state["page_state"] = page_state
    page_state["event"] = event
    next_state["event_name"] = str(event.get("name") or next_state.get("event_name") or "JUPR Live Session")
    next_state["event_type"] = str(event.get("type") or next_state.get("event_type") or "round_robin")
    return next_state


LIVE_SESSION_SELECT = (
    "club_id,session_key,title,status,state,version,edit_token_hash,creation_operation_key,"
    "last_operation_key,last_request_fingerprint,pending_operation_key,pending_operation_action,"
    "created_at,updated_at,last_seen_at,expires_at,completed_at"
)


def get_public_live_session_row(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
) -> dict[str, Any] | None:
    return _safe_first(
        supabase.table("live_sessions")
        .select(LIVE_SESSION_SELECT)
        .eq("club_id", str(club_id))
        .eq("session_key", str(session_key))
        .limit(1)
        .execute()
    )


def _get_creation_row(supabase: Any, *, club_id: str, operation_key: str) -> dict[str, Any] | None:
    return _safe_first(
        supabase.table("live_sessions")
        .select(LIVE_SESSION_SELECT)
        .eq("club_id", str(club_id))
        .eq("creation_operation_key", str(operation_key))
        .limit(1)
        .execute()
    )


def _event_for_create(
    *,
    event_type: str,
    title: str,
    names: list[str],
    total_rounds: int,
    court_sizes: list[int] | None,
    resolved_ids: dict[str, int] | None = None,
) -> dict[str, Any]:
    if event_type == "round_robin":
        if len(names) < MIN_PUBLIC_RR_PLAYERS:
            raise PublicLiveSessionError("Enter at least 4 unique player names.")
        if len(names) > MAX_PUBLIC_RR_PLAYERS:
            raise PublicLiveSessionError("Public JUPR Live supports up to 20 players.")
        return create_round_robin_event(name=title, participant_names=names, resolved_ids=resolved_ids)
    if len(names) < 4 or len(names) > MAX_PUBLIC_RR_PLAYERS:
        raise PublicLiveSessionError("League / Ladder supports 4 to 20 unique players.")
    try:
        return create_league_event(
            name=title,
            participant_names=names,
            total_rounds=max(1, min(int(total_rounds or 3), 20)),
            court_sizes=[int(value) for value in (court_sizes or [])] or None,
            resolved_ids=resolved_ids,
        )
    except ValueError as exc:
        raise PublicLiveSessionError(str(exc)) from exc


def _resolve_public_participant_ids(
    supabase: Any,
    *,
    club_id: str,
    names: list[str],
    requested_links: dict[str, int] | None,
    require_social_duplicate_preflight: bool,
) -> dict[str, int]:
    requested = {
        _normalize_name(name).casefold(): int(player_id)
        for name, player_id in (requested_links or {}).items()
        if _normalize_name(name)
    }
    if not requested and not require_social_duplicate_preflight:
        return {}
    try:
        player_rows = _safe_rows(
            supabase.table("players")
            .select("id,club_id,name,active,inactive_at")
            .eq("club_id", str(club_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Current-player validation is unavailable; no live session was created.") from exc
    players_by_id = {
        int(row["id"]): row
        for row in player_rows
        if row.get("id") is not None and str(row.get("id")).strip()
    }
    ctx = SimpleNamespace(df_players_all=pd.DataFrame(player_rows))
    resolved: dict[str, int] = {}
    for name in names:
        normalized = _normalize_name(name).casefold()
        requested_id = requested.get(normalized)
        if requested_id is not None:
            player = players_by_id.get(int(requested_id))
            if player is None or _normalize_name(player.get("name")).casefold() != normalized:
                raise PublicLiveSessionError(
                    f"The selected current-player profile for '{name}' is stale. Search and select it again."
                )
            resolved[name] = int(requested_id)
            continue
        if require_social_duplicate_preflight:
            duplicates = _find_strong_duplicate_candidates(ctx, display_name=name)
            if duplicates:
                raise PublicLiveSessionError(
                    f"'{name}' closely matches current player '{duplicates[0].get('name')}'. "
                    "Add that player through current-player search before creating Club Social."
                )
    return resolved


def create_public_live_session(
    supabase: Any,
    *,
    club_id: str,
    event_name: str,
    event_type: str,
    participant_names: list[Any],
    live_mode: str,
    total_rounds: int,
    court_sizes: list[int] | None,
    host_name: str | None,
    skill_levels: list[str] | None,
    participant_player_ids: dict[str, int] | None = None,
    idempotency_key: str,
    requester_hash: str,
    token_secret: str | None = None,
) -> dict[str, Any]:
    names = normalize_public_participant_names(participant_names)
    normalized_type = _normalize_event_type(event_type)
    normalized_mode = _normalize_live_mode(live_mode)
    clean_host = _normalize_name(host_name)[:160]
    if normalized_mode == "club_social" and not clean_host:
        raise PublicLiveSessionError("Host / Submitter Name is required for Club Social.")
    levels = normalize_skill_levels(skill_levels, default_all=normalized_mode == "club_social")
    title = _normalize_name(event_name)[:160] or (
        "JUPR Live League" if normalized_type == "league_ladder" else "JUPR Live Round Robin"
    )
    request_payload = {
        "event_name": title,
        "event_type": normalized_type,
        "participant_names": names,
        "live_mode": normalized_mode,
        "total_rounds": max(1, min(int(total_rounds or 3), 20)),
        "court_sizes": [int(value) for value in (court_sizes or [])],
        "host_name": clean_host or None,
        "skill_levels": levels,
    }
    provided_links = {
        _normalize_name(name): int(player_id)
        for name, player_id in (participant_player_ids or {}).items()
        if normalized_mode == "club_social" and _normalize_name(name)
    }
    request_payload["participant_player_ids"] = provided_links
    event = _event_for_create(
        event_type=normalized_type,
        title=title,
        names=names,
        total_rounds=request_payload["total_rounds"],
        court_sizes=request_payload["court_sizes"],
    )
    secret = _token_secret(token_secret)
    operation, existed = begin_public_live_operation(
        supabase,
        club_id=str(club_id),
        session_key=None,
        action="create",
        idempotency_key=idempotency_key,
        requester_hash=requester_hash,
        expected_version=None,
        request_payload=request_payload,
    )
    operation_key = str(operation.get("operation_key") or "")
    session_key = operation_key[:32]
    edit_token = _deterministic_edit_token(secret=secret, operation_key=operation_key)
    existing_row = _get_creation_row(supabase, club_id=str(club_id), operation_key=operation_key)
    if existing_row is not None:
        result = {"edit_token": edit_token, "session": public_live_session_detail(existing_row)}
        if str(operation.get("status") or "") != "completed":
            update_public_live_operation(
                supabase,
                club_id=str(club_id),
                operation_key_value=operation_key,
                status="completed",
                result={},
            )
        return {**result, "idempotent_replay": bool(existed)}
    if existed and completed_operation_result(operation) is not None:
        raise PublicLiveRecoveryRequiredError(
            "The create operation completed but its session row is unavailable. Stop and contact an administrator."
        )

    try:
        resolved_ids = _resolve_public_participant_ids(
            supabase,
            club_id=str(club_id),
            names=names,
            requested_links=provided_links,
            require_social_duplicate_preflight=normalized_mode == "club_social",
        )
    except PublicLiveSessionError as exc:
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="rejected",
            error_text=str(exc),
        )
        raise
    except Exception as exc:
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="recovery_required",
            error_text=str(exc),
        )
        raise
    for participant in event.get("participants") or []:
        participant_name = str(participant.get("name") or "")
        if participant_name in resolved_ids:
            participant["player_id"] = int(resolved_ids[participant_name])

    state = _state_from_event(
        club_id=str(club_id),
        session_key=session_key,
        event=event,
        participant_names=names,
        live_mode=normalized_mode,
        host_name=clean_host or None,
        skill_levels=levels,
    )
    now = _now_iso()
    row_payload = {
        "club_id": str(club_id),
        "session_key": session_key,
        "title": title,
        "status": "active",
        "state": state,
        "source": "public_web",
        "version": 1,
        "edit_token_hash": hash_edit_token(edit_token),
        "creation_operation_key": operation_key,
        "last_operation_key": operation_key,
        "last_request_fingerprint": str(operation.get("request_fingerprint") or ""),
        "updated_at": now,
        "last_seen_at": now,
        "expires_at": _expires_at_iso(),
    }
    update_public_live_operation(
        supabase,
        club_id=str(club_id),
        operation_key_value=operation_key,
        status="running",
    )
    try:
        created = _safe_first(supabase.table("live_sessions").insert(row_payload).execute())
    except Exception as exc:
        created = _get_creation_row(supabase, club_id=str(club_id), operation_key=operation_key)
        if created is None:
            update_public_live_operation(
                supabase,
                club_id=str(club_id),
                operation_key_value=operation_key,
                status="recovery_required",
                error_text=str(exc),
            )
            raise PublicLiveRecoveryRequiredError(
                "JUPR Live creation did not return a verified outcome. Retry the same create request to reconcile it."
            ) from exc
    if created is None:
        raise PublicLiveRecoveryRequiredError("JUPR Live creation returned no recoverable session row.")
    result = {"edit_token": edit_token, "session": public_live_session_detail(created)}
    update_public_live_operation(
        supabase,
        club_id=str(club_id),
        operation_key_value=operation_key,
        status="completed",
        result={},
    )
    return {**result, "idempotent_replay": False}


def create_public_round_robin_session(
    supabase: Any,
    *,
    club_id: str,
    event_name: str,
    participant_names: list[Any],
    idempotency_key: str,
    requester_hash: str,
    token_secret: str | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for the original public Round Robin service."""

    return create_public_live_session(
        supabase,
        club_id=club_id,
        event_name=event_name,
        event_type="round_robin",
        participant_names=participant_names,
        live_mode="quick",
        total_rounds=1,
        court_sizes=None,
        host_name=None,
        skill_levels=None,
        idempotency_key=idempotency_key,
        requester_hash=requester_hash,
        token_secret=token_secret,
    )


def _validate_edit_token(row: dict[str, Any] | None, *, edit_token: str) -> dict[str, Any]:
    if not row:
        raise PublicLiveSessionError("Live session not found.")
    if not is_public_live_session_row(row):
        raise PublicLiveSessionError("This live session is not available for public editing.")
    if not edit_token_matches(edit_token, str(row.get("edit_token_hash") or "")):
        raise PermissionError("Invalid edit token for this public live session.")
    return row


def _validate_editable_row(row: dict[str, Any] | None, *, edit_token: str) -> dict[str, Any]:
    validated = _validate_edit_token(row, edit_token=edit_token)
    if str(validated.get("pending_operation_key") or ""):
        raise PublicLiveRecoveryRequiredError(
            "This live session has an unfinished completion. Retry that completion before making another change."
        )
    return validated


def _update_row_with_cas(
    supabase: Any,
    *,
    row: dict[str, Any],
    patch: dict[str, Any],
    operation: dict[str, Any],
) -> dict[str, Any]:
    club_id = str(row.get("club_id") or "")
    session_key = str(row.get("session_key") or "")
    expected_version = int(row.get("version") or 1)
    operation_key = str(operation.get("operation_key") or "")
    fingerprint = str(operation.get("request_fingerprint") or "")
    payload = {
        **patch,
        "version": expected_version + 1,
        "last_operation_key": operation_key,
        "last_request_fingerprint": fingerprint,
        "updated_at": _now_iso(),
        "last_seen_at": _now_iso(),
    }
    try:
        updated = _safe_first(
            supabase.table("live_sessions")
            .update(payload)
            .eq("club_id", club_id)
            .eq("session_key", session_key)
            .eq("version", expected_version)
            .execute()
        )
    except Exception as exc:
        updated = get_public_live_session_row(supabase, club_id=club_id, session_key=session_key)
        if updated and str(updated.get("last_operation_key") or "") == operation_key:
            return updated
        raise PublicLiveRecoveryRequiredError(
            "The live-session write may have completed. Reload and retry with the same operation key."
        ) from exc
    if updated is not None:
        return updated
    current = get_public_live_session_row(supabase, club_id=club_id, session_key=session_key)
    if current and str(current.get("last_operation_key") or "") == operation_key:
        return current
    raise PublicLiveConflictError("This live session changed while it was being saved. Reload before retrying.")


def _run_session_mutation(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    edit_token: str,
    expected_version: int,
    idempotency_key: str,
    requester_hash: str,
    action: str,
    request_payload: dict[str, Any],
    mutate: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any] | None],
) -> dict[str, Any]:
    operation, existed = begin_public_live_operation(
        supabase,
        club_id=str(club_id),
        session_key=str(session_key),
        action=str(action),
        idempotency_key=idempotency_key,
        requester_hash=requester_hash,
        expected_version=int(expected_version),
        request_payload=request_payload,
    )
    operation_key = str(operation.get("operation_key") or "")
    try:
        row = _validate_editable_row(
            get_public_live_session_row(supabase, club_id=str(club_id), session_key=str(session_key)),
            edit_token=edit_token,
        )
    except (PermissionError, PublicLiveSessionError, PublicLiveRecoveryRequiredError) as exc:
        if str(operation.get("status") or "") != "completed":
            update_public_live_operation(
                supabase,
                club_id=str(club_id),
                operation_key_value=operation_key,
                status="rejected",
                error_text=str(exc),
            )
        raise
    completed = completed_operation_result(operation)
    if completed is not None:
        return {
            **{key: value for key, value in completed.items() if key != "session"},
            "session": public_live_session_detail(row),
            "idempotent_replay": True,
        }
    fingerprint = str(operation.get("request_fingerprint") or "")
    if existed and str(row.get("last_operation_key") or "") == operation_key:
        result = {"session": public_live_session_detail(row), "idempotent_replay": True}
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="completed",
            result={},
        )
        return result
    if str(row.get("status") or "") != "active":
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="rejected",
            error_text="session is not active",
        )
        raise PublicLiveSessionError("This live session is complete and cannot be edited.")
    if int(row.get("version") or 1) != int(expected_version):
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="rejected",
            error_text="stale authoritative version",
        )
        raise PublicLiveConflictError("This live session changed after it was loaded. Reload it before continuing.")

    update_public_live_operation(
        supabase,
        club_id=str(club_id),
        operation_key_value=operation_key,
        status="running",
    )
    state = _state(row)
    event = _event_from_state(state)
    if not event:
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="rejected",
            error_text="session has no recoverable event state",
        )
        raise PublicLiveSessionError("This live session has no recoverable event state.")
    try:
        extra = mutate(state, event, row) or {}
    except (PublicLiveSessionError, ValueError) as exc:
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="rejected",
            error_text=str(exc),
        )
        if isinstance(exc, PublicLiveSessionError):
            raise
        raise PublicLiveSessionError(str(exc)) from exc

    next_state = _put_event(state, event)
    patch: dict[str, Any] = {"state": next_state}
    if "status" in extra:
        patch["status"] = extra["status"]
    if "completed_at" in extra:
        patch["completed_at"] = extra["completed_at"]
    try:
        updated = _update_row_with_cas(supabase, row=row, patch=patch, operation=operation)
    except PublicLiveConflictError:
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="rejected",
            error_text="stale authoritative version",
        )
        raise
    except Exception as exc:
        current = get_public_live_session_row(supabase, club_id=str(club_id), session_key=str(session_key))
        if current and str(current.get("last_operation_key") or "") == operation_key and str(current.get("last_request_fingerprint") or "") == fingerprint:
            updated = current
            extra = {}
        else:
            update_public_live_operation(
                supabase,
                club_id=str(club_id),
                operation_key_value=operation_key,
                status="recovery_required",
                error_text=str(exc),
            )
            if isinstance(exc, (PublicLiveSessionError, PermissionError)):
                raise
            raise PublicLiveRecoveryRequiredError(
                "The JUPR Live write did not return a verified result. Retry the identical request to reconcile it."
            ) from exc
    result = {"session": public_live_session_detail(updated), **{key: value for key, value in extra.items() if key not in {"status", "completed_at"}}}
    operation_result = {
        key: value
        for key, value in result.items()
        if key in {"advanced_to_round", "changed_scores"}
    }
    update_public_live_operation(
        supabase,
        club_id=str(club_id),
        operation_key_value=operation_key,
        status="completed",
        result=operation_result,
    )
    return {**result, "idempotent_replay": bool(existed)}


def update_public_live_scores(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    edit_token: str,
    expected_version: int,
    idempotency_key: str,
    requester_hash: str,
    scores: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_scores: list[dict[str, Any]] = []
    for score in scores or []:
        match_id = str(score.get("match_id") or "").strip()
        if not match_id:
            continue
        raw_a, raw_b = score.get("score_a"), score.get("score_b")
        score_a = None if raw_a in (None, "") else int(raw_a)
        score_b = None if raw_b in (None, "") else int(raw_b)
        if score_a is not None and not 0 <= score_a <= 99:
            raise PublicLiveSessionError("Scores must be between 0 and 99.")
        if score_b is not None and not 0 <= score_b <= 99:
            raise PublicLiveSessionError("Scores must be between 0 and 99.")
        normalized_scores.append({"match_id": match_id, "score_a": score_a, "score_b": score_b})
    if not normalized_scores:
        raise PublicLiveSessionError("Provide at least one match score.")

    def mutate(_state: dict[str, Any], event: dict[str, Any], _row: dict[str, Any]) -> dict[str, Any]:
        event_type = str(event.get("type") or "")
        current_round = int(event.get("currentRoundNumber") or 1)
        current_match_ids = {str(match.get("id")) for match in matches_for_round(event, current_round)}
        for score in normalized_scores:
            match = find_match_by_id(event, score["match_id"])
            if match is None:
                raise PublicLiveSessionError("One or more score rows no longer belong to this session.")
            if event_type == "league" and score["match_id"] not in current_match_ids:
                raise PublicLiveSessionError("Earlier league rounds are locked after court movement. Reload the current round.")
            set_match_score(match, score["score_a"], score["score_b"])
        return {"changed_scores": len(normalized_scores)}

    return _run_session_mutation(
        supabase,
        club_id=club_id,
        session_key=session_key,
        edit_token=edit_token,
        expected_version=expected_version,
        idempotency_key=idempotency_key,
        requester_hash=requester_hash,
        action="scores",
        request_payload={"scores": normalized_scores},
        mutate=mutate,
    )


def update_public_round_robin_scores(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    edit_token: str,
    expected_version: int,
    idempotency_key: str,
    requester_hash: str,
    scores: list[dict[str, Any]],
) -> dict[str, Any]:
    return update_public_live_scores(
        supabase,
        club_id=club_id,
        session_key=session_key,
        edit_token=edit_token,
        expected_version=expected_version,
        idempotency_key=idempotency_key,
        requester_hash=requester_hash,
        scores=scores,
    )


def advance_public_live_session(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    edit_token: str,
    expected_version: int,
    idempotency_key: str,
    requester_hash: str,
) -> dict[str, Any]:
    def mutate(_state: dict[str, Any], event: dict[str, Any], _row: dict[str, Any]) -> dict[str, Any]:
        if str(event.get("type") or "") != "league":
            raise PublicLiveSessionError("Only League / Ladder sessions advance between rounds.")
        start_next_league_round(event)
        return {"advanced_to_round": int(event.get("currentRoundNumber") or 1)}

    return _run_session_mutation(
        supabase,
        club_id=club_id,
        session_key=session_key,
        edit_token=edit_token,
        expected_version=expected_version,
        idempotency_key=idempotency_key,
        requester_hash=requester_hash,
        action="advance",
        request_payload={},
        mutate=mutate,
    )


def substitute_public_live_participant(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    edit_token: str,
    expected_version: int,
    idempotency_key: str,
    requester_hash: str,
    scope: str,
    round_number: int,
    original_participant_id: str,
    substitute_name: str,
    substitute_player_id: int | None = None,
    match_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    clean_scope = str(scope or "round").strip().lower()
    clean_name = _normalize_name(substitute_name)[:80]
    if clean_scope not in {"round", "game"}:
        raise PublicLiveSessionError("Substitution scope must be round or game.")
    if not clean_name:
        raise PublicLiveSessionError("Substitute name is required.")
    request_payload = {
        "scope": clean_scope,
        "round_number": int(round_number),
        "match_id": str(match_id or "") or None,
        "original_participant_id": str(original_participant_id),
        "substitute_name": clean_name,
        "substitute_player_id": int(substitute_player_id) if substitute_player_id is not None else None,
        "note": str(note or "")[:300] or None,
    }

    def mutate(_state: dict[str, Any], event: dict[str, Any], _row: dict[str, Any]) -> dict[str, Any]:
        social = _state.get("social") if isinstance(_state.get("social"), dict) else {}
        if bool(social.get("enabled")):
            raise PublicLiveSessionError(
                "Club Social substitutions are not supported because moderation must preserve the original participant identities. Create a new event with the correct roster."
            )
        common = {
            "round_number": int(round_number),
            "original_participant_id": str(original_participant_id),
            "substitute_player_id": substitute_player_id,
            "substitute_name": clean_name,
            "created_by": "public organizer",
            "created_at": _now_iso(),
            "note": str(note or "")[:300],
        }
        if clean_scope == "game":
            if not match_id:
                raise PublicLiveSessionError("A match is required for a single-game substitution.")
            substitution = apply_single_game_substitution(event, match_id=str(match_id), **common)
        else:
            substitution = apply_round_substitution(event, **common)
        event.setdefault("substitutions", []).append(substitution)
        return {"substitution": substitution}

    return _run_session_mutation(
        supabase,
        club_id=club_id,
        session_key=session_key,
        edit_token=edit_token,
        expected_version=expected_version,
        idempotency_key=idempotency_key,
        requester_hash=requester_hash,
        action="substitute",
        request_payload=request_payload,
        mutate=mutate,
    )


def _event_complete(event: dict[str, Any]) -> bool:
    if str(event.get("type") or "") == "round_robin":
        matches = round_robin_matches(event)
        return bool(matches) and all(match_is_scored(match) for match in matches)
    if str(event.get("type") or "") == "league":
        current = int(event.get("currentRoundNumber") or 1)
        total = int(event.get("totalRounds") or 1)
        return current >= total and is_league_round_complete(event, current)
    return False


def _submit_social_event(supabase: Any, *, club_id: str, state: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    try:
        player_rows = _safe_rows(
            supabase.table("players").select("*").eq("club_id", str(club_id)).execute()
        )
    except Exception as exc:
        raise RuntimeError("Club Social player resolution is unavailable.") from exc
    social = state.get("social") if isinstance(state.get("social"), dict) else {}
    ctx = SimpleNamespace(
        supabase=supabase,
        df_players_all=pd.DataFrame(player_rows),
        admin_name="",
        user_name="",
    )
    return save_resolved_social_live_event(
        ctx,
        event,
        target_club_id=str(club_id),
        submission_mode="public",
        host_name=str(social.get("host_name") or "guest"),
        skill_levels=social.get("skill_levels") or ["All"],
    )


def complete_public_live_session(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    edit_token: str,
    expected_version: int,
    idempotency_key: str,
    requester_hash: str,
    social_submitter: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    submitter = social_submitter or _submit_social_event
    operation, existed = begin_public_live_operation(
        supabase,
        club_id=str(club_id),
        session_key=str(session_key),
        action="complete",
        idempotency_key=idempotency_key,
        requester_hash=requester_hash,
        expected_version=int(expected_version),
        request_payload={},
    )
    operation_key = str(operation.get("operation_key") or "")
    try:
        row = _validate_edit_token(
            get_public_live_session_row(supabase, club_id=str(club_id), session_key=str(session_key)),
            edit_token=edit_token,
        )
    except (PermissionError, PublicLiveSessionError) as exc:
        if str(operation.get("status") or "") != "completed":
            update_public_live_operation(
                supabase,
                club_id=str(club_id),
                operation_key_value=operation_key,
                status="rejected",
                error_text=str(exc),
            )
        raise
    completed = completed_operation_result(operation)
    if completed is not None:
        return {
            **{key: value for key, value in completed.items() if key != "session"},
            "session": public_live_session_detail(row),
            "idempotent_replay": True,
        }

    fingerprint = str(operation.get("request_fingerprint") or "")
    pending_key = str(row.get("pending_operation_key") or "")
    if pending_key and pending_key != operation_key:
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="rejected",
            error_text="another completion is unfinished",
        )
        raise PublicLiveRecoveryRequiredError(
            "Another completion is unfinished for this live session. Resume it with its original operation key."
        )
    if str(row.get("status") or "") == "completed":
        if str(row.get("last_operation_key") or "") != operation_key:
            update_public_live_operation(
                supabase,
                club_id=str(club_id),
                operation_key_value=operation_key,
                status="rejected",
                error_text="session is already complete",
            )
            raise PublicLiveSessionError("This live session is already complete.")
        state = _state(row)
        social = state.get("social") if isinstance(state.get("social"), dict) else {}
        result = {
            "session": public_live_session_detail(row),
            "social_submission": social.get("submission") if isinstance(social, dict) else None,
        }
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="completed",
            result={},
        )
        return {**result, "idempotent_replay": True}

    if not str(row.get("pending_operation_key") or ""):
        preflight_event = _event_from_state(_state(row))
        if not preflight_event or not _event_complete(preflight_event):
            update_public_live_operation(
                supabase,
                club_id=str(club_id),
                operation_key_value=operation_key,
                status="rejected",
                error_text="not all scores are complete",
            )
            raise PublicLiveSessionError("Complete every scheduled score before closing this live session.")

    claim_public_live_completion_executor(
        supabase,
        club_id=str(club_id),
        operation_key_value=operation_key,
    )

    if not pending_key:
        if int(row.get("version") or 1) != int(expected_version):
            update_public_live_operation(
                supabase,
                club_id=str(club_id),
                operation_key_value=operation_key,
                status="rejected",
                error_text="stale authoritative version",
            )
            raise PublicLiveConflictError("This live session changed after it was loaded. Reload it before continuing.")
        initial_event = _event_from_state(_state(row))
        if not initial_event or not _event_complete(initial_event):
            update_public_live_operation(
                supabase,
                club_id=str(club_id),
                operation_key_value=operation_key,
                status="rejected",
                error_text="not all scores are complete",
            )
            raise PublicLiveSessionError("Complete every scheduled score before closing this live session.")
        reserved = _update_row_with_cas(
            supabase,
            row=row,
            patch={
                "pending_operation_key": operation_key,
                "pending_operation_action": "complete",
            },
            operation=operation,
        )
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="applied",
            result={"reserved_version": int(reserved.get("version") or 1)},
        )
    else:
        if (
            str(row.get("last_operation_key") or "") != operation_key
            or str(row.get("last_request_fingerprint") or "") != fingerprint
        ):
            raise PublicLiveRecoveryRequiredError(
                "The pending completion marker does not match its recovery request. Stop and contact an administrator."
            )
        reserved = row
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="running",
        )

    submission: dict[str, Any] | None = None
    try:
        state = _state(reserved)
        event = _event_from_state(state)
        if not event or not _event_complete(event):
            raise PublicLiveRecoveryRequiredError(
                "The reserved completion no longer has a complete score state. Stop and contact an administrator."
            )
        social = state.get("social") if isinstance(state.get("social"), dict) else {}
        if bool(social.get("enabled")):
            submission = submitter(supabase, club_id=str(club_id), state=state, event=event)
            social["submission"] = submission
            state["social"] = social
            event["saved_rounds"] = list(submission.get("saved_rounds") or [])
        completed_at = _now_iso()
        updated = _update_row_with_cas(
            supabase,
            row=reserved,
            patch={
                "state": _put_event(state, event),
                "status": "completed",
                "completed_at": completed_at,
                "pending_operation_key": None,
                "pending_operation_action": None,
            },
            operation=operation,
        )
    except Exception as exc:
        current = get_public_live_session_row(supabase, club_id=str(club_id), session_key=str(session_key))
        if current and str(current.get("status") or "") == "completed" and str(current.get("last_operation_key") or "") == operation_key:
            updated = current
            current_state = _state(current)
            current_social = current_state.get("social") if isinstance(current_state.get("social"), dict) else {}
            submission = current_social.get("submission") if isinstance(current_social, dict) else None
        else:
            update_public_live_operation(
                supabase,
                club_id=str(club_id),
                operation_key_value=operation_key,
                status="recovery_required",
                error_text=str(exc),
            )
            if isinstance(exc, PublicLiveRecoveryRequiredError):
                raise
            raise PublicLiveRecoveryRequiredError(
                "Completion is reserved but did not return a verified result. Retry the identical completion request to reconcile it."
            ) from exc

    result = {"session": public_live_session_detail(updated), "social_submission": submission}
    update_public_live_operation(
        supabase,
        club_id=str(club_id),
        operation_key_value=operation_key,
        status="completed",
        result={},
    )
    return {**result, "idempotent_replay": bool(existed)}


def _formula_safe(value: Any) -> str:
    text = str(value or "")
    return f"'{text}" if text[:1] in {"=", "+", "-", "@"} else text


def build_public_live_export(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    export_format: str,
) -> dict[str, Any]:
    row = get_public_live_session_row(supabase, club_id=str(club_id), session_key=str(session_key))
    if row is None or not is_public_live_session_row(row):
        raise PublicLiveSessionError("Live session not found.")
    session = public_live_session_detail(row)
    clean_format = str(export_format or "csv").strip().lower()
    if clean_format == "json":
        import json

        return {
            "content": json.dumps(session, indent=2, sort_keys=True),
            "media_type": "application/json",
            "filename": f"jupr-live-{session_key}.json",
        }
    if clean_format != "csv":
        raise PublicLiveSessionError("Export format must be csv or json.")
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["round", "court", "mini_round", "match_id", "team_a", "score_a", "score_b", "team_b"])
    for round_data in session.get("rounds") or []:
        matches = round_data.get("matches") or []
        for match in matches:
            writer.writerow(
                [
                    round_data.get("number"),
                    match.get("court_number") or "",
                    match.get("mini_round_number") or "",
                    _formula_safe(match.get("id")),
                    _formula_safe(" / ".join(match.get("team_a") or [])),
                    match.get("score_a") if match.get("score_a") is not None else "",
                    match.get("score_b") if match.get("score_b") is not None else "",
                    _formula_safe(" / ".join(match.get("team_b") or [])),
                ]
            )
    return {
        "content": output.getvalue(),
        "media_type": "text/csv; charset=utf-8",
        "filename": f"jupr-live-{session_key}.csv",
    }
