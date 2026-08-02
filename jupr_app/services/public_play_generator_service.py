from __future__ import annotations

import base64
import copy
import csv
import hashlib
import hmac
import io
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from jupr_app.domain.adaptive_play_engine import (
    advance_generator_event,
    create_generator_preview,
    generator_event_standings,
    mutate_generator_roster,
    save_generator_round,
    schedule_export_rows,
    skip_generator_round,
    start_generator_event,
)
from jupr_app.services.public_live_operation_service import (
    PublicLiveConflictError,
    PublicLiveRecoveryRequiredError,
    begin_public_live_operation,
    completed_operation_result,
    edit_token_matches,
    hash_edit_token,
    update_public_live_operation,
)


PUBLIC_GENERATOR_MODE = "public_play_generator"
GENERATOR_KINDS = {"round_robin", "ladder"}
PLAY_FORMATS = {"singles", "doubles"}


class PublicPlayGeneratorError(ValueError):
    """User-correctable public generator error."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ttl_iso() -> str:
    try:
        hours = int(os.getenv("JUPR_PUBLIC_PLAY_GENERATOR_TTL_HOURS", "48"))
    except ValueError:
        hours = 48
    hours = max(1, min(hours, 168))
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _first(response: Any) -> dict[str, Any] | None:
    rows = _safe_rows(response)
    return rows[0] if rows else None


def _clean(value: Any, *, limit: int = 160) -> str:
    return " ".join(str(value or "").replace("<", "").replace(">", "").split()).strip()[:limit]


def _normalize_kind(value: Any) -> str:
    kind = str(value or "").strip().lower().replace("-", "_")
    if kind == "league_ladder":
        kind = "ladder"
    if kind not in GENERATOR_KINDS:
        raise PublicPlayGeneratorError("Choose Round-Robin Generator or Ladder Generator.")
    return kind


def _normalize_format(value: Any) -> str:
    play_format = str(value or "").strip().lower()
    if play_format not in PLAY_FORMATS:
        raise PublicPlayGeneratorError("Choose Singles or Doubles.")
    return play_format


def _normalize_names(values: list[Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        name = _clean(raw)
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def _token_secret(explicit: str | None = None) -> str:
    secret = str(explicit or os.getenv("JUPR_PUBLIC_LIVE_TOKEN_SECRET", "")).strip()
    if len(secret) < 32:
        raise RuntimeError("Public generator edit-token protection is not configured.")
    return secret


def _edit_token(*, secret: str, operation_key: str) -> str:
    digest = hmac.new(
        secret.encode("utf-8"),
        f"jupr:public-play-generator:v1:{operation_key}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


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
    page_state = next_state.get("page_state")
    if not isinstance(page_state, dict):
        page_state = {}
    page_state.update(
        {
            "event": copy.deepcopy(event),
            "event_name": str(event.get("name") or "Play session"),
            "event_type": str(event.get("type") or ""),
            "generator_kind": str(event.get("generatorKind") or ""),
            "play_format": str(event.get("playFormat") or ""),
            "current_round_number": int(event.get("currentRoundNumber") or 1),
            "participant_count": len(event.get("participants") or []),
        }
    )
    next_state["page_state"] = page_state
    next_state["event_name"] = page_state["event_name"]
    next_state["event_type"] = page_state["event_type"]
    next_state["generator_kind"] = page_state["generator_kind"]
    next_state["play_format"] = page_state["play_format"]
    return next_state


def _base_state(*, club_id: str, event: dict[str, Any]) -> dict[str, Any]:
    return _put_event(
        {
            "version": 4,
            "mode": PUBLIC_GENERATOR_MODE,
            "source": "public_web",
            "club_id": str(club_id),
            "generator_kind": str(event.get("generatorKind") or ""),
            "play_format": str(event.get("playFormat") or ""),
            "page_state": {},
        },
        event,
    )


def _is_generator_row(row: dict[str, Any], generator_kind: str | None = None) -> bool:
    state = _state(row)
    if str(state.get("mode") or "") != PUBLIC_GENERATOR_MODE:
        return False
    if generator_kind:
        return str(state.get("generator_kind") or "") == str(generator_kind)
    return True


def _not_expired(row: dict[str, Any]) -> bool:
    if str(row.get("status") or "") != "active":
        return True
    raw = str(row.get("expires_at") or "").strip()
    if not raw:
        return True
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc) > datetime.now(timezone.utc)
    except ValueError:
        return True


def public_play_generator_session_payload(row: dict[str, Any]) -> dict[str, Any]:
    state = _state(row)
    event = _event_from_state(state)
    return {
        "session_key": str(row.get("session_key") or ""),
        "title": _clean(row.get("title") or event.get("name")),
        "status": str(row.get("status") or event.get("status") or "active"),
        "version": int(row.get("version") or 1),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "last_seen_at": row.get("last_seen_at"),
        "expires_at": row.get("expires_at"),
        "completed_at": row.get("completed_at"),
        "generator_kind": str(event.get("generatorKind") or state.get("generator_kind") or ""),
        "play_format": str(event.get("playFormat") or state.get("play_format") or ""),
        "current_round_number": int(event.get("currentRoundNumber") or 1) if event else None,
        "total_rounds": int(event.get("totalRounds") or 0) if event else None,
        "event": event,
        "schedule_rows": schedule_export_rows(event) if event else [],
        "standings_sort": str(event.get("standingsSort") or "wins") if event else "wins",
        "standings": generator_event_standings(event) if event else [],
        "unrated": True,
    }


def _player_ids_for_names(
    supabase: Any,
    *,
    club_id: str,
    names: list[str],
    requested: dict[str, int] | None,
) -> list[int]:
    requested_by_name = {
        _clean(name).casefold(): int(player_id)
        for name, player_id in (requested or {}).items()
        if _clean(name)
    }
    if not requested_by_name:
        return []
    if any(name.casefold() not in requested_by_name for name in names):
        raise PublicPlayGeneratorError("Link every entered player or leave all player links blank.")
    ids = [requested_by_name[name.casefold()] for name in names]
    try:
        rows = _safe_rows(
            supabase.table("players")
            .select("id,name,club_id,active,inactive_at")
            .eq("club_id", str(club_id))
            .in_("id", sorted(set(ids)))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Current-player validation is unavailable; nothing was written.") from exc
    by_id = {int(row["id"]): row for row in rows if row.get("id") is not None}
    for name, player_id in zip(names, ids):
        row = by_id.get(int(player_id))
        if row is None or _clean(row.get("name")).casefold() != name.casefold():
            raise PublicPlayGeneratorError(f"The linked player for {name} is stale. Select the player again.")
    return ids


def preview_public_play_generator(
    supabase: Any,
    *,
    club_id: str,
    generator_kind: str,
    play_format: str,
    title: str,
    participant_names: list[Any],
    participant_player_ids: dict[str, int] | None,
    total_rounds: int,
    court_count: int,
    standings_sort: str = "wins",
) -> dict[str, Any]:
    kind = _normalize_kind(generator_kind)
    fmt = _normalize_format(play_format)
    names = _normalize_names(participant_names)
    ids = _player_ids_for_names(
        supabase,
        club_id=str(club_id),
        names=names,
        requested=participant_player_ids,
    )
    try:
        preview = create_generator_preview(
            generator_kind=kind,
            play_format=fmt,
            title=_clean(title) or ("Round-Robin Generator" if kind == "round_robin" else "Ladder Generator"),
            participant_names=names,
            player_ids=ids,
            total_rounds=max(1, min(int(total_rounds or 1), 50)),
            court_count=max(0, min(int(court_count or 0), 20)),
            standings_sort=standings_sort,
        )
    except ValueError as exc:
        raise PublicPlayGeneratorError(str(exc)) from exc
    return {"ok": True, "preview": preview, "schedule_rows": schedule_export_rows(preview)}


def _find_creation_row(supabase: Any, *, club_id: str, operation_key: str) -> dict[str, Any] | None:
    try:
        row = _first(
            supabase.table("live_sessions")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("creation_operation_key", str(operation_key))
            .limit(1)
            .execute()
        )
    except Exception:
        row = None
    return row if row and _is_generator_row(row) else None


def create_public_play_generator_session(
    supabase: Any,
    *,
    club_id: str,
    generator_kind: str,
    play_format: str,
    title: str,
    participant_names: list[Any],
    participant_player_ids: dict[str, int] | None,
    total_rounds: int,
    court_count: int,
    preview_fingerprint: str | None,
    idempotency_key: str,
    requester_hash: str,
    token_secret: str | None = None,
    standings_sort: str = "wins",
) -> dict[str, Any]:
    preview_result = preview_public_play_generator(
        supabase,
        club_id=str(club_id),
        generator_kind=generator_kind,
        play_format=play_format,
        title=title,
        participant_names=participant_names,
        participant_player_ids=participant_player_ids,
        total_rounds=total_rounds,
        court_count=court_count,
        standings_sort=standings_sort,
    )
    preview = preview_result["preview"]
    supplied = str(preview_fingerprint or "").strip()
    if supplied and supplied != str(preview.get("previewFingerprint") or ""):
        raise PublicPlayGeneratorError("The roster or settings changed after preview. Preview again before starting.")
    request_payload = {
        "event_type": str(preview.get("generatorKind") or ""),
        "play_format": str(preview.get("playFormat") or ""),
        "participant_names": [row.get("name") for row in preview.get("participants") or []],
        "participant_player_ids": participant_player_ids or {},
        "total_rounds": int(preview.get("totalRounds") or 1),
        "court_sizes": [int(preview.get("courtCount") or 0)],
        "standings_sort": str(preview.get("standingsSort") or "wins"),
        "live_mode": "quick",
    }
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
    token = _edit_token(secret=_token_secret(token_secret), operation_key=operation_key)
    existing = _find_creation_row(supabase, club_id=str(club_id), operation_key=operation_key)
    if existing is not None:
        if str(operation.get("status") or "") != "completed":
            update_public_live_operation(
                supabase,
                club_id=str(club_id),
                operation_key_value=operation_key,
                status="completed",
                result={},
            )
        return {
            "ok": True,
            "edit_token": token,
            "session": public_play_generator_session_payload(existing),
            "idempotent_replay": bool(existed),
        }
    if existed and completed_operation_result(operation) is not None:
        raise PublicLiveRecoveryRequiredError("The completed create operation has no recoverable session row.")

    event = start_generator_event(preview)
    now = _now_iso()
    payload = {
        "club_id": str(club_id),
        "session_key": session_key,
        "title": _clean(title) or str(event.get("name") or "Play session"),
        "status": "active",
        "state": _base_state(club_id=str(club_id), event=event),
        "source": "public_play_generator",
        "version": 1,
        "edit_token_hash": hash_edit_token(token),
        "creation_operation_key": operation_key,
        "last_operation_key": operation_key,
        "last_request_fingerprint": str(operation.get("request_fingerprint") or ""),
        "created_at": now,
        "updated_at": now,
        "last_seen_at": now,
        "expires_at": _ttl_iso(),
    }
    update_public_live_operation(
        supabase,
        club_id=str(club_id),
        operation_key_value=operation_key,
        status="running",
    )
    try:
        created = _first(supabase.table("live_sessions").insert(payload).execute())
    except Exception as exc:
        created = _find_creation_row(supabase, club_id=str(club_id), operation_key=operation_key)
        if created is None:
            update_public_live_operation(
                supabase,
                club_id=str(club_id),
                operation_key_value=operation_key,
                status="recovery_required",
                error_text=str(exc),
            )
            raise PublicLiveRecoveryRequiredError(
                "The generator may have been created. Retry the identical request to reconcile it."
            ) from exc
    if created is None:
        raise PublicLiveRecoveryRequiredError("The generator returned no recoverable session row.")
    update_public_live_operation(
        supabase,
        club_id=str(club_id),
        operation_key_value=operation_key,
        status="completed",
        result={},
    )
    return {
        "ok": True,
        "edit_token": token,
        "session": public_play_generator_session_payload(created),
        "idempotent_replay": False,
    }


def build_public_play_generator_status(
    supabase: Any | None,
    *,
    club_id: str,
    writes_enabled: bool,
) -> dict[str, Any]:
    counts = {"round_robin": 0, "ladder": 0, "active": 0, "completed": 0}
    if supabase is not None:
        try:
            rows = _safe_rows(
                supabase.table("live_sessions")
                .select("club_id,status,state,expires_at")
                .eq("club_id", str(club_id))
                .limit(500)
                .execute()
            )
        except Exception:
            rows = []
        for row in rows:
            if not _is_generator_row(row) or not _not_expired(row):
                continue
            kind = str(_state(row).get("generator_kind") or "")
            status = str(row.get("status") or "")
            if kind in counts:
                counts[kind] += 1
            if status in counts:
                counts[status] += 1
    return {
        "enabled": True,
        "writes_enabled": bool(writes_enabled),
        "status": "ready_for_public_play_generators" if writes_enabled else "read_only",
        "warnings": [] if writes_enabled else ["Public generator creation and editing are paused."],
        "counts": counts,
        "official_publish": False,
    }


def list_public_play_generator_sessions(
    supabase: Any,
    *,
    club_id: str,
    generator_kind: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    try:
        rows = _safe_rows(
            supabase.table("live_sessions")
            .select("*")
            .eq("club_id", str(club_id))
            .order("updated_at", desc=True)
            .limit(max(1, min(int(limit or 50), 100)))
            .execute()
        )
    except Exception:
        rows = []
    kind = _normalize_kind(generator_kind) if generator_kind else None
    sessions = [
        public_play_generator_session_payload(row)
        for row in rows
        if _is_generator_row(row, kind) and _not_expired(row)
    ]
    return {"ok": True, "sessions": sessions, "count": len(sessions)}


def _get_row(supabase: Any, *, club_id: str, session_key: str) -> dict[str, Any]:
    row = _first(
        supabase.table("live_sessions")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("session_key", str(session_key))
        .limit(1)
        .execute()
    )
    if row is None or not _is_generator_row(row) or not _not_expired(row):
        raise PublicPlayGeneratorError("Generator session not found.")
    return row


def get_public_play_generator_session(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
) -> dict[str, Any]:
    return {
        "ok": True,
        "session": public_play_generator_session_payload(
            _get_row(supabase, club_id=str(club_id), session_key=str(session_key))
        ),
    }


def _validate_editable(row: dict[str, Any], *, edit_token: str) -> None:
    if not edit_token_matches(edit_token, str(row.get("edit_token_hash") or "")):
        raise PermissionError("Invalid organizer edit token.")
    if str(row.get("status") or "") != "active":
        raise PublicPlayGeneratorError("This generator session is complete and view-only.")
    if str(row.get("pending_operation_key") or ""):
        raise PublicLiveRecoveryRequiredError("This session has an unfinished operation. Retry that preserved action first.")


def _update_cas(
    supabase: Any,
    *,
    row: dict[str, Any],
    patch: dict[str, Any],
    operation: dict[str, Any],
) -> dict[str, Any]:
    expected = int(row.get("version") or 1)
    operation_key = str(operation.get("operation_key") or "")
    payload = {
        **patch,
        "version": expected + 1,
        "last_operation_key": operation_key,
        "last_request_fingerprint": str(operation.get("request_fingerprint") or ""),
        "updated_at": _now_iso(),
        "last_seen_at": _now_iso(),
    }
    try:
        updated = _first(
            supabase.table("live_sessions")
            .update(payload)
            .eq("club_id", str(row.get("club_id") or ""))
            .eq("session_key", str(row.get("session_key") or ""))
            .eq("version", expected)
            .execute()
        )
    except Exception as exc:
        current = _get_row(
            supabase,
            club_id=str(row.get("club_id") or ""),
            session_key=str(row.get("session_key") or ""),
        )
        if str(current.get("last_operation_key") or "") == operation_key:
            return current
        raise PublicLiveRecoveryRequiredError(
            "The write may have completed. Retry the identical preserved request to reconcile it."
        ) from exc
    if updated is not None:
        return updated
    current = _get_row(
        supabase,
        club_id=str(row.get("club_id") or ""),
        session_key=str(row.get("session_key") or ""),
    )
    if str(current.get("last_operation_key") or "") == operation_key:
        return current
    raise PublicLiveConflictError("This generator session changed. Reload it before continuing.")


def _run_mutation(
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
    mutate: Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    operation, existed = begin_public_live_operation(
        supabase,
        club_id=str(club_id),
        session_key=str(session_key),
        action=f"generator_{action}",
        idempotency_key=idempotency_key,
        requester_hash=requester_hash,
        expected_version=int(expected_version),
        request_payload=request_payload,
    )
    operation_key = str(operation.get("operation_key") or "")
    row = _get_row(supabase, club_id=str(club_id), session_key=str(session_key))
    try:
        _validate_editable(row, edit_token=edit_token)
    except Exception as exc:
        if str(operation.get("status") or "") != "completed":
            update_public_live_operation(
                supabase,
                club_id=str(club_id),
                operation_key_value=operation_key,
                status="rejected",
                error_text=str(exc),
            )
        raise
    if completed_operation_result(operation) is not None:
        return {
            "ok": True,
            "session": public_play_generator_session_payload(row),
            "idempotent_replay": True,
        }
    if existed and str(row.get("last_operation_key") or "") == operation_key:
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="completed",
            result={},
        )
        return {
            "ok": True,
            "session": public_play_generator_session_payload(row),
            "idempotent_replay": True,
        }
    if int(row.get("version") or 1) != int(expected_version):
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="rejected",
            error_text="stale version",
        )
        raise PublicLiveConflictError("This generator session changed. Reload it before continuing.")
    update_public_live_operation(
        supabase,
        club_id=str(club_id),
        operation_key_value=operation_key,
        status="running",
    )
    state = _state(row)
    event = _event_from_state(state)
    try:
        next_event, extra_patch = mutate(event)
    except (ValueError, PublicPlayGeneratorError) as exc:
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="rejected",
            error_text=str(exc),
        )
        raise PublicPlayGeneratorError(str(exc)) from exc
    patch = {"state": _put_event(state, next_event), **dict(extra_patch or {})}
    try:
        updated = _update_cas(supabase, row=row, patch=patch, operation=operation)
    except PublicLiveConflictError:
        update_public_live_operation(
            supabase,
            club_id=str(club_id),
            operation_key_value=operation_key,
            status="rejected",
            error_text="stale version",
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
    update_public_live_operation(
        supabase,
        club_id=str(club_id),
        operation_key_value=operation_key,
        status="completed",
        result={},
    )
    return {
        "ok": True,
        "session": public_play_generator_session_payload(updated),
        "idempotent_replay": False,
    }


def save_public_play_generator_round(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    round_number: int,
    scores: list[dict[str, Any]],
    edit_token: str,
    expected_version: int,
    idempotency_key: str,
    requester_hash: str,
) -> dict[str, Any]:
    return _run_mutation(
        supabase,
        club_id=club_id,
        session_key=session_key,
        edit_token=edit_token,
        expected_version=expected_version,
        idempotency_key=idempotency_key,
        requester_hash=requester_hash,
        action="scores",
        request_payload={"round_number": int(round_number), "score_count": len(scores or [])},
        mutate=lambda event: (
            save_generator_round(event, round_number=int(round_number), scores=scores),
            {},
        ),
    )


def skip_public_play_generator_round(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    round_number: int,
    reason: str,
    edit_token: str,
    expected_version: int,
    idempotency_key: str,
    requester_hash: str,
) -> dict[str, Any]:
    return _run_mutation(
        supabase,
        club_id=club_id,
        session_key=session_key,
        edit_token=edit_token,
        expected_version=expected_version,
        idempotency_key=idempotency_key,
        requester_hash=requester_hash,
        action="skip",
        request_payload={"round_number": int(round_number), "reason": _clean(reason, limit=300)},
        mutate=lambda event: (
            skip_generator_round(event, round_number=int(round_number), reason=reason),
            {},
        ),
    )


def advance_public_play_generator_session(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    edit_token: str,
    expected_version: int,
    idempotency_key: str,
    requester_hash: str,
) -> dict[str, Any]:
    def mutate(event: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        next_event = advance_generator_event(event)
        if str(next_event.get("status") or "") == "completed":
            now = _now_iso()
            return next_event, {"status": "completed", "completed_at": now}
        return next_event, {}

    return _run_mutation(
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


def mutate_public_play_generator_roster(
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
    edit_token: str,
    expected_version: int,
    idempotency_key: str,
    requester_hash: str,
) -> dict[str, Any]:
    if player_id is not None:
        _player_ids_for_names(
            supabase,
            club_id=str(club_id),
            names=[_clean(name) or "Player"],
            requested={_clean(name) or "Player": int(player_id)},
        )
    request_payload = {
        "action": str(action),
        "participant_id": participant_id,
        "name_supplied": bool(_clean(name)),
        "linked_player": player_id is not None,
        "substitute_scope": substitute_scope,
        "roster_count": len(roster_order or []),
    }
    return _run_mutation(
        supabase,
        club_id=club_id,
        session_key=session_key,
        edit_token=edit_token,
        expected_version=expected_version,
        idempotency_key=idempotency_key,
        requester_hash=requester_hash,
        action="roster",
        request_payload=request_payload,
        mutate=lambda event: (
            mutate_generator_roster(
                event,
                action=action,
                participant_id=participant_id,
                name=name,
                player_id=player_id,
                substitute_scope=substitute_scope,
                roster_order=roster_order,
            ),
            {},
        ),
    )


def complete_public_play_generator_session(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    edit_token: str,
    expected_version: int,
    idempotency_key: str,
    requester_hash: str,
) -> dict[str, Any]:
    def mutate(event: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        current = int(event.get("currentRoundNumber") or 1)
        row = next(
            (item for item in event.get("rounds") or [] if int(item.get("number") or 0) == current),
            None,
        )
        if row and str(row.get("status") or "") not in {"saved", "skipped"}:
            raise PublicPlayGeneratorError("Save or skip the current round before completing the session.")
        next_event = copy.deepcopy(event)
        now = _now_iso()
        next_event["status"] = "completed"
        next_event["completedAt"] = now
        return next_event, {"status": "completed", "completed_at": now}

    return _run_mutation(
        supabase,
        club_id=club_id,
        session_key=session_key,
        edit_token=edit_token,
        expected_version=expected_version,
        idempotency_key=idempotency_key,
        requester_hash=requester_hash,
        action="complete",
        request_payload={},
        mutate=mutate,
    )


def build_public_play_generator_export(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    export_format: str,
) -> dict[str, Any]:
    session = get_public_play_generator_session(
        supabase,
        club_id=str(club_id),
        session_key=str(session_key),
    )["session"]
    clean_format = str(export_format or "csv").strip().lower()
    if clean_format == "json":
        return {
            "content": json.dumps(session, indent=2, sort_keys=True),
            "media_type": "application/json",
            "filename": f"play-generator-{session_key}.json",
        }
    if clean_format != "csv":
        raise PublicPlayGeneratorError("Export format must be csv or json.")
    event = session.get("event") or {}
    participants = {
        str(row.get("id")): str(row.get("name") or row.get("id") or "")
        for row in event.get("participants") or []
    }
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["round", "court", "side_a", "score_a", "score_b", "side_b", "byes", "status"])
    for round_row in event.get("rounds") or []:
        matches = list(round_row.get("matches") or []) or [
            match
            for court in round_row.get("courts") or []
            for match in court.get("matches") or []
        ]
        byes = " / ".join(participants.get(str(pid), str(pid)) for pid in round_row.get("byeParticipantIds") or [])
        for match in matches:
            side_a = " / ".join(participants.get(str(pid), str(pid)) for pid in match.get("sideA") or match.get("teamA") or [])
            side_b = " / ".join(participants.get(str(pid), str(pid)) for pid in match.get("sideB") or match.get("teamB") or [])
            writer.writerow(
                [
                    round_row.get("number"),
                    match.get("court") or "",
                    side_a,
                    match.get("scoreA") if match.get("scoreA") is not None else "",
                    match.get("scoreB") if match.get("scoreB") is not None else "",
                    side_b,
                    byes,
                    round_row.get("status") or "",
                ]
            )
    return {
        "content": output.getvalue(),
        "media_type": "text/csv; charset=utf-8",
        "filename": f"play-generator-{session_key}.csv",
    }
