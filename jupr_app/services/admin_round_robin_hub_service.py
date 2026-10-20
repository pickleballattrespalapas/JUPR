"""One durable court authority for independent Round-Robin Generator sessions."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from jupr_app.services.public_live_operation_service import PublicLiveConflictError

HUB_TABLE = "round_robin_hubs"
OPERATION_TABLE = "round_robin_hub_operations"
GENERATOR_MODES = {"admin_play_generator", "public_play_generator"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _retained_expiry(row: dict[str, Any], requested: str | None = None) -> str:
    expiry = datetime.now(timezone.utc) + timedelta(days=30)
    for value in (row.get("expires_at"), requested):
        if value:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            expiry = max(expiry, parsed)
    return expiry.isoformat()


def _rows(response: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in (response.data or [])]


def _first(response: Any) -> dict[str, Any] | None:
    return next(iter(_rows(response)), None)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _event(row: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy((row.get("state") or {}).get("page_state", {}).get("event") or {})


def _hub(db: Any, club_id: str, hub_id: str) -> dict[str, Any]:
    row = _first(db.table(HUB_TABLE).select("*").eq("club_id", club_id).eq("id", hub_id).limit(1).execute())
    if row is None:
        raise LookupError("Round Robin Hub not found.")
    return row


def _session_rows(db: Any, club_id: str, keys: list[str]) -> dict[str, dict[str, Any]]:
    if not keys:
        return {}
    rows = _rows(db.table("live_sessions").select("*").eq("club_id", club_id).in_("session_key", sorted(set(keys))).execute())
    result = {str(row["session_key"]): row for row in rows}
    if set(keys) - set(result):
        raise ValueError("A round robin is no longer available. Refresh the hub.")
    for row in result.values():
        if (row.get("state") or {}).get("mode") not in GENERATOR_MODES or _event(row).get("generatorKind") != "round_robin":
            raise ValueError("Only Round-Robin Generator sessions can join this hub.")
    return result


def _group_keys(state: dict[str, Any]) -> list[str]:
    return [str(group["session_key"]) for group in state.get("groups", [])]


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in ("id", "title", "status", "version", "created_at", "updated_at")}


def _is_locked(row: dict[str, Any]) -> bool:
    state = row.get("state") or {}
    published = state.get("official_publish") or {}
    return bool(state.get("generator_submission") or published.get("published_at") or published.get("published_match_ids"))


def list_round_robin_hubs(db: Any, *, club_id: str) -> dict[str, Any]:
    hubs = _rows(db.table(HUB_TABLE).select("*").eq("club_id", club_id).order("updated_at", desc=True).limit(100).execute())
    candidates = _rows(db.table("live_sessions").select("*").eq("club_id", club_id).eq("status", "active").order("updated_at", desc=True).limit(300).execute())
    sessions = []
    for row in candidates:
        state, event = row.get("state") or {}, _event(row)
        if state.get("mode") not in GENERATOR_MODES or event.get("generatorKind") != "round_robin" or _is_locked(row):
            continue
        if state.get("mode") == "public_play_generator" and row.get("expires_at"):
            if datetime.fromisoformat(str(row["expires_at"]).replace("Z", "+00:00")) < datetime.now(timezone.utc):
                continue
        sessions.append({
            "session_key": row["session_key"], "title": row.get("title") or event.get("name"),
            "status": row["status"], "rating_mode": event.get("ratingMode", "unrated"),
            "event_format": event.get("eventFormat", "standard"), "scoring_mode": event.get("scoringMode", "scored"),
            "round_robin_hub_id": state.get("round_robin_hub_id"),
            "current_round_number": event.get("currentRoundNumber", 1),
        })
    return {"ok": True, "hubs": [_summary(row) for row in hubs], "sessions": sessions}


def get_round_robin_hub(db: Any, *, club_id: str, hub_id: str) -> dict[str, Any]:
    from jupr_app.domain.round_robin_hub import build_hub_snapshot
    row = _hub(db, club_id, hub_id)
    state = copy.deepcopy(row.get("state") or {})
    sessions = _session_rows(db, club_id, _group_keys(state))
    events = {key: _event(value) for key, value in sessions.items()}
    workspace = build_hub_snapshot(state, events, now=_now())
    for group in workspace.get("groups", []):
        session = sessions[str(group["session_key"])]
        group["rating_mode"] = _event(session).get("ratingMode", "unrated")
        group["results_locked"] = _is_locked(session)
        group["source_mode"] = (session.get("state") or {}).get("mode")
    return {"ok": True, "hub": _summary(row), "workspace": workspace}


def _replayed(db: Any, club_id: str, hub_id: str, key: str, fingerprint: str) -> bool:
    operation = _first(db.table(OPERATION_TABLE).select("*").eq("club_id", club_id).eq("hub_id", hub_id).eq("operation_key", key).limit(1).execute())
    if operation is None:
        return False
    if str(operation.get("fingerprint") or operation.get("request_fingerprint") or "") != fingerprint:
        raise PublicLiveConflictError("This request identifier was already used for a different change.")
    return True


def _rpc_commit(db: Any, *, club_id: str, hub_id: str, expected_version: int,
                operation_key: str, fingerprint: str, title: str, status: str,
                state: dict[str, Any], sessions: list[dict[str, Any]], actor_email: str) -> dict[str, Any]:
    try:
        result = db.rpc("commit_round_robin_hub", {
            "p_club_id": club_id, "p_hub_id": hub_id, "p_expected_version": expected_version,
            "p_operation_key": operation_key, "p_fingerprint": fingerprint,
            "p_title": title, "p_status": status, "p_state": state,
            "p_sessions": sessions, "p_actor_email": actor_email,
        }).execute().data
    except Exception as exc:
        code = str(getattr(exc, "code", ""))
        message = str(exc)
        if code == "23505":
            raise PublicLiveConflictError("A court name or player is already in use by another active hub. Review the court setup and assignments.") from exc
        if code in {"40001", "P0001"} or any(term in message.lower() for term in ("changed", "conflict", "already assigned", "leased")):
            raise PublicLiveConflictError("The hub or one of its round robins changed. Refresh before continuing.") from exc
        if code in {"22023", "55000", "23514"}:
            raise ValueError("This change cannot be applied to the current hub. Refresh and resolve any pending round-robin operation first.") from exc
        raise
    if isinstance(result, list):
        result = next(iter(result), None)
    if not isinstance(result, dict) or not result.get("hub"):
        raise RuntimeError("The hub update could not be confirmed. Retry the same action.")
    return result


def _session_patch(row: dict[str, Any], event: dict[str, Any], hub_id: str | None) -> dict[str, Any]:
    from jupr_app.services.admin_play_generator_service import _put_event
    updated_event = copy.deepcopy(event)
    if hub_id:
        updated_event["roundRobinHubId"] = hub_id
    else:
        updated_event.pop("roundRobinHubId", None)
    state = _put_event(row.get("state") or {}, updated_event)
    if hub_id:
        state["round_robin_hub_id"] = hub_id
    else:
        state.pop("round_robin_hub_id", None)
    patch: dict[str, Any] = {"state": state}
    if updated_event.get("status") in {"active", "completed"}:
        patch["status"] = updated_event["status"]
        patch["completed_at"] = updated_event.get("completedAt") if updated_event["status"] == "completed" else None
    if all(row.get(key) == value for key, value in patch.items()):
        return {}
    return patch


def _read_set(rows: dict[str, dict[str, Any]], events: dict[str, dict[str, Any]],
              state: dict[str, Any], hub_id: str) -> list[dict[str, Any]]:
    members = set(_group_keys(state))
    result = []
    for key, row in sorted(rows.items()):
        patch = _session_patch(row, events.get(key, _event(row)), hub_id if key in members else None)
        if patch and key in members:
            patch["expires_at"] = _retained_expiry(row)
        result.append({"session_key": key, "expected_version": row["updated_at"],
                       "expected_numeric_version": int(row.get("version") or 1), "patch": patch})
    return result


def create_round_robin_hub(db: Any, *, club_id: str, title: str, courts: list[dict[str, Any]],
                          idempotency_key: str, actor_email: str) -> dict[str, Any]:
    from jupr_app.domain.round_robin_hub import apply_hub_command
    hub_id = str(uuid5(NAMESPACE_URL, f"jupr:rr-hub:{club_id}:{idempotency_key}"))
    fingerprint = _fingerprint({"action": "create", "title": title, "courts": courts})
    if not _replayed(db, club_id, hub_id, idempotency_key, fingerprint):
        state = {"id": hub_id, "courts": [], "groups": [], "players": {}}
        state, _ = apply_hub_command(state, {}, action="configure_courts", payload={"courts": courts}, now=_now())
        _rpc_commit(db, club_id=club_id, hub_id=hub_id, expected_version=0,
                    operation_key=idempotency_key, fingerprint=fingerprint, title=title.strip(),
                    status="active", state=state, sessions=[], actor_email=actor_email)
    return get_round_robin_hub(db, club_id=club_id, hub_id=hub_id)


def command_round_robin_hub(db: Any, *, club_id: str, hub_id: str, action: str,
                           payload: dict[str, Any], expected_version: int,
                           idempotency_key: str, actor_email: str) -> dict[str, Any]:
    from jupr_app.domain.round_robin_hub import apply_hub_command
    fingerprint = _fingerprint({"action": action, "payload": payload, "expected_version": expected_version})
    if _replayed(db, club_id, hub_id, idempotency_key, fingerprint):
        return {**get_round_robin_hub(db, club_id=club_id, hub_id=hub_id), "replayed": True}
    row = _hub(db, club_id, hub_id)
    if int(row["version"]) != int(expected_version):
        raise PublicLiveConflictError("The hub changed. Refresh before continuing.")
    if row["status"] != "active":
        raise ValueError("This hub is complete. Create a new hub to run more round robins.")
    state = copy.deepcopy(row.get("state") or {})
    keys = _group_keys(state)
    if action == "add_group":
        keys = list(dict.fromkeys([*keys, str(payload.get("session_key") or "")]))
    sessions = _session_rows(db, club_id, keys)
    events = {key: _event(value) for key, value in sessions.items()}
    if action == "add_group":
        candidate = sessions[str(payload["session_key"])]
        other_hub = (candidate.get("state") or {}).get("round_robin_hub_id")
        if other_hub and other_hub != hub_id:
            raise ValueError("This round robin already belongs to another hub.")
        if candidate.get("status") != "active" or _is_locked(candidate):
            raise ValueError("Choose an active round robin whose results have not been submitted.")
        if candidate.get("pending_operation_key"):
            raise ValueError("Finish the pending change to this round robin before adding it.")
        if (candidate.get("state") or {}).get("mode") == "public_play_generator" and candidate.get("expires_at"):
            if datetime.fromisoformat(str(candidate["expires_at"]).replace("Z", "+00:00")) < datetime.now(timezone.utc):
                raise ValueError("This organizer session has expired.")
    status = str(row["status"])
    if action == "finish_hub":
        if any(court.get("current") or court.get("next") for court in state.get("courts", [])):
            raise ValueError("Resolve the playing and reserved matches before finishing this hub.")
        if any(event.get("status") != "completed" for event in events.values()):
            raise ValueError("Finish or remove each round robin before finishing the hub.")
        status = "completed"
    elif action == "rename_hub":
        title = str(payload.get("title") or "").strip()
        if not title or len(title) > 160:
            raise ValueError("Enter an event name of up to 160 characters.")
        row["title"] = title
    else:
        target_key = str(payload.get("session_key") or "")
        court = next((court for court in state.get("courts", []) if court.get("id") == payload.get("court_id")), None)
        if not target_key and court and court.get("current"):
            target_key = str(court["current"].get("session_key") or "")
        if target_key in sessions and _is_locked(sessions[target_key]) and action not in {"remove_group", "cancel_reservation", "release_game"}:
            raise ValueError("Submitted results are locked for administrator review.")
        state, events = apply_hub_command(state, events, action=action, payload=payload, now=_now())
    _rpc_commit(db, club_id=club_id, hub_id=hub_id, expected_version=expected_version,
                operation_key=idempotency_key, fingerprint=fingerprint, title=str(row["title"]),
                status=status, state=state, sessions=_read_set(sessions, events, state, hub_id), actor_email=actor_email)
    return get_round_robin_hub(db, club_id=club_id, hub_id=hub_id)


def persist_linked_generator_patch(db: Any, *, before: dict[str, Any], patch: dict[str, Any],
                                   expected_version: str) -> dict[str, Any]:
    """Route every original generator mutation through the same court transaction."""
    from jupr_app.domain.round_robin_hub import reconcile_hub_events
    hub_id = str((before.get("state") or {}).get("round_robin_hub_id") or "")
    club_id, session_key = str(before["club_id"]), str(before["session_key"])
    if not hub_id:
        raise ValueError("This session is not connected to a hub.")
    stable_patch = {key: value for key, value in patch.items() if key not in {"updated_at", "last_seen_at"}}
    fingerprint = _fingerprint({"session_key": session_key, "expected_version": expected_version, "patch": stable_patch})
    operation_key = f"generator:{session_key}:{fingerprint[:48]}"
    if _replayed(db, club_id, hub_id, operation_key, fingerprint):
        return _session_rows(db, club_id, [session_key])[session_key]
    row = _hub(db, club_id, hub_id)
    state = copy.deepcopy(row["state"])
    sessions = _session_rows(db, club_id, _group_keys(state))
    current = sessions.get(session_key)
    if not current or str(current["updated_at"]) != str(expected_version) or int(current.get("version") or 1) != int(before.get("version") or 1):
        raise PublicLiveConflictError("The round robin changed. Refresh before continuing.")
    before_events = {key: _event(value) for key, value in sessions.items()}
    after_events = copy.deepcopy(before_events)
    after_events[session_key] = _event({**current, **patch})
    if row.get("status") == "completed" and after_events[session_key].get("status") != "completed":
        raise ValueError("This hub is complete. Create a new event before starting more games.")
    state, after_events = reconcile_hub_events(state, before_events, after_events, now=_now())
    read_set = _read_set(sessions, after_events, state, hub_id)
    for item in read_set:
        if item["session_key"] == session_key:
            normalized = _session_patch(current, after_events[session_key], hub_id)
            requested_state = copy.deepcopy(patch.get("state", current.get("state") or {}))
            if normalized.get("state"):
                requested_state["page_state"] = normalized["state"]["page_state"]
            requested_state["round_robin_hub_id"] = hub_id
            item["patch"] = {**patch, **normalized, "state": requested_state,
                             "expires_at": _retained_expiry(current, patch.get("expires_at"))}
    # Retry identity is tied to the original row and requested change, not volatile hub data.
    committed = _rpc_commit(db, club_id=club_id, hub_id=hub_id, expected_version=int(row["version"]),
                            operation_key=operation_key, fingerprint=fingerprint, title=str(row["title"]),
                            status=str(row["status"]), state=state, sessions=read_set,
                            actor_email=str(before.get("created_by_email") or ""))
    updated = next((value for value in committed.get("sessions", []) if value.get("session_key") == session_key), None)
    if updated is None:
        updated = _session_rows(db, club_id, [session_key])[session_key]
    return updated
