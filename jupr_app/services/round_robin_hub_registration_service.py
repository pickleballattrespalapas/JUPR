"""Durable check-in intake and atomic admission to registration-driven groups."""
from __future__ import annotations

from copy import deepcopy
import unicodedata
from typing import Any

from jupr_app.services.admin_round_robin_hub_service import (
    HUB_TABLE, _event, _first, _group_keys, _is_locked, _rows, _session_rows,
)

CHECKIN_TABLE = "round_robin_hub_checkins"


class CheckinRateLimitError(ValueError):
    pass


def clean_checkin_name(value: str) -> str:
    value = " ".join(unicodedata.normalize("NFKC", value).split())
    if not 2 <= len(value) <= 120 or any(unicodedata.category(c).startswith("C") for c in value):
        raise ValueError("Enter your full name (2–120 characters).")
    return value


def name_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _accepting_group(row: dict[str, Any]) -> bool:
    event = _event(row)
    return (row.get("status") == "active" and event.get("status") == "active"
            and not _is_locked(row) and event.get("hubFinishAfterRound") is None)


def public_hub_registration(db: Any, *, hub_id: str) -> dict[str, Any]:
    row = _first(db.table(HUB_TABLE).select("*").eq("id", hub_id).limit(1).execute())
    if row is None:
        raise LookupError("This event check-in page was not found.")
    state = row.get("state") or {}
    open_now = row.get("status") == "active" and state.get("registration_open") is True
    # No session state, roster, profile IDs, staff email, or organizer credentials
    # may cross this public boundary. Closed events do not expose their groups.
    groups = []
    if open_now:
        sessions = _session_rows(db, str(row["club_id"]), _group_keys(state))
        groups = [{"session_key": key, "title": str(session.get("title") or _event(session).get("name") or "Round robin")}
                  for key, session in sessions.items() if _accepting_group(session)]
    return {"ok": True, "event": {"id": row["id"], "title": row["title"],
            "registration_open": open_now, "finished": row.get("status") != "active"}, "groups": groups}


def submit_hub_checkin(db: Any, *, hub_id: str, session_key: str, name: str) -> dict[str, Any]:
    name = clean_checkin_name(name)
    try:
        result = db.rpc("submit_round_robin_hub_checkin", {
            "p_hub_id": hub_id, "p_session_key": session_key,
            "p_name": name, "p_name_key": name_key(name),
        }).execute().data
    except Exception as exc:
        code = str(getattr(exc, "code", ""))
        if code == "P0002":
            raise LookupError("This event check-in page was not found.") from exc
        if code == "P0003":
            raise CheckinRateLimitError("Check-in is busy. Please try again in a minute or ask the event organizer.") from exc
        if code in {"22023", "55000"}:
            raise ValueError("Check-in is closed for this round robin. Refresh the page or ask the event organizer.") from exc
        raise
    if isinstance(result, list):
        result = next(iter(result), None)
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError("Your check-in could not be confirmed. Please retry; this will not create a duplicate.")
    _process_saved_checkins(db, hub_id=hub_id)
    # Never expose roster membership or another check-in to a public caller.
    return {"ok": True, "name": name, "message": "Your check-in is saved. Watch the court board for your match. The organizer will help if your name needs review."}


def _automatic_group(row: dict[str, Any]) -> bool:
    event = _event(row)
    return (_accepting_group(row) and event.get("hubRegistrationDriven") is True
            and event.get("eventFormat", "standard") == "standard" and not event.get("playoff"))


def _pending_automatic(db: Any, row: dict[str, Any], sessions: dict[str, Any]) -> list[dict[str, Any]]:
    decisions = (row.get("state") or {}).get("checkin_decisions") or {}
    checkins = _rows(db.table(CHECKIN_TABLE).select("id,session_key,name,created_at")
                    .eq("club_id", row["club_id"]).eq("hub_id", row["id"])
                    .order("created_at").limit(1000).execute())
    return [c for c in checkins if str(c["id"]) not in decisions
            and c["session_key"] in sessions and _automatic_group(sessions[c["session_key"]])]


def process_automatic_checkins(db: Any, *, row: dict[str, Any], state: dict[str, Any],
                              sessions: dict[str, Any], events: dict[str, Any], now: str
                              ) -> tuple[dict[str, Any], dict[str, Any]]:
    """The admission decision, roster, and generated games commit in one hub CAS."""
    from jupr_app.domain.round_robin_hub import apply_hub_command
    excluded = [key for key, session in sessions.items() if _is_locked(session)]
    for checkin in _pending_automatic(db, row, sessions):
        candidate = deepcopy(state)
        key, checkin_id = str(checkin["session_key"]), str(checkin["id"])
        try:
            action, payload = prepare_checkin_review(db, club_id=row["club_id"], hub_id=row["id"],
                state=candidate, events=events, sessions=sessions,
                payload={"checkin_id": checkin_id, "disposition": "add"},
                actor_email="front-desk-check-in", now=now)
            prior_ids = {p["id"] for p in events[key].get("participants") or []}
            candidate, changed = apply_hub_command(candidate, events, action=action,
                payload={**payload, "excluded_session_keys": excluded}, now=now)
            candidate["checkin_decisions"][checkin_id]["participant_id"] = next(
                p["id"] for p in changed[key]["participants"] if p["id"] not in prior_ids)
            state, events = candidate, changed
        except ValueError as exc:
            # Name conflicts, capacity and roster restrictions need staff review.
            # Save that outcome so polling never endlessly retries the same input.
            state.setdefault("checkin_decisions", {})[checkin_id] = {
                "status": "pending", "session_key": key, "review_reason": str(exc), "reviewed_at": now}
    return state, events


def _process_saved_checkins(db: Any, *, hub_id: str) -> None:
    from jupr_app.services.admin_round_robin_hub_service import command_round_robin_hub
    from jupr_app.services.public_live_operation_service import PublicLiveConflictError
    for _ in range(4):
        row = _first(db.table(HUB_TABLE).select("*").eq("id", hub_id).limit(1).execute())
        if row is None or row["status"] != "active":
            return
        sessions = _session_rows(db, str(row["club_id"]), _group_keys(row.get("state") or {}))
        if not _pending_automatic(db, row, sessions):
            return
        try:
            command_round_robin_hub(db, club_id=str(row["club_id"]), hub_id=hub_id,
                action="process_checkins", payload={}, expected_version=row["version"],
                idempotency_key=f"automatic-checkins-v{row['version']}", actor_email="front-desk-check-in")
            return
        except PublicLiveConflictError:
            continue  # Another check-in or court action won; reread before retrying.
    # Intake is durable. The signed-in hub also retries unprocessed arrivals.


def admin_registration(db: Any, row: dict[str, Any]) -> dict[str, Any]:
    checkins = _rows(db.table(CHECKIN_TABLE).select("id,session_key,name,created_at")
                    .eq("club_id", row["club_id"]).eq("hub_id", row["id"])
                    .order("created_at").limit(1000).execute())
    decisions = (row.get("state") or {}).get("checkin_decisions") or {}
    for checkin in checkins:
        decision = decisions.get(str(checkin["id"])) or {}
        checkin["status"] = decision.get("status", "pending")
        checkin["participant_id"] = decision.get("participant_id")
        checkin["review_reason"] = decision.get("review_reason")
    sessions = _session_rows(db, str(row["club_id"]), _group_keys(row.get("state") or {}))
    return {"open": row.get("status") == "active" and (row.get("state") or {}).get("registration_open") is True,
            "checkins": checkins, "pending_count": sum(c["status"] == "pending" for c in checkins),
            "automatic_pending_count": sum(str(c["id"]) not in decisions and _automatic_group(sessions[c["session_key"]])
                for c in checkins if c["session_key"] in sessions) if row.get("status") == "active" else 0}


def prepare_checkin_review(db: Any, *, club_id: str, hub_id: str, state: dict[str, Any],
                          events: dict[str, Any], sessions: dict[str, Any], payload: dict[str, Any],
                          actor_email: str, now: str) -> tuple[str | None, dict[str, Any]]:
    """Build one atomic hub command: the decision and roster share a commit."""
    checkin_id = str(payload.get("checkin_id") or "")
    checkin = _first(db.table(CHECKIN_TABLE).select("*").eq("club_id", club_id).eq("hub_id", hub_id)
                     .eq("id", checkin_id).limit(1).execute())
    if checkin is None:
        raise ValueError("This check-in was not found in this event.")
    decisions = state.setdefault("checkin_decisions", {})
    if decisions.get(checkin_id, {}).get("status") == "accepted":
        raise ValueError("This check-in has already been added or matched to the roster.")
    disposition = payload.get("disposition")
    key = str(checkin["session_key"])
    decision = {"reviewed_at": now, "reviewed_by": actor_email, "session_key": key}
    if disposition == "dismiss":
        decisions[checkin_id] = {**decision, "status": "dismissed"}
        return None, {}
    if disposition == "restore":
        decisions.pop(checkin_id, None)
        return None, {}
    if key not in sessions or not _accepting_group(sessions[key]):
        raise ValueError("This round robin is no longer accepting players. Review its status in the hub.")
    if disposition == "existing":
        participant_id = str(payload.get("participant_id") or "")
        from jupr_app.domain.round_robin_hub import build_hub_snapshot
        group = next(g for g in build_hub_snapshot(state, events, now=now)["groups"] if g["session_key"] == key)
        participant = next((p for p in group["players"] if p["participant_id"] == participant_id and (p.get("active") or p.get("can_remove"))), None)
        if participant is None:
            raise ValueError("Choose a player already on this round robin's roster.")
        decisions[checkin_id] = {**decision, "status": "accepted", "participant_id": participant_id}
        return None, {}
    if disposition != "add":
        raise ValueError("Choose how to handle this check-in.")
    command = {"session_key": key, "name": clean_checkin_name(str(payload.get("name") or checkin["name"]))}
    if payload.get("player_id") is not None:
        command["player_id"] = payload["player_id"]
    # Names alone are not enough to assert that two people are the same. Staff
    # must explicitly match an existing participant instead of adding a duplicate.
    if any(name_key(str(p.get("name") or "")) == name_key(command["name"]) for p in events[key].get("participants") or []):
        raise ValueError("A player with this name is already on the roster. Match that player, or use a distinguishing name for a different person.")
    decisions[checkin_id] = {**decision, "status": "accepted"}
    return "add_player", command
