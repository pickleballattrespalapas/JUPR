"""Organizer submissions, sealed review, and retry-safe publication of generators."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any
from uuid import uuid4

from jupr_app.data.load import load_data
from jupr_app.data.paged_reads import read_all_rows
from jupr_app.domain.player_visibility import is_merged_player
from jupr_app.services.admin_guarded_write_service import required_audit_event
from jupr_app.services.leaderboard_service import _leaderboard_settings
from jupr_app.services.admin_play_generator_service import (
    _audit, _publish_payloads, _saved_matches, _session_payload,
)
from jupr_app.services.direct_match_entry_service import submit_atomic_direct_matches
from jupr_app.services.public_live_operation_service import (
    PublicLiveConflictError, begin_public_live_operation, edit_token_matches,
    update_public_live_operation,
)
from jupr_app.services.public_play_generator_service import public_play_generator_session_payload

ADMIN_ROLES = {"administrator", "club_owner", "super_admin"}
GENERATOR_MODES = {"public_play_generator", "admin_play_generator"}
RATING_FIELDS = ("rating", "starting_rating", "wins", "losses", "matches_played")
SINGLES_FIELDS = ("singles_rating", "singles_wins", "singles_losses", "singles_matches_played", "singles_replay_baseline")
RATING_PLAYER_COLUMNS = ",".join(("id", "name", *RATING_FIELDS, *SINGLES_FIELDS))


def _rating_snapshot(player: dict, match_format: str) -> dict:
    fields = SINGLES_FIELDS if match_format == "singles" else RATING_FIELDS
    return {"id": player["id"], "name": player.get("name"),
            **{field: player.get(field) for field in fields}}


def _review_player(player: dict) -> dict:
    ratings = {}
    for match_format in ("doubles", "singles"):
        prefix = "singles_" if match_format == "singles" else ""
        snapshot = _rating_snapshot(player, match_format)
        rating = player.get(f"{prefix}rating")
        if match_format == "singles" and rating is None:
            rating = (player.get("singles_replay_baseline") or {}).get("rating")
        ratings[match_format] = {
            "rating_jupr": float(rating) / 400 if rating is not None else None,
            "matches_played": int(player.get(f"{prefix}matches_played") or 0),
            "can_set_starting_rating": match_format == "doubles" and not any(
                player.get(field) for field in ("wins", "losses", "matches_played")),
            "state_fingerprint": hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(),
        }
    return {"id": player["id"], "name": player.get("name"), "ratings": ratings}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rows(query) -> list[dict]:
    return [copy.deepcopy(row) for row in (query.execute().data or [])]


def _row(db, club_id: str, session_key: str) -> dict:
    rows = _rows(db.table("live_sessions").select("*").eq("club_id", club_id)
                 .eq("session_key", session_key).limit(1))
    if not rows or (rows[0].get("state") or {}).get("mode") not in GENERATOR_MODES:
        raise ValueError("Generator session not found.")
    return rows[0]


def _event(row: dict) -> dict:
    return copy.deepcopy(((row.get("state") or {}).get("page_state") or {}).get("event") or {})


def _session(row: dict) -> dict:
    if row["state"]["mode"] == "public_play_generator":
        return public_play_generator_session_payload(row)
    return _session_payload(row)


def _save(db, row: dict, state: dict, **patch) -> dict:
    updated = _rows(db.table("live_sessions").update({
        "state": state, "version": int(row.get("version") or 1) + 1,
        "updated_at": _now(), **patch,
    }).eq("club_id", row["club_id"]).eq("session_key", row["session_key"])
        .eq("version", int(row.get("version") or 1)))
    if not updated:
        raise PublicLiveConflictError("This session changed. Refresh before continuing.")
    return updated[0]


def _date(value: str) -> str:
    try:
        parsed = date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError("Choose the date these games were played.") from exc
    if parsed > datetime.now(timezone.utc).date() + timedelta(days=1):
        raise ValueError("The match date cannot be in the future.")
    return parsed.isoformat()


def submit_generator_session(
    db, *, club_id: str, session_key: str, expected_version: str | int,
    organizer_name: str, match_date: str, edit_token: str | None = None,
    idempotency_key: str = "", requester_hash: str = "", actor_email: str = "",
) -> dict:
    row = _row(db, club_id, session_key)
    public = row["state"]["mode"] == "public_play_generator"
    if public:
        if not edit_token_matches(edit_token or "", row.get("edit_token_hash") or ""):
            raise PermissionError("Only the organizer can submit this session.")
    elif not actor_email or edit_token is not None:
        raise PermissionError("Sign in to submit this staff session.")
    state = copy.deepcopy(row["state"])
    if state.get("generator_submission"):
        return {"ok": True, "session": _session(row), "idempotent_replay": True}
    version = int(row.get("version") or 1) if public else str(row.get("updated_at") or "")
    if str(version) != str(expected_version):
        raise PublicLiveConflictError("This session changed. Refresh before submitting.")
    if row.get("pending_operation_key"):
        raise PublicLiveConflictError("A previous change is pending. Retry it first.")
    if row.get("status") != "completed":
        raise ValueError("Finish the session before submitting its results.")
    if (state.get("official_publish") or {}).get("published_match_ids"):
        raise ValueError("This session already has published matches.")
    event = _event(row)
    saved = _saved_matches(event)
    if event.get("scoringMode") == "unscored" or not saved:
        raise ValueError("Save scored games before submitting results for approval.")
    name = " ".join(str(organizer_name or "").replace("<", "").replace(">", "").split())[:160]
    if not name:
        raise ValueError("Enter the organizer’s name.")
    played_on = _date(match_date)
    operation = None
    if public:
        operation, _ = begin_public_live_operation(
            db, club_id=club_id, session_key=session_key, action="generator_submit",
            idempotency_key=idempotency_key, requester_hash=requester_hash,
            expected_version=int(expected_version),
            request_payload={"organizer_name": name, "match_date": played_on},
        )
    state["generator_submission"] = {
        "id": str(uuid4()), "status": "pending", "organizer_name": name,
        "submitted_at": _now(), "submitted_by_email": actor_email or None,
        "match_date": played_on, "match_count": len(saved),
        "rating_mode": event.get("ratingMode") or ("unrated" if public else "rated"),
    }
    patch = {"last_operation_key": operation["operation_key"]} if operation else {}
    updated = _save(db, row, state, **patch)
    if operation:
        update_public_live_operation(db, club_id=club_id,
            operation_key_value=operation["operation_key"], status="completed", result={})
    return {"ok": True, "session": _session(updated), "idempotent_replay": False}


def require_reviewer(role: str) -> None:
    if role not in ADMIN_ROLES:
        raise PermissionError("Only a club administrator can approve or reject results.")


def list_generator_submissions(db, *, club_id: str, actor_role: str, status: str = "pending") -> dict:
    require_reviewer(actor_role)
    statuses = ["pending", "processing"] if status == "pending" else [status]
    rows = _rows(db.table("live_sessions").select("*").eq("club_id", club_id)
        .in_("state->generator_submission->>status", statuses).order("updated_at", desc=True).limit(200))
    submissions = []
    for row in rows:
        if (row.get("state") or {}).get("mode") not in GENERATOR_MODES:
            continue
        event = _event(row)
        saved = _saved_matches(event)
        played = {str(pid) for _, match in saved for side in ("A", "B")
                  for pid in (match.get(f"side{side}") or match.get(f"team{side}") or [])}
        submissions.append({
            "session_key": row["session_key"], "title": row.get("title") or event.get("name"),
            "version": int(row.get("version") or 1), "generator_kind": event.get("generatorKind"),
            "play_format": event.get("playFormat"), **{k: v for k, v in row["state"]["generator_submission"].items() if k not in {"review", "results"}},
            "review": {"request": (row["state"]["generator_submission"].get("review") or {}).get("request")} if row["state"]["generator_submission"].get("review") else None,
            "participants": [p for p in event.get("participants", []) if str(p.get("id")) in played],
            "matches": [{"round_number": number, **match} for number, match in saved],
        })
    players = [_review_player(p) for p in read_all_rows(
        lambda: db.table("players").select(RATING_PLAYER_COLUMNS).eq("club_id", club_id)
    ) if not is_merged_player(p)]
    players.sort(key=lambda p: (str(p.get("name") or "").casefold(), p["id"]))
    return {"ok": True, "submissions": submissions, "players": players}


def update_generator_submission_title(
    db, *, club_id: str, session_key: str, expected_version: int, title: str,
    actor_email: str, actor_role: str,
) -> dict:
    """Correct pending results' label without reopening or rebuilding the games."""
    require_reviewer(actor_role)
    clean_title = " ".join(str(title or "").replace("<", "").replace(">", "").split())
    if not clean_title or len(clean_title) > 160:
        raise ValueError("Enter a title between 1 and 160 characters.")
    row = _row(db, club_id, session_key)
    if int(row.get("version") or 1) != int(expected_version):
        raise PublicLiveConflictError("This submission changed. Refresh before editing its title.")
    if row.get("pending_operation_key"):
        raise PublicLiveConflictError("A previous change is pending. Retry it first.")
    state = copy.deepcopy(row["state"])
    submission = state.get("generator_submission") or {}
    if (row.get("status") != "completed" or submission.get("status") != "pending"
            or (state.get("official_publish") or {}).get("published_match_ids")):
        raise ValueError("The title can only be edited while results are awaiting approval.")
    event = _event(row)
    before = {"title": row.get("title"), "event_name": event.get("name"),
              "version": int(row.get("version") or 1)}
    event["name"] = clean_title
    state["page_state"]["event"] = event
    state["page_state"]["event_name"] = clean_title
    state["event_name"] = clean_title
    audit = dict(club_id=club_id, actor_email=actor_email, actor_role=actor_role,
        entity_type="play_generator_session", entity_id=session_key,
        source="generator_submission_title", before=before,
        after={"title": clean_title, "event_name": clean_title, "version": before["version"] + 1})
    required_audit_event(db, **audit, action_type="update_generator_submission_title_intent", intent=True)
    updated = _save(db, row, state, title=clean_title)
    required_audit_event(db, **audit, action_type="update_generator_submission_title", intent=False)
    return {"ok": True, "session": _session(updated), "title": clean_title,
            "version": int(updated["version"])}


def set_generator_starting_rating(
    db, *, club_id: str, session_key: str, participant_id: str, player_id: int,
    match_format: str, rating_jupr: float, expected_state_fingerprint: str,
    actor_email: str, actor_role: str,
) -> dict:
    """Set a new doubles player's current and replay baseline before approval.

    This is an absolute assignment. After an interrupted response, refresh to
    read the saved level; stale retries cannot overwrite later edits or games.
    Singles replay baselines are immutable and remain outside this workflow.
    """
    require_reviewer(actor_role)
    if match_format != "doubles":
        raise ValueError("Starting levels in this review are for doubles.")
    if not math.isfinite(rating_jupr) or not 1 <= rating_jupr <= 7:
        raise ValueError("Starting level must be between 1.0 and 7.0.")
    row = _row(db, club_id, session_key)
    submission = row["state"].get("generator_submission") or {}
    if row.get("status") != "completed" or submission.get("status") != "pending" or submission.get("rating_mode") != "rated":
        raise ValueError("Starting levels can only be set before approving rated results.")
    if row.get("pending_operation_key"):
        raise PublicLiveConflictError("A review is already in progress. Refresh before continuing.")
    relevant = False
    for _, match in _saved_matches(_event(row)):
        side_a = match.get("sideA") or match.get("teamA") or []
        side_b = match.get("sideB") or match.get("teamB") or []
        if len(side_a) == len(side_b) == 2 and str(participant_id) in {str(pid) for pid in side_a + side_b}:
            relevant = True
            break
    if not relevant:
        raise ValueError("Choose a player who has scored doubles games.")
    players = _rows(db.table("players").select(RATING_PLAYER_COLUMNS).eq("club_id", club_id).eq("id", player_id).limit(1))
    if not players or is_merged_player(players[0]):
        raise ValueError("Choose a current player profile in this club.")
    before = players[0]
    rating = _review_player(before)["ratings"]["doubles"]
    if not rating["can_set_starting_rating"]:
        raise ValueError("This player already has rated games. Keep their current rating or use Player Editor for a reviewed correction.")
    if rating["state_fingerprint"] != expected_state_fingerprint:
        raise PublicLiveConflictError("This player's rating changed. Refresh and check their level before saving.")
    patch = {"rating": round(rating_jupr * 400, 4), "starting_rating": round(rating_jupr * 400, 4)}
    audit = dict(club_id=club_id, actor_email=actor_email, actor_role=actor_role,
        entity_type="player", entity_id=str(player_id), source="generator_starting_rating",
        before=before, after={"session_key": session_key, "match_format": match_format, "patch": patch})
    required_audit_event(db, **audit, action_type="set_generator_starting_rating_intent", intent=True)
    query = db.table("players").update(patch).eq("club_id", club_id)
    for field, value in _rating_snapshot(before, "doubles").items():
        query = query.is_(field, "null") if value is None else query.eq(field, value)
    updated = _rows(query)
    if not updated:
        raise PublicLiveConflictError("This player's rating changed. Refresh and check their level before saving.")
    required_audit_event(db, **audit, action_type="set_generator_starting_rating", intent=False)
    return {"ok": True, "player": _review_player(updated[0])}


def review_generator_submission(
    db, *, club_id: str, session_key: str, expected_version: int, action: str,
    player_ids: dict[str, int], match_date: str, reason: str,
    actor_email: str, actor_role: str,
) -> dict:
    require_reviewer(actor_role)
    if action not in {"approve", "reject"}:
        raise ValueError("Choose approve or reject.")
    row = _row(db, club_id, session_key)
    state = copy.deepcopy(row["state"])
    submission = state.get("generator_submission") or {}
    if not submission or row.get("status") != "completed":
        raise ValueError("This session has not been submitted for approval.")
    rating_mode = submission["rating_mode"]
    request = {"action": action, "player_ids": {str(k): int(v) for k, v in player_ids.items()},
               "match_date": _date(match_date), "reason": str(reason or "").strip()[:500]}
    fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    review = submission.get("review") or {}
    if submission["status"] != "pending":
        if review.get("fingerprint") != fingerprint:
            raise PublicLiveConflictError("A decision has already started. Refresh to see it; retry that same decision if needed.")
        if submission["status"] in {"approved", "rejected"}:
            return {"ok": True, "status": submission["status"], "idempotent_replay": True}
    elif int(row.get("version") or 1) != int(expected_version):
        raise PublicLiveConflictError("This submission changed. Refresh before reviewing it.")

    operation_key = hashlib.sha256(f"generator-approval:{club_id}:{submission['id']}".encode()).hexdigest()
    if submission["status"] == "pending":
        if row.get("pending_operation_key"):
            raise PublicLiveConflictError("A previous change is pending. Retry it first.")
        payloads = []
        if action != "reject":
            event = _event(row)
            for participant in event.get("participants", []):
                pid = str(participant.get("id"))
                participant["player_id"] = player_ids.get(pid, participant.get("player_id"))
            zone = ZoneInfo(_leaderboard_settings(db, club_id).get("timezone") or "America/Mazatlan")
            played_at = datetime.combine(date.fromisoformat(request["match_date"]), time(12), zone).isoformat()
            payloads = _publish_payloads(event, session_key=session_key,
                match_date=played_at, operation_key=operation_key, published_ids=set())
            if not payloads:
                raise ValueError("There are no scored games to approve.")
            ids = {int(m[key]) for m in payloads for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2") if m.get(key) is not None}
            players = _rows(db.table("players").select(RATING_PLAYER_COLUMNS).eq("club_id", club_id).in_("id", sorted(ids)))
            if {int(p["id"]) for p in players if not is_merged_player(p)} != ids:
                raise ValueError("Match every player to a profile in this club before approving.")
            by_id = {int(p["id"]): p for p in players}
            for match in payloads:
                match_ids = [match[k] for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2") if match.get(k) is not None]
                if len(set(match_ids)) != len(match_ids):
                    raise ValueError("A player cannot appear more than once in the same game.")
                if rating_mode == "rated" and match["match_format"] == "doubles":
                    for pid in match_ids:
                        value = by_id[pid].get("rating")
                        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
                            raise ValueError(f"Set the doubles starting level for {by_id[pid]['name']} before approving.")
                match["rating_scope"] = "unrated" if rating_mode == "unrated" else "overall_only"
        review = {"request": request, "fingerprint": fingerprint, "payloads": payloads,
                  "actor_email": actor_email, "actor_role": actor_role, "started_at": _now()}
        submission.update({"status": "processing", "review": review})
        row = _save(db, row, state, pending_operation_key=operation_key,
                    pending_operation_action="generator_approval")
    elif row.get("pending_operation_key") != operation_key:
        raise PublicLiveConflictError("This review needs recovery before it can continue.")

    # Reservation freezes the chosen mode and identity mapping. Stable receipts let
    # an interrupted approval resume, including a mixed singles/doubles session.
    results = []
    if action != "reject":
        for match_format in ("doubles", "singles"):
            matches = [m for m in review["payloads"] if m["match_format"] == match_format]
            for start in range(0, len(matches), 200):
                # Every batch needs current ratings and shared activity fields,
                # including when a player appears in both singles and doubles.
                data = load_data(db, club_id)
                results.append(submit_atomic_direct_matches(
                    db, club_id=club_id, matches=matches[start:start + 200], match_format=match_format,
                    idempotency_key=f"generator-approval-{operation_key}:{match_format}:{start // 200}",
                    actor_email=review["actor_email"], actor_role=review["actor_role"],
                    source="generator_admin_approval", name_to_id=data[7],
                    df_players_all=data[0], df_leagues=data[2], df_meta=data[4],
                ))
    submission.update({"status": "rejected" if action == "reject" else "approved",
        "approved_mode": None if action == "reject" else rating_mode, "reviewed_at": _now(),
        "rejection_reason": request["reason"] if action == "reject" else None,
        "results": results})
    updated = _save(db, row, state, pending_operation_key=None, pending_operation_action=None,
                    last_operation_key=operation_key)
    _audit(db, club_id=club_id, actor_email=review["actor_email"], actor_role=review["actor_role"],
        action_type="review_generator_submission", entity_id=session_key,
        before_json={"status": "pending"}, after_json={"status": submission["status"],
            "approved_mode": submission["approved_mode"], "match_count": submission["match_count"]},
        source="generator_admin_approval")
    return {"ok": True, "status": submission["status"], "session": _session(updated), "warnings": [w for r in results for w in r.get("warnings", [])]}
