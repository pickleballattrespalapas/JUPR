"""Session-scoped viewing and revocable operation of existing round robins."""
from __future__ import annotations

import base64
import copy
import hmac
import hashlib
from typing import Any
from uuid import uuid4

from jupr_app.domain.adaptive_play_engine import (
    advance_generator_event, complete_generator_event, mark_generator_round_played,
    mutate_generator_roster, reopen_generator_round, save_generator_round,
    select_generator_round, skip_generator_round,
)
from jupr_app.domain.generator_playoffs import start_generator_playoff
from jupr_app.services.public_live_operation_service import (
    PublicLiveConflictError, PublicLiveRecoveryRequiredError, begin_public_live_operation,
    completed_operation_result, edit_token_matches, update_public_live_operation,
)
from jupr_app.services.public_play_generator_service import (
    _event_from_state, _first, _not_expired, _now_iso, _player_ids_for_names,
    _put_event, _state, _token_secret, _ttl_iso, public_play_generator_session_payload,
)

MODES = {"admin_play_generator", "public_play_generator"}


def sharing_row(supabase: Any, *, club_id: str, session_key: str) -> dict[str, Any]:
    row = _first(supabase.table("live_sessions").select("*")
                 .eq("club_id", str(club_id)).eq("session_key", str(session_key)).limit(1).execute())
    if (not row or _state(row).get("mode") not in MODES
            or _event_from_state(_state(row)).get("generatorKind") != "round_robin"
            or row.get("status") not in {"active", "completed"}):
        raise LookupError("We couldn’t find this round robin.")
    if _state(row).get("mode") == "public_play_generator" and not _not_expired(row):
        raise LookupError("We couldn’t find this round robin.")
    return copy.deepcopy(row)


def _sharing(row: dict[str, Any]) -> dict[str, Any]:
    return dict(_state(row).get("generator_sharing") or {})


def _handoff_token(row: dict[str, Any]) -> str:
    nonce = str(_sharing(row).get("handoff_nonce") or "")
    if not nonce:
        return ""
    scope = f"jupr:rr-handoff:v1:{row['club_id']}:{row['session_key']}:{nonce}"
    digest = hmac.new(_token_secret().encode(), scope.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def _version(row: dict[str, Any]) -> str:
    # updated_at is shared by both admin (timestamp leases) and public (integer leases).
    return str(row.get("updated_at") or row.get("created_at") or "")


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    result = public_play_generator_session_payload(row)
    result["version"] = _version(row)
    return result


def _cas(supabase: Any, row: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    # A revoke/owner edit racing a handed-off write wins via the same row lease.
    if _state(row).get("round_robin_hub_id"):
        from jupr_app.services.admin_round_robin_hub_service import persist_linked_generator_patch
        return persist_linked_generator_patch(
            supabase, before=row,
            patch={**patch, "updated_at": _now_iso(), "version": int(row.get("version") or 1) + 1},
            expected_version=_version(row),
        )
    updated = _first(supabase.table("live_sessions").update({
        **patch, "updated_at": _now_iso(), "version": int(row.get("version") or 1) + 1,
    }).eq("club_id", row["club_id"]).eq("session_key", row["session_key"])
        .eq("updated_at", _version(row)).eq("version", int(row.get("version") or 1)).execute())
    if not updated:
        raise PublicLiveConflictError("This session changed. Refresh before continuing.")
    return updated


def manage_generator_sharing(supabase: Any, *, club_id: str, session_key: str,
                             action: str, expected_version: str | int,
                             owner_token: str | None = None,
                             admin_authorized: bool = False) -> dict[str, Any]:
    row = sharing_row(supabase, club_id=club_id, session_key=session_key)
    mode = _state(row)["mode"]
    if not admin_authorized and not (
        mode == "public_play_generator"
        and edit_token_matches(owner_token or "", str(row.get("edit_token_hash") or ""))
    ):
        raise PermissionError("Only the organizer can manage sharing for this round robin.")
    if action not in {"view", "handoff", "revoke"}:
        raise ValueError("Choose a sharing action.")
    club = _first(supabase.table("clubs").select("slug").eq("id", str(club_id)).limit(1).execute())
    if not club or not club.get("slug"):
        raise ValueError("This club’s public link is unavailable.")
    supplied_version = str(expected_version)
    # The admin session projection leases updated_at even when a public-created
    # round robin is managed through its hub. Its original owner still leases
    # the integer version through the public organizer page.
    admin_version = admin_authorized or mode == "admin_play_generator"
    valid_version = _version(row) if admin_version else str(row.get("version") or 1)
    if supplied_version != valid_version:
        raise PublicLiveConflictError("This session changed. Refresh before sharing.")
    state = _state(row)
    settings = _sharing(row)
    before = dict(settings)
    if action != "revoke":
        settings["enabled"] = True
    if action == "handoff":
        if _payload(row)["results_locked"]:
            raise ValueError("Submitted results are locked for administrator review.")
        # Stable until revoked: copy again without disabling the current operator.
        if not settings.get("handoff_nonce"):
            _token_secret()  # Fail before persisting if the signing configuration is absent.
            settings["handoff_nonce"] = uuid4().hex
    elif action == "revoke":
        settings.pop("handoff_nonce", None)
    if settings != before:
        state["generator_sharing"] = settings
        row = _cas(supabase, row, {"state": state})
    session = public_play_generator_session_payload(row)
    if admin_version:
        session["version"] = _version(row)
    return {"ok": True, "session": session,
            "sharing": {"enabled": bool(settings.get("enabled")), "club_slug": str(club["slug"]),
                        "handoff_active": bool(settings.get("handoff_nonce")),
                        "handoff_token": _handoff_token(row) if action == "handoff" else None}}


def get_shared_generator(supabase: Any, *, club_id: str, session_key: str) -> dict[str, Any]:
    row = sharing_row(supabase, club_id=club_id, session_key=session_key)
    if not _sharing(row).get("enabled"):
        raise LookupError("This round robin hasn’t been shared.")
    return {"ok": True, "session": _payload(row)}


def _authorized_operator(row: dict[str, Any], token: str) -> None:
    expected = _handoff_token(row)
    if not _sharing(row).get("enabled") or not expected or not hmac.compare_digest(expected, str(token)):
        raise PermissionError("This handoff link is no longer valid.")


def _mutate_event(supabase: Any, row: dict[str, Any], action: str,
                  body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    event = _event_from_state(_state(row))
    number = int(body.get("round_number") or 1)
    patch: dict[str, Any] = {}
    if action == "scores":
        event = save_generator_round(event, round_number=number, scores=body["scores"])
    elif action == "played":
        event = mark_generator_round_played(event, round_number=number)
        if int(event.get("currentRoundNumber") or 1) == number:
            event = advance_generator_event(event)
    elif action == "skip":
        event = skip_generator_round(event, round_number=number, reason=body.get("reason", ""))
    elif action == "reopen":
        event = reopen_generator_round(event, round_number=number)
    elif action == "select":
        event = select_generator_round(event, round_number=number)
    elif action == "advance":
        event = advance_generator_event(event)
    elif action == "complete":
        event = complete_generator_event(event)
    elif action == "playoff":
        event = start_generator_playoff(event, playoff_format=body["playoff_format"])
    elif action == "roster":
        if body.get("player_id") is not None:
            name = str(body.get("name") or "Player")
            _player_ids_for_names(supabase, club_id=row["club_id"], names=[name],
                                  requested={name: int(body["player_id"])})
        event = mutate_generator_roster(event, **{key: body.get(key) for key in (
            "action", "participant_id", "name", "player_id", "substitute_scope",
            "roster_order", "participant_ids", "court_number")})
    else:
        raise ValueError("This handoff only permits running the round robin.")
    status = str(event.get("status") or row.get("status"))
    patch.update(status=status, completed_at=event.get("completedAt") if status == "completed" else None)
    if status == "active" and _state(row)["mode"] == "public_play_generator":
        patch["expires_at"] = _ttl_iso()
    return event, patch


def mutate_shared_generator(supabase: Any, *, club_id: str, session_key: str,
                            edit_token: str, expected_version: str,
                            idempotency_key: str, requester_hash: str,
                            action: str, body: dict[str, Any]) -> dict[str, Any]:
    row = sharing_row(supabase, club_id=club_id, session_key=session_key)
    _authorized_operator(row, edit_token)
    # Link holders never submit, approve, publish, manage access or change club profiles.
    if _payload(row)["results_locked"]:
        raise ValueError("Submitted results are locked for administrator review.")
    if row.get("pending_operation_key"):
        raise PublicLiveRecoveryRequiredError("A previous change is still pending. Ask the organizer to recover it.")
    operation, existed = begin_public_live_operation(
        supabase, club_id=club_id, session_key=session_key, action=f"shared_generator_{action}",
        idempotency_key=idempotency_key, requester_hash=requester_hash,
        # Shared clients lease updated_at; retain that immutable lease in the
        # request fingerprint so response-loss retries survive a version increment.
        expected_version=1,
        request_payload={"session_key": session_key, "expected_version": expected_version, **body},
    )
    operation_key = str(operation["operation_key"])
    if completed_operation_result(operation) is not None or (
        existed and row.get("last_operation_key") == operation_key
    ):
        update_public_live_operation(supabase, club_id=club_id, operation_key_value=operation_key,
                                     status="completed", result={})
        return {"ok": True, "session": _payload(row), "idempotent_replay": True}
    try:
        if str(expected_version) != _version(row):
            raise PublicLiveConflictError("This session changed. Refresh before continuing.")
        event, patch = _mutate_event(supabase, row, action, body)
        updated = _cas(supabase, row, {
            **patch, "state": _put_event(_state(row), event), "last_seen_at": _now_iso(),
            "last_operation_key": operation_key,
            "last_request_fingerprint": operation["request_fingerprint"],
        })
    except (ValueError, PublicLiveConflictError) as exc:
        update_public_live_operation(supabase, club_id=club_id, operation_key_value=operation_key,
                                     status="rejected", error_text=str(exc))
        raise
    except Exception:
        # An ambiguous response is recovered by the row's durable operation receipt.
        update_public_live_operation(supabase, club_id=club_id, operation_key_value=operation_key,
                                     status="recovery_required", error_text="Shared RR change needs recovery.")
        raise
    update_public_live_operation(supabase, club_id=club_id, operation_key_value=operation_key,
                                 status="completed", result={})
    return {"ok": True, "session": _payload(updated), "idempotent_replay": False}
