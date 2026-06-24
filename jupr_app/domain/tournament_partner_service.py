from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

PENDING_STATUS = "PENDING"
ACCEPTED_STATUS = "ACCEPTED"
DECLINED_STATUS = "DECLINED"
CANCELLED_STATUS = "CANCELLED"
ADMIN_CONFIRMED_STATUS = "ADMIN_CONFIRMED"
ACTIVE_MEMBER_STATUS = "ACTIVE"
CONFIRMED_LINK_STATUSES = {"CONFIRMED", "ADMIN_CONFIRMED"}


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_data(resp: Any) -> list[dict[str, Any]]:
    try:
        return list(resp.data or [])
    except Exception:
        return []


def _safe_first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_data(resp)
    return rows[0] if rows else None


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _table(supabase, table_name: str):
    return supabase.table(table_name)


def _get_selection(supabase, selection_id: str) -> dict[str, Any]:
    selection_id = _safe_text(selection_id)
    if not selection_id:
        raise ValueError("Selection ID is required.")
    row = _safe_first(
        _table(supabase, "tournament_registration_selections")
        .select("*")
        .eq("id", selection_id)
        .limit(1)
        .execute()
    )
    if not row:
        raise ValueError("Registration selection was not found.")
    return row


def _get_registration(supabase, registration_id: str) -> dict[str, Any]:
    registration_id = _safe_text(registration_id)
    if not registration_id:
        raise ValueError("Registration ID is required.")
    row = _safe_first(
        _table(supabase, "tournament_registrations")
        .select("*")
        .eq("id", registration_id)
        .limit(1)
        .execute()
    )
    if not row:
        raise ValueError("Registration was not found.")
    return row


def _get_request(supabase, request_id: str) -> dict[str, Any]:
    request_id = _safe_text(request_id)
    if not request_id:
        raise ValueError("Partner request ID is required.")
    row = _safe_first(
        _table(supabase, "tournament_registration_partner_requests")
        .select("*")
        .eq("id", request_id)
        .limit(1)
        .execute()
    )
    if not row:
        raise ValueError("Partner request was not found.")
    return row


def _active_team_members_for_selection(supabase, *, event_option_id: str, selection_id: str) -> list[dict[str, Any]]:
    return _safe_data(
        _table(supabase, "tournament_registration_team_members")
        .select("*")
        .eq("event_option_id", _safe_text(event_option_id))
        .eq("selection_id", _safe_text(selection_id))
        .eq("status", ACTIVE_MEMBER_STATUS)
        .execute()
    )


def _ensure_not_confirmed(supabase, *, event_option_id: str, selection_id: str) -> None:
    if _active_team_members_for_selection(supabase, event_option_id=event_option_id, selection_id=selection_id):
        raise ValueError("Selection is already on a confirmed team for this division.")


def _pending_requests_for_event(supabase, *, event_option_id: str) -> list[dict[str, Any]]:
    return _safe_data(
        _table(supabase, "tournament_registration_partner_requests")
        .select("*")
        .eq("event_option_id", _safe_text(event_option_id))
        .eq("status", PENDING_STATUS)
        .execute()
    )


def _update_selection_partner_mode(supabase, *, selection_id: str, partner_mode: str) -> dict[str, Any]:
    updated = _safe_first(
        _table(supabase, "tournament_registration_selections")
        .update({"partner_mode": partner_mode})
        .eq("id", _safe_text(selection_id))
        .execute()
    )
    if not updated:
        raise ValueError("Registration selection could not be updated.")
    return updated


def _update_request_status(supabase, *, request_id: str, status: str, actor_user_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status, "updated_at": _now_iso()}
    if status in {ACCEPTED_STATUS, DECLINED_STATUS, CANCELLED_STATUS, ADMIN_CONFIRMED_STATUS}:
        payload["responded_at"] = payload["updated_at"]
    if actor_user_id:
        payload["created_by_user_id"] = actor_user_id
    updated = _safe_first(
        _table(supabase, "tournament_registration_partner_requests")
        .update(payload)
        .eq("id", _safe_text(request_id))
        .execute()
    )
    if not updated:
        raise ValueError("Partner request could not be updated.")
    return updated


def _cancel_competing_pending_requests(
    supabase,
    *,
    event_option_id: str,
    selection_ids: set[str],
    exclude_request_id: str | None = None,
) -> list[dict[str, Any]]:
    cancelled: list[dict[str, Any]] = []
    for request in _pending_requests_for_event(supabase, event_option_id=event_option_id):
        request_id = _safe_text(request.get("id"))
        if exclude_request_id and request_id == _safe_text(exclude_request_id):
            continue
        requester = _safe_text(request.get("requester_selection_id"))
        target = _safe_text(request.get("target_selection_id"))
        if requester in selection_ids or target in selection_ids:
            cancelled.append(_update_request_status(supabase, request_id=request_id, status=CANCELLED_STATUS))
    return cancelled


def _validate_same_event_pair(selection1: dict[str, Any], selection2: dict[str, Any]) -> None:
    if _safe_text(selection1.get("id")) == _safe_text(selection2.get("id")):
        raise ValueError("A player cannot request themselves as a partner.")
    if _safe_text(selection1.get("tournament_id")) != _safe_text(selection2.get("tournament_id")):
        raise ValueError("Partner selections must be in the same tournament.")
    if _safe_text(selection1.get("event_option_id")) != _safe_text(selection2.get("event_option_id")):
        raise ValueError("Partner selections must be in the same division.")


def _request_targets_same_pending(request: dict[str, Any], *, target_selection_id: str | None, target_player_id: Any | None) -> bool:
    if target_selection_id and _safe_text(request.get("target_selection_id")) == _safe_text(target_selection_id):
        return True
    if target_player_id not in (None, "") and _safe_text(request.get("target_player_id")) == _safe_text(target_player_id):
        return True
    if not target_selection_id and target_player_id in (None, "") and not request.get("target_selection_id") and not request.get("target_player_id"):
        return True
    return False


def _selection_player_id(selection: dict[str, Any], registration: dict[str, Any]) -> Any | None:
    return selection.get("player_id") or registration.get("player_id")


def mark_needs_partner(supabase, *, selection_id: str) -> dict[str, Any]:
    selection = _get_selection(supabase, selection_id)
    _ensure_not_confirmed(
        supabase,
        event_option_id=_safe_text(selection.get("event_option_id")),
        selection_id=_safe_text(selection.get("id")),
    )
    _cancel_competing_pending_requests(
        supabase,
        event_option_id=_safe_text(selection.get("event_option_id")),
        selection_ids={_safe_text(selection.get("id"))},
    )
    return _update_selection_partner_mode(supabase, selection_id=_safe_text(selection.get("id")), partner_mode="NEEDS_PARTNER")


def create_partner_request(
    supabase,
    *,
    tournament_id: str,
    event_option_id: str,
    requester_selection_id: str,
    target_selection_id: str | None = None,
    target_player_id: Any | None = None,
    target_display_name_snapshot: str | None = None,
    source: str,
) -> dict[str, Any]:
    requester_selection = _get_selection(supabase, requester_selection_id)
    requester_registration = _get_registration(supabase, _safe_text(requester_selection.get("registration_id")))
    if _safe_text(requester_selection.get("tournament_id")) != _safe_text(tournament_id):
        raise ValueError("Requester selection is not in this tournament.")
    if _safe_text(requester_selection.get("event_option_id")) != _safe_text(event_option_id):
        raise ValueError("Requester selection is not in this division.")

    target_selection = None
    target_registration = None
    if target_selection_id:
        target_selection = _get_selection(supabase, target_selection_id)
        _validate_same_event_pair(requester_selection, target_selection)
        target_registration = _get_registration(supabase, _safe_text(target_selection.get("registration_id")))
        _ensure_not_confirmed(supabase, event_option_id=_safe_text(event_option_id), selection_id=_safe_text(target_selection.get("id")))
    elif target_player_id not in (None, "") and _safe_text(target_player_id) == _safe_text(_selection_player_id(requester_selection, requester_registration)):
        raise ValueError("A player cannot request themselves as a partner.")

    _ensure_not_confirmed(supabase, event_option_id=_safe_text(event_option_id), selection_id=_safe_text(requester_selection.get("id")))

    for request in _pending_requests_for_event(supabase, event_option_id=_safe_text(event_option_id)):
        if _safe_text(request.get("requester_selection_id")) != _safe_text(requester_selection_id):
            continue
        if _request_targets_same_pending(request, target_selection_id=target_selection_id, target_player_id=target_player_id):
            raise ValueError("A pending partner request already exists for this target.")

    payload = {
        "id": _uid("preq"),
        "tournament_id": _safe_text(tournament_id),
        "event_option_id": _safe_text(event_option_id),
        "requester_selection_id": _safe_text(requester_selection.get("id")),
        "requester_registration_id": _safe_text(requester_selection.get("registration_id")),
        "requester_player_id": _selection_player_id(requester_selection, requester_registration),
        "target_selection_id": _safe_text(target_selection.get("id")) if target_selection else None,
        "target_registration_id": _safe_text(target_selection.get("registration_id")) if target_selection else None,
        "target_player_id": _selection_player_id(target_selection, target_registration) if target_selection and target_registration else target_player_id,
        "target_display_name_snapshot": _safe_text(target_display_name_snapshot) or None,
        "status": PENDING_STATUS,
        "source": _safe_text(source),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "created_by_registration_id": _safe_text(requester_selection.get("registration_id")),
    }
    created = _safe_first(_table(supabase, "tournament_registration_partner_requests").insert(payload).execute())
    if not created:
        raise ValueError("Partner request could not be created.")
    return created


def _create_team_link(
    supabase,
    *,
    tournament_id: str,
    event_option_id: str,
    selection1: dict[str, Any],
    selection2: dict[str, Any],
    request: dict[str, Any] | None,
    status: str,
    admin_user_id: str | None = None,
) -> dict[str, Any]:
    reg1 = _get_registration(supabase, _safe_text(selection1.get("registration_id")))
    reg2 = _get_registration(supabase, _safe_text(selection2.get("registration_id")))
    link_id = _uid("tlink")
    now = _now_iso()
    link_payload = {
        "id": link_id,
        "tournament_id": _safe_text(tournament_id),
        "event_option_id": _safe_text(event_option_id),
        "registration1_id": _safe_text(selection1.get("registration_id")),
        "registration2_id": _safe_text(selection2.get("registration_id")),
        "selection1_id": _safe_text(selection1.get("id")),
        "selection2_id": _safe_text(selection2.get("id")),
        "player1_id": _selection_player_id(selection1, reg1),
        "player2_id": _selection_player_id(selection2, reg2),
        "status": status,
        "accepted_request_id": _safe_text((request or {}).get("id")) or None,
        "created_at": now,
        "updated_at": now,
        "created_by_user_id": admin_user_id,
    }
    created_link = _safe_first(_table(supabase, "tournament_registration_team_links").insert(link_payload).execute())
    if not created_link:
        raise ValueError("Partner team link could not be created.")

    member_rows = [
        {
            "id": _uid("tmem"),
            "team_link_id": link_id,
            "tournament_id": _safe_text(tournament_id),
            "event_option_id": _safe_text(event_option_id),
            "selection_id": _safe_text(selection1.get("id")),
            "registration_id": _safe_text(selection1.get("registration_id")),
            "player_id": _selection_player_id(selection1, reg1),
            "player_order": 1,
            "status": ACTIVE_MEMBER_STATUS,
            "created_at": now,
        },
        {
            "id": _uid("tmem"),
            "team_link_id": link_id,
            "tournament_id": _safe_text(tournament_id),
            "event_option_id": _safe_text(event_option_id),
            "selection_id": _safe_text(selection2.get("id")),
            "registration_id": _safe_text(selection2.get("registration_id")),
            "player_id": _selection_player_id(selection2, reg2),
            "player_order": 2,
            "status": ACTIVE_MEMBER_STATUS,
            "created_at": now,
        },
    ]
    _table(supabase, "tournament_registration_team_members").insert(member_rows).execute()
    return created_link


def accept_partner_request(supabase, *, request_id: str, accepted_by_selection_id: str) -> dict[str, Any]:
    request = _get_request(supabase, request_id)
    if _safe_text(request.get("status")) != PENDING_STATUS:
        raise ValueError("Only pending partner requests can be accepted.")
    if _safe_text(request.get("target_selection_id")) != _safe_text(accepted_by_selection_id):
        raise ValueError("Only the requested partner can accept this request.")

    requester_selection = _get_selection(supabase, _safe_text(request.get("requester_selection_id")))
    target_selection = _get_selection(supabase, _safe_text(request.get("target_selection_id")))
    _validate_same_event_pair(requester_selection, target_selection)
    event_option_id = _safe_text(request.get("event_option_id"))
    _ensure_not_confirmed(supabase, event_option_id=event_option_id, selection_id=_safe_text(requester_selection.get("id")))
    _ensure_not_confirmed(supabase, event_option_id=event_option_id, selection_id=_safe_text(target_selection.get("id")))

    link = _create_team_link(
        supabase,
        tournament_id=_safe_text(request.get("tournament_id")),
        event_option_id=event_option_id,
        selection1=requester_selection,
        selection2=target_selection,
        request=request,
        status="CONFIRMED",
    )
    _update_request_status(supabase, request_id=_safe_text(request.get("id")), status=ACCEPTED_STATUS)
    _update_selection_partner_mode(supabase, selection_id=_safe_text(requester_selection.get("id")), partner_mode="HAS_PARTNER")
    _update_selection_partner_mode(supabase, selection_id=_safe_text(target_selection.get("id")), partner_mode="HAS_PARTNER")
    _cancel_competing_pending_requests(
        supabase,
        event_option_id=event_option_id,
        selection_ids={_safe_text(requester_selection.get("id")), _safe_text(target_selection.get("id"))},
        exclude_request_id=_safe_text(request.get("id")),
    )
    return link


def decline_partner_request(supabase, *, request_id: str, declined_by_selection_id: str) -> dict[str, Any]:
    request = _get_request(supabase, request_id)
    if _safe_text(request.get("target_selection_id")) != _safe_text(declined_by_selection_id):
        raise ValueError("Only the requested partner can decline this request.")
    if _safe_text(request.get("status")) != PENDING_STATUS:
        raise ValueError("Only pending partner requests can be declined.")
    return _update_request_status(supabase, request_id=_safe_text(request.get("id")), status=DECLINED_STATUS)


def cancel_partner_request(
    supabase,
    *,
    request_id: str,
    cancelled_by_selection_id: str | None = None,
    admin_user_id: str | None = None,
) -> dict[str, Any]:
    request = _get_request(supabase, request_id)
    if _safe_text(request.get("status")) != PENDING_STATUS:
        raise ValueError("Only pending partner requests can be cancelled.")
    if not admin_user_id:
        actor = _safe_text(cancelled_by_selection_id)
        if actor != _safe_text(request.get("requester_selection_id")):
            raise ValueError("Only the requester or an admin can cancel this request.")
    return _update_request_status(
        supabase,
        request_id=_safe_text(request.get("id")),
        status=CANCELLED_STATUS,
        actor_user_id=admin_user_id,
    )


def admin_confirm_partner_link(
    supabase,
    *,
    tournament_id: str,
    event_option_id: str,
    selection1_id: str,
    selection2_id: str,
    admin_user_id: str | None = None,
    source: str = "ADMIN_RECONCILIATION",
) -> dict[str, Any]:
    selection1 = _get_selection(supabase, selection1_id)
    selection2 = _get_selection(supabase, selection2_id)
    _validate_same_event_pair(selection1, selection2)
    if _safe_text(selection1.get("tournament_id")) != _safe_text(tournament_id):
        raise ValueError("Selections are not in this tournament.")
    if _safe_text(selection1.get("event_option_id")) != _safe_text(event_option_id):
        raise ValueError("Selections are not in this division.")
    _ensure_not_confirmed(supabase, event_option_id=_safe_text(event_option_id), selection_id=_safe_text(selection1.get("id")))
    _ensure_not_confirmed(supabase, event_option_id=_safe_text(event_option_id), selection_id=_safe_text(selection2.get("id")))
    reg1 = _get_registration(supabase, _safe_text(selection1.get("registration_id")))
    reg2 = _get_registration(supabase, _safe_text(selection2.get("registration_id")))

    now = _now_iso()
    request_payload = {
        "id": _uid("preq"),
        "tournament_id": _safe_text(tournament_id),
        "event_option_id": _safe_text(event_option_id),
        "requester_selection_id": _safe_text(selection1.get("id")),
        "requester_registration_id": _safe_text(selection1.get("registration_id")),
        "requester_player_id": _selection_player_id(selection1, reg1),
        "target_selection_id": _safe_text(selection2.get("id")),
        "target_registration_id": _safe_text(selection2.get("registration_id")),
        "target_player_id": _selection_player_id(selection2, reg2),
        "target_display_name_snapshot": _safe_text(reg2.get("display_name")) or None,
        "status": ADMIN_CONFIRMED_STATUS,
        "source": _safe_text(source),
        "created_at": now,
        "updated_at": now,
        "responded_at": now,
        "created_by_user_id": admin_user_id,
    }
    request = _safe_first(_table(supabase, "tournament_registration_partner_requests").insert(request_payload).execute())
    if not request:
        raise ValueError("Admin partner confirmation request audit row could not be created.")
    link = _create_team_link(
        supabase,
        tournament_id=_safe_text(tournament_id),
        event_option_id=_safe_text(event_option_id),
        selection1=selection1,
        selection2=selection2,
        request=request,
        status=ADMIN_CONFIRMED_STATUS,
        admin_user_id=admin_user_id,
    )
    _update_selection_partner_mode(supabase, selection_id=_safe_text(selection1.get("id")), partner_mode="HAS_PARTNER")
    _update_selection_partner_mode(supabase, selection_id=_safe_text(selection2.get("id")), partner_mode="HAS_PARTNER")
    _cancel_competing_pending_requests(
        supabase,
        event_option_id=_safe_text(event_option_id),
        selection_ids={_safe_text(selection1.get("id")), _safe_text(selection2.get("id"))},
    )
    return link
