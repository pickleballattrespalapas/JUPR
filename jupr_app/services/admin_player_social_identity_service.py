from __future__ import annotations

from typing import Any
import os

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.live_social import normalize_person_name
from jupr_app.services.admin_player_editor_service import is_admin_player_editor_enabled

CONFIRM_LINK_SOCIAL = "LINK SOCIAL"
TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _fetch_players(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    rows = _safe_rows(
        supabase.table("players")
        .select("id,name,active,inactive_at")
        .eq("club_id", str(club_id))
        .order("name", desc=False)
        .execute()
    )
    return [row for row in rows if _safe_int(row.get("id")) is not None]


def _fetch_club_people(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    rows = _safe_rows(
        supabase.table("club_people")
        .select("id,club_id,display_name,normalized_name,linked_player_id,first_seen_on,last_seen_on")
        .eq("club_id", str(club_id))
        .order("last_seen_on", desc=True)
        .execute()
    )
    return rows


def _person_payload(row: dict[str, Any], *, player_names: dict[int, str]) -> dict[str, Any]:
    linked_id = _safe_int(row.get("linked_player_id"))
    return {
        "id": str(row.get("id") or ""),
        "club_id": str(row.get("club_id") or ""),
        "display_name": _clean_text(row.get("display_name"), limit=160),
        "normalized_name": _clean_text(row.get("normalized_name"), limit=160),
        "linked_player_id": linked_id,
        "linked_player_name": player_names.get(int(linked_id)) if linked_id is not None else None,
        "first_seen_on": row.get("first_seen_on"),
        "last_seen_on": row.get("last_seen_on"),
    }


def _player_payload(row: dict[str, Any]) -> dict[str, Any]:
    pid = _safe_int(row.get("id"))
    return {
        "id": int(pid or 0),
        "name": _clean_text(row.get("name"), limit=160),
        "active": bool(row.get("active", True)) and not bool(row.get("inactive_at")),
    }


def list_admin_player_social_identities(supabase: Any, *, club_id: str) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    players = _fetch_players(supabase, club_id=str(club_id))
    player_names = {int(row["id"]): _clean_text(row.get("name"), limit=160) for row in players if _safe_int(row.get("id")) is not None}
    people = [_person_payload(row, player_names=player_names) for row in _fetch_club_people(supabase, club_id=str(club_id))]
    linked = len([row for row in people if row.get("linked_player_id") is not None])
    return {
        "ok": True,
        "mode": "player_social_identity_list",
        "people": people,
        "players": [_player_payload(row) for row in players],
        "summary": {"people": len(people), "linked": linked, "unlinked": max(0, len(people) - linked)},
    }


def _fetch_person(supabase: Any, *, club_id: str, club_person_id: str) -> dict[str, Any] | None:
    rows = _safe_rows(
        supabase.table("club_people")
        .select("id,club_id,display_name,normalized_name,linked_player_id,first_seen_on,last_seen_on")
        .eq("club_id", str(club_id))
        .eq("id", str(club_person_id))
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def _fetch_player(supabase: Any, *, club_id: str, player_id: int) -> dict[str, Any] | None:
    rows = _safe_rows(
        supabase.table("players")
        .select("id,name,active,inactive_at")
        .eq("club_id", str(club_id))
        .eq("id", int(player_id))
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def update_admin_player_social_identity(
    supabase: Any,
    *,
    club_id: str,
    club_person_id: str,
    linked_player_id: Any = None,
    display_name: str | None = None,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    source: str = "next_player_editor_social_identity",
) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_LINK_SOCIAL:
        raise ValueError(f"Type {CONFIRM_LINK_SOCIAL} to save social identity links.")
    clean_id = _clean_text(club_person_id, limit=120)
    if not clean_id:
        raise ValueError("club_person_id is required")
    before = _fetch_person(supabase, club_id=str(club_id), club_person_id=clean_id)
    if not before:
        raise ValueError("social identity not found")

    patch: dict[str, Any] = {}
    player_id = _safe_int(linked_player_id)
    if linked_player_id not in (None, "") and player_id is None:
        raise ValueError("linked_player_id must be a player id or blank to unlink.")
    if player_id is not None:
        player = _fetch_player(supabase, club_id=str(club_id), player_id=player_id)
        if not player:
            raise ValueError("linked player not found")
        patch["linked_player_id"] = int(player_id)
    else:
        patch["linked_player_id"] = None

    if display_name is not None:
        clean_display = _clean_text(display_name, limit=160)
        if not clean_display:
            raise ValueError("display_name cannot be blank.")
        patch["display_name"] = clean_display
        patch["normalized_name"] = normalize_person_name(clean_display)

    updated = _safe_rows(
        supabase.table("club_people")
        .update(patch)
        .eq("club_id", str(club_id))
        .eq("id", clean_id)
        .execute()
    )
    players = _fetch_players(supabase, club_id=str(club_id))
    player_names = {int(row["id"]): _clean_text(row.get("name"), limit=160) for row in players if _safe_int(row.get("id")) is not None}
    after_raw = updated[0] if updated else _fetch_person(supabase, club_id=str(club_id), club_person_id=clean_id)
    after = _person_payload(after_raw or {**before, **patch}, player_names=player_names)
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="update_player_social_identity_admin",
        entity_type="club_people",
        entity_id=clean_id,
        before_json={"club_person": before},
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "patch": patch, "club_person": after},
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "player_social_identity_update", "club_person": after, "warnings": warnings}


def auto_link_admin_player_social_identities(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_player_editor_social_auto_link",
) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_LINK_SOCIAL:
        raise ValueError(f"Type {CONFIRM_LINK_SOCIAL} to auto-link social identities.")
    players = _fetch_players(supabase, club_id=str(club_id))
    by_norm: dict[str, list[dict[str, Any]]] = {}
    for player in players:
        by_norm.setdefault(normalize_person_name(player.get("name")), []).append(player)
    people = _fetch_club_people(supabase, club_id=str(club_id))
    linked: list[dict[str, Any]] = []
    skipped = 0
    for person in people:
        if _safe_int(person.get("linked_player_id")) is not None:
            skipped += 1
            continue
        normalized = normalize_person_name(person.get("normalized_name") or person.get("display_name"))
        candidates = by_norm.get(normalized, [])
        if len(candidates) != 1:
            skipped += 1
            continue
        player_id = int(candidates[0]["id"])
        rows = _safe_rows(
            supabase.table("club_people")
            .update({"linked_player_id": player_id})
            .eq("club_id", str(club_id))
            .eq("id", str(person.get("id")))
            .execute()
        )
        linked.append(rows[0] if rows else {**person, "linked_player_id": player_id})
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="auto_link_player_social_identities_admin",
        entity_type="club_people",
        entity_id="auto_link",
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "linked_count": len(linked), "skipped_count": skipped},
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "player_social_identity_auto_link", "linked_count": len(linked), "skipped_count": skipped, "warnings": warnings}
