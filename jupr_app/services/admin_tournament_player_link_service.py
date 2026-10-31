from __future__ import annotations

from typing import Any
from datetime import datetime

from postgrest.exceptions import APIError

from jupr_app.domain.tournament_registration_repo import StaleTournamentRegistrationAdminError
from jupr_app.services.admin_tournament_service import is_admin_tournament_admin_enabled


def unlinked_draw_registrations(
    registrations: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    draws: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose only confirmed entrants in the displayed draws, without contact data."""
    result = []
    for registration in registrations:
        if registration.get("player_id") is not None:
            continue
        if str(registration.get("status") or "").lower() not in {"confirmed", "admin_confirmed"}:
            continue
        entries = [row for row in selections if row.get("registration_id") == registration.get("id")
                   and str(row.get("partner_mode") or "").upper() != "NEEDS_PARTNER"]
        draw_ids = sorted({str(draw["id"]) for draw in draws for entry in entries
                           if (not draw.get("event_option_id") or str(draw["event_option_id"]) == str(entry.get("event_option_id") or ""))
                           and (not draw.get("registration_day_id") or str(draw["registration_day_id"]) == str(entry.get("registration_day_id") or ""))})
        if draw_ids:
            result.append({
                "id": registration["id"],
                "display_name": registration.get("display_name") or " ".join(filter(None, [registration.get("first_name"), registration.get("last_name")])) or "Unnamed registrant",
                "updated_at": registration.get("updated_at"),
                "draw_ids": draw_ids,
            })
    return sorted(result, key=lambda row: (row["display_name"].casefold(), row["id"]))


def link_admin_tournament_registration_player(
    supabase: Any, *, club_id: str, tournament_id: str, registration_id: str,
    player_id: int, expected_updated_at: str, actor_email: str, actor_role: str,
    confirmation_text: str,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if confirmation_text.strip().upper() != "LINK PLAYER":
        raise ValueError("Confirm the selected profile before linking this registration.")
    try:
        datetime.fromisoformat(expected_updated_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Reload the draw before linking this registration.") from exc
    try:
        result = supabase.rpc("admin_link_tournament_registration_player_v1", {
            "p_club_id": str(club_id), "p_tournament_id": str(tournament_id),
            "p_registration_id": str(registration_id), "p_player_id": player_id,
            "p_expected_updated_at": expected_updated_at,
            "p_actor_email": actor_email, "p_actor_role": actor_role,
        }).execute().data
    except APIError as exc:
        # The DB also protects identity across different divisions and against
        # concurrent legacy registration edits. Its transaction rolls back.
        if exc.code == "23505" and "uq_tournament_registrations_tournament_player" in str(exc):
            raise ValueError("That player already has a registration in this tournament. Resolve the duplicate registration before linking.") from exc
        raise
    if not isinstance(result, dict):
        raise RuntimeError("The profile link could not be verified. Reload the draw to check it before trying again.")
    if not result.get("ok"):
        message = str(result.get("message") or "The registration could not be linked.")
        if result.get("code") == "STALE":
            raise StaleTournamentRegistrationAdminError(message)
        raise ValueError(message)
    return result
