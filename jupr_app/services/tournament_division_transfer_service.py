"""Read-only checks shared by organizer and token-bound division changes.

The database RPC repeats the checks under locks and performs the complete move.
No individual selection, team link, or payment is mutated here.
"""
from __future__ import annotations

from typing import Any
from decimal import Decimal, InvalidOperation

from jupr_app.domain.tournament_registration_repo import (
    TournamentRegistrationEditConflictError,
    TournamentRegistrationImportedDrawError,
    TournamentRegistrationRelationshipLockedError,
    registration_is_imported_to_draw,
)


PARTNER_FIELDS = (
    "partner_mode", "partner_name", "partner_email", "partner_phone", "partner_dupr_id",
    "partner_skill", "partner_age", "partner_gender", "partner_note", "show_on_partner_board",
)


def division_family_key(event: dict[str, Any]) -> tuple[str, str, str, bool, str]:
    """An event family can have divisions on different registration days."""
    return (
        " ".join(str(event.get("event_family_label") or event.get("label") or "Event").lower().split()),
        str(event.get("event_type") or "").strip().upper(),
        str(event.get("gender_restriction") or "ANY").strip().upper(),
        str(event.get("partner_required") or "").lower() in {"1", "true", "yes", "on"},
        str(event.get("competition_format") or "STANDARD").strip().upper(),
    )


def _rows(query: Any) -> list[dict[str, Any]]:
    # Relationship lookup failures must not be treated as unpaired entries.
    return [dict(row) for row in (query.execute().data or [])]


def confirmed_division_team(supabase: Any, *, tournament_id: str, selection: dict[str, Any]) -> dict[str, Any] | None:
    selection_id = str(selection.get("id") or "")
    links: dict[str, dict[str, Any]] = {}
    for side in ("1", "2"):
        for link in _rows(supabase.table("tournament_registration_team_links").select("*")
                          .eq("tournament_id", tournament_id).eq(f"selection{side}_id", selection_id)
                          .in_("status", ["CONFIRMED", "ADMIN_CONFIRMED"])):
            links[str(link.get("id") or "")] = link
    if not links:
        return None
    if len(links) != 1:
        raise TournamentRegistrationRelationshipLockedError("This entry has conflicting partner teams. Please contact tournament staff.")
    link = next(iter(links.values()))
    side = "2" if str(link.get("selection1_id") or "") == selection_id else "1"
    partner_selection_id = str(link.get(f"selection{side}_id") or "")
    partner_registration_id = str(link.get(f"registration{side}_id") or "")
    entries = _rows(supabase.table("tournament_registration_selections").select("*")
                    .eq("tournament_id", tournament_id).eq("id", partner_selection_id))
    registrations = _rows(supabase.table("tournament_registrations").select("*")
                          .eq("tournament_id", tournament_id).eq("id", partner_registration_id))
    partner_entry = entries[0] if len(entries) == 1 else {}
    partner = registrations[0] if len(registrations) == 1 else {}
    if (not partner or not partner_entry
            or str(partner_entry.get("registration_id") or "") != partner_registration_id
            or str(partner_entry.get("event_option_id") or "") != str(selection.get("event_option_id") or "")
            or str(link.get("event_option_id") or "") != str(selection.get("event_option_id") or "")
            or partner_registration_id == str(selection.get("registration_id") or "")):
        raise TournamentRegistrationRelationshipLockedError("The confirmed team no longer matches this entry. Please contact tournament staff.")
    context = {
        "team_link_id": str(link.get("id") or ""),
        "team_link_updated_at": str(link.get("updated_at") or ""),
        "partner_selection_id": partner_selection_id,
        "partner_selection_updated_at": str(partner_entry.get("updated_at") or ""),
        "partner_registration_id": partner_registration_id,
        "partner_registration_updated_at": str(partner.get("updated_at") or ""),
    }
    return {"link": link, "partner_selection": partner_entry, "partner_registration": partner, "context": context}


def hydrated_partner_selection(selection: dict[str, Any], partner: dict[str, Any]) -> dict[str, Any]:
    return {**selection, "partner_mode": "HAS_PARTNER",
            "partner_name": partner.get("display_name") or " ".join(str(partner.get(key) or "").strip() for key in ("first_name", "last_name")).strip(),
            "partner_email": partner.get("email"), "partner_skill": partner.get("doubles_skill"),
            "partner_age": partner.get("age"), "partner_gender": partner.get("gender"),
            "show_on_partner_board": False}


def validate_confirmed_division_transfer(
    supabase: Any, *, club_id: str, tournament_id: str, selection: dict[str, Any],
    registration: dict[str, Any], source_event: dict[str, Any], target_event: dict[str, Any],
    settings: dict[str, Any], team: dict[str, Any], expected_context: Any,
) -> dict[str, Any]:
    from jupr_app.services.public_tournament_registration_service import (
        build_tournament_registration_player_profile, validate_and_clean_tournament_selection,
    )

    if division_family_key(source_event) != division_family_key(target_event):
        raise ValueError("Choose a division within the same event, such as Men's Doubles 4.0 to 4.5.")
    try:
        same_price = Decimal(str(source_event.get("price_usd") or 0)) == Decimal(str(target_event.get("price_usd") or 0))
    except InvalidOperation as exc:
        raise ValueError("Tournament staff must review the division fees before moving this team.") from exc
    if not same_price:
        raise ValueError("These divisions have different entry fees. Tournament staff must review payment before moving the team.")
    if not isinstance(expected_context, dict) or any(
        str(expected_context.get(key) or "") != value for key, value in team["context"].items()
    ):
        raise TournamentRegistrationEditConflictError("This team changed after it was loaded. Refresh and try again.")
    partner = team["partner_registration"]
    partner_entry = team["partner_selection"]
    if any(str(row.get("status") or "").lower() == "cancelled" for row in (registration, partner)):
        raise ValueError("A cancelled registration cannot change divisions.")
    for entry in (selection, partner_entry):
        if registration_is_imported_to_draw(supabase, tournament_id=tournament_id, selection_id=str(entry["id"])):
            raise TournamentRegistrationImportedDrawError("This team is already imported into a draw. Tournament staff must update the draw before changing divisions.")
        for field in ("requester_selection_id", "target_selection_id"):
            if _rows(supabase.table("tournament_registration_partner_requests").select("id")
                     .eq("tournament_id", tournament_id).eq(field, str(entry["id"])).eq("status", "PENDING")):
                raise TournamentRegistrationRelationshipLockedError("Resolve the pending partner request before changing divisions.")
    events = {str(row.get("id") or ""): row for row in _rows(
        supabase.table("tournament_event_options").select("*").eq("tournament_id", tournament_id))}
    for player, current_entry in ((registration, selection), (partner, partner_entry)):
        entries = _rows(supabase.table("tournament_registration_selections").select("id,registration_id,event_option_id")
                        .eq("tournament_id", tournament_id).eq("registration_id", str(player["id"])))
        if any(str(row.get("id") or "") != str(current_entry["id"])
               and division_family_key(events.get(str(row.get("event_option_id") or "")) or {}) == division_family_key(target_event)
               for row in entries):
            raise ValueError("A team member already has another entry in this event. Tournament staff must review it before moving the team.")
    for entry, player, teammate in ((selection, registration, partner), (partner_entry, partner, registration)):
        candidate = {**hydrated_partner_selection(entry, teammate), "event_option_id": str(target_event["id"]),
                     "registration_day_id": str(target_event.get("registration_day_id") or "")}
        validate_and_clean_tournament_selection(
            supabase, club_id=club_id, tournament_id=tournament_id, event=target_event,
            raw_selection=candidate, settings=settings, primary_registration_id=str(player["id"]),
            player_profile=build_tournament_registration_player_profile(supabase, club_id=club_id, registration=player),
        )
    return dict(team["context"])
