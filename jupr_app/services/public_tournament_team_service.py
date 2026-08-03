from __future__ import annotations

import os
from typing import Any, Callable

from jupr_app.domain.notifications.tournament_team_invitation_email import (
    send_team_invitation_email,
)
from jupr_app.domain.tournament_admin_operations import (
    build_tournament_admin_operation_request,
)
from jupr_app.domain.tournament_four_player_team import build_team_standings
from jupr_app.domain.tournament_registration_confirmation_tokens import (
    verify_registration_confirmation_token,
)
from jupr_app.domain.tournament_team_invitation_tokens import (
    tournament_team_invitation_token_hash,
    verify_tournament_team_invitation_token,
)
from jupr_app.services.admin_tournament_team_competition_service import (
    _rpc_data,
    _rows,
    create_four_player_team,
    is_admin_team_tournament_enabled,
)
from jupr_app.services.staging_write_guard import staging_write_wave_allows
from jupr_app.services.production_tournament_guard import require_production_tournament_writes

PUBLIC_TEAM_WRITE_WAVE = "public-intake-auth"
PUBLIC_INTAKE_WRITE_FLAG = "JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES"
TRUTHY = {"1", "true", "yes", "y", "on"}


def require_public_team_tournament_mutation_runtime() -> None:
    if not is_admin_team_tournament_enabled():
        raise PermissionError("Four-player team registration is disabled.")
    environment = os.getenv("JUPR_ENV", "").strip().lower()
    if environment == "production":
        require_production_tournament_writes()
        if os.getenv(PUBLIC_INTAKE_WRITE_FLAG, "").strip().lower() not in TRUTHY:
            raise PermissionError(f"Production four-player team writes require {PUBLIC_INTAKE_WRITE_FLAG}=1.")
    elif environment == "staging":
        if not staging_write_wave_allows(PUBLIC_TEAM_WRITE_WAVE) or os.getenv(PUBLIC_INTAKE_WRITE_FLAG, "").strip().lower() not in TRUTHY:
            raise PermissionError("Four-player team writes are disabled.")
    else:
        return
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        raise RuntimeError("Four-player team hosted writes require the server-only database credential.")


def _one(
    supabase: Any,
    table: str,
    *,
    filters: tuple[tuple[str, Any], ...],
) -> dict[str, Any] | None:
    rows = _rows(supabase, table, filters=filters, limit=2)
    return rows[0] if rows else None


def _public_team(team: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": team.get("id"),
        "name": team.get("name"),
        "draw_id": team.get("draw_id"),
        "status": team.get("status"),
    }


def _public_member(member: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": member.get("id"),
        "team_id": member.get("team_id"),
        "slot": member.get("slot"),
        "display_name": member.get("display_name_snapshot") or "Player",
        "status": member.get("status"),
    }


def _public_invitation_response(value: dict[str, Any]) -> dict[str, Any]:
    team = value.get("team") if isinstance(value.get("team"), dict) else {}
    invitation_value = (
        value.get("invitation")
        if isinstance(value.get("invitation"), dict)
        else value.get("member")
        if isinstance(value.get("member"), dict)
        else {}
    )
    return {
        "ok": bool(value.get("ok")),
        "status": str(
            value.get("status") or invitation_value.get("status") or ""
        ),
        "team": {
            "id": team.get("id"),
            "name": team.get("name"),
            "status": team.get("status"),
            "version": team.get("version"),
        },
        "invitation": {
            "member_id": invitation_value.get("member_id")
            or invitation_value.get("id"),
            "slot": invitation_value.get("slot"),
            "status": invitation_value.get("status"),
            "invitation_version": invitation_value.get("invitation_version"),
        },
    }


def _public_team_creation_response(value: dict[str, Any]) -> dict[str, Any]:
    team = value.get("team") if isinstance(value.get("team"), dict) else {}
    members = value.get("members") if isinstance(value.get("members"), list) else []
    deliveries = (
        value.get("invitation_deliveries")
        if isinstance(value.get("invitation_deliveries"), list)
        else []
    )
    return {
        "ok": bool(value.get("ok")),
        "team": {
            "id": team.get("id"),
            "name": team.get("name"),
            "status": team.get("status"),
            "version": team.get("version"),
        },
        "members": [
            {
                "member_id": member.get("id"),
                "slot": member.get("slot"),
                "display_name": member.get("display_name_snapshot")
                or member.get("display_name"),
                "status": member.get("status"),
                "invitation_version": member.get("invitation_version"),
            }
            for member in members
            if isinstance(member, dict)
        ],
        "invitation_deliveries": [
            {
                "member_id": delivery.get("member_id"),
                "status": delivery.get("status"),
            }
            for delivery in deliveries
            if isinstance(delivery, dict)
        ],
    }


def _active_team(team: dict[str, Any]) -> bool:
    return (
        str(team.get("status") or "").upper() == "CONFIRMED"
        and str(team.get("eligibility_state") or "").upper()
        in {"ELIGIBLE", "NOT_REQUIRED"}
    )


def build_public_team_tournament_index(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str | None = None,
) -> dict[str, Any]:
    tournament_filters: tuple[tuple[str, Any], ...] = (("club_id", str(club_id)),)
    if tournament_id:
        tournament_filters = (*tournament_filters, ("id", str(tournament_id)))
    tournaments = _rows(supabase, "tournaments", filters=tournament_filters)
    tournament_by_id = {str(row.get("id") or ""): row for row in tournaments}
    draws: list[dict[str, Any]] = []
    for current_id, tournament in tournament_by_id.items():
        event_options = _rows(
            supabase,
            "tournament_event_options",
            filters=(("tournament_id", current_id),),
        )
        event_by_id = {str(row.get("id") or ""): row for row in event_options}
        teams = _rows(
            supabase,
            "tournament_four_player_teams",
            filters=(("tournament_id", current_id),),
        )
        for draw in _rows(
            supabase,
            "tournament_event_draws",
            filters=(("tournament_id", current_id),),
        ):
            if (
                str(draw.get("draw_kind") or "").upper() != "TEAM_PARENT"
                or str(draw.get("status") or "").lower() != "published"
            ):
                continue
            event = event_by_id.get(str(draw.get("event_option_id") or "")) or {}
            draws.append(
                {
                    "id": draw.get("id"),
                    "name": draw.get("name"),
                    "status": "published",
                    "tournament_id": current_id,
                    "tournament_name": tournament.get("name"),
                    "event_option_id": draw.get("event_option_id"),
                    "event_family_label": event.get("event_family_label"),
                    "division_name": event.get("division_name"),
                    "team_count": sum(
                        1
                        for team in teams
                        if _active_team(team)
                        and str(team.get("draw_id") or "") == str(draw.get("id") or "")
                    ),
                }
            )
    return {
        "tournaments": [
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "start_date": row.get("start_date"),
                "end_date": row.get("end_date"),
            }
            for row in tournaments
            if any(str(draw.get("tournament_id")) == str(row.get("id")) for draw in draws)
        ],
        "draws": sorted(
            draws,
            key=lambda row: (
                str(row.get("tournament_name") or ""),
                str(row.get("name") or ""),
                str(row.get("id") or ""),
            ),
        ),
    }


def build_public_team_tournament_results(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
) -> dict[str, Any]:
    tournament = _one(
        supabase,
        "tournaments",
        filters=(("id", str(tournament_id)), ("club_id", str(club_id))),
    )
    if not tournament:
        raise ValueError("tournament results not found")
    draw = _one(
        supabase,
        "tournament_event_draws",
        filters=(("id", str(draw_id)), ("tournament_id", str(tournament_id))),
    )
    if (
        not draw
        or str(draw.get("draw_kind") or "").upper() != "TEAM_PARENT"
        or str(draw.get("status") or "").lower() != "published"
    ):
        raise ValueError("team tournament results are not published")
    event = _one(
        supabase,
        "tournament_event_options",
        filters=(
            ("id", str(draw.get("event_option_id") or "")),
            ("tournament_id", str(tournament_id)),
        ),
    ) or {}
    teams = [
        row
        for row in _rows(
            supabase,
            "tournament_four_player_teams",
            filters=(("tournament_id", str(tournament_id)), ("draw_id", str(draw_id))),
        )
        if _active_team(row)
    ]
    active_ids = {str(row.get("id") or "") for row in teams}
    members = [
        row
        for row in _rows(
            supabase,
            "tournament_four_player_team_members",
            filters=(("tournament_id", str(tournament_id)),),
        )
        if str(row.get("team_id") or "") in active_ids
        and str(row.get("status") or "").upper() == "ACCEPTED"
    ]
    matchups = [
        row
        for row in _rows(
            supabase,
            "tournament_team_matchups",
            filters=(("tournament_id", str(tournament_id)), ("draw_id", str(draw_id))),
        )
        if (
            not row.get("team_a_id")
            or str(row.get("team_a_id")) in active_ids
        )
        and (
            not row.get("team_b_id")
            or str(row.get("team_b_id")) in active_ids
        )
    ]
    standings = build_team_standings(teams, matchups)
    podium = [
        {
            "placement": row.get("placement"),
            "team_id": row.get("team_id"),
            "team_name": next(
                (
                    team.get("name")
                    for team in teams
                    if str(team.get("id")) == str(row.get("team_id"))
                ),
                "Team",
            ),
        }
        for row in _rows(
            supabase,
            "tournament_four_player_podium",
            filters=(("tournament_id", str(tournament_id)), ("draw_id", str(draw_id))),
        )
        if row.get("published_at") and str(row.get("team_id") or "") in active_ids
    ]
    return {
        "tournament": {
            "id": tournament.get("id"),
            "name": tournament.get("name"),
            "start_date": tournament.get("start_date"),
            "end_date": tournament.get("end_date"),
        },
        "draw": {
            "id": draw.get("id"),
            "name": draw.get("name"),
            "status": "published",
            "event_option_id": draw.get("event_option_id"),
            "event_family_label": event.get("event_family_label"),
            "division_name": event.get("division_name"),
            "team_playoff_format": event.get("team_playoff_format"),
        },
        "teams": [
            {
                **_public_team(team),
                "members": [
                    _public_member(member)
                    for member in members
                    if str(member.get("team_id")) == str(team.get("id"))
                ],
            }
            for team in teams
        ],
        "standings": standings,
        "bracket": [
            {
                "id": row.get("id"),
                "round_number": row.get("round_number"),
                "slot_number": row.get("slot_number"),
                "playoff_game_code": row.get("playoff_game_code"),
                "team_a_id": row.get("team_a_id"),
                "team_b_id": row.get("team_b_id"),
                "team_a_source": row.get("team_a_source"),
                "team_b_source": row.get("team_b_source"),
                "status": row.get("status"),
                "team_a_game_wins": row.get("team_a_game_wins"),
                "team_b_game_wins": row.get("team_b_game_wins"),
                "winner_team_id": row.get("winner_team_id"),
                "loser_team_id": row.get("loser_team_id"),
            }
            for row in matchups
            if str(row.get("stage") or "").upper() == "PLAYOFF"
        ],
        "podium": sorted(podium, key=lambda row: int(row.get("placement") or 99)),
    }


def create_public_four_player_team(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    event_option_id: str,
    team_name: str,
    captain_registration_id: str,
    confirmation_token: str,
    members: list[dict[str, Any]],
    idempotency_key: str,
    public_base_url: str | None = None,
    sender: Callable[..., dict[str, str]] = send_team_invitation_email,
) -> dict[str, Any]:
    captain = _one(
        supabase,
        "tournament_registrations",
        filters=(
            ("id", str(captain_registration_id)),
            ("tournament_id", str(tournament_id)),
        ),
    )
    if not captain or str(captain.get("status") or "").upper() != "CONFIRMED":
        raise ValueError("A confirmed captain registration is required.")
    try:
        verify_registration_confirmation_token(
            confirmation_token,
            expected_tournament_id=str(tournament_id),
            expected_registration_id=str(captain_registration_id),
            expected_email=str(captain.get("email") or ""),
        )
    except ValueError as exc:
        raise PermissionError(
            "A valid captain confirmation token is required to create this team."
        ) from exc
    selection = _one(
        supabase,
        "tournament_registration_selections",
        filters=(
            ("tournament_id", str(tournament_id)),
            ("registration_id", str(captain_registration_id)),
            ("event_option_id", str(event_option_id)),
        ),
    )
    if not selection or str(selection.get("partner_mode") or "NONE").upper() != "NONE":
        raise ValueError("Captain must select this exact four-player team event.")
    result = create_four_player_team(
        supabase,
        club_id=club_id,
        tournament_id=tournament_id,
        event_option_id=event_option_id,
        team_name=team_name,
        captain_registration_id=captain_registration_id,
        members=members,
        actor_email=str(captain.get("email") or "").strip().lower(),
        idempotency_key=idempotency_key,
        public_base_url=public_base_url,
        sender=sender,
    )
    return _public_team_creation_response(result)


def build_public_four_player_team_setup_recovery(
    supabase: Any,
    *,
    club_id: str,
    confirmation_token: str,
) -> dict[str, Any]:
    """Return captain-authorized, durable team-setup state.

    The registration selection is the durable setup intent.  Completed team
    rows and the existing operation ledger distinguish a committed request
    whose browser response was lost from a setup that still needs submission.
    No operation payload, token hash, or audit actor is returned.
    """

    claims = verify_registration_confirmation_token(confirmation_token)
    tournament_id = str(claims.get("tournament_id") or "")
    registration_id = str(claims.get("registration_id") or "")
    registration = _one(
        supabase,
        "tournament_registrations",
        filters=(
            ("id", registration_id),
            ("tournament_id", tournament_id),
        ),
    )
    if not registration:
        raise ValueError("Registration confirmation is no longer available.")
    verify_registration_confirmation_token(
        confirmation_token,
        expected_tournament_id=tournament_id,
        expected_registration_id=registration_id,
        expected_email=str(registration.get("email") or ""),
    )
    tournament = _one(
        supabase,
        "tournaments",
        filters=(("id", tournament_id), ("club_id", str(club_id))),
    )
    if not tournament:
        raise ValueError("Registration confirmation is for a different club.")

    selections = _rows(
        supabase,
        "tournament_registration_selections",
        filters=(
            ("tournament_id", tournament_id),
            ("registration_id", registration_id),
        ),
    )
    selected_event_ids = {
        str(selection.get("event_option_id") or "")
        for selection in selections
        if str(selection.get("event_option_id") or "")
    }
    event_options = [
        event
        for event in _rows(
            supabase,
            "tournament_event_options",
            filters=(("tournament_id", tournament_id),),
        )
        if str(event.get("id") or "") in selected_event_ids
        and str(event.get("competition_format") or "").upper()
        == "FOUR_PLAYER_TEAM"
    ]
    active_teams = [
        team
        for team in _rows(
            supabase,
            "tournament_four_player_teams",
            filters=(
                ("tournament_id", tournament_id),
                ("captain_registration_id", registration_id),
            ),
        )
        if str(team.get("status") or "").upper()
        not in {"WITHDRAWN", "CANCELLED"}
    ]
    team_by_event: dict[str, dict[str, Any]] = {}
    for team in active_teams:
        event_option_id = str(team.get("event_option_id") or "")
        if event_option_id in team_by_event:
            raise RuntimeError(
                "Multiple active team setups exist for the same captain and event."
            )
        team_by_event[event_option_id] = team
    active_team_ids = {str(team.get("id") or "") for team in active_teams}
    members = [
        member
        for member in _rows(
            supabase,
            "tournament_four_player_team_members",
            filters=(("tournament_id", tournament_id),),
        )
        if str(member.get("team_id") or "") in active_team_ids
        and str(member.get("status") or "").upper() != "REMOVED"
    ]

    operation_query = (
        supabase.table("tournament_team_operations")
        .select("entity_id,status,updated_at")
        .eq("tournament_id", tournament_id)
        .eq("surface", "registration")
        .eq("action", "four_player_team_create")
        .eq("actor", str(registration.get("email") or "").strip().lower())
        .limit(100)
    )
    operation_data = getattr(operation_query.execute(), "data", None)
    operation_rows = (
        [dict(row) for row in operation_data if isinstance(row, dict)]
        if isinstance(operation_data, list)
        else []
    )
    if len(operation_rows) >= 100:
        raise RuntimeError("Team setup recovery exceeded its safe operation bound.")
    operation_status_by_event: dict[str, str] = {}
    operation_rank = {"INTENT": 1, "RECOVERY_REQUIRED": 2, "COMPLETED": 3}
    for operation in operation_rows:
        event_option_id = str(operation.get("entity_id") or "")
        status = str(operation.get("status") or "").upper()
        current = operation_status_by_event.get(event_option_id, "")
        if operation_rank.get(status, 0) >= operation_rank.get(current, 0):
            operation_status_by_event[event_option_id] = status

    recovery_events: list[dict[str, Any]] = []
    for event in sorted(
        event_options,
        key=lambda row: (
            int(row.get("sort_order") or 0),
            str(row.get("label") or row.get("division_name") or ""),
        ),
    ):
        event_id = str(event.get("id") or "")
        team = team_by_event.get(event_id)
        operation_status = operation_status_by_event.get(event_id)
        if team:
            setup_state = "COMPLETE"
        elif operation_status in {"INTENT", "RECOVERY_REQUIRED", "COMPLETED"}:
            # A committed operation without its business row is inconsistent;
            # keep the public surface fail-closed and direct staff recovery.
            setup_state = "STAFF_RECOVERY_REQUIRED"
        else:
            setup_state = "SETUP_REQUIRED"
        recovery_events.append(
            {
                "id": event_id,
                "registration_day_id": event.get("registration_day_id"),
                "label": event.get("label")
                or event.get("division_name")
                or "Four-player team",
                "event_family_label": event.get("event_family_label")
                or event.get("label")
                or "Team event",
                "division_name": event.get("division_name")
                or event.get("label")
                or "Division",
                "event_type": event.get("event_type"),
                "competition_format": "FOUR_PLAYER_TEAM",
                "team_allow_substitutes": bool(
                    event.get("team_allow_substitutes")
                ),
                "setup_state": setup_state,
                "operation_status": operation_status,
                "team": (
                    {
                        "id": team.get("id"),
                        "name": team.get("name"),
                        "status": team.get("status"),
                        "eligibility_state": team.get("eligibility_state"),
                        "version": team.get("version"),
                        "members": [
                            {
                                "member_id": member.get("id"),
                                "slot": member.get("slot"),
                                "display_name": member.get(
                                    "display_name_snapshot"
                                )
                                or "Player",
                                "status": member.get("status"),
                                "invitation_version": member.get(
                                    "invitation_version"
                                ),
                            }
                            for member in sorted(
                                (
                                    row
                                    for row in members
                                    if str(row.get("team_id") or "")
                                    == str(team.get("id") or "")
                                ),
                                key=lambda row: str(row.get("slot") or ""),
                            )
                        ],
                    }
                    if team
                    else None
                ),
            }
        )

    return {
        "ok": True,
        "tournament": {
            "id": tournament.get("id"),
            "name": tournament.get("name"),
        },
        "captain": {
            "registration_id": registration_id,
            "display_name": registration.get("display_name")
            or " ".join(
                part
                for part in (
                    str(registration.get("first_name") or "").strip(),
                    str(registration.get("last_name") or "").strip(),
                )
                if part
            )
            or "Team captain",
            "email": str(registration.get("email") or "").strip().lower(),
            "gender": registration.get("gender"),
            "registration_status": registration.get("status"),
        },
        "events": recovery_events,
    }


def _invitation_context(
    supabase: Any,
    *,
    club_id: str,
    token: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    claims = verify_tournament_team_invitation_token(token)
    team = _one(
        supabase,
        "tournament_four_player_teams",
        filters=(
            ("id", claims["team_id"]),
            ("tournament_id", claims["tournament_id"]),
        ),
    )
    if not team:
        raise ValueError("Team invitation is no longer available.")
    tournament = _one(
        supabase,
        "tournaments",
        filters=(("id", claims["tournament_id"]), ("club_id", str(club_id))),
    )
    if not tournament:
        raise ValueError("Team invitation is for a different club.")
    member = _one(
        supabase,
        "tournament_four_player_team_members",
        filters=(("id", claims["member_id"]), ("team_id", claims["team_id"])),
    )
    if not member:
        raise ValueError("Team invitation is no longer available.")
    verify_tournament_team_invitation_token(
        token,
        expected_tournament_id=str(tournament.get("id") or ""),
        expected_team_id=str(team.get("id") or ""),
        expected_member_id=str(member.get("id") or ""),
        expected_invited_email=str(member.get("invited_email") or ""),
        expected_invitation_version=int(member.get("invitation_version") or 0),
    )
    stored_token_hash = str(member.get("invitation_token_hash") or "")
    accepted_token_consumed = (
        str(member.get("status") or "").upper() == "ACCEPTED"
        and not stored_token_hash
    )
    if (
        stored_token_hash != tournament_team_invitation_token_hash(token)
        and not accepted_token_consumed
    ):
        raise ValueError("Team invitation is no longer current.")
    return claims, tournament, team, member


def build_public_team_invitation(
    supabase: Any,
    *,
    club_id: str,
    token: str,
) -> dict[str, Any]:
    _claims, tournament, team, member = _invitation_context(
        supabase, club_id=club_id, token=token
    )
    if str(member.get("status") or "").upper() not in {"INVITED", "ACCEPTED"}:
        raise ValueError("Team invitation has already been resolved.")
    registrations = _rows(
        supabase,
        "tournament_registrations",
        filters=(("tournament_id", str(tournament.get("id") or "")),),
    )
    matching = [
        row
        for row in registrations
        if str(row.get("email") or "").strip().lower()
        == str(member.get("invited_email") or "").strip().lower()
        and str(row.get("status") or "").upper() == "CONFIRMED"
    ]
    if len(matching) != 1:
        raise ValueError(
            "A single confirmed registration with the invited email is required."
        )
    registration = matching[0]
    return {
        "tournament": {
            "id": tournament.get("id"),
            "name": tournament.get("name"),
        },
        "team": {"id": team.get("id"), "name": team.get("name")},
        "invitation": {
            "member_id": member.get("id"),
            "slot": member.get("slot"),
            "status": member.get("status"),
            "invitation_version": member.get("invitation_version"),
            "invited_name": member.get("display_name_snapshot"),
        },
        "registration": {
            "id": registration.get("id"),
            "display_name": registration.get("display_name")
            or " ".join(
                filter(
                    None,
                    [
                        str(registration.get("first_name") or "").strip(),
                        str(registration.get("last_name") or "").strip(),
                    ],
                )
            ),
        },
    }


def respond_public_team_invitation(
    supabase: Any,
    *,
    club_id: str,
    token: str,
    action: str,
    registration_id: str,
    idempotency_key: str,
) -> dict[str, Any]:
    claims, tournament, team, member = _invitation_context(
        supabase, club_id=club_id, token=token
    )
    normalized_action = str(action or "").upper()
    if normalized_action not in {"ACCEPT", "DECLINE"}:
        raise ValueError("Invitation response must be accept or decline.")
    registration = _one(
        supabase,
        "tournament_registrations",
        filters=(
            ("id", str(registration_id)),
            ("tournament_id", str(tournament.get("id") or "")),
        ),
    )
    if (
        not registration
        or str(registration.get("status") or "").upper() != "CONFIRMED"
        or str(registration.get("email") or "").strip().lower()
        != str(member.get("invited_email") or "").strip().lower()
    ):
        raise ValueError("Invitation identity does not match a confirmed registration.")
    operation = build_tournament_admin_operation_request(
        club_id=str(club_id),
        surface="public_registration",
        action=f"four_player_invite_{normalized_action.lower()}",
        entity_type="tournament_four_player_team_member",
        entity_id=str(member.get("id") or ""),
        lock_scope=str(tournament.get("id") or ""),
        expected_state=str(member.get("invitation_version") or ""),
        payload={
            "registration_id": str(registration_id),
            "action": normalized_action,
        },
        idempotency_key=idempotency_key,
    )
    result = _rpc_data(
        supabase.rpc(
            "server_respond_tournament_four_player_invite",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament.get("id") or ""),
                "p_team_id": str(team.get("id") or ""),
                "p_member_id": str(member.get("id") or ""),
                "p_registration_id": str(registration_id),
                "p_invitation_version": int(claims["invitation_version"]),
                "p_invitation_token_hash": tournament_team_invitation_token_hash(
                    token
                ),
                "p_action": normalized_action,
                "p_operation_key": operation["operation_key"],
                "p_request_fingerprint": operation["request_fingerprint"],
                "p_actor": str(registration.get("email") or "").strip().lower(),
            },
        ).execute()
    )
    if not isinstance(result, dict):
        raise RuntimeError("Invitation response did not return the expected result.")
    return _public_invitation_response(result)


def public_team_tournament_runtime_ready() -> bool:
    return is_admin_team_tournament_enabled() and bool(
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )


# Compatibility names used by route tests and prior prototypes.
build_public_tournament_team_index = build_public_team_tournament_index
build_public_tournament_team_results = build_public_team_tournament_results
build_public_tournament_team_invitation = build_public_team_invitation
respond_to_public_tournament_team_invitation = respond_public_team_invitation
