from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from typing import Any, Callable
from urllib.parse import quote, urlsplit
from uuid import uuid4

from jupr_app.config import get_email_mode, get_env_or_default
from jupr_app.domain.notifications.tournament_team_invitation_email import (
    send_team_invitation_email,
)
from jupr_app.domain.tournament_admin_operations import (
    build_tournament_admin_operation_request,
)
from jupr_app.domain.tournament_four_player_team import (
    build_team_playoff_matchups,
    build_team_round_robin_matchups,
    build_team_standings,
    calculate_team_podium,
    validate_four_player_roster,
)
from jupr_app.domain.tournament_team_invitation_tokens import (
    build_tournament_team_invitation_token,
    tournament_team_invitation_email_hash,
    tournament_team_invitation_token_hash,
)
from jupr_app.domain.tournament_team_canonical_publish import (
    classify_team_child_publish_state,
)
from jupr_app.services.admin_tournament_service import (
    is_admin_tournament_admin_enabled,
)
from jupr_app.services.production_tournament_guard import require_production_tournament_writes

TRUTHY = {"1", "true", "yes", "y", "on"}
TEAM_TABLES = (
    "tournament_event_options",
    "tournament_event_draws",
    "tournament_registrations",
    "tournament_registration_selections",
    "tournament_four_player_teams",
    "tournament_four_player_team_members",
    "tournament_rating_verifications",
    "tournament_rating_eligibility_reviews",
    "tournament_team_matchups",
    "tournament_team_lineup_submissions",
    "tournament_team_match_games",
    "tournament_four_player_podium",
    "tournament_team_audit_events",
    "tournament_team_operations",
    "tournament_team_invitation_deliveries",
    "tournament_games",
)
SECRET_KEYS = {
    "invitation_token",
    "invitation_token_hash",
    "raw_invitation_token",
}


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in TRUTHY


def is_admin_team_tournament_enabled() -> bool:
    explicit = os.getenv("JUPR_ENABLE_TOURNAMENT_TEAM_COMPETITION")
    return is_admin_tournament_admin_enabled() if explicit is None else _truthy(explicit)


def require_admin_team_tournament_runtime() -> None:
    if not is_admin_team_tournament_enabled():
        raise PermissionError("Team tournament management is disabled.")
    if os.getenv("JUPR_ENV", "").strip().lower() == "production":
        require_production_tournament_writes()
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        raise PermissionError("Team tournament management requires the server database credential.")


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    data = getattr(response, "data", None)
    if isinstance(data, list):
        return [dict(row) for row in data if isinstance(row, dict)]
    return []


def _rows(
    supabase: Any,
    table: str,
    *,
    filters: tuple[tuple[str, Any], ...] = (),
    limit: int = 5000,
) -> list[dict[str, Any]]:
    query = supabase.table(table).select("*")
    for key, value in filters:
        query = query.eq(key, value)
    if hasattr(query, "order"):
        query = query.order("id", desc=False)
    query = query.limit(limit)
    rows = _safe_rows(query.execute())
    if len(rows) >= limit:
        raise RuntimeError(f"{table} exceeded the safe tournament snapshot bound.")
    return rows


def _one(
    supabase: Any,
    table: str,
    *,
    filters: tuple[tuple[str, Any], ...],
) -> dict[str, Any] | None:
    rows = _rows(supabase, table, filters=filters, limit=2)
    return rows[0] if rows else None


def _redact_browser_secrets(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact_browser_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_browser_secrets(item) for item in value)
    if not isinstance(value, dict):
        return value
    return {
        key: _redact_browser_secrets(item)
        for key, item in value.items()
        if key not in SECRET_KEYS and not str(key).lower().endswith("_token_hash")
    }


def _rpc_data(response: Any) -> Any:
    data = getattr(response, "data", None)
    if isinstance(data, list) and len(data) == 1:
        return data[0]
    return data


def _call_rpc(supabase: Any, name: str, params: dict[str, Any]) -> dict[str, Any]:
    result = _rpc_data(supabase.rpc(name, params).execute())
    if not isinstance(result, dict):
        raise RuntimeError(f"{name} did not return the expected result.")
    return _redact_browser_secrets(dict(result))


def _operation(
    *,
    club_id: str,
    tournament_id: str,
    surface: str,
    action: str,
    entity_type: str,
    entity_id: str,
    expected_state: Any,
    payload: dict[str, Any],
    idempotency_key: str,
) -> dict[str, Any]:
    return build_tournament_admin_operation_request(
        club_id=str(club_id),
        surface=surface,
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id),
        lock_scope=str(tournament_id),
        expected_state=str(expected_state or ""),
        payload=payload,
        idempotency_key=str(idempotency_key),
    )


def _display_name(row: dict[str, Any]) -> str:
    return (
        str(row.get("display_name") or "").strip()
        or " ".join(
            part
            for part in (
                str(row.get("first_name") or "").strip(),
                str(row.get("last_name") or "").strip(),
            )
            if part
        )
        or str(row.get("name") or "").strip()
        or "Player"
    )


def get_admin_team_tournament_snapshot(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
) -> dict[str, Any]:
    """Build one complete private management snapshot.

    Route authorization is manage-only. This service also strips invitation
    hashes recursively so accidental pass-through cannot expose a usable secret.
    """

    tournament = _one(
        supabase,
        "tournaments",
        filters=(("id", str(tournament_id)), ("club_id", str(club_id))),
    )
    if not tournament:
        raise ValueError("tournament not found")
    tables: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []
    for table in TEAM_TABLES:
        try:
            tables[table] = _rows(
                supabase,
                table,
                filters=(("tournament_id", str(tournament_id)),),
            )
        except Exception as exc:  # noqa: BLE001 - report optional migration/table gaps
            tables[table] = []
            warnings.append(f"{table} unavailable: {exc.__class__.__name__}")
    try:
        players = _rows(supabase, "players", filters=(("club_id", str(club_id)),))
    except Exception as exc:  # noqa: BLE001
        players = []
        warnings.append(f"players unavailable: {exc.__class__.__name__}")
    try:
        canonical_matches = _rows(
            supabase,
            "matches",
            filters=(("club_id", str(club_id)), ("tournament_id", str(tournament_id))),
        )
    except Exception as exc:  # noqa: BLE001
        canonical_matches = []
        warnings.append(f"matches unavailable: {exc.__class__.__name__}")

    registrations = {
        str(row.get("id") or ""): row
        for row in tables["tournament_registrations"]
    }
    player_by_id = {str(row.get("id") or ""): row for row in players}
    members: list[dict[str, Any]] = []
    for member in tables["tournament_four_player_team_members"]:
        registration = registrations.get(str(member.get("registration_id") or "")) or {}
        player = player_by_id.get(str(member.get("player_id") or "")) or {}
        members.append(
            {
                **member,
                "display_name": (
                    str(member.get("display_name_snapshot") or "").strip()
                    or _display_name(registration)
                    or _display_name(player)
                ),
            }
        )
    teams = tables["tournament_four_player_teams"]
    matchups = tables["tournament_team_matchups"]
    draws = tables["tournament_event_draws"]
    events = tables["tournament_event_options"]
    active_teams = [
        row
        for row in teams
        if str(row.get("status") or "").upper() == "CONFIRMED"
        and str(row.get("eligibility_state") or "").upper()
        in {"ELIGIBLE", "NOT_REQUIRED"}
    ]
    standings_by_draw: dict[str, list[dict[str, Any]]] = {}
    calculated_podium_by_draw: dict[str, list[dict[str, Any]] | None] = {}
    for draw in draws:
        draw_id = str(draw.get("id") or "")
        draw_teams = [row for row in active_teams if str(row.get("draw_id") or "") == draw_id]
        draw_matchups = [row for row in matchups if str(row.get("draw_id") or "") == draw_id]
        standings = build_team_standings(draw_teams, draw_matchups)
        standings_by_draw[draw_id] = standings
        event = next(
            (
                row
                for row in events
                if str(row.get("id") or "") == str(draw.get("event_option_id") or "")
            ),
            {},
        )
        try:
            calculated_podium_by_draw[draw_id] = calculate_team_podium(
                playoff_format=str(event.get("team_playoff_format") or "NONE"),
                standings=standings,
                playoff_matchups=[
                    row
                    for row in draw_matchups
                    if str(row.get("stage") or "").upper() == "PLAYOFF"
                ],
            )
        except ValueError:
            calculated_podium_by_draw[draw_id] = None

    canonical_by_game: dict[str, list[dict[str, Any]]] = {}
    for match in canonical_matches:
        game_id = str(match.get("tournament_game_id") or "")
        if game_id:
            canonical_by_game.setdefault(game_id, []).append(match)
    tournament_games_by_id = {
        str(row.get("id") or ""): row
        for row in tables["tournament_games"]
        if row.get("id")
    }
    game_publish_state: dict[str, str] = {}
    for child in tables["tournament_team_match_games"]:
        child_id = str(child.get("id") or "")
        tournament_game_id = str(child.get("tournament_game_id") or "")
        game_publish_state[child_id] = classify_team_child_publish_state(
            child=child,
            tournament_game=tournament_games_by_id.get(tournament_game_id),
            canonical_matches=canonical_by_game.get(tournament_game_id, []),
        )

    review_audit_by_selection: dict[str, list[dict[str, Any]]] = {}
    for audit in tables["tournament_team_audit_events"]:
        after = audit.get("after_json") if isinstance(audit.get("after_json"), dict) else {}
        selection_id = str(after.get("selection_id") or "")
        if selection_id:
            review_audit_by_selection.setdefault(selection_id, []).append(
                {
                    "id": audit.get("id"),
                    "action": audit.get("action"),
                    "actor": audit.get("actor"),
                    "created_at": audit.get("created_at"),
                    "before": audit.get("before_json"),
                    "after": after,
                }
            )
    for history in review_audit_by_selection.values():
        history.sort(
            key=lambda row: (
                str(row.get("created_at") or ""),
                str(row.get("id") or ""),
            ),
            reverse=True,
        )
    rating_entries: list[dict[str, Any]] = []
    for review in tables["tournament_rating_eligibility_reviews"]:
        registration = registrations.get(str(review.get("registration_id") or "")) or {}
        partner = registrations.get(str(review.get("partner_registration_id") or "")) or {}
        rating_entries.append(
            {
                **review,
                "registration_name": _display_name(registration),
                "partner_name": _display_name(partner) if partner else "Partner not selected",
                "history": review_audit_by_selection.get(
                    str(review.get("selection_id") or ""), []
                ),
            }
        )
    return _redact_browser_secrets(
        {
            "tournament": tournament,
            "event_options": events,
            "draws": draws,
            "registrations": tables["tournament_registrations"],
            "selections": tables["tournament_registration_selections"],
            "players": players,
            "teams": teams,
            "members": members,
            "rating_verifications": tables["tournament_rating_verifications"],
            "rating_reviews": rating_entries,
            "combined_rating_entries": rating_entries,
            "matchups": matchups,
            "lineups": tables["tournament_team_lineup_submissions"],
            "games": tables["tournament_team_match_games"],
            "canonical_matches": canonical_matches,
            "podium": [
                row
                for row in tables["tournament_four_player_podium"]
                if any(
                    str(team.get("id") or "") == str(row.get("team_id") or "")
                    for team in active_teams
                )
            ],
            "audit_events": tables["tournament_team_audit_events"],
            "operations": tables["tournament_team_operations"],
            "invitation_deliveries": tables[
                "tournament_team_invitation_deliveries"
            ],
            "standings_by_draw": standings_by_draw,
            "calculated_podium_by_draw": calculated_podium_by_draw,
            "game_publish_state": game_publish_state,
            "warnings": warnings,
        }
    )


def update_tournament_competition_config(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    event_option_id: str,
    expected_updated_at: str,
    patch: dict[str, Any],
    actor_email: str,
    idempotency_key: str,
) -> dict[str, Any]:
    require_admin_team_tournament_runtime()
    normalized = dict(patch)
    if str(normalized.get("competition_format") or "").upper() == "FOUR_PLAYER_TEAM":
        normalized.setdefault("team_allow_substitutes", False)
    op = _operation(
        club_id=club_id,
        tournament_id=tournament_id,
        surface="setup",
        action="competition_config_update",
        entity_type="tournament_event_option",
        entity_id=event_option_id,
        expected_state=expected_updated_at,
        payload=normalized,
        idempotency_key=idempotency_key,
    )
    return _call_rpc(
        supabase,
        "admin_update_tournament_competition_config_cas",
        {
            "p_club_id": str(club_id),
            "p_tournament_id": str(tournament_id),
            "p_event_option_id": str(event_option_id),
            "p_expected_updated_at": expected_updated_at,
            "p_patch": normalized,
            "p_operation_key": op["operation_key"],
            "p_request_fingerprint": op["request_fingerprint"],
            "p_actor": actor_email,
        },
    )


def upsert_rating_verification(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    event_option_id: str,
    registration_id: str,
    rating: float,
    note: str,
    expected_version: int | None,
    actor_email: str,
    idempotency_key: str,
) -> dict[str, Any]:
    require_admin_team_tournament_runtime()
    payload = {"rating": rating, "note": note}
    op = _operation(
        club_id=club_id,
        tournament_id=tournament_id,
        surface="registration",
        action="rating_verification_upsert",
        entity_type="tournament_registration",
        entity_id=registration_id,
        expected_state=expected_version,
        payload=payload,
        idempotency_key=idempotency_key,
    )
    return _call_rpc(
        supabase,
        "admin_upsert_tournament_rating_verification_cas",
        {
            "p_club_id": str(club_id),
            "p_tournament_id": str(tournament_id),
            "p_event_option_id": str(event_option_id),
            "p_registration_id": str(registration_id),
            "p_rating": rating,
            "p_note": note,
            "p_expected_version": expected_version,
            "p_operation_key": op["operation_key"],
            "p_request_fingerprint": op["request_fingerprint"],
            "p_actor": actor_email,
        },
    )


def record_combined_rating_review(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    event_option_id: str,
    selection_id: str,
    review_phase: str,
    override_state: str | None,
    override_reason: str | None,
    expected_selection_updated_at: str,
    actor_email: str,
    idempotency_key: str,
) -> dict[str, Any]:
    require_admin_team_tournament_runtime()
    initial_reviews = _rows(
        supabase,
        "tournament_rating_eligibility_reviews",
        filters=(
            ("tournament_id", str(tournament_id)),
            ("event_option_id", str(event_option_id)),
            ("selection_id", str(selection_id)),
            ("review_phase", "INITIAL"),
        ),
        limit=2,
    )
    if len(initial_reviews) != 1:
        raise ValueError(
            "The current combined-rating review is unavailable. Verify both "
            "players and refresh before saving the review."
        )
    initial = initial_reviews[0]
    payload = {
        "selection_id": str(selection_id),
        "registration_id": initial.get("registration_id"),
        "partner_registration_id": initial.get("partner_registration_id"),
        "review_phase": review_phase,
        "state": initial.get("state"),
        "player_rating": initial.get("player_rating"),
        "partner_rating": initial.get("partner_rating"),
        "combined_rating": initial.get("combined_rating"),
        "player_rating_source": initial.get("player_rating_source"),
        "partner_rating_source": initial.get("partner_rating_source"),
        "player_verification_id": initial.get("player_verification_id"),
        "partner_verification_id": initial.get("partner_verification_id"),
        "override_state": override_state,
        "override_reason": override_reason,
        "expected_selection_updated_at": expected_selection_updated_at,
    }
    op = _operation(
        club_id=club_id,
        tournament_id=tournament_id,
        surface="registration",
        action="combined_rating_review_record",
        entity_type="tournament_registration_selection",
        entity_id=selection_id,
        expected_state=expected_selection_updated_at,
        payload=payload,
        idempotency_key=idempotency_key,
    )
    return _call_rpc(
        supabase,
        "admin_record_tournament_rating_review_cas",
        {
            "p_club_id": str(club_id),
            "p_tournament_id": str(tournament_id),
            "p_event_option_id": str(event_option_id),
            "p_review": payload,
            "p_operation_key": op["operation_key"],
            "p_request_fingerprint": op["request_fingerprint"],
            "p_actor": actor_email,
        },
    )


def close_combined_rating_reviews(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    event_option_id: str,
    entries: list[dict[str, Any]],
    actor_email: str,
    idempotency_key: str,
) -> dict[str, Any]:
    require_admin_team_tournament_runtime()
    results: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if str(entry.get("registration_status") or "CONFIRMED").upper() != "CONFIRMED":
            continue
        results.append(
            record_combined_rating_review(
                supabase,
                club_id=club_id,
                tournament_id=tournament_id,
                event_option_id=event_option_id,
                selection_id=str(entry["selection_id"]),
                review_phase="REGISTRATION_CLOSE",
                override_state=entry.get("override_state"),
                override_reason=entry.get("override_reason"),
                expected_selection_updated_at=str(
                    entry["expected_selection_updated_at"]
                ),
                actor_email=actor_email,
                idempotency_key=f"{idempotency_key}:{index}",
            )
        )
    return {"ok": True, "reviews": results, "review_count": len(results)}


def _invitation_url(
    *,
    public_base_url: str,
    token: str,
) -> str:
    return (
        f"{str(public_base_url).rstrip('/')}/tournament-team-invitation"
        f"#{quote(token, safe='')}"
    )


def _validated_team_invitation_base_url(value: str) -> str:
    candidate = str(value or "").strip().rstrip("/")
    parsed = urlsplit(candidate)
    local_host = (parsed.hostname or "").lower() in {"localhost", "127.0.0.1"}
    if (
        parsed.scheme not in ({"http", "https"} if local_host else {"https"})
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("A safe server-configured tournament invitation URL is required.")
    return candidate


def resolve_team_invitation_base_url(
    supabase: Any,
    *,
    club_id: str,
) -> str:
    """Resolve an invitation destination from server-owned club/config state."""

    club: dict[str, Any] = {}
    try:
        club = _one(
            supabase,
            "clubs",
            filters=(("id", str(club_id)),),
        ) or {}
    except Exception:
        club = {}
    configured_origin = ""
    for name in (
        "JUPR_NEXT_WEB_BASE_URL",
        "JUPR_WEB_BASE_URL",
        "STAGING_WEB_BASE_URL",
        "NEXT_PUBLIC_JUPR_WEB_BASE_URL",
    ):
        configured_origin = get_env_or_default(name)
        if configured_origin:
            break
    environment = os.getenv("JUPR_ENV", "").strip().lower()
    club_slug = str(club.get("slug") or club_id).strip().replace("_", "-")
    if environment in {"staging", "production"}:
        if not configured_origin:
            raise RuntimeError(
                "Tournament team invitations require the hosted web origin."
            )
        origin = _validated_team_invitation_base_url(configured_origin)
        return f"{origin}/clubs/{quote(club_slug, safe='-')}"

    configured_club_url = str(club.get("public_base_url") or "").strip()
    if configured_club_url:
        return _validated_team_invitation_base_url(configured_club_url)
    if not configured_origin and environment in {"", "local", "test", "development", "dev"}:
        configured_origin = "http://localhost:3000"
    if not configured_origin:
        raise RuntimeError(
            "Tournament team invitations require a server-configured public web origin."
        )
    origin = _validated_team_invitation_base_url(configured_origin)
    return f"{origin}/clubs/{quote(club_slug, safe='-')}"


def _delivery_operation(
    *,
    club_id: str,
    tournament_id: str,
    member: dict[str, Any],
    batch_idempotency_key: str,
) -> dict[str, Any]:
    return _operation(
        club_id=club_id,
        tournament_id=tournament_id,
        surface="registration",
        action="team_invitation_delivery",
        entity_type="tournament_four_player_team_member",
        entity_id=str(member.get("id") or ""),
        expected_state=member.get("invitation_version"),
        payload={
            "member_id": member.get("id"),
            "invitation_version": member.get("invitation_version"),
            "recipient_email_hash": tournament_team_invitation_email_hash(
                str(member.get("invited_email") or "")
            ),
        },
        idempotency_key=f"{batch_idempotency_key}:{member.get('id')}:{member.get('invitation_version')}",
    )


def _deliver_invitations(
    supabase: Any,
    *,
    club_id: str,
    tournament: dict[str, Any],
    team: dict[str, Any],
    members: list[dict[str, Any]],
    actor_email: str,
    public_base_url: str,
    captain_name: str,
    batch_idempotency_key: str,
    sender: Callable[..., dict[str, str]] = send_team_invitation_email,
) -> list[dict[str, Any]]:
    deliveries: list[dict[str, Any]] = []
    for member in members:
        if str(member.get("status") or "").upper() != "INVITED":
            continue
        mode = get_email_mode()
        existing_deliveries = _rows(
            supabase,
            "tournament_team_invitation_deliveries",
            filters=(
                ("member_id", str(member.get("id") or "")),
                (
                    "invitation_version",
                    int(member.get("invitation_version") or 1),
                ),
                ("email_mode", mode),
            ),
            limit=3,
        )
        completed_delivery = next(
            (
                row
                for row in existing_deliveries
                if str(row.get("status") or "").lower()
                in {"dry_run", "staging_redirect", "sent", "skipped"}
            ),
            None,
        )
        if completed_delivery:
            deliveries.append(
                {
                    "ok": True,
                    "send_required": False,
                    "recovered_by_business_identity": True,
                    "status": completed_delivery.get("status"),
                }
            )
            continue
        if any(
            str(row.get("status") or "").lower() == "pending"
            for row in existing_deliveries
        ):
            deliveries.append(
                {
                    "ok": False,
                    "send_required": False,
                    "recovery_required": True,
                    "status": "pending",
                }
            )
            continue
        token = build_tournament_team_invitation_token(
            tournament_id=str(tournament.get("id") or ""),
            team_id=str(team.get("id") or ""),
            member_id=str(member.get("id") or ""),
            invited_email=str(member.get("invited_email") or ""),
            invitation_version=int(member.get("invitation_version") or 1),
        )
        operation = _delivery_operation(
            club_id=club_id,
            tournament_id=str(tournament.get("id") or ""),
            member=member,
            batch_idempotency_key=batch_idempotency_key,
        )
        claim = _call_rpc(
            supabase,
            "server_claim_tournament_team_invitation_delivery",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament.get("id") or ""),
                "p_team_id": str(team.get("id") or ""),
                "p_member_id": str(member.get("id") or ""),
                "p_invitation_version": int(member.get("invitation_version") or 1),
                "p_email_mode": mode,
                "p_recipient_email_hash": tournament_team_invitation_email_hash(
                    str(member.get("invited_email") or "")
                ),
                "p_invitation_token_hash": tournament_team_invitation_token_hash(
                    token
                ),
                "p_operation_key": operation["operation_key"],
                "p_request_fingerprint": operation["request_fingerprint"],
                "p_actor": actor_email,
            },
        )
        if not claim.get("send_required"):
            deliveries.append(claim)
            continue
        try:
            sent = sender(
                target_email=str(member.get("invited_email") or ""),
                tournament_name=str(tournament.get("name") or "Tournament"),
                team_name=str(team.get("name") or "Team"),
                captain_name=captain_name,
                invited_name=str(member.get("display_name_snapshot") or ""),
                invitation_url=_invitation_url(
                    public_base_url=public_base_url, token=token
                ),
            )
        except Exception:
            # A provider exception leaves the durable claim pending. Retrying
            # must return recovery_required and must never send a second email.
            raise
        completed = _call_rpc(
            supabase,
            "server_complete_tournament_team_invitation_delivery",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament.get("id") or ""),
                "p_delivery_id": str(claim.get("delivery_id") or ""),
                "p_status": str(sent.get("status") or "failed"),
                "p_provider_message_id": str(
                    sent.get("provider_message_id") or ""
                ),
                "p_operation_key": operation["operation_key"],
                "p_request_fingerprint": operation["request_fingerprint"],
                "p_actor": actor_email,
            },
        )
        deliveries.append(completed)
    return deliveries


def tournament_team_creation_fingerprint(
    *,
    event_option_id: str,
    team_name: str,
    captain_registration_id: str,
    members: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    normalized_members = sorted(
        [
            {
                "slot": str(row.get("slot") or "").upper(),
                "registration_id": (
                    str(row.get("registration_id") or "").strip() or None
                ),
                "email": str(
                    row.get("email") or row.get("invited_email") or ""
                )
                .strip()
                .lower(),
                "display_name": str(
                    row.get("display_name")
                    or row.get("display_name_snapshot")
                    or ""
                ).strip(),
                "gender": str(
                    row.get("gender") or row.get("gender_snapshot") or ""
                ).strip(),
            }
            for row in members
        ],
        key=lambda row: row["slot"],
    )
    business_payload = {
        "event_option_id": str(event_option_id),
        "team_name": str(team_name or "").strip(),
        "captain_registration_id": str(captain_registration_id),
        "members": normalized_members,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            business_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return fingerprint, normalized_members


def create_four_player_team(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    event_option_id: str,
    team_name: str,
    captain_registration_id: str,
    members: list[dict[str, Any]],
    actor_email: str,
    idempotency_key: str,
    public_base_url: str | None = None,
    sender: Callable[..., dict[str, str]] = send_team_invitation_email,
) -> dict[str, Any]:
    require_admin_team_tournament_runtime()
    creation_fingerprint, normalized_members = (
        tournament_team_creation_fingerprint(
            event_option_id=event_option_id,
            team_name=team_name,
            captain_registration_id=captain_registration_id,
            members=members,
        )
    )
    payload = {
        "event_option_id": event_option_id,
        "team_name": team_name,
        "captain_registration_id": captain_registration_id,
        "members": normalized_members,
    }
    op = _operation(
        club_id=club_id,
        tournament_id=tournament_id,
        surface="registration",
        action="four_player_team_create",
        entity_type="tournament_event_option",
        entity_id=event_option_id,
        expected_state="new-team",
        payload=payload,
        idempotency_key=idempotency_key,
    )
    raw = _rpc_data(
        supabase.rpc(
            "admin_create_tournament_four_player_team",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament_id),
                "p_event_option_id": str(event_option_id),
                "p_team_name": team_name,
                "p_captain_registration_id": str(captain_registration_id),
                "p_members": normalized_members,
                "p_creation_fingerprint": creation_fingerprint,
                "p_operation_key": op["operation_key"],
                "p_request_fingerprint": op["request_fingerprint"],
                "p_actor": actor_email,
            },
        ).execute()
    )
    if not isinstance(raw, dict):
        raise RuntimeError("Team creation did not return the expected result.")
    tournament = _one(
        supabase,
        "tournaments",
        filters=(("id", str(tournament_id)), ("club_id", str(club_id))),
    )
    if not tournament:
        raise RuntimeError("Tournament disappeared after team creation.")
    result = dict(raw)
    captain = next(
        (
            row
            for row in result.get("members") or []
            if str(row.get("registration_id") or "") == str(captain_registration_id)
        ),
        {},
    )
    # ``public_base_url`` remains a compatibility argument for older internal
    # callers, but is deliberately ignored. Browser input never owns email links.
    _ = public_base_url
    invitation_base_url = resolve_team_invitation_base_url(
        supabase,
        club_id=club_id,
    )
    deliveries = _deliver_invitations(
        supabase,
        club_id=club_id,
        tournament=tournament,
        team=dict(result["team"]),
        members=[dict(row) for row in result.get("members") or []],
        actor_email=actor_email,
        public_base_url=invitation_base_url,
        captain_name=str(captain.get("display_name_snapshot") or "Team captain"),
        batch_idempotency_key=idempotency_key,
        sender=sender,
    )
    return _redact_browser_secrets(
        {**result, "invitation_deliveries": deliveries}
    )


def reissue_four_player_team_invitation(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    team_id: str,
    member_id: str,
    expected_invitation_version: int,
    invited_email: str,
    actor_email: str,
    idempotency_key: str,
    public_base_url: str | None = None,
    sender: Callable[..., dict[str, str]] = send_team_invitation_email,
) -> dict[str, Any]:
    require_admin_team_tournament_runtime()
    operation = _operation(
        club_id=club_id,
        tournament_id=tournament_id,
        surface="registration",
        action="four_player_invite_reissue",
        entity_type="tournament_four_player_team_member",
        entity_id=member_id,
        expected_state=expected_invitation_version,
        payload={"team_id": team_id, "invited_email": invited_email},
        idempotency_key=idempotency_key,
    )
    result = _call_rpc(
        supabase,
        "admin_reissue_tournament_four_player_invite_cas",
        {
            "p_club_id": str(club_id),
            "p_tournament_id": str(tournament_id),
            "p_team_id": str(team_id),
            "p_member_id": str(member_id),
            "p_expected_invitation_version": int(expected_invitation_version),
            "p_invited_email": str(invited_email).strip().lower(),
            "p_operation_key": operation["operation_key"],
            "p_request_fingerprint": operation["request_fingerprint"],
            "p_actor": actor_email,
        },
    )
    tournament = _one(
        supabase,
        "tournaments",
        filters=(("id", str(tournament_id)), ("club_id", str(club_id))),
    )
    if not tournament:
        raise RuntimeError("Tournament not found after invitation reissue.")
    team = dict(result.get("team") or {})
    member = dict(result.get("member") or {})
    _ = public_base_url
    invitation_base_url = resolve_team_invitation_base_url(
        supabase,
        club_id=club_id,
    )
    deliveries = _deliver_invitations(
        supabase,
        club_id=club_id,
        tournament=tournament,
        team=team,
        members=[member],
        actor_email=actor_email,
        public_base_url=invitation_base_url,
        captain_name="Team captain",
        batch_idempotency_key=idempotency_key,
        sender=sender,
    )
    return {**result, "invitation_deliveries": deliveries}


def replace_team_round_robin(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    event_option_id: str,
    draw_id: str,
    team_ids: list[str],
    expected_draw_updated_at: str,
    actor_email: str,
    idempotency_key: str,
) -> dict[str, Any]:
    require_admin_team_tournament_runtime()
    matchups = build_team_round_robin_matchups(team_ids)
    op = _operation(
        club_id=club_id,
        tournament_id=tournament_id,
        surface="operations",
        action="team_matchups_replace",
        entity_type="tournament_event_draw",
        entity_id=draw_id,
        expected_state=expected_draw_updated_at,
        payload={"matchups": matchups},
        idempotency_key=idempotency_key,
    )
    return _call_rpc(
        supabase,
        "admin_replace_tournament_team_matchups_cas",
        {
            "p_club_id": str(club_id),
            "p_tournament_id": str(tournament_id),
            "p_event_option_id": str(event_option_id),
            "p_draw_id": str(draw_id),
            "p_expected_draw_updated_at": expected_draw_updated_at,
            "p_matchups": matchups,
            "p_operation_key": op["operation_key"],
            "p_request_fingerprint": op["request_fingerprint"],
            "p_actor": actor_email,
        },
    )


def create_team_playoffs(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    playoff_format: str,
    expected_draw_updated_at: str,
    actor_email: str,
    idempotency_key: str,
) -> dict[str, Any]:
    require_admin_team_tournament_runtime()
    snapshot = get_admin_team_tournament_snapshot(
        supabase, club_id=club_id, tournament_id=tournament_id
    )
    draw = next(
        (row for row in snapshot["draws"] if str(row.get("id") or "") == str(draw_id)),
        None,
    )
    if not draw:
        raise ValueError("team draw not found")
    event = next(
        (
            row
            for row in snapshot["event_options"]
            if str(row.get("id") or "") == str(draw.get("event_option_id") or "")
        ),
        None,
    )
    configured = str((event or {}).get("team_playoff_format") or "NONE").upper()
    requested = str(playoff_format or "").upper()
    if configured != requested:
        raise ValueError("Playoff format must match the event setup.")
    matchups = build_team_playoff_matchups(
        snapshot["standings_by_draw"].get(str(draw_id), []),
        playoff_format=requested,
    )
    op = _operation(
        club_id=club_id,
        tournament_id=tournament_id,
        surface="operations",
        action="team_playoffs_append",
        entity_type="tournament_event_draw",
        entity_id=draw_id,
        expected_state=expected_draw_updated_at,
        payload={"playoff_format": requested, "matchups": matchups},
        idempotency_key=idempotency_key,
    )
    return _call_rpc(
        supabase,
        "admin_append_tournament_team_playoffs_cas",
        {
            "p_club_id": str(club_id),
            "p_tournament_id": str(tournament_id),
            "p_draw_id": str(draw_id),
            "p_expected_draw_updated_at": expected_draw_updated_at,
            "p_matchups": matchups,
            "p_operation_key": op["operation_key"],
            "p_request_fingerprint": op["request_fingerprint"],
            "p_actor": actor_email,
        },
    )


def lock_team_lineup(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    matchup_id: str,
    team_id: str,
    mixed_pairing: str,
    singles_tiebreak_player_id: int | None,
    expected_matchup_version: int,
    expected_lineup_version: int | None,
    actor_email: str,
    idempotency_key: str,
) -> dict[str, Any]:
    require_admin_team_tournament_runtime()
    payload = {
        "team_id": team_id,
        "mixed_pairing": str(mixed_pairing).upper(),
        "singles_tiebreak_player_id": singles_tiebreak_player_id,
    }
    op = _operation(
        club_id=club_id,
        tournament_id=tournament_id,
        surface="operations",
        action="team_lineup_lock",
        entity_type="tournament_team_matchup",
        entity_id=matchup_id,
        expected_state=f"{expected_matchup_version}:{expected_lineup_version}",
        payload=payload,
        idempotency_key=idempotency_key,
    )
    return _call_rpc(
        supabase,
        "admin_lock_tournament_team_lineup_cas",
        {
            "p_club_id": str(club_id),
            "p_tournament_id": str(tournament_id),
            "p_matchup_id": str(matchup_id),
            "p_team_id": str(team_id),
            "p_mixed_pairing": str(mixed_pairing).upper(),
            "p_singles_tiebreak_player_id": singles_tiebreak_player_id,
            "p_expected_matchup_version": int(expected_matchup_version),
            "p_expected_lineup_version": expected_lineup_version,
            "p_operation_key": op["operation_key"],
            "p_request_fingerprint": op["request_fingerprint"],
            "p_actor": actor_email,
        },
    )


def score_team_match_game(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    match_game_id: str,
    score_a: int,
    score_b: int,
    expected_game_version: int,
    expected_matchup_version: int,
    actor_email: str,
    idempotency_key: str,
) -> dict[str, Any]:
    require_admin_team_tournament_runtime()
    payload = {"score_a": int(score_a), "score_b": int(score_b)}
    op = _operation(
        club_id=club_id,
        tournament_id=tournament_id,
        surface="operations",
        action="team_match_game_score",
        entity_type="tournament_team_match_game",
        entity_id=match_game_id,
        expected_state=f"{expected_game_version}:{expected_matchup_version}",
        payload=payload,
        idempotency_key=idempotency_key,
    )
    return _call_rpc(
        supabase,
        "admin_score_tournament_team_match_game_cas",
        {
            "p_club_id": str(club_id),
            "p_tournament_id": str(tournament_id),
            "p_match_game_id": str(match_game_id),
            "p_score_a": int(score_a),
            "p_score_b": int(score_b),
            "p_expected_game_version": int(expected_game_version),
            "p_expected_matchup_version": int(expected_matchup_version),
            "p_operation_key": op["operation_key"],
            "p_request_fingerprint": op["request_fingerprint"],
            "p_actor": actor_email,
        },
    )


def reconcile_team_match_game(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    match_game_id: str,
    official_match_id: str,
    expected_official_row_version: int,
    expected_game_version: int,
    expected_matchup_version: int,
    reason: str,
    actor_email: str,
    idempotency_key: str,
) -> dict[str, Any]:
    require_admin_team_tournament_runtime()
    if not str(reason or "").strip():
        raise ValueError("A reconciliation reason is required.")
    payload = {"official_match_id": official_match_id, "reason": reason}
    op = _operation(
        club_id=club_id,
        tournament_id=tournament_id,
        surface="operations",
        action="team_match_game_reconcile",
        entity_type="tournament_team_match_game",
        entity_id=match_game_id,
        expected_state=(
            f"{expected_official_row_version}:{expected_game_version}:"
            f"{expected_matchup_version}"
        ),
        payload=payload,
        idempotency_key=idempotency_key,
    )
    return _call_rpc(
        supabase,
        "admin_reconcile_tournament_team_match_game_cas",
        {
            "p_club_id": str(club_id),
            "p_tournament_id": str(tournament_id),
            "p_match_game_id": str(match_game_id),
            "p_match_id": str(official_match_id),
            "p_expected_match_row_version": int(expected_official_row_version),
            "p_expected_game_version": int(expected_game_version),
            "p_expected_matchup_version": int(expected_matchup_version),
            "p_reason": reason,
            "p_operation_key": op["operation_key"],
            "p_request_fingerprint": op["request_fingerprint"],
            "p_actor": actor_email,
        },
    )


def amend_four_player_team_roster(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    team_id: str,
    expected_team_version: int,
    action: str,
    members: list[dict[str, Any]],
    reason: str,
    actor_email: str,
    idempotency_key: str,
    sender: Callable[..., dict[str, str]] = send_team_invitation_email,
) -> dict[str, Any]:
    require_admin_team_tournament_runtime()
    normalized_action = str(action or "").upper()
    if normalized_action == "REPLACE":
        # Registration-backed rosters are fully checked by the database; this
        # local validation catches malformed admin requests early.
        slots = {str(row.get("slot") or "").upper() for row in members}
        if slots != {"MAN_1", "MAN_2", "WOMAN_1", "WOMAN_2"}:
            raise ValueError("Replacement roster must fill all four team slots.")
    payload = {"action": normalized_action, "members": members, "reason": reason}
    op = _operation(
        club_id=club_id,
        tournament_id=tournament_id,
        surface="registration",
        action=f"four_player_roster_{normalized_action.lower()}",
        entity_type="tournament_four_player_team",
        entity_id=team_id,
        expected_state=expected_team_version,
        payload=payload,
        idempotency_key=idempotency_key,
    )
    result = _call_rpc(
        supabase,
        "admin_amend_tournament_four_player_roster_cas",
        {
            "p_club_id": str(club_id),
            "p_tournament_id": str(tournament_id),
            "p_team_id": str(team_id),
            "p_expected_team_version": int(expected_team_version),
            "p_action": normalized_action,
            "p_members": members,
            "p_reason": reason,
            "p_operation_key": op["operation_key"],
            "p_request_fingerprint": op["request_fingerprint"],
            "p_actor": actor_email,
        },
    )
    if normalized_action != "REPLACE":
        return result
    tournament = _one(
        supabase,
        "tournaments",
        filters=(("id", str(tournament_id)), ("club_id", str(club_id))),
    )
    if not tournament:
        raise RuntimeError("Tournament not found after roster replacement.")
    saved_members = [
        dict(row)
        for row in result.get("members") or []
        if isinstance(row, dict)
    ]
    captain = next(
        (
            row
            for row in saved_members
            if str(row.get("registration_id") or "")
            == str((result.get("team") or {}).get("captain_registration_id") or "")
        ),
        {},
    )
    deliveries = _deliver_invitations(
        supabase,
        club_id=club_id,
        tournament=tournament,
        team=dict(result.get("team") or {}),
        members=saved_members,
        actor_email=actor_email,
        public_base_url=resolve_team_invitation_base_url(
            supabase,
            club_id=club_id,
        ),
        captain_name=str(
            captain.get("display_name_snapshot") or "Team captain"
        ),
        batch_idempotency_key=idempotency_key,
        sender=sender,
    )
    return {**result, "invitation_deliveries": deliveries}


def replace_team_podium(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    expected_draw_updated_at: str,
    publish: bool,
    reason: str,
    actor_email: str,
    idempotency_key: str,
    podium: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    require_admin_team_tournament_runtime()
    snapshot = get_admin_team_tournament_snapshot(
        supabase, club_id=club_id, tournament_id=tournament_id
    )
    derived = snapshot["calculated_podium_by_draw"].get(str(draw_id))
    if not derived:
        raise ValueError("Podium cannot be calculated until required results are final.")
    if podium not in (None, [], derived):
        raise ValueError("Podium placements must match calculated results.")
    draw = next(
        (row for row in snapshot["draws"] if str(row.get("id") or "") == str(draw_id)),
        {},
    )
    was_public = str(draw.get("status") or "").lower() == "published"
    existing = [
        {"placement": row.get("placement"), "team_id": row.get("team_id")}
        for row in snapshot["podium"]
        if str(row.get("draw_id") or "") == str(draw_id)
    ]
    if (was_public and (not publish or existing != derived)) and not str(reason).strip():
        raise ValueError("A correction reason is required for a published podium change.")
    op = _operation(
        club_id=club_id,
        tournament_id=tournament_id,
        surface="operations",
        action="team_podium_replace",
        entity_type="tournament_event_draw",
        entity_id=draw_id,
        expected_state=expected_draw_updated_at,
        payload={"podium": derived, "publish": publish, "reason": reason},
        idempotency_key=idempotency_key,
    )
    return _call_rpc(
        supabase,
        "admin_replace_tournament_team_podium_cas",
        {
            "p_club_id": str(club_id),
            "p_tournament_id": str(tournament_id),
            "p_draw_id": str(draw_id),
            "p_expected_draw_updated_at": expected_draw_updated_at,
            # The database derives and checks this again. Empty caller input is
            # supported by the final hardening migration.
            "p_podium": [],
            "p_publish": bool(publish),
            "p_reason": reason,
            "p_operation_key": op["operation_key"],
            "p_request_fingerprint": op["request_fingerprint"],
            "p_actor": actor_email,
        },
    )


def build_admin_team_tournament_status(
    supabase: Any | None,
    *,
    club_id: str,
) -> dict[str, Any]:
    return {
        "enabled": is_admin_team_tournament_enabled(),
        "club_id": str(club_id),
        "email_mode": get_email_mode(),
        "service_role_configured": bool(
            os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        ),
        "checked_at": datetime.now(UTC).isoformat(),
    }


def new_team_tournament_idempotency_key() -> str:
    return str(uuid4())


def invitation_batch_fingerprint(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()
