from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jupr_app.data.load import load_data
from jupr_app.domain.league_analytics import compute_team_league_standings
from jupr_app.domain.team_league_roster import (
    TEAM_LEAGUE_TEAM_SIZES,
    normalize_player_gender,
    normalize_roster_settings,
    normalize_team_category,
    validate_playing_lineup,
    validate_team_members,
)
from jupr_app.domain.notifications.team_league_partner_invitation_email import (
    send_team_league_partner_invitation_email,
)
from jupr_app.services.direct_match_entry_service import (
    DirectMatchConflictError,
    DirectMatchRecoveryRequiredError,
    submit_atomic_direct_matches,
)
from jupr_app.services.staging_write_guard import (
    require_staging_admin_team_league_writes,
    require_staging_public_team_league_writes,
)


TEAM_SIGNUP_CONFIRMATION = "REGISTER TEAM"
SOLO_SIGNUP_CONFIRMATION = "JOIN PARTNER WAITLIST"
SAVE_SETTINGS_CONFIRMATION = "SAVE TEAM LEAGUE"
CREATE_TEAM_CONFIRMATION = "CREATE TEAM"
UPDATE_ROSTER_CONFIRMATION = "UPDATE TEAM ROSTER"
UPDATE_SUBSTITUTE_POOL_CONFIRMATION = "UPDATE SUBSTITUTE POOL"
PAIR_WAITLIST_CONFIRMATION = "PAIR WAITLIST PLAYERS"
WITHDRAW_WAITLIST_CONFIRMATION = "WITHDRAW WAITLIST PLAYERS"
COMMIT_SCHEDULE_CONFIRMATION = "PUBLISH TEAM LEAGUE SCHEDULE"
COMMIT_PLAYOFFS_CONFIRMATION = "PUBLISH TEAM LEAGUE PLAYOFFS"
SCORE_FIXTURE_CONFIRMATION = "SAVE TEAM LEAGUE RESULT"
FORFEIT_FIXTURE_CONFIRMATION = "SAVE TEAM LEAGUE FORFEIT"
RECONCILE_FIXTURE_CONFIRMATION = "RECONCILE TEAM LEAGUE RESULT"
FINALIZE_OPERATION_CONFIRMATION = "FINALIZE TEAM LEAGUE RECOVERY"
COMPENSATE_OPERATION_CONFIRMATION = "COMPENSATE TEAM LEAGUE RECOVERY"
IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$")
PUBLIC_VISIBLE_STATUSES = {
    "registration_open",
    "registration_closed",
    "scheduled",
    "active",
    "playoffs",
    "complete",
    "archived",
}
PUBLIC_MANAGER_VISIBLE_STATUSES = {"active", "ended"}


class TeamLeagueConflictError(RuntimeError):
    pass


class TeamLeagueRecoveryRequiredError(RuntimeError):
    def __init__(self, message: str, *, operation_id: str | None = None):
        super().__init__(message)
        self.operation_id = operation_id


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _text(value: Any, limit: int = 240) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _email(value: Any) -> str:
    result = _text(value, 320).lower()
    if len(result) < 3 or "@" not in result:
        raise ValueError("Enter a valid email address.")
    return result


def _int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
        result = int(number)
        return result if number == result else None
    except Exception:
        return None


def _data(response: Any) -> list[dict[str, Any]]:
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return [dict(data)]
    return [dict(row) for row in (data or []) if isinstance(row, Mapping)]


def _payload(response: Any) -> dict[str, Any]:
    rows = _data(response)
    return rows[0] if rows else {}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def confirmed_roster_fingerprint(
    teams: Sequence[Mapping[str, Any]],
    members: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    confirmed_ids = {
        str(row.get("id")) for row in teams if row.get("status") == "confirmed"
    }
    normalized = [
        row
        for row in (members or [])
        if str(row.get("team_id")) in confirmed_ids
        and str(row.get("status") or "").lower() == "active"
    ]
    if normalized:
        evidence = "|".join(
            f"{row.get('team_id')}:{int(row['player_id'])}:{row.get('role')}"
            for row in sorted(
                normalized,
                key=lambda row: (
                    str(row.get("team_id") or ""),
                    str(row.get("role") or ""),
                    int(row.get("player_id") or 0),
                ),
            )
        )
    else:
        evidence = "|".join(
            (
                f"{team.get('id')}:{int(team['captain_player_id'])}:captain|"
                f"{team.get('id')}:{int(team['partner_player_id'])}:primary"
            )
            for team in sorted(
                [row for row in teams if row.get("status") == "confirmed"],
                key=lambda row: str(row.get("id") or ""),
            )
        )
    return hashlib.sha256(evidence.encode("utf-8")).hexdigest()


def _operation_key(value: Any) -> str:
    key = _text(value, 160)
    if not IDEMPOTENCY_KEY_RE.fullmatch(key):
        raise ValueError(
            "idempotency_key must be 8–160 letters, numbers, dots, colons, "
            "underscores, or hyphens."
        )
    return key


def _confirm(value: Any, expected: str) -> None:
    if _text(value, 100) != expected:
        raise ValueError(f"Type {expected} to continue.")


def _assert_public_write_enabled() -> None:
    require_staging_public_team_league_writes()


def _assert_admin_write_enabled() -> None:
    require_staging_admin_team_league_writes()


def partner_token_secret() -> str:
    secret = os.getenv("JUPR_TEAM_LEAGUE_PARTNER_TOKEN_SECRET", "").strip()
    if not secret:
        secret = os.getenv("JUPR_PUBLIC_REGISTRATION_TOKEN_SECRET", "").strip()
    if not secret:
        secret = os.getenv("JUPR_REGISTRATION_EDIT_SECRET", "").strip()
    if len(secret) < 32:
        raise PermissionError(
            "Configure JUPR_TEAM_LEAGUE_PARTNER_TOKEN_SECRET (or the public "
            "registration/edit token fallback) with at least 32 characters."
        )
    return secret


def partner_token_hash(token: str) -> str:
    clean = str(token or "").strip()
    if len(clean) < 24:
        raise ValueError("The partner invitation token is invalid.")
    return hmac.new(
        partner_token_secret().encode("utf-8"),
        f"team-league-partner:v1:{clean}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _new_partner_token() -> tuple[str, str]:
    token = secrets.token_urlsafe(36)
    return token, partner_token_hash(token)


def _rpc(supabase: Any, name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = _payload(supabase.rpc(name, dict(params)).execute())
    except Exception as exc:
        detail = str(exc)
        if any(
            marker in detail
            for marker in (
                "VERSION_CONFLICT",
                "IDEMPOTENCY_CONFLICT",
                "IDENTITY_CONFLICT",
                "CHANGED",
                "LOCKED",
                "ALREADY",
                "RESERVATION_CONFLICT",
            )
        ):
            raise TeamLeagueConflictError(
                "Team-league data changed. Reload before trying again."
            ) from exc
        if "RECOVERY_REQUIRED" in detail or "DELIVERY_IN_PROGRESS" in detail:
            raise TeamLeagueRecoveryRequiredError(
                "This operation needs recovery before another attempt."
            ) from exc
        if "NOT_FOUND" in detail:
            raise ValueError("The requested team-league record was not found.") from exc
        if "CLOSED" in detail or "EXPIRED" in detail:
            raise ValueError("Registration or this invitation is closed.") from exc
        raise
    if not result:
        raise TeamLeagueRecoveryRequiredError(
            "The server returned no durable receipt. Retry only with the exact "
            "same request and idempotency key."
        )
    return result


def _fetch_rows(
    supabase: Any,
    table_name: str,
    *,
    filters: Mapping[str, Any],
    order: str | None = None,
    page_size: int = 500,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = supabase.table(table_name).select("*")
        for field, value in filters.items():
            query = query.eq(field, value)
        if order:
            query = query.order(order)
        ranged = getattr(query, "range", None)
        if callable(ranged):
            batch = _data(ranged(offset, offset + page_size - 1).execute())
        else:
            batch = _data(query.execute())
        rows.extend(batch)
        if not callable(ranged) or len(batch) < page_size:
            return rows
        offset += page_size


def _one(
    supabase: Any, table_name: str, *, filters: Mapping[str, Any]
) -> dict[str, Any] | None:
    query = supabase.table(table_name).select("*")
    for field, value in filters.items():
        query = query.eq(field, value)
    rows = _data(query.limit(1).execute())
    return rows[0] if rows else None


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except Exception:
        return None


def _registration_is_open(settings: Mapping[str, Any]) -> bool:
    if not bool(settings.get("registration_open")):
        return False
    if str(settings.get("status") or "") != "registration_open":
        return False
    closes_at = _parse_timestamp(settings.get("registration_closes_at"))
    return closes_at is None or closes_at > _now()


def _team_size(settings: Mapping[str, Any]) -> int:
    """Return the backward-compatible configured primary roster size."""

    value = _int(settings.get("team_size"))
    return value if value in TEAM_LEAGUE_TEAM_SIZES else 2


def _team_category(value: Any) -> str:
    return normalize_team_category(value)


def _normalized_player_gender(value: Any) -> str | None:
    """Normalize the legacy player gender spellings used by club imports."""

    return normalize_player_gender(value)


def _enforce_team_category(
    category: Any, player_rows: Sequence[Mapping[str, Any]]
) -> None:
    """Fail closed when a two-player roster does not satisfy its category."""

    try:
        validate_playing_lineup(category=category, player_rows=player_rows)
    except ValueError as exc:
        clean_category = _team_category(category)
        label = {
            "mens": "Men's",
            "womens": "Women's",
            "mixed": "Mixed",
        }.get(clean_category, "Team")
        raise ValueError(f"{label} team eligibility: {exc}") from exc


def _online_team_registration_supported(settings: Mapping[str, Any]) -> bool:
    """The durable public registration record still confirms one pair only."""

    return _team_size(settings) == 2


def _registration_unavailable_reason(settings: Mapping[str, Any]) -> str | None:
    if not _registration_is_open(settings):
        return "Registration is closed. The schedule and results remain available below."
    if not _online_team_registration_supported(settings):
        return (
            f"Online registration for {_team_size(settings)}-player team rosters "
            "is not available yet. Contact league staff to register the full roster."
        )
    return None


def _public_settings(row: Mapping[str, Any]) -> dict[str, Any]:
    configured_open = _registration_is_open(row)
    online_registration_supported = _online_team_registration_supported(row)
    return {
        key: row.get(key)
        for key in (
            "league_name",
            "status",
            "playoff_format",
            "playoff_team_count",
            "start_date",
            "weekday",
            "start_time",
            "timezone",
            "venue",
            "registration_closes_at",
            "schedule_version",
            "standings_version",
        )
    } | {
        "team_size": _team_size(row),
        "team_category": _team_category(row.get("team_category")),
        "max_alternates": _int(row.get("max_alternates")) or 0,
        "substitute_pool_enabled": bool(row.get("substitute_pool_enabled", False)),
        "mixed_required_men": _int(row.get("mixed_required_men")) or 1,
        "mixed_required_women": _int(row.get("mixed_required_women")) or 1,
        "allow_substitutes": bool(row.get("allow_substitutes", False)),
        "registration_configured_open": configured_open,
        "registration_open": configured_open
        and online_registration_supported,
        "online_team_registration_supported": online_registration_supported,
    }


def _legacy_member_rows(teams: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for team in teams:
        for role, field in (
            ("captain", "captain_player_id"),
            ("primary", "partner_player_id"),
        ):
            player_id = _int(team.get(field))
            if player_id is None:
                continue
            rows.append(
                {
                    "id": f"legacy:{team.get('id')}:{player_id}",
                    "team_id": str(team.get("id")),
                    "club_id": team.get("club_id"),
                    "league_name": team.get("league_name"),
                    "player_id": player_id,
                    "role": role,
                    "status": (
                        "active"
                        if team.get("status") == "confirmed"
                        or (
                            team.get("status") == "pending_partner"
                            and role == "captain"
                        )
                        else "invited"
                        if team.get("status") == "pending_partner"
                        else "declined"
                        if team.get("status") == "declined"
                        else "removed"
                    ),
                    "legacy_projection": True,
                }
            )
    return rows


def _normalized_members(
    teams: Sequence[Mapping[str, Any]],
    members: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = [dict(row) for row in members]
    team_ids = {str(row.get("team_id")) for row in normalized}
    normalized.extend(
        row
        for row in _legacy_member_rows(teams)
        if str(row.get("team_id")) not in team_ids
    )
    return normalized


def _team_members(
    members: Sequence[Mapping[str, Any]], team_id: Any
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in members
        if str(row.get("team_id")) == str(team_id)
        and str(row.get("status") or "").lower() in {"invited", "active"}
    ]


def _team_roster_complete(
    team: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> bool:
    active_primary = [
        row
        for row in _team_members(members, team.get("id"))
        if str(row.get("status") or "").lower() == "active"
        and str(row.get("role") or "primary").lower() in {"captain", "primary"}
    ]
    return len(active_primary) == _team_size(settings)


def _manager_league_status(row: Mapping[str, Any] | None) -> str:
    if not row:
        return "draft"
    status = _text(row.get("status"), 40).lower()
    if status in {"active", "running", "live"}:
        return "active"
    if status in {"ended", "complete", "completed", "done"}:
        return "ended"
    if status == "archived":
        return "archived"
    if status == "paused":
        return "paused"
    if status in {"draft", "planned"}:
        return "draft"
    if row.get("ended_at"):
        return "ended"
    return "active" if bool(row.get("is_active")) else "draft"


def _manager_league_is_public(row: Mapping[str, Any] | None) -> bool:
    """Keep draft, paused, and archived manager records off public routes."""

    return _manager_league_status(row) in PUBLIC_MANAGER_VISIBLE_STATUSES


def _manager_league_accepts_registration(row: Mapping[str, Any] | None) -> bool:
    return _manager_league_status(row) == "active"


def list_public_team_leagues(
    supabase: Any, *, club_id: str
) -> dict[str, Any]:
    settings = _fetch_rows(
        supabase,
        "team_league_settings",
        filters={"club_id": str(club_id)},
        order="start_date",
    )
    manager_rows = _fetch_rows(
        supabase,
        "leagues_metadata",
        filters={"club_id": str(club_id)},
        order="league_name",
    )
    manager_by_name = {
        _text(row.get("league_name"), 120).casefold(): row
        for row in manager_rows
        if _text(row.get("league_name"), 120)
    }
    visible = [
        _public_settings(row)
        for row in settings
        if _manager_league_is_public(
            manager_by_name.get(_text(row.get("league_name"), 120).casefold())
        )
        and str(row.get("status") or "") in PUBLIC_VISIBLE_STATUSES
    ]
    return {"ok": True, "leagues": visible, "league_count": len(visible)}


def get_public_team_league(
    supabase: Any, *, club_id: str, league_name: str
) -> dict[str, Any]:
    settings = _one(
        supabase,
        "team_league_settings",
        filters={"club_id": str(club_id), "league_name": _text(league_name, 120)},
    )
    manager = _one(
        supabase,
        "leagues_metadata",
        filters={"club_id": str(club_id), "league_name": _text(league_name, 120)},
    )
    if (
        not settings
        or not _manager_league_is_public(manager)
        or str(settings.get("status") or "") not in PUBLIC_VISIBLE_STATUSES
    ):
        raise ValueError("Team league not found.")
    teams = _fetch_rows(
        supabase,
        "team_league_teams",
        filters={
            "club_id": str(club_id),
            "league_name": _text(league_name, 120),
        },
        order="team_name",
    )
    member_rows = _normalized_members(
        teams,
        _fetch_rows(
            supabase,
            "team_league_team_members",
            filters={
                "club_id": str(club_id),
                "league_name": _text(league_name, 120),
            },
            order="created_at",
        ),
    )
    confirmed = [
        row
        for row in teams
        if row.get("status") == "confirmed"
        and _team_roster_complete(row, member_rows, settings)
    ]
    fixtures = _fetch_rows(
        supabase,
        "team_league_fixtures",
        filters={
            "club_id": str(club_id),
            "league_name": _text(league_name, 120),
        },
        order="scheduled_at",
    )
    confirmed_ids = {str(team.get("id")) for team in confirmed}
    player_ids = {
        int(row["player_id"])
        for row in member_rows
        if str(row.get("team_id")) in confirmed_ids
        and str(row.get("status") or "").lower() == "active"
        and _int(row.get("player_id")) is not None
    }
    all_player_rows = _fetch_rows(
        supabase, "players", filters={"club_id": str(club_id)}, order="name"
    )
    names: dict[int, str] = {}
    for row in all_player_rows:
        player_id = _int(row.get("id"))
        if player_id in player_ids:
            names[int(player_id)] = _text(row.get("name"), 160)
    public_teams = [
        {
            "id": str(team.get("id")),
            "team_name": _text(team.get("team_name"), 120),
            "players": [
                {
                    "player_id": int(member["player_id"]),
                    "player_name": names.get(
                        int(member["player_id"]),
                        f"Player {int(member['player_id'])}",
                    ),
                    "role": str(member.get("role") or "primary"),
                }
                for member in sorted(
                    _team_members(member_rows, team.get("id")),
                    key=lambda row: (
                        {"captain": 0, "primary": 1, "alternate": 2}.get(
                            str(row.get("role")), 9
                        ),
                        int(row.get("player_id") or 0),
                    ),
                )
                if str(member.get("status") or "").lower() == "active"
            ],
            "team_size": _team_size(settings),
            "roster_complete": True,
        }
        for team in confirmed
    ]
    public_fixtures = [
        {
            key: fixture.get(key)
            for key in (
                "id",
                "phase",
                "round_number",
                "week_number",
                "bracket_slot",
                "scheduled_at",
                "team_a_id",
                "team_b_id",
                "status",
                "team_a_score",
                "team_b_score",
                "winner_team_id",
                "resolution",
            )
        }
        for fixture in fixtures
    ]
    registration_supported = _online_team_registration_supported(settings)
    registration_open = _registration_is_open(settings) and registration_supported
    from jupr_app.services.admin_league_awards_service import (
        get_public_league_award_progress,
    )

    return {
        "ok": True,
        "league": _public_settings(settings),
        "teams": public_teams,
        "fixtures": public_fixtures,
        "standings": compute_team_league_standings(fixtures, confirmed),
        "award_progress": get_public_league_award_progress(
            supabase,
            club_id=str(club_id),
            league_name=_text(league_name, 120),
        ),
        "registration": {
            "open": registration_open,
            "payment_mode": "offline",
            "signup_types": (
                ["team", "solo_waitlist"] if registration_open else []
            ),
            "partner_confirmation_required": registration_supported,
            "online_team_registration_supported": registration_supported,
            "unavailable_reason": _registration_unavailable_reason(settings),
        },
        "registration_players": [
            {
                "player_id": int(row["id"]),
                "player_name": _text(row.get("name"), 160),
                "rating_jupr": (
                    round(float(row.get("rating")) / 400.0, 3)
                    if row.get("rating") not in (None, "")
                    else None
                ),
                "gender": row.get("gender"),
            }
            for row in all_player_rows
            if _int(row.get("id")) is not None
            and bool(row.get("active", True))
            and not row.get("inactive_at")
        ],
    }


def register_public_team_league(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    signup_type: str,
    player_id: Any,
    contact_email: str,
    idempotency_key: str,
    confirmation_text: str,
    partner_player_id: Any = None,
    team_name: str = "",
    partner_email: str = "",
    note: str = "",
    public_base_url: str,
    club_name: str,
) -> dict[str, Any]:
    _assert_public_write_enabled()
    clean_league_name = _text(league_name, 120)
    settings = _one(
        supabase,
        "team_league_settings",
        filters={"club_id": str(club_id), "league_name": clean_league_name},
    )
    manager = _one(
        supabase,
        "leagues_metadata",
        filters={"club_id": str(club_id), "league_name": clean_league_name},
    )
    if (
        not settings
        or not _manager_league_accepts_registration(manager)
        or not _registration_is_open(settings)
    ):
        raise ValueError("Registration is not open for this team league.")
    if not _online_team_registration_supported(settings):
        raise ValueError(
            f"Online registration for {_team_size(settings)}-player team rosters "
            "is not available yet. Contact league staff to register the full roster."
        )
    clean_type = _text(signup_type, 20).lower()
    if clean_type not in {"team", "solo"}:
        raise ValueError("Choose team registration or the solo partner waitlist.")
    _confirm(
        confirmation_text,
        TEAM_SIGNUP_CONFIRMATION
        if clean_type == "team"
        else SOLO_SIGNUP_CONFIRMATION,
    )
    clean_player_id = _int(player_id)
    if clean_player_id is None:
        raise ValueError("Choose your player profile.")
    clean_partner_id = _int(partner_player_id)
    if clean_type == "team" and (
        clean_partner_id is None or clean_partner_id == clean_player_id
    ):
        raise ValueError("Choose a different partner profile.")
    if clean_type == "team" and _team_category(settings.get("team_category")) != "open":
        player_rows = [
            _one(
                supabase,
                "players",
                filters={"club_id": str(club_id), "id": selected_player_id},
            )
            for selected_player_id in (clean_player_id, clean_partner_id)
        ]
        if any(row is None for row in player_rows):
            raise ValueError(
                "Team eligibility could not load both player profiles. "
                "Choose active club players and try again."
            )
        _enforce_team_category(
            settings.get("team_category"),
            [row for row in player_rows if row is not None],
        )
    clean_email = _email(contact_email)
    clean_partner_email = _email(partner_email) if clean_type == "team" else ""
    clean_team_name = _text(team_name, 120) if clean_type == "team" else ""
    if clean_type == "team" and not clean_team_name:
        raise ValueError("Enter a team name.")
    key = _operation_key(idempotency_key)
    token, token_hash = _new_partner_token() if clean_type == "team" else ("", "")
    expires_at = _now() + timedelta(days=7)
    request = {
        "club_id": str(club_id),
        "league_name": clean_league_name,
        "signup_type": clean_type,
        "player_id": clean_player_id,
        "partner_player_id": clean_partner_id,
        "team_name": clean_team_name,
        "contact_email": clean_email,
        "partner_email": clean_partner_email,
        "note": _text(note, 500),
    }
    request_fingerprint = _fingerprint(request)
    operation_id = str(uuid4())
    recovery_params = {
        "p_club_id": str(club_id),
        "p_league_name": request["league_name"],
        "p_request_fingerprint": request_fingerprint,
        "p_signup_type": clean_type,
        "p_player_id": clean_player_id,
        "p_partner_player_id": clean_partner_id,
        "p_invite_token_hash": token_hash or None,
        "p_invite_expires_at": expires_at.isoformat()
        if clean_type == "team"
        else None,
        "p_source": "public_team_league_registration_recovery",
    }

    def recover_exact_registration() -> dict[str, Any]:
        return _rpc(
            supabase,
            "team_league_recover_public_registration_v1",
            recovery_params,
        )

    result = recover_exact_registration()
    if not bool(result.get("found")):
        try:
            result = _rpc(
                supabase,
                "team_league_register_public_v1",
                {
                    "p_operation_id": operation_id,
                    "p_club_id": str(club_id),
                    "p_league_name": request["league_name"],
                    "p_idempotency_key": key,
                    "p_request_fingerprint": request_fingerprint,
                    "p_signup_type": clean_type,
                    "p_player_id": clean_player_id,
                    "p_partner_player_id": clean_partner_id,
                    "p_team_name": clean_team_name,
                    "p_contact_email": clean_email,
                    "p_partner_email": clean_partner_email,
                    "p_note": request["note"],
                    "p_invite_token_hash": token_hash or None,
                    "p_invite_expires_at": expires_at.isoformat()
                    if clean_type == "team"
                    else None,
                    "p_source": "public_team_league_registration",
                },
            )
        except Exception as registration_error:
            try:
                recovered = recover_exact_registration()
            except TeamLeagueConflictError:
                raise
            except Exception:
                raise registration_error
            if not bool(recovered.get("found")):
                raise registration_error
            result = recovered
    if clean_type != "team" or (
        bool(result.get("idempotent"))
        and not bool(result.get("invitation_send_required"))
    ):
        return {**result, "payment_mode": "offline"}
    team_id = str(result.get("team_id") or "")
    claim_token = str(uuid4())
    claim = _rpc(
        supabase,
        "team_league_claim_partner_invitation_v1",
        {
            "p_team_id": team_id,
            "p_token_hash": token_hash,
            "p_claim_token": claim_token,
        },
    )
    if not bool(claim.get("send_required")):
        return {
            **result,
            "invitation_delivery_status": claim.get("status"),
            "payment_mode": "offline",
        }
    players = _fetch_rows(
        supabase, "players", filters={"club_id": str(club_id)}, order="name"
    )
    names = {
        int(row["id"]): _text(row.get("name"), 160)
        for row in players
        if _int(row.get("id")) is not None
    }
    base = str(public_base_url or "").strip().rstrip("/")
    confirmation_url = (
        f"{base}/team-league-partner-confirmation"
        f"?team={team_id}#token={token}"
    )
    try:
        delivery = send_team_league_partner_invitation_email(
            target_email=clean_partner_email,
            club_name=_text(club_name, 160) or str(club_id),
            league_name=request["league_name"],
            team_name=clean_team_name,
            captain_name=names.get(clean_player_id, f"Player {clean_player_id}"),
            partner_name=names.get(
                int(clean_partner_id), f"Player {int(clean_partner_id)}"
            ),
            confirmation_url=confirmation_url,
            expires_label=expires_at.strftime("%B %-d, %Y at %-I:%M %p UTC"),
        )
        _rpc(
            supabase,
            "team_league_finish_partner_invitation_v1",
            {
                "p_team_id": team_id,
                "p_claim_token": claim_token,
                "p_delivery_status": delivery["status"],
                "p_provider_message_id": delivery.get("provider_message_id"),
                "p_delivery_error": None,
                "p_source": "public_team_league_partner_invitation",
            },
        )
    except Exception as exc:
        _rpc(
            supabase,
            "team_league_finish_partner_invitation_v1",
            {
                "p_team_id": team_id,
                "p_claim_token": claim_token,
                "p_delivery_status": "failed",
                "p_provider_message_id": None,
                "p_delivery_error": _text(str(exc), 500),
                "p_source": "public_team_league_partner_invitation",
            },
        )
        raise TeamLeagueRecoveryRequiredError(
            "The team was saved, but the partner invitation needs a safe "
            "delivery retry.",
            operation_id=str(result.get("operation_id") or operation_id),
        ) from exc
    return {
        **result,
        "invitation_delivery_status": delivery["status"],
        "payment_mode": "offline",
    }


def confirm_public_team_league_partner(
    supabase: Any,
    *,
    club_id: str,
    team_id: str,
    token: str,
    accept: bool,
    idempotency_key: str,
) -> dict[str, Any]:
    _assert_public_write_enabled()
    key = _operation_key(idempotency_key)
    if not _one(
        supabase,
        "team_league_teams",
        filters={"id": str(team_id), "club_id": str(club_id)},
    ):
        raise ValueError("The requested team-league record was not found.")
    request = {"team_id": str(team_id), "accept": bool(accept)}
    return _rpc(
        supabase,
        "team_league_confirm_partner_public_v1",
        {
            "p_operation_id": str(uuid4()),
            "p_team_id": str(team_id),
            "p_token_hash": partner_token_hash(token),
            "p_accept": bool(accept),
            "p_idempotency_key": key,
            "p_request_fingerprint": _fingerprint(request),
            "p_source": "public_team_league_partner_confirmation",
        },
    )


def list_admin_team_leagues(
    supabase: Any, *, club_id: str
) -> dict[str, Any]:
    settings = _fetch_rows(
        supabase,
        "team_league_settings",
        filters={"club_id": str(club_id)},
        order="start_date",
    )
    return {
        "ok": True,
        "leagues": settings,
        "league_count": len(settings),
        "writes_staging_only": True,
    }


def get_admin_team_league(
    supabase: Any, *, club_id: str, league_name: str
) -> dict[str, Any]:
    clean_league = _text(league_name, 120)
    settings = _one(
        supabase,
        "team_league_settings",
        filters={"club_id": str(club_id), "league_name": clean_league},
    )
    if not settings:
        raise ValueError("Team league not found.")
    teams = _fetch_rows(
        supabase,
        "team_league_teams",
        filters={"club_id": str(club_id), "league_name": clean_league},
        order="team_name",
    )
    members = _normalized_members(
        teams,
        _fetch_rows(
            supabase,
            "team_league_team_members",
            filters={"club_id": str(club_id), "league_name": clean_league},
            order="created_at",
        ),
    )
    substitute_pool = _fetch_rows(
        supabase,
        "team_league_substitute_pool",
        filters={"club_id": str(club_id), "league_name": clean_league},
        order="created_at",
    )
    waitlist = _fetch_rows(
        supabase,
        "team_league_solo_waitlist",
        filters={"club_id": str(club_id), "league_name": clean_league},
        order="created_at",
    )
    fixtures = _fetch_rows(
        supabase,
        "team_league_fixtures",
        filters={"club_id": str(club_id), "league_name": clean_league},
        order="scheduled_at",
    )
    operations = _fetch_rows(
        supabase,
        "team_league_operations",
        filters={"club_id": str(club_id), "league_name": clean_league},
        order="started_at",
    )
    players = [
        {
            "id": int(row["id"]),
            "name": _text(row.get("name"), 160),
            "rating": row.get("rating"),
            "gender": row.get("gender"),
            "active": bool(row.get("active", True))
            and not bool(row.get("inactive_at")),
        }
        for row in _fetch_rows(
            supabase, "players", filters={"club_id": str(club_id)}, order="name"
        )
        if _int(row.get("id")) is not None
    ]
    pending_operations = [
        {
            key: operation.get(key)
            for key in (
                "id",
                "operation_type",
                "status",
                "idempotency_key",
                "request_json",
                "recovery_note",
                "started_at",
                "updated_at",
            )
        }
        for operation in operations
        if operation.get("status") == "started"
    ]
    player_by_id = {int(row["id"]): row for row in players}
    for team in teams:
        team["members"] = [
            {
                **row,
                "player_name": player_by_id.get(
                    int(row.get("player_id") or 0), {}
                ).get("name"),
                "gender": player_by_id.get(
                    int(row.get("player_id") or 0), {}
                ).get("gender"),
            }
            for row in _team_members(members, team.get("id"))
        ]
        team["roster_complete"] = _team_roster_complete(team, members, settings)
    roster_fingerprint = confirmed_roster_fingerprint(teams, members)
    return {
        "ok": True,
        "settings": settings,
        "teams": teams,
        "members": members,
        "substitute_pool": substitute_pool,
        "waitlist": waitlist,
        "players": players,
        "fixtures": fixtures,
        "standings": compute_team_league_standings(fixtures, teams),
        "pending_operations": pending_operations,
        "confirmed_roster_fingerprint": roster_fingerprint,
        "recovery_required": bool(pending_operations),
    }


def save_admin_team_league_settings(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    settings: Mapping[str, Any],
    expected_settings_version: int,
    idempotency_key: str,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    _assert_admin_write_enabled()
    _confirm(confirmation_text, SAVE_SETTINGS_CONFIRMATION)
    key = _operation_key(idempotency_key)
    roster_policy = normalize_roster_settings(settings)
    team_size = int(roster_policy["team_size"])
    team_category = str(roster_policy["team_category"])
    allow_substitutes = bool(settings.get("allow_substitutes"))
    if roster_policy["substitute_pool_enabled"] and not allow_substitutes:
        raise ValueError("Enable substitutes before enabling the substitute pool.")
    clean = {
        "registration_open": bool(settings.get("registration_open")),
        "team_size": team_size,
        "team_category": team_category,
        "max_alternates": int(roster_policy["max_alternates"]),
        "substitute_pool_enabled": bool(
            roster_policy["substitute_pool_enabled"]
        ),
        "mixed_required_men": int(roster_policy["mixed_required_men"]),
        "mixed_required_women": int(roster_policy["mixed_required_women"]),
        "allow_substitutes": allow_substitutes,
        "playoff_format": _text(settings.get("playoff_format"), 80) or "none",
        "playoff_team_count": _int(settings.get("playoff_team_count")),
        "start_date": _text(settings.get("start_date"), 20) or None,
        "weekday": _int(settings.get("weekday")) or 0,
        "start_time": _text(settings.get("start_time"), 20) or "18:00",
        "timezone": _text(settings.get("timezone"), 80) or "UTC",
        "venue": _text(settings.get("venue"), 240) or None,
        "registration_closes_at": _text(
            settings.get("registration_closes_at"), 80
        )
        or None,
    }
    if clean["playoff_format"] not in {
        "none",
        "top_2_final",
        "top_4_single_elimination",
        "all_team_single_elimination",
    }:
        raise ValueError("Choose a supported playoff format.")
    try:
        ZoneInfo(str(clean["timezone"]))
    except ZoneInfoNotFoundError as exc:
        raise ValueError("Choose a valid timezone.") from exc
    request = {
        "league_name": _text(league_name, 120),
        "settings": clean,
        "expected_settings_version": int(expected_settings_version),
    }
    return _rpc(
        supabase,
        "team_league_save_settings_v2",
        {
            "p_operation_id": str(uuid4()),
            "p_club_id": str(club_id),
            "p_league_name": request["league_name"],
            "p_idempotency_key": key,
            "p_request_fingerprint": _fingerprint(request),
            "p_expected_settings_version": int(expected_settings_version),
            "p_settings": clean,
            "p_actor_email": _text(actor_email, 320),
            "p_actor_role": _text(actor_role, 80),
            "p_source": _text(source, 160),
        },
    )


def create_admin_team_league_team(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    team_name: str,
    captain_player_id: Any,
    captain_contact_email: str,
    expected_roster_version: int,
    idempotency_key: str,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    initial_primary_player_id: Any = None,
    initial_primary_contact_email: str = "",
    source: str = "next_team_league_create_team",
) -> dict[str, Any]:
    """Create a forming normalized team with a captain and optional primary."""

    _assert_admin_write_enabled()
    _confirm(confirmation_text, CREATE_TEAM_CONFIRMATION)
    clean_name = _text(team_name, 120)
    if not clean_name:
        raise ValueError("Enter a team name.")
    captain_id = _int(captain_player_id)
    primary_id = _int(initial_primary_player_id)
    if captain_id is None:
        raise ValueError("Choose an active captain.")
    if primary_id is not None and primary_id == captain_id:
        raise ValueError("The captain and initial primary must be different players.")
    captain_email = _email(captain_contact_email)
    primary_email = (
        _email(initial_primary_contact_email)
        if str(initial_primary_contact_email or "").strip()
        else ""
    )
    detail = get_admin_team_league(
        supabase, club_id=str(club_id), league_name=league_name
    )
    player_by_id = {int(row["id"]): row for row in detail["players"]}
    selected_ids = [captain_id] + ([primary_id] if primary_id is not None else [])
    if any(
        player_id not in player_by_id or not player_by_id[player_id].get("active")
        for player_id in selected_ids
    ):
        raise ValueError("Choose active club players for the forming team.")
    candidates = [
        {
            "player_id": captain_id,
            "player_name": player_by_id[captain_id].get("name"),
            "gender": player_by_id[captain_id].get("gender"),
            "role": "captain",
            "status": "active",
        }
    ]
    if primary_id is not None:
        candidates.append(
            {
                "player_id": primary_id,
                "player_name": player_by_id[primary_id].get("name"),
                "gender": player_by_id[primary_id].get("gender"),
                "role": "primary",
                "status": "active",
            }
        )
    validate_team_members(
        settings=detail["settings"],
        members=candidates,
        require_complete=len(candidates) == _team_size(detail["settings"]),
    )
    clean_league_name = _text(league_name, 120)
    request = {
        "league_name": clean_league_name,
        "team_name": clean_name,
        "captain_player_id": captain_id,
        "captain_contact_email": captain_email,
        "initial_primary_player_id": primary_id,
        "initial_primary_contact_email": primary_email,
        "expected_roster_version": int(expected_roster_version),
    }
    return _rpc(
        supabase,
        "team_league_create_team_v1",
        {
            "p_operation_id": str(uuid4()),
            "p_club_id": str(club_id),
            "p_league_name": clean_league_name,
            "p_team_name": clean_name,
            "p_captain_player_id": captain_id,
            "p_captain_contact_email": captain_email,
            "p_initial_primary_player_id": primary_id,
            "p_initial_primary_contact_email": primary_email or None,
            "p_expected_roster_version": int(expected_roster_version),
            "p_idempotency_key": _operation_key(idempotency_key),
            "p_request_fingerprint": _fingerprint(request),
            "p_actor_email": _text(actor_email, 320),
            "p_actor_role": _text(actor_role, 80),
            "p_source": _text(source, 160),
        },
    )


def admin_team_league_roster_action(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    action: str,
    player_id: Any,
    expected_roster_version: int,
    idempotency_key: str,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    team_id: str | None = None,
    member_role: str = "primary",
    member_status: str = "active",
    contact_email: str = "",
    note: str = "",
    source: str = "next_team_league_roster",
) -> dict[str, Any]:
    """Apply one normalized assigned-roster or substitute-pool mutation."""

    _assert_admin_write_enabled()
    clean_action = _text(action, 40).lower()
    if clean_action not in {"add_member", "remove_member", "set_pool"}:
        raise ValueError("Choose add member, remove member, or update substitute pool.")
    _confirm(
        confirmation_text,
        UPDATE_SUBSTITUTE_POOL_CONFIRMATION
        if clean_action == "set_pool"
        else UPDATE_ROSTER_CONFIRMATION,
    )
    clean_player_id = _int(player_id)
    if clean_player_id is None:
        raise ValueError("Choose an active club player.")
    detail = get_admin_team_league(
        supabase, club_id=str(club_id), league_name=league_name
    )
    player = next(
        (row for row in detail["players"] if int(row["id"]) == clean_player_id),
        None,
    )
    if not player or not player.get("active"):
        raise ValueError("Choose an active club player.")

    clean_team_id = str(team_id or "").strip() or None
    clean_role = _text(member_role, 20).lower()
    clean_status = _text(member_status, 20).lower()
    if clean_action in {"add_member", "remove_member"}:
        team = next(
            (row for row in detail["teams"] if str(row.get("id")) == clean_team_id),
            None,
        )
        if not team:
            raise ValueError("Choose a team in this league.")
        if clean_action == "add_member":
            if clean_role not in {"captain", "primary", "alternate"}:
                raise ValueError("Choose captain, primary, or alternate.")
            if clean_role == "captain" and clean_player_id != int(
                team.get("captain_player_id") or 0
            ):
                raise ValueError(
                    "The legacy team captain is stable; add this player as a primary or alternate."
                )
            if clean_player_id == int(team.get("captain_player_id") or 0) and (
                clean_role != "captain"
            ):
                raise ValueError(
                    "The team captain must keep the captain role."
                )
            if clean_status not in {"invited", "active"}:
                raise ValueError("Choose invited or active member status.")
            if any(
                int(row.get("player_id") or 0) == clean_player_id
                and str(row.get("status") or "")
                in {"available", "unavailable"}
                for row in detail.get("substitute_pool") or []
            ):
                raise ValueError(
                    "Withdraw this player from the substitute pool before assigning them to a team."
                )
            candidate = [
                row
                for row in _team_members(detail.get("members") or [], clean_team_id)
                if int(row.get("player_id") or 0) != clean_player_id
            ] + [
                {
                    "player_id": clean_player_id,
                    "player_name": player.get("name"),
                    "gender": player.get("gender"),
                    "role": clean_role,
                    "status": clean_status,
                }
            ]
            active_primary_count = len(
                [
                    row
                    for row in candidate
                    if str(row.get("status")) == "active"
                    and str(row.get("role")) in {"captain", "primary"}
                ]
            )
            validate_team_members(
                settings=detail["settings"],
                members=candidate,
                require_complete=active_primary_count
                == _team_size(detail["settings"]),
            )
    else:
        if clean_status not in {"available", "unavailable", "withdrawn"}:
            raise ValueError("Choose available, unavailable, or withdrawn.")
        if clean_status != "withdrawn" and not bool(
            detail["settings"].get("substitute_pool_enabled")
        ):
            raise ValueError("Enable the substitute pool in Team League settings first.")
        if clean_status != "withdrawn" and any(
            int(row.get("player_id") or 0) == clean_player_id
            and str(row.get("status") or "") in {"invited", "active"}
            for row in detail.get("members") or []
        ):
            raise ValueError(
                "A player assigned to a team cannot also join the substitute pool."
            )

    clean_email = _email(contact_email) if str(contact_email or "").strip() else ""
    clean_league_name = _text(league_name, 120)
    request = {
        "league_name": clean_league_name,
        "action": clean_action,
        "team_id": clean_team_id,
        "player_id": clean_player_id,
        "member_role": clean_role if clean_action == "add_member" else None,
        "member_status": clean_status,
        "contact_email": clean_email,
        "note": _text(note, 500),
        "expected_roster_version": int(expected_roster_version),
    }
    return _rpc(
        supabase,
        "team_league_apply_roster_action_v1",
        {
            "p_operation_id": str(uuid4()),
            "p_club_id": str(club_id),
            "p_league_name": clean_league_name,
            "p_action": clean_action,
            "p_team_id": clean_team_id,
            "p_player_id": clean_player_id,
            "p_member_role": clean_role if clean_action == "add_member" else None,
            "p_member_status": clean_status,
            "p_contact_email": clean_email or None,
            "p_note": request["note"] or None,
            "p_expected_roster_version": int(expected_roster_version),
            "p_idempotency_key": _operation_key(idempotency_key),
            "p_request_fingerprint": _fingerprint(request),
            "p_actor_email": _text(actor_email, 320),
            "p_actor_role": _text(actor_role, 80),
            "p_source": _text(source, 160),
        },
    )


def admin_team_league_waitlist_action(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    action: str,
    waitlist_ids: Sequence[str],
    idempotency_key: str,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    team_name: str = "",
    source: str = "next_team_league_waitlist",
) -> dict[str, Any]:
    _assert_admin_write_enabled()
    clean_action = _text(action, 20).lower()
    if clean_action not in {"pair", "withdraw"}:
        raise ValueError("Choose pair or withdraw.")
    _confirm(
        confirmation_text,
        PAIR_WAITLIST_CONFIRMATION
        if clean_action == "pair"
        else WITHDRAW_WAITLIST_CONFIRMATION,
    )
    ids = [str(value) for value in waitlist_ids if str(value).strip()]
    if clean_action == "pair" and (len(ids) != 2 or len(set(ids)) != 2):
        raise ValueError("Select exactly two waiting players to pair.")
    if clean_action == "withdraw" and not ids:
        raise ValueError("Select at least one waitlist entry.")
    clean_league_name = _text(league_name, 120)
    if clean_action == "pair":
        settings = _one(
            supabase,
            "team_league_settings",
            filters={
                "club_id": str(club_id),
                "league_name": clean_league_name,
            },
        )
        if not settings:
            raise ValueError("Team league not found.")
        if _team_size(settings) != 2:
            raise ValueError(
                "Pairing two waitlisted players is available only for a two-player roster. "
                "Manage larger team rosters directly until partial-team intake is implemented."
            )
        if _team_category(settings.get("team_category")) != "open":
            waitlist_rows = [
                _one(
                    supabase,
                    "team_league_solo_waitlist",
                    filters={
                        "id": waitlist_id,
                        "club_id": str(club_id),
                        "league_name": clean_league_name,
                    },
                )
                for waitlist_id in ids
            ]
            if any(
                row is None or _text(row.get("status"), 40).lower() != "waiting"
                for row in waitlist_rows
            ):
                raise ValueError(
                    "One or more selected waitlist entries are no longer available. "
                    "Reload the league and try again."
                )
            player_rows = [
                _one(
                    supabase,
                    "players",
                    filters={
                        "club_id": str(club_id),
                        "id": _int(row.get("player_id")),
                    },
                )
                for row in waitlist_rows
                if row is not None
            ]
            if len(player_rows) != 2 or any(row is None for row in player_rows):
                raise ValueError(
                    "Team eligibility could not load both waitlisted player profiles. "
                    "Reload the league and try again."
                )
            _enforce_team_category(
                settings.get("team_category"),
                [row for row in player_rows if row is not None],
            )
    request = {
        "action": clean_action,
        "waitlist_ids": sorted(ids),
        "team_name": _text(team_name, 120),
    }
    return _rpc(
        supabase,
        "team_league_admin_waitlist_action_v1",
        {
            "p_operation_id": str(uuid4()),
            "p_club_id": str(club_id),
            "p_league_name": clean_league_name,
            "p_action": clean_action,
            "p_waitlist_ids": ids,
            "p_team_name": request["team_name"] or None,
            "p_idempotency_key": _operation_key(idempotency_key),
            "p_request_fingerprint": _fingerprint(request),
            "p_actor_email": _text(actor_email, 320),
            "p_actor_role": _text(actor_role, 80),
            "p_source": _text(source, 160),
        },
    )


def _scheduled_at(
    start_date: str, start_time: str, timezone_name: str, week_offset: int
) -> str:
    try:
        day = date.fromisoformat(str(start_date))
        clock = time.fromisoformat(str(start_time))
        zone = ZoneInfo(str(timezone_name))
    except Exception as exc:
        raise ValueError(
            "A valid start date, start time, and timezone are required."
        ) from exc
    local = datetime.combine(day + timedelta(days=7 * week_offset), clock, zone)
    return local.astimezone(timezone.utc).isoformat()


def generate_round_robin_fixtures(
    team_ids: Sequence[str],
    *,
    start_date: str,
    start_time: str = "18:00",
    timezone_name: str = "UTC",
) -> list[dict[str, Any]]:
    teams = [str(team_id) for team_id in team_ids if str(team_id).strip()]
    if len(teams) < 2 or len(set(teams)) != len(teams):
        raise ValueError("At least two distinct confirmed teams are required.")
    rotation: list[str | None] = teams + ([None] if len(teams) % 2 else [])
    rounds = len(rotation) - 1
    fixtures: list[dict[str, Any]] = []
    for round_index in range(rounds):
        scheduled_at = _scheduled_at(
            start_date, start_time, timezone_name, round_index
        )
        slot = 1
        for index in range(len(rotation) // 2):
            team_a = rotation[index]
            team_b = rotation[-1 - index]
            if team_a is None or team_b is None:
                bye_team = team_b if team_a is None else team_a
                fixtures.append(
                    {
                        "round_number": round_index + 1,
                        "week_number": round_index + 1,
                        "bracket_slot": slot,
                        "scheduled_at": scheduled_at,
                        "team_a_id": bye_team,
                        "team_b_id": None,
                        "team_a_source": None,
                        "team_b_source": None,
                        "status": "bye",
                    }
                )
                slot += 1
                continue
            # Reverse alternating rounds to keep the presentation balanced.
            if round_index % 2:
                team_a, team_b = team_b, team_a
            fixtures.append(
                {
                    "round_number": round_index + 1,
                    "week_number": round_index + 1,
                    "bracket_slot": slot,
                    "scheduled_at": scheduled_at,
                    "team_a_id": team_a,
                    "team_b_id": team_b,
                    "team_a_source": None,
                    "team_b_source": None,
                    "status": "scheduled",
                }
            )
            slot += 1
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return fixtures


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < value:
        result *= 2
    return result


def _seed_positions(bracket_size: int) -> list[int]:
    positions = [1, 2]
    while len(positions) < bracket_size:
        next_size = len(positions) * 2
        positions = [
            candidate
            for seed in positions
            for candidate in (seed, next_size + 1 - seed)
        ]
    return positions


def generate_playoff_fixtures(
    standings: Sequence[Mapping[str, Any]],
    *,
    playoff_format: str,
    playoff_team_count: int | None = None,
) -> list[dict[str, Any]]:
    clean_format = _text(playoff_format, 80)
    if clean_format == "none":
        raise ValueError("This league is not configured for playoffs.")
    count = {
        "top_2_final": 2,
        "top_4_single_elimination": 4,
    }.get(clean_format)
    if count is None:
        if clean_format != "all_team_single_elimination":
            raise ValueError("Choose a supported playoff format.")
        count = int(playoff_team_count or len(standings))
    seeded = [dict(row) for row in standings[: min(count, len(standings))]]
    if len(seeded) < 2:
        raise ValueError("At least two ranked teams are required for playoffs.")
    bracket_size = _next_power_of_two(len(seeded))
    positions = _seed_positions(bracket_size)
    seed_map = {
        index + 1: str(row.get("team_id")) for index, row in enumerate(seeded)
    }
    fixtures: list[dict[str, Any]] = []
    round_slots = bracket_size // 2
    for slot in range(round_slots):
        seed_a = positions[slot * 2]
        seed_b = positions[slot * 2 + 1]
        team_a, team_b = seed_map.get(seed_a), seed_map.get(seed_b)
        if team_a is None and team_b is not None:
            team_a, team_b = team_b, None
            seed_a, seed_b = seed_b, seed_a
        fixtures.append(
            {
                "round_number": 1,
                "week_number": None,
                "bracket_slot": slot + 1,
                "scheduled_at": None,
                "team_a_id": team_a,
                "team_b_id": team_b,
                "team_a_source": f"seed:{seed_a}",
                "team_b_source": f"seed:{seed_b}" if team_b else None,
                "status": "scheduled" if team_b else "bye",
            }
        )
    previous_slots = round_slots
    round_number = 2
    while previous_slots > 1:
        next_slots = previous_slots // 2
        for slot in range(1, next_slots + 1):
            fixtures.append(
                {
                    "round_number": round_number,
                    "week_number": None,
                    "bracket_slot": slot,
                    "scheduled_at": None,
                    "team_a_id": None,
                    "team_b_id": None,
                    "team_a_source": f"winner:{round_number - 1}:{slot * 2 - 1}",
                    "team_b_source": f"winner:{round_number - 1}:{slot * 2}",
                    "status": "scheduled",
                }
            )
        previous_slots = next_slots
        round_number += 1
    return fixtures


def _fixture_match_date(
    scheduled_at: Any,
    *,
    timezone_name: Any,
) -> str:
    scheduled = _parse_timestamp(scheduled_at) or _now()
    try:
        local_timezone = ZoneInfo(_text(timezone_name, 120) or "UTC")
    except ZoneInfoNotFoundError as exc:
        raise ValueError("Choose a valid IANA timezone.") from exc
    return scheduled.astimezone(local_timezone).date().isoformat()


def build_admin_team_league_schedule_preview(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    phase: str,
) -> dict[str, Any]:
    detail = get_admin_team_league(
        supabase, club_id=str(club_id), league_name=league_name
    )
    settings = dict(detail["settings"])
    teams = [
        row
        for row in detail["teams"]
        if str(row.get("status") or "") == "confirmed"
        and bool(row.get("roster_complete"))
    ]
    roster_fingerprint = confirmed_roster_fingerprint(teams, detail.get("members"))
    clean_phase = _text(phase, 20).lower()
    if clean_phase == "regular":
        fixtures = generate_round_robin_fixtures(
            [str(row["id"]) for row in teams],
            start_date=str(settings.get("start_date") or ""),
            start_time=str(settings.get("start_time") or "18:00"),
            timezone_name=str(settings.get("timezone") or "UTC"),
        )
    elif clean_phase == "playoff":
        fixtures = generate_playoff_fixtures(
            detail["standings"],
            playoff_format=str(settings.get("playoff_format") or "none"),
            playoff_team_count=_int(settings.get("playoff_team_count")),
        )
    else:
        raise ValueError("Choose the regular season or playoffs.")
    names = {str(row["id"]): _text(row.get("team_name"), 120) for row in teams}
    current = [
        row
        for row in detail["fixtures"]
        if str(row.get("phase") or "") == clean_phase
    ]
    return {
        "ok": True,
        "phase": clean_phase,
        "league_name": _text(league_name, 120),
        "current_fixtures": current,
        "proposed_fixtures": fixtures,
        "team_names": names,
        "expected_schedule_version": int(settings.get("schedule_version") or 0),
        "expected_standings_version": int(
            settings.get("standings_version") or 0
        ),
        "expected_roster_version": int(settings.get("roster_version") or 0),
        "confirmed_roster_fingerprint": roster_fingerprint,
        "preview_fingerprint": _fingerprint(
            {
                "phase": clean_phase,
                "fixtures": fixtures,
                "schedule_version": int(settings.get("schedule_version") or 0),
                "standings_version": int(
                    settings.get("standings_version") or 0
                ),
                "roster_version": int(settings.get("roster_version") or 0),
                "confirmed_roster_fingerprint": roster_fingerprint,
            }
        ),
    }


def commit_admin_team_league_schedule(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    phase: str,
    fixtures: Sequence[Mapping[str, Any]],
    expected_schedule_version: int,
    expected_standings_version: int,
    expected_roster_version: int,
    confirmed_roster_fingerprint_value: str,
    preview_fingerprint: str,
    idempotency_key: str,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    _assert_admin_write_enabled()
    clean_phase = _text(phase, 20).lower()
    if clean_phase not in {"regular", "playoff"}:
        raise ValueError("Choose the regular season or playoffs.")
    _confirm(
        confirmation_text,
        COMMIT_SCHEDULE_CONFIRMATION
        if clean_phase == "regular"
        else COMMIT_PLAYOFFS_CONFIRMATION,
    )
    proposed = [dict(row) for row in fixtures]
    review = {
        "phase": clean_phase,
        "fixtures": proposed,
        "schedule_version": int(expected_schedule_version),
        "standings_version": int(expected_standings_version),
        "roster_version": int(expected_roster_version),
        "confirmed_roster_fingerprint": str(
            confirmed_roster_fingerprint_value
        ),
    }
    if _fingerprint(review) != str(preview_fingerprint):
        raise TeamLeagueConflictError(
            "The reviewed schedule is stale or was changed after preview."
        )
    request = {
        "phase": clean_phase,
        "expected_schedule_version": int(expected_schedule_version),
        "expected_standings_version": int(expected_standings_version),
        "expected_roster_version": int(expected_roster_version),
        "confirmed_roster_fingerprint": str(
            confirmed_roster_fingerprint_value
        ),
        "fixtures": proposed,
    }
    return _rpc(
        supabase,
        "team_league_replace_schedule_v2",
        {
            "p_operation_id": str(uuid4()),
            "p_club_id": str(club_id),
            "p_league_name": _text(league_name, 120),
            "p_phase": clean_phase,
            "p_idempotency_key": _operation_key(idempotency_key),
            "p_request_fingerprint": _fingerprint(request),
            "p_expected_schedule_version": int(expected_schedule_version),
            "p_expected_standings_version": int(expected_standings_version),
            "p_expected_roster_version": int(expected_roster_version),
            "p_confirmed_roster_fingerprint": str(
                confirmed_roster_fingerprint_value
            ),
            "p_fixtures": proposed,
            "p_actor_email": _text(actor_email, 320),
            "p_actor_role": _text(actor_role, 80),
            "p_source": _text(source, 160),
        },
    )


def _active_player_rows(
    supabase: Any, *, club_id: str
) -> dict[int, dict[str, Any]]:
    return {
        int(row["id"]): row
        for row in _fetch_rows(
            supabase, "players", filters={"club_id": str(club_id)}, order="name"
        )
        if _int(row.get("id")) is not None
        and bool(row.get("active", True))
        and not row.get("inactive_at")
    }


def _validate_fixture_players(
    *,
    players: Mapping[int, Mapping[str, Any]],
    teams: Sequence[Mapping[str, Any]],
    fixture: Mapping[str, Any],
    team_a_player_ids: Sequence[int],
    team_b_player_ids: Sequence[int],
    members: Sequence[Mapping[str, Any]],
    substitute_pool: Sequence[Mapping[str, Any]],
    team_category: Any,
    allow_substitutes: bool,
    substitute_pool_enabled: bool,
) -> list[dict[str, Any]]:
    side_a = [int(value) for value in team_a_player_ids]
    side_b = [int(value) for value in team_b_player_ids]
    if len(side_a) != 2 or len(side_b) != 2 or len(set(side_a + side_b)) != 4:
        raise ValueError("Each side needs two distinct active players.")
    if any(player_id not in players for player_id in side_a + side_b):
        raise ValueError("Every participant must be an active club player.")
    by_id = {str(team.get("id")): team for team in teams}
    team_a = by_id.get(str(fixture.get("team_a_id")))
    team_b = by_id.get(str(fixture.get("team_b_id")))
    if not team_a or not team_b:
        raise ValueError("Both scheduled teams are required.")
    validate_playing_lineup(
        category=team_category,
        player_rows=[players[player_id] for player_id in side_a],
    )
    validate_playing_lineup(
        category=team_category,
        player_rows=[players[player_id] for player_id in side_b],
    )
    roster_a = {
        int(row["player_id"])
        for row in _team_members(members, team_a.get("id"))
        if str(row.get("status") or "") == "active"
    }
    roster_b = {
        int(row["player_id"])
        for row in _team_members(members, team_b.get("id"))
        if str(row.get("status") or "") == "active"
    }
    if not roster_a:
        roster_a = {
            int(team_a["captain_player_id"]),
            int(team_a["partner_player_id"]),
        }
    if not roster_b:
        roster_b = {
            int(team_b["captain_player_id"]),
            int(team_b["partner_player_id"]),
        }
    active_team_players = {
        int(row["player_id"])
        for row in members
        if str(row.get("status") or "") in {"invited", "active"}
    }
    if not active_team_players:
        active_team_players = {
            int(team[field])
            for team in teams
            if team.get("status") in {"confirmed", "pending_partner"}
            for field in ("captain_player_id", "partner_player_id")
        }
    available_pool = {
        int(row["player_id"])
        for row in substitute_pool
        if str(row.get("status") or "") == "available"
    }
    substitutions: list[dict[str, Any]] = []
    for team, side, scheduled in (
        (team_a, side_a, roster_a),
        (team_b, side_b, roster_b),
    ):
        extras = set(side) - scheduled
        if extras and not allow_substitutes:
            raise ValueError("Substitutes are disabled for this league.")
        if extras and not substitute_pool_enabled:
            raise ValueError("Enable the shared substitute pool before using a substitute.")
        for incoming in sorted(extras):
            if incoming in active_team_players:
                raise ValueError(
                    "A substitute cannot be registered to another team in this league."
                )
            if incoming not in available_pool:
                raise ValueError(
                    "Every substitute must be available in this league's substitute pool."
                )
            substitutions.append(
                {
                    "incoming_player_id": incoming,
                    "team_id": str(team.get("id")),
                    "source": "substitute_pool",
                }
            )
    return substitutions


def _insert_score_operation(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    fixture_id: str,
    idempotency_key: str,
    request: Mapping[str, Any],
    actor_email: str,
    actor_role: str,
    source: str,
) -> tuple[str, dict[str, Any] | None]:
    existing = _one(
        supabase,
        "team_league_operations",
        filters={"club_id": str(club_id), "idempotency_key": idempotency_key},
    )
    fingerprint = _fingerprint(request)
    if existing:
        if str(existing.get("request_fingerprint") or "") != fingerprint:
            raise TeamLeagueConflictError(
                "This idempotency key belongs to a different result."
            )
        if existing.get("status") == "complete" and isinstance(
            existing.get("result_json"), Mapping
        ):
            return str(existing["id"]), {
                **dict(existing["result_json"]),
                "idempotent": True,
            }
        raise TeamLeagueRecoveryRequiredError(
            "A previous result submission is unfinished. Reconcile it before "
            "trying another score.",
            operation_id=str(existing.get("id") or ""),
        )
    operation_id = str(uuid4())
    payload = {
        "id": operation_id,
        "club_id": str(club_id),
        "league_name": _text(league_name, 120),
        "idempotency_key": idempotency_key,
        "request_fingerprint": fingerprint,
        "operation_type": "admin_score_fixture",
        "status": "started",
        "request_json": dict(request),
        "actor_email": _text(actor_email, 320),
        "actor_role": _text(actor_role, 80),
        "source": _text(source, 160),
    }
    try:
        rows = _data(supabase.table("team_league_operations").insert(payload).execute())
    except Exception as exc:
        raise TeamLeagueRecoveryRequiredError(
            "The score operation could not be reserved safely.",
            operation_id=operation_id,
        ) from exc
    if len(rows) != 1:
        raise TeamLeagueRecoveryRequiredError(
            "The score operation returned no durable receipt.",
            operation_id=operation_id,
        )
    return operation_id, None


def score_admin_team_league_fixture(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    fixture_id: str,
    status: str,
    team_a_score: Any,
    team_b_score: Any,
    winner_team_id: str,
    team_a_player_ids: Sequence[Any],
    team_b_player_ids: Sequence[Any],
    idempotency_key: str,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    score_note: str = "",
    source: str = "next_team_league_score",
) -> dict[str, Any]:
    _assert_admin_write_enabled()
    clean_status = _text(status, 20).lower()
    if clean_status not in {"complete", "forfeit"}:
        raise ValueError("Choose a played result or a forfeit.")
    _confirm(
        confirmation_text,
        SCORE_FIXTURE_CONFIRMATION
        if clean_status == "complete"
        else FORFEIT_FIXTURE_CONFIRMATION,
    )
    key = _operation_key(idempotency_key)
    detail = get_admin_team_league(
        supabase, club_id=str(club_id), league_name=league_name
    )
    fixture = next(
        (
            row
            for row in detail["fixtures"]
            if str(row.get("id") or "") == str(fixture_id)
        ),
        None,
    )
    if not fixture:
        raise ValueError("Fixture not found.")
    if str(fixture.get("status") or "") != "scheduled":
        raise TeamLeagueConflictError(
            "This fixture is no longer open for a new result."
        )
    if str(winner_team_id) not in {
        str(fixture.get("team_a_id") or ""),
        str(fixture.get("team_b_id") or ""),
    }:
        raise ValueError("Choose one of the scheduled teams as winner.")

    clean_a = [_int(value) for value in team_a_player_ids]
    clean_b = [_int(value) for value in team_b_player_ids]
    substitutions: list[dict[str, Any]] = []
    score_a = _int(team_a_score)
    score_b = _int(team_b_score)
    if clean_status == "complete":
        if (
            any(value is None for value in clean_a + clean_b)
            or score_a is None
            or score_b is None
            or min(score_a, score_b) < 0
            or score_a == score_b
        ):
            raise ValueError(
                "Enter two complete lineups and a non-tied non-negative score."
            )
        expected_winner = (
            str(fixture.get("team_a_id"))
            if score_a > score_b
            else str(fixture.get("team_b_id"))
        )
        if str(winner_team_id) != expected_winner:
            raise ValueError("The selected winner does not match the score.")
        substitutions = _validate_fixture_players(
            players=_active_player_rows(supabase, club_id=str(club_id)),
            teams=detail["teams"],
            fixture=fixture,
            team_a_player_ids=[int(value) for value in clean_a if value is not None],
            team_b_player_ids=[int(value) for value in clean_b if value is not None],
            members=detail.get("members") or [],
            substitute_pool=detail.get("substitute_pool") or [],
            team_category=detail["settings"].get("team_category"),
            allow_substitutes=bool(
                detail["settings"].get("allow_substitutes")
            ),
            substitute_pool_enabled=bool(
                detail["settings"].get("substitute_pool_enabled")
            ),
        )
    else:
        clean_a, clean_b = [None, None], [None, None]
        score_a = score_b = None

    request = {
        "fixture_id": str(fixture_id),
        "status": clean_status,
        "team_a_score": score_a,
        "team_b_score": score_b,
        "winner_team_id": str(winner_team_id),
        "team_a_player_ids": clean_a,
        "team_b_player_ids": clean_b,
        "substitutions": substitutions,
        "score_note": _text(score_note, 500),
    }
    operation_id, prior_result = _insert_score_operation(
        supabase,
        club_id=str(club_id),
        league_name=league_name,
        fixture_id=str(fixture_id),
        idempotency_key=key,
        request=request,
        actor_email=actor_email,
        actor_role=actor_role,
        source=source,
    )
    if prior_result is not None:
        return prior_result
    _rpc(
        supabase,
        "team_league_reserve_fixture_score_v1",
        {
            "p_operation_id": operation_id,
            "p_club_id": str(club_id),
            "p_league_name": _text(league_name, 120),
            "p_fixture_id": str(fixture_id),
            "p_team_a_id": str(fixture.get("team_a_id")),
            "p_team_b_id": str(fixture.get("team_b_id")),
        },
    )
    official_match_id: int | None = None
    match_receipt: dict[str, Any] | None = None
    if clean_status == "complete":
        match_payload = {
            "date": _fixture_match_date(
                fixture.get("scheduled_at"),
                timezone_name=detail["settings"].get("timezone"),
            ),
            "league": _text(league_name, 120),
            "week_tag": (
                f"Week {int(fixture.get('week_number'))}"
                if _int(fixture.get("week_number"))
                else (
                    f"Playoff Round {int(fixture.get('round_number') or 1)}"
                )
            ),
            "match_type": "Team League",
            "t1_p1": int(clean_a[0]),
            "t1_p2": int(clean_a[1]),
            "t2_p1": int(clean_b[0]),
            "t2_p2": int(clean_b[1]),
            "s1": int(score_a),
            "s2": int(score_b),
        }
        try:
            (
                df_players_all,
                _df_players_active,
                df_leagues,
                _df_matches,
                df_meta,
                _df_badges,
                _df_player_badges,
                name_to_id,
                _id_to_name,
                _schema_degraded,
                _schema_degraded_reason,
            ) = load_data(supabase, str(club_id))
            match_receipt = submit_atomic_direct_matches(
                supabase,
                club_id=str(club_id),
                matches=[match_payload],
                match_format="doubles",
                idempotency_key=f"teamfx:{fixture_id}",
                actor_email=_text(actor_email, 320),
                actor_role=_text(actor_role, 80),
                source=_text(source, 160),
                name_to_id=name_to_id,
                df_players_all=df_players_all,
                df_leagues=df_leagues,
                df_meta=df_meta,
            )
        except DirectMatchConflictError as exc:
            try:
                _rpc(
                    supabase,
                    "team_league_resolve_operation_v2",
                    {
                        "p_operation_id": operation_id,
                        "p_club_id": str(club_id),
                        "p_resolution": "compensate",
                        "p_result": None,
                        "p_recovery_note": "Canonical match rejected before commit.",
                        "p_actor_email": _text(actor_email, 320),
                        "p_actor_role": _text(actor_role, 80),
                        "p_source": _text(source, 160),
                    },
                )
            except Exception:
                pass
            raise TeamLeagueConflictError(str(exc)) from exc
        except DirectMatchRecoveryRequiredError as exc:
            raise TeamLeagueRecoveryRequiredError(
                str(exc), operation_id=operation_id
            ) from exc
        match_ids = list(
            dict(match_receipt.get("operation") or {}).get("match_ids") or []
        )
        official_match_id = _int(match_ids[0]) if match_ids else None
        if official_match_id is None:
            raise TeamLeagueRecoveryRequiredError(
                "The canonical match committed without a usable match receipt.",
                operation_id=operation_id,
            )
    try:
        finalized = _rpc(
            supabase,
            "team_league_finalize_fixture_v2",
            {
                "p_operation_id": operation_id,
                "p_club_id": str(club_id),
                "p_fixture_id": str(fixture_id),
                "p_status": clean_status,
                "p_team_a_score": score_a,
                "p_team_b_score": score_b,
                "p_winner_team_id": str(winner_team_id),
                "p_official_match_id": official_match_id,
                "p_team_a_player_1_id": clean_a[0],
                "p_team_a_player_2_id": clean_a[1],
                "p_team_b_player_1_id": clean_b[0],
                "p_team_b_player_2_id": clean_b[1],
                "p_substitutions": substitutions,
                "p_score_note": request["score_note"],
                "p_actor_email": _text(actor_email, 320),
                "p_actor_role": _text(actor_role, 80),
                "p_source": _text(source, 160),
            },
        )
    except Exception as exc:
        raise TeamLeagueRecoveryRequiredError(
            "The canonical result may be committed, but fixture finalization "
            "needs recovery.",
            operation_id=operation_id,
        ) from exc
    return {
        **finalized,
        "match_receipt": match_receipt,
        "recovery": {
            "operation_id": operation_id,
            "stable_match_key": f"teamfx:{fixture_id}"
            if clean_status == "complete"
            else None,
        },
    }


def _insert_reconcile_operation(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    fixture_id: str,
    idempotency_key: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> str:
    request = {"fixture_id": str(fixture_id)}
    existing = _one(
        supabase,
        "team_league_operations",
        filters={
            "club_id": str(club_id),
            "idempotency_key": idempotency_key,
        },
    )
    fingerprint = _fingerprint(request)
    if existing:
        if str(existing.get("request_fingerprint") or "") != fingerprint:
            raise TeamLeagueConflictError(
                "This idempotency key belongs to a different reconciliation."
            )
        return str(existing["id"])
    operation_id = str(uuid4())
    rows = _data(
        supabase.table("team_league_operations")
        .insert(
            {
                "id": operation_id,
                "club_id": str(club_id),
                "league_name": _text(league_name, 120),
                "idempotency_key": idempotency_key,
                "request_fingerprint": fingerprint,
                "operation_type": "admin_reconcile_fixture",
                "status": "started",
                "request_json": request,
                "actor_email": _text(actor_email, 320),
                "actor_role": _text(actor_role, 80),
                "source": _text(source, 160),
            }
        )
        .execute()
    )
    if len(rows) != 1:
        raise TeamLeagueRecoveryRequiredError(
            "The reconciliation operation was not durably reserved.",
            operation_id=operation_id,
        )
    return operation_id


def reconcile_admin_team_league_fixture(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    fixture_id: str,
    idempotency_key: str,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    source: str = "next_team_league_reconcile",
) -> dict[str, Any]:
    _assert_admin_write_enabled()
    _confirm(confirmation_text, RECONCILE_FIXTURE_CONFIRMATION)
    detail = get_admin_team_league(
        supabase, club_id=str(club_id), league_name=league_name
    )
    fixture = next(
        (
            row
            for row in detail["fixtures"]
            if str(row.get("id") or "") == str(fixture_id)
        ),
        None,
    )
    if not fixture or not fixture.get("official_match_id"):
        raise ValueError("This fixture has no canonical match to reconcile.")
    if str(fixture.get("phase") or "") == "regular" and any(
        str(row.get("phase") or "") == "playoff"
        for row in detail["fixtures"]
    ):
        raise TeamLeagueConflictError(
            "Regular-season results are locked after playoff seeding."
        )
    operation_id = _insert_reconcile_operation(
        supabase,
        club_id=str(club_id),
        league_name=league_name,
        fixture_id=str(fixture_id),
        idempotency_key=_operation_key(idempotency_key),
        actor_email=actor_email,
        actor_role=actor_role,
        source=source,
    )
    try:
        return _rpc(
            supabase,
            "team_league_reconcile_fixture_v2",
            {
                "p_operation_id": operation_id,
                "p_club_id": str(club_id),
                "p_fixture_id": str(fixture_id),
                "p_actor_email": _text(actor_email, 320),
                "p_actor_role": _text(actor_role, 80),
                "p_source": _text(source, 160),
            },
        )
    except Exception as exc:
        raise TeamLeagueRecoveryRequiredError(
            "Reconciliation needs review before another correction.",
            operation_id=operation_id,
        ) from exc


def inspect_admin_team_league_operation(
    supabase: Any, *, club_id: str, operation_id: str
) -> dict[str, Any]:
    operation = _one(
        supabase,
        "team_league_operations",
        filters={"club_id": str(club_id), "id": str(operation_id)},
    )
    if not operation:
        raise ValueError("Team-league operation not found.")
    fixture: dict[str, Any] | None = None
    direct_receipt: dict[str, Any] | None = None
    request = dict(operation.get("request_json") or {})
    fixture_id = str(request.get("fixture_id") or "")
    if fixture_id:
        fixture = _one(
            supabase,
            "team_league_fixtures",
            filters={"club_id": str(club_id), "id": fixture_id},
        )
        direct_receipt = _one(
            supabase,
            "admin_direct_match_entry_operations",
            filters={
                "club_id": str(club_id),
                "idempotency_key": f"teamfx:{fixture_id}",
            },
        )
    stable_commit = bool(
        direct_receipt
        and isinstance(direct_receipt.get("result_json"), Mapping)
        and dict(direct_receipt["result_json"]).get("committed")
    )
    if operation.get("status") == "complete":
        safe_action = "none"
    elif operation.get("operation_type") == "admin_score_fixture":
        safe_action = "finalize" if stable_commit else "compensate_or_retry"
    else:
        safe_action = "review"
    return {
        "ok": True,
        "operation": {
            key: operation.get(key)
            for key in (
                "id",
                "league_name",
                "operation_type",
                "status",
                "request_json",
                "result_json",
                "recovery_note",
                "created_at",
                "updated_at",
            )
        },
        "fixture": fixture,
        "stable_direct_match_receipt": {
            "found": bool(direct_receipt),
            "committed": stable_commit,
            "match_ids": (
                list(dict(direct_receipt.get("result_json") or {}).get("match_ids") or [])
                if direct_receipt
                else []
            ),
            "idempotency_key": f"teamfx:{fixture_id}" if fixture_id else None,
        },
        "safe_action": safe_action,
    }


def resolve_admin_team_league_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_id: str,
    resolution: str,
    recovery_note: str,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    source: str = "next_team_league_recovery",
) -> dict[str, Any]:
    _assert_admin_write_enabled()
    clean_resolution = _text(resolution, 20).lower()
    if clean_resolution not in {"finalize", "compensate"}:
        raise ValueError("Choose finalize or compensate.")
    _confirm(
        confirmation_text,
        FINALIZE_OPERATION_CONFIRMATION
        if clean_resolution == "finalize"
        else COMPENSATE_OPERATION_CONFIRMATION,
    )
    evidence = inspect_admin_team_league_operation(
        supabase, club_id=str(club_id), operation_id=str(operation_id)
    )
    operation = dict(evidence["operation"])
    if operation.get("status") == "complete":
        return {
            **dict(operation.get("result_json") or {}),
            "idempotent": True,
            "safe_action": "none",
        }
    if clean_resolution == "compensate":
        if bool(evidence["stable_direct_match_receipt"].get("committed")):
            raise TeamLeagueConflictError(
                "The canonical match is already committed; this operation "
                "must be finalized instead of compensated."
            )
        return _rpc(
            supabase,
            "team_league_resolve_operation_v2",
            {
                "p_operation_id": str(operation_id),
                "p_club_id": str(club_id),
                "p_resolution": "compensate",
                "p_result": None,
                "p_recovery_note": _text(recovery_note, 500),
                "p_actor_email": _text(actor_email, 320),
                "p_actor_role": _text(actor_role, 80),
                "p_source": _text(source, 160),
            },
        )
    if operation.get("operation_type") != "admin_score_fixture":
        raise TeamLeagueConflictError(
            "Only score operations with canonical commit evidence can use "
            "automatic finalization."
        )
    request = dict(operation.get("request_json") or {})
    receipt = dict(evidence["stable_direct_match_receipt"])
    status = str(request.get("status") or "")
    official_match_id = (
        _int((receipt.get("match_ids") or [None])[0])
        if status == "complete"
        else None
    )
    if status == "complete" and (
        not receipt.get("committed") or official_match_id is None
    ):
        raise TeamLeagueConflictError(
            "No committed canonical match receipt is available to finalize."
        )
    team_a_players = list(request.get("team_a_player_ids") or [None, None])
    team_b_players = list(request.get("team_b_player_ids") or [None, None])
    return _rpc(
        supabase,
        "team_league_finalize_fixture_v2",
        {
            "p_operation_id": str(operation_id),
            "p_club_id": str(club_id),
            "p_fixture_id": str(request.get("fixture_id") or ""),
            "p_status": status,
            "p_team_a_score": request.get("team_a_score"),
            "p_team_b_score": request.get("team_b_score"),
            "p_winner_team_id": request.get("winner_team_id"),
            "p_official_match_id": official_match_id,
            "p_team_a_player_1_id": team_a_players[0],
            "p_team_a_player_2_id": team_a_players[1],
            "p_team_b_player_1_id": team_b_players[0],
            "p_team_b_player_2_id": team_b_players[1],
            "p_substitutions": list(request.get("substitutions") or []),
            "p_score_note": _text(request.get("score_note"), 500),
            "p_actor_email": _text(actor_email, 320),
            "p_actor_role": _text(actor_role, 80),
            "p_source": _text(source, 160),
        },
    )
