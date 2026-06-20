from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from postgrest.exceptions import APIError

ROLE_SUPER_ADMIN = "super_admin"
ROLE_CLUB_OWNER = "club_owner"
ROLE_ORGANIZER = "organizer"
ROLE_SCOREKEEPER = "scorekeeper"
ROLE_READ_ONLY = "read_only"

ALL_ROLES: tuple[str, ...] = (
    ROLE_SUPER_ADMIN,
    ROLE_CLUB_OWNER,
    ROLE_ORGANIZER,
    ROLE_SCOREKEEPER,
    ROLE_READ_ONLY,
)

PERMISSION_MANAGE_ROLES = "manage_roles"
PERMISSION_MANAGE_PLAYERS = "manage_players"
PERMISSION_MANAGE_MATCHES = "manage_matches"
PERMISSION_DELETE_MATCHES = "delete_matches"
PERMISSION_RUN_REPLAY = "run_replay"
PERMISSION_MANAGE_TOURNAMENTS = "manage_tournaments"
PERMISSION_MANAGE_SUBSCRIPTIONS = "manage_subscriptions"
PERMISSION_VIEW_AUDIT_LOG = "view_audit_log"
PERMISSION_ENTER_SCORES = "enter_scores"

ROLE_PERMISSION_MATRIX: dict[str, frozenset[str]] = {
    ROLE_SUPER_ADMIN: frozenset(
        {
            PERMISSION_MANAGE_ROLES,
            PERMISSION_MANAGE_PLAYERS,
            PERMISSION_MANAGE_MATCHES,
            PERMISSION_DELETE_MATCHES,
            PERMISSION_RUN_REPLAY,
            PERMISSION_MANAGE_TOURNAMENTS,
            PERMISSION_MANAGE_SUBSCRIPTIONS,
            PERMISSION_VIEW_AUDIT_LOG,
            PERMISSION_ENTER_SCORES,
        }
    ),
    ROLE_CLUB_OWNER: frozenset(
        {
            PERMISSION_MANAGE_PLAYERS,
            PERMISSION_MANAGE_MATCHES,
            PERMISSION_DELETE_MATCHES,
            PERMISSION_MANAGE_TOURNAMENTS,
            PERMISSION_MANAGE_SUBSCRIPTIONS,
            PERMISSION_VIEW_AUDIT_LOG,
            PERMISSION_ENTER_SCORES,
        }
    ),
    ROLE_ORGANIZER: frozenset(
        {
            PERMISSION_MANAGE_TOURNAMENTS,
            PERMISSION_ENTER_SCORES,
        }
    ),
    ROLE_SCOREKEEPER: frozenset(
        {
            PERMISSION_ENTER_SCORES,
        }
    ),
    ROLE_READ_ONLY: frozenset(
        {
            PERMISSION_VIEW_AUDIT_LOG,
        }
    ),
}


@dataclass(frozen=True)
class AdminRoleResolution:
    role: str
    source: str
    table_available: bool


def normalize_role(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in ALL_ROLES else ROLE_READ_ONLY


def is_roles_table_missing_error(exc: Exception) -> bool:
    if isinstance(exc, APIError):
        code = getattr(exc, "code", None)
        if code in {"42P01", "PGRST205"}:
            return True
        if exc.args and isinstance(exc.args[0], dict):
            payload = exc.args[0]
            if payload.get("code") in {"42P01", "PGRST205"}:
                return True
    return False


def resolve_admin_role(
    *,
    supabase: Any,
    club_id: str,
    email: str,
    user_id: str | None,
    allowlist: set[str],
) -> AdminRoleResolution:
    normalized_email = str(email or "").strip().lower()
    normalized_user_id = str(user_id or "").strip() or None

    try:
        response = (
            supabase.table("admin_role_assignments")
            .select("role,user_id")
            .eq("club_id", str(club_id or "").strip())
            .eq("email", normalized_email)
            .execute()
        )
        rows = response.data or []
        if rows:
            preferred_row = None
            if normalized_user_id:
                preferred_row = next(
                    (
                        row
                        for row in rows
                        if str(row.get("user_id") or "").strip() == normalized_user_id
                    ),
                    None,
                )
            if preferred_row is None:
                preferred_row = next(
                    (
                        row
                        for row in rows
                        if str(row.get("user_id") or "").strip() == ""
                    ),
                    None,
                )
            if preferred_row is None:
                preferred_row = rows[0]
            return AdminRoleResolution(
                role=normalize_role(preferred_row.get("role")),
                source="admin_role_assignments",
                table_available=True,
            )
    except Exception as exc:  # noqa: BLE001 - graceful fallback required
        if is_roles_table_missing_error(exc):
            # Keep legacy allowlist fallback only for Tres Palapas during migration rollout.
            if str(club_id or "").strip() == "tres_palapas" and normalized_email in allowlist:
                return AdminRoleResolution(
                    role=ROLE_SUPER_ADMIN,
                    source="allowlist_fallback_missing_table",
                    table_available=False,
                )
            return AdminRoleResolution(
                role=ROLE_READ_ONLY,
                source="missing_table_default",
                table_available=False,
            )
        raise

    # Keep legacy allowlist fallback only for Tres Palapas.
    if str(club_id or "").strip() == "tres_palapas" and normalized_email in allowlist:
        return AdminRoleResolution(
            role=ROLE_SUPER_ADMIN,
            source="allowlist_default",
            table_available=True,
        )

    return AdminRoleResolution(
        role=ROLE_READ_ONLY,
        source="default",
        table_available=True,
    )


def has_permission(role: str, permission: str) -> bool:
    normalized_role = normalize_role(role)
    return permission in ROLE_PERMISSION_MATRIX.get(normalized_role, frozenset())


def can_manage_roles(role: str) -> bool:
    return has_permission(role, PERMISSION_MANAGE_ROLES)


def can_manage_players(role: str) -> bool:
    return has_permission(role, PERMISSION_MANAGE_PLAYERS)


def can_manage_matches(role: str) -> bool:
    return has_permission(role, PERMISSION_MANAGE_MATCHES)


def can_delete_matches(role: str) -> bool:
    return has_permission(role, PERMISSION_DELETE_MATCHES)


def can_run_replay(role: str) -> bool:
    return has_permission(role, PERMISSION_RUN_REPLAY)


def can_manage_tournaments(role: str) -> bool:
    return has_permission(role, PERMISSION_MANAGE_TOURNAMENTS)


def can_manage_subscriptions(role: str) -> bool:
    return has_permission(role, PERMISSION_MANAGE_SUBSCRIPTIONS)


def can_view_audit_log(role: str) -> bool:
    return has_permission(role, PERMISSION_VIEW_AUDIT_LOG)


def can_enter_scores(role: str) -> bool:
    return has_permission(role, PERMISSION_ENTER_SCORES)
