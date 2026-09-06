from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from postgrest.exceptions import APIError

ROLE_ADMINISTRATOR = "administrator"
ROLE_OPERATOR = "operator"
ROLE_SUPER_ADMIN = "super_admin"
ROLE_CLUB_OWNER = "club_owner"
ROLE_ORGANIZER = "organizer"
ROLE_SCOREKEEPER = "scorekeeper"
ROLE_READ_ONLY = "read_only"
# Internal fail-closed sentinel. It is intentionally excluded from ALL_ROLES so it
# cannot be stored as an assignment or selected as an operator-facing role.
ROLE_UNASSIGNED = "__unassigned__"

ALL_ROLES: tuple[str, ...] = (
    ROLE_ADMINISTRATOR,
    ROLE_OPERATOR,
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


ROLE_PERMISSION_MATRIX[ROLE_ADMINISTRATOR] = (
    ROLE_PERMISSION_MATRIX[ROLE_SUPER_ADMIN] - {PERMISSION_MANAGE_ROLES}
) | {"manage_club_staff"}
ROLE_PERMISSION_MATRIX[ROLE_OPERATOR] = frozenset({
    PERMISSION_MANAGE_PLAYERS, PERMISSION_MANAGE_MATCHES,
    PERMISSION_MANAGE_TOURNAMENTS, PERMISSION_ENTER_SCORES,
})


@dataclass(frozen=True)
class AdminRoleResolution:
    role: str
    source: str
    table_available: bool
    assigned: bool


def normalize_role(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in ALL_ROLES else ROLE_READ_ONLY


def _api_error_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, APIError) and exc.args and isinstance(exc.args[0], dict):
        return exc.args[0]
    return {}


def is_roles_table_missing_error(exc: Exception) -> bool:
    if isinstance(exc, APIError):
        code = getattr(exc, "code", None)
        if code in {"42P01", "PGRST205"}:
            return True
        payload = _api_error_payload(exc)
        if payload.get("code") in {"42P01", "PGRST205"}:
            return True
    return False


def is_missing_column_error(exc: Exception, column: str) -> bool:
    if not isinstance(exc, APIError):
        return False
    payload = _api_error_payload(exc)
    code = getattr(exc, "code", None) or payload.get("code")
    message = " ".join(str(payload.get(key) or "") for key in ("message", "details", "hint"))
    return code in {"42703", "PGRST204"} and column in message


def _select_role_assignment_rows(*, supabase: Any, club_id: str, email: str) -> list[dict[str, Any]]:
    try:
        response = (
            supabase.table("admin_role_assignments")
            .select("*")
            .eq("club_id", str(club_id or "").strip())
            .eq("email", email)
            .execute()
        )
        return response.data or []
    except Exception as exc:  # noqa: BLE001 - tolerate older role table schema during migration
        if is_missing_column_error(exc, "user_id"):
            response = (
                supabase.table("admin_role_assignments")
                .select("role")
                .eq("club_id", str(club_id or "").strip())
                .eq("email", email)
                .execute()
            )
            rows = response.data or []
            for row in rows:
                row.setdefault("user_id", None)
            return rows
        raise


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
    assignment_user_id_mismatch = False

    try:
        rows = _select_role_assignment_rows(supabase=supabase, club_id=club_id, email=normalized_email)
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
            if preferred_row is None and normalized_user_id is None:
                preferred_row = rows[0]
            if preferred_row is not None:
                from jupr_app.domain.admin.staff_policy import assignment_active
                if not assignment_active(preferred_row):
                    return AdminRoleResolution(ROLE_UNASSIGNED, "expired_assignment", True, False)
                if preferred_row.get("role") == ROLE_OPERATOR:
                    from services.api.staff_access import authorize_operator_request
                    authorize_operator_request(supabase, str(club_id), preferred_row)
                return AdminRoleResolution(
                    role=normalize_role(preferred_row.get("role")),
                    source="admin_role_assignments",
                    table_available=True,
                    assigned=True,
                )
            assignment_user_id_mismatch = True
    except Exception as exc:  # noqa: BLE001 - graceful fallback required
        if is_roles_table_missing_error(exc):
            # Keep legacy allowlist fallback only for Tres Palapas during migration rollout.
            if str(club_id or "").strip() == "tres_palapas" and normalized_email in allowlist:
                return AdminRoleResolution(
                    role=ROLE_SUPER_ADMIN,
                    source="allowlist_fallback_missing_table",
                    table_available=False,
                    assigned=True,
                )
            return AdminRoleResolution(
                role=ROLE_UNASSIGNED,
                source="missing_table_default",
                table_available=False,
                assigned=False,
            )
        raise

    # Keep legacy allowlist fallback only for Tres Palapas.
    if str(club_id or "").strip() == "tres_palapas" and normalized_email in allowlist:
        return AdminRoleResolution(
            role=ROLE_SUPER_ADMIN,
            source="allowlist_default",
            table_available=True,
            assigned=True,
        )

    return AdminRoleResolution(
        role=ROLE_UNASSIGNED,
        source="admin_role_user_id_mismatch" if assignment_user_id_mismatch else "default",
        table_available=True,
        assigned=False,
    )


def has_permission(role: str, permission: str) -> bool:
    if str(role or "").strip().lower() == ROLE_UNASSIGNED:
        return False
    normalized_role = normalize_role(role)
    if normalized_role == ROLE_OPERATOR:
        from jupr_app.domain.admin.staff_policy import operator_request_authorized
        if not operator_request_authorized.get():
            return False
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
