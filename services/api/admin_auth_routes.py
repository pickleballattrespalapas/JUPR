from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, Query

from jupr_app.domain.admin.roles import ALL_ROLES, ROLE_PERMISSION_MATRIX
from services.api.auth import authenticate_bearer, auth_header


def _admin_access_denied() -> HTTPException:
    # Do not reveal whether an email, role assignment, or club exists.
    return HTTPException(status_code=403, detail="admin access denied")


def _matching_assignments(
    *,
    rows: list[dict[str, Any]],
    user_id: str,
) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    for row in rows:
        row_user_id = str(row.get("user_id") or "").strip()
        if row_user_id and row_user_id != user_id:
            continue
        role = str(row.get("role") or "").strip().lower()
        club_id = str(row.get("club_id") or "").strip()
        if role not in ALL_ROLES or not club_id:
            continue
        assignments.append(
            {
                "club_id": club_id,
                "role": role,
                "permissions": sorted(ROLE_PERMISSION_MATRIX.get(role, frozenset())),
            }
        )
    return sorted(assignments, key=lambda item: (item["club_id"], item["role"]))


def require_admin_assignments(
    *,
    get_supabase_client: Callable[[], Any],
    authorization: str | None,
    requested_club_id: str | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Verify a Supabase JWT and require at least one bound JUPR admin role.

    This is the shared read boundary for admin surfaces that are not tied to a
    single permission. It deliberately preserves generic denial and backend
    failure messages so role, club, and user-assignment details are not leaked.
    """

    user = authenticate_bearer(authorization)
    normalized_club_id = str(requested_club_id or "").strip() or None
    try:
        rows = (
            get_supabase_client()
            .table("admin_role_assignments")
            .select("club_id,role,user_id")
            .eq("email", user.email)
            .execute()
            .data
            or []
        )
    except Exception as exc:  # noqa: BLE001 - fail closed without leaking backend details
        raise HTTPException(status_code=503, detail="admin access check unavailable") from exc

    assignments = _matching_assignments(rows=list(rows), user_id=user.user_id)
    if normalized_club_id:
        assignments = [
            row for row in assignments if row["club_id"] == normalized_club_id
        ]
    if not assignments:
        raise _admin_access_denied()
    return user, assignments


def install_admin_auth_routes(app, *, get_supabase_client) -> None:
    """Install the verified JWT -> JUPR capability boundary used by admin login."""

    @app.get("/admin/auth/capabilities")
    def get_admin_auth_capabilities(
        club_id: str | None = Query(default=None),
        authorization: str | None = auth_header(),
    ) -> dict[str, Any]:
        requested_club_id = str(club_id or "").strip() or None
        user, assignments = require_admin_assignments(
            get_supabase_client=get_supabase_client,
            authorization=authorization,
            requested_club_id=requested_club_id,
        )

        return {
            "authorized": True,
            "user": {"email": user.email},
            "requested_club_id": requested_club_id,
            "assignments": assignments,
        }
