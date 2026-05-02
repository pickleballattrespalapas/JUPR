from __future__ import annotations

from jupr_app.domain.admin.roles import ROLE_SUPER_ADMIN, normalize_role

_ALLOWED_VIEW_AS_ROLES = frozenset({"club_owner", "organizer", "scorekeeper", "read_only"})


def can_use_view_as(real_role: str) -> bool:
    return normalize_role(real_role) == ROLE_SUPER_ADMIN


def sanitize_view_as_role(real_role: str, requested_view_as_role: str | None) -> str | None:
    if not can_use_view_as(real_role):
        return None
    candidate = normalize_role(requested_view_as_role)
    if candidate in _ALLOWED_VIEW_AS_ROLES:
        return candidate
    return None


def resolve_effective_admin_role(
    real_role: str,
    real_role_source: str,
    requested_view_as_role: str | None,
) -> tuple[str, str, str | None]:
    normalized_real_role = normalize_role(real_role)
    sanitized_view_as_role = sanitize_view_as_role(normalized_real_role, requested_view_as_role)

    if sanitized_view_as_role:
        return sanitized_view_as_role, "super_admin_view_as", sanitized_view_as_role

    return normalized_real_role, str(real_role_source or "admin_role_assignments"), None
