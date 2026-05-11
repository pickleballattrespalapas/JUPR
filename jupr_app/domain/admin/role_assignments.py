from __future__ import annotations

from typing import Any

from jupr_app.domain.admin.roles import ROLE_SUPER_ADMIN, is_roles_table_missing_error, normalize_role


def normalize_email(email: str | None) -> str:
    return str(email or "").strip().lower()


def list_role_assignments(supabase: Any, club_id: str) -> list[dict[str, Any]]:
    response = (
        supabase.table("admin_role_assignments")
        .select("club_id,email,role,user_id,created_at,updated_at")
        .eq("club_id", str(club_id or "").strip())
        .order("email")
        .execute()
    )
    rows = list(response.data or [])
    for row in rows:
        row["email"] = normalize_email(row.get("email"))
        row["role"] = normalize_role(row.get("role"))
    return rows


def upsert_role_assignment(supabase: Any, club_id: str, email: str, role: str, user_id: str | None = None) -> None:
    payload = {
        "club_id": str(club_id or "").strip(),
        "email": normalize_email(email),
        "role": normalize_role(role),
        "user_id": str(user_id or "").strip() or None,
    }
    supabase.table("admin_role_assignments").upsert(payload, on_conflict="club_id,email").execute()


def delete_role_assignment(supabase: Any, club_id: str, email: str) -> None:
    supabase.table("admin_role_assignments").delete().eq("club_id", str(club_id or "").strip()).eq("email", normalize_email(email)).execute()


def count_super_admin_assignments(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if normalize_role(row.get("role")) == ROLE_SUPER_ADMIN)


def has_other_super_admin_support(
    *,
    rows: list[dict[str, Any]],
    target_email: str,
    admin_allowlist: set[str],
) -> bool:
    normalized_target = normalize_email(target_email)
    table_super_admins = {
        normalize_email(row.get("email"))
        for row in rows
        if normalize_role(row.get("role")) == ROLE_SUPER_ADMIN
    }
    remaining_table_super_admins = {email for email in table_super_admins if email != normalized_target}
    fallback_super_admins = {normalize_email(email) for email in admin_allowlist if normalize_email(email) != normalized_target}
    return bool(remaining_table_super_admins or fallback_super_admins)


def is_role_table_missing_error(exc: Exception) -> bool:
    return is_roles_table_missing_error(exc)
