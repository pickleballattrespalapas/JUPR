from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import resolve_admin_role
from jupr_app.domain.admin.staff_policy import ADMIN_ROLES, validate_scopes
from services.api.auth import authenticate_bearer, auth_header


class StaffAssignment(BaseModel):
    email: str = Field(min_length=3, max_length=254, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    role: Literal["administrator", "operator"]
    scopes: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    expires_at: datetime | None = None
    revoke: bool = False


def install_admin_staff_routes(app, *, get_supabase_client):
    def administrator(club_id, authorization):
        user = authenticate_bearer(authorization)
        db = get_supabase_client()
        role = resolve_admin_role(supabase=db, club_id=club_id, email=user.email,
                                  user_id=user.user_id, allowlist=set())
        if not role.assigned or role.role not in ADMIN_ROLES:
            raise HTTPException(403, "Club administrator access required.")
        return db, user

    @app.get("/admin/clubs/{club_id}/staff")
    def get_staff(club_id: str, authorization: str | None = auth_header()):
        db, _ = administrator(club_id, authorization)
        return {"staff": db.table("admin_role_assignments").select(
            "email,role,scopes,expires_at,revoked_at,updated_at"
        ).eq("club_id", club_id).order("email").execute().data or []}

    @app.get("/admin/clubs/{club_id}/staff/targets")
    def staff_targets(club_id: str, authorization: str | None = auth_header()):
        db, _ = administrator(club_id, authorization)
        targets = []
        for table, columns, program, key, label in (
            ("leagues_metadata", "league_name", "leagues", "league_name", "league_name"),
            ("tournaments", "id,name", "tournaments", "id", "name"),
            ("league_live_sessions", "id,league_name,week_tag", "leagues", "id", "league_name"),
            ("live_sessions", "session_key,title,state", "live_play", "session_key", "title"),
        ):
            for row in db.table(table).select(columns).eq("club_id", club_id).limit(1000).execute().data or []:
                kind = str((row.get("state") or {}).get("generator_kind") or program)
                targets.append({"program_type": kind, "resource_id": str(row[key]),
                                "label": str(row.get(label) or row[key]) + (" — " + str(row["week_tag"]) if row.get("week_tag") else "")})
        return {"targets": targets}

    @app.put("/admin/clubs/{club_id}/staff")
    def save_staff(club_id: str, payload: StaffAssignment, authorization: str | None = auth_header()):
        db, user = administrator(club_id, authorization)
        try:
            scopes = validate_scopes(payload.scopes) if payload.role == "operator" and not payload.revoke else []
            end = payload.expires_at
            if end and (end.tzinfo is None or end <= datetime.now(timezone.utc)) and not payload.revoke:
                raise ValueError("Choose a future expiration date with a timezone.")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            result = db.rpc("pcs_save_staff", {
                "p_club_id": club_id, "p_actor_email": user.email, "p_actor_id": user.user_id,
                "p_email": payload.email.strip().lower(), "p_role": payload.role,
                "p_scopes": scopes, "p_expires_at": end.isoformat() if end else None,
                "p_revoke": payload.revoke,
            }).execute()
        except Exception as exc:
            # The database repeats authorization and last-administrator checks atomically.
            message = str(exc)
            if "Keep at least one administrator" in message:
                raise HTTPException(409, "Keep at least one administrator.") from exc
            if "Platform access" in message or "Administrator access" in message:
                raise HTTPException(403, "This staff assignment cannot be changed by this account.") from exc
            raise HTTPException(503, "Staff access could not be saved. Refresh the list before retrying.") from exc
        return {"ok": True, "staff": result.data}
