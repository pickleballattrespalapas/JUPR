from __future__ import annotations

from datetime import date, datetime, timezone
import logging
from uuid import UUID
from zoneinfo import ZoneInfoNotFoundError

from fastapi import BackgroundTasks, HTTPException
from postgrest.exceptions import APIError
from pydantic import BaseModel, Field

from jupr_app.domain.admin.roles import resolve_admin_role
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue_until_empty
from jupr_app.services.admin_badge_diagnostics_service import is_admin_badge_diagnostics_enabled
from jupr_app.services.admin_badge_management_service import badge_management_options, save_badge_management
from services.api.auth import authenticate_bearer, auth_header


def check_saved_season(supabase, club_id):
    try:
        process_badge_eval_queue_until_empty(supabase, club_id, max_total_jobs=20, max_wall_clock_seconds=10)
    except Exception:
        # The season and its durable evaluation job have already committed.
        logging.getLogger(__name__).exception("Unable to finish season badge check for club %s", club_id)


class CommunityAwardRequest(BaseModel):
    operation_id: UUID
    player_id: int = Field(gt=0)
    badge_id: str
    criteria: list[str] = Field(min_length=1)
    note: str = Field(min_length=1, max_length=1000)
    contribution_date: date


class BadgeSeasonRequest(BaseModel):
    operation_id: UUID
    id: UUID
    name: str = Field(min_length=1, max_length=100)
    start_date: date
    end_date: date
    timezone: str = Field(default="UTC", max_length=100)
    expected_revision: int = Field(ge=0)


def install_admin_badge_management_routes(app, *, get_supabase_client):
    def authorize(club_id, authorization):
        if not is_admin_badge_diagnostics_enabled():
            raise HTTPException(403, "Badge management is not enabled.")
        user = authenticate_bearer(authorization)
        supabase = get_supabase_client()
        role = resolve_admin_role(supabase=supabase, club_id=club_id, email=user.email, user_id=user.user_id, allowlist=set())
        if not role.assigned or role.role not in {"administrator", "super_admin", "club_owner"}:
            raise HTTPException(403, "Only club administrators can manage badge awards and seasons.")
        assignments = supabase.table("admin_role_assignments").select("user_id,role,revoked_at,expires_at").eq("club_id", club_id).eq("email", user.email.strip().lower()).execute().data or []
        now = datetime.now(timezone.utc)
        if not any(row.get("role") == role.role and not row.get("revoked_at")
                   and (not row.get("user_id") or str(row["user_id"]) == str(user.user_id))
                   and (not row.get("expires_at") or datetime.fromisoformat(str(row["expires_at"]).replace("Z", "+00:00")) > now)
                   for row in assignments):
            raise HTTPException(403, "A current club administrator assignment is required.")
        return supabase, user, role.role

    def save(club_id, payload, authorization, action, background_tasks=None):
        supabase, user, role = authorize(club_id, authorization)
        data = payload.model_dump(mode="json")
        operation_id = data.pop("operation_id")
        try:
            result = save_badge_management(supabase, club_id=club_id, actor_email=user.email,
                actor_user_id=user.user_id, actor_role=role, operation_id=operation_id, action=action, payload=data)
            if action == "save_season" and background_tasks is not None:
                background_tasks.add_task(check_saved_season, supabase, club_id)
            return result
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, "Unable to verify the save. Retry the same request.") from exc
        except APIError as exc:
            if exc.code in {"22023", "40001", "23505", "42501"}:
                raise HTTPException(403 if exc.code == "42501" else 409, exc.message) from exc
            raise HTTPException(503, "Unable to verify the save. Retry the same request.") from exc

    @app.get("/admin/clubs/{club_id}/badge-management")
    def options(club_id: str, authorization: str | None = auth_header()):
        supabase, _, _ = authorize(club_id, authorization)
        return badge_management_options(supabase, club_id)

    @app.post("/admin/clubs/{club_id}/badge-management/awards")
    def award(club_id: str, payload: CommunityAwardRequest, authorization: str | None = auth_header()):
        return save(club_id, payload, authorization, "award_community")

    @app.post("/admin/clubs/{club_id}/badge-management/seasons")
    def season(club_id: str, payload: BadgeSeasonRequest, background_tasks: BackgroundTasks, authorization: str | None = auth_header()):
        return save(club_id, payload, authorization, "save_season", background_tasks)
