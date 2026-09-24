"""Club administrator controls for published leaderboard cards and seasons."""
from __future__ import annotations

from fastapi import HTTPException
from pydantic import Field

from jupr_app.domain.admin.roles import resolve_admin_role
from jupr_app.domain.admin.staff_policy import ADMIN_ROLES
from services.api.auth import authenticate_bearer, auth_header
from services.api.club_site_models import LeaderboardSettings, StrictModel


class SettingsAction(StrictModel):
    revision: int = Field(ge=0)


class SettingsSave(SettingsAction):
    settings: LeaderboardSettings


def install_club_leaderboard_settings_routes(app, *, get_supabase_client):
    def administrator(club_id, authorization):
        user = authenticate_bearer(authorization)
        db = get_supabase_client()
        role = resolve_admin_role(supabase=db, club_id=club_id, email=user.email,
                                  user_id=user.user_id, allowlist=set())
        if not role.assigned or role.role not in ADMIN_ROLES:
            raise HTTPException(403, "Club administrator access required.")
        return db, user

    def response(row):
        row = row or {}
        return {"revision": row.get("revision", 0),
                "draft": LeaderboardSettings.model_validate(row.get("draft") or {}).model_dump(mode="json"),
                "published": row.get("published"), "published_at": row.get("published_at")}

    def change(club_id, payload, authorization, action):
        db, user = administrator(club_id, authorization)
        try:
            result = db.rpc("save_club_leaderboard_settings", {
                "p_club_id": club_id, "p_actor_id": user.user_id, "p_actor_email": user.email,
                "p_expected_revision": payload.revision, "p_action": action,
                "p_settings": payload.settings.model_dump(mode="json") if action == "save" else None,
            }).execute().data
        except Exception as exc:
            code = str(getattr(exc, "code", ""))
            if code == "40001":
                raise HTTPException(409, "Settings changed. Reload before saving again.") from exc
            if code == "42501":
                raise HTTPException(403, "Club administrator access required.") from exc
            raise HTTPException(503, "The save could not be confirmed. Reload before retrying.") from exc
        if not isinstance(result, dict) or "revision" not in result:
            raise HTTPException(503, "The save could not be confirmed. Reload before retrying.")
        return response(result)

    @app.get("/admin/clubs/{club_id}/leaderboard-settings")
    def get_settings(club_id: str, authorization: str | None = auth_header()):
        db, _ = administrator(club_id, authorization)
        rows = db.table("club_leaderboard_settings").select("revision,draft,published,published_at").eq("club_id", club_id).execute().data or []
        return response(rows[0] if rows else None)

    @app.put("/admin/clubs/{club_id}/leaderboard-settings")
    def save_settings(club_id: str, payload: SettingsSave, authorization: str | None = auth_header()):
        return change(club_id, payload, authorization, "save")

    @app.post("/admin/clubs/{club_id}/leaderboard-settings/publish")
    def publish_settings(club_id: str, payload: SettingsAction, authorization: str | None = auth_header()):
        return change(club_id, payload, authorization, "publish")

    @app.post("/admin/clubs/{club_id}/leaderboard-settings/discard")
    def discard_settings(club_id: str, payload: SettingsAction, authorization: str | None = auth_header()):
        return change(club_id, payload, authorization, "discard")
