"""Club administrator profile editing and setup submission."""
from fastapi import HTTPException
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from jupr_app.domain.admin.staff_policy import ADMIN_ROLES
from services.api.admin_auth_routes import require_admin_assignments
from services.api.auth import auth_header


CLUB_FIELDS = "id,slug,name,tagline,support_email,is_active,onboarding_status,updated_at"
SETUP_STATES = {"draft", "in_progress", "ready_for_review"}


class ClubSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    expected_updated_at: AwareDatetime
    name: str = Field(min_length=1, max_length=120)
    tagline: str = Field(default="", max_length=240)
    support_email: str = Field(default="", max_length=254)
    submit_for_review: bool = False

    @field_validator("support_email")
    @classmethod
    def contact_email(cls, value: str) -> str:
        import re
        if value and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("Enter a valid contact email.")
        return value.lower()


def settings_response(club):
    # Never return billing settings, feature flags, creator identity or staff rows.
    public = {key: club.get(key) for key in CLUB_FIELDS.split(",")}
    missing = [] if str(club.get("support_email") or "").strip() else ["Contact email"]
    return {"club": public, "setup": {
        "missing": missing,
        "can_submit": not club.get("is_active") and club.get("onboarding_status") in SETUP_STATES,
    }}


def install_club_settings_routes(app, *, get_supabase_client):
    def administrator(club_id, authorization):
        db = get_supabase_client()
        user, assignments = require_admin_assignments(
            get_supabase_client=lambda: db, authorization=authorization,
            requested_club_id=club_id,
        )
        if not any(row["role"] in ADMIN_ROLES for row in assignments):
            raise HTTPException(403, "Club administrator access required.")
        return db, user

    @app.get("/admin/clubs/{club_id}/settings")
    def settings(club_id: str, authorization: str | None = auth_header()):
        db, _ = administrator(club_id, authorization)
        try:
            rows = db.table("clubs").select(CLUB_FIELDS).eq("id", club_id).limit(1).execute().data or []
        except Exception as exc:
            raise HTTPException(503, "Club settings are temporarily unavailable.") from exc
        if not rows:
            raise HTTPException(404, "Club not found.")
        return settings_response(rows[0])

    @app.put("/admin/clubs/{club_id}/settings")
    def save_settings(club_id: str, payload: ClubSettingsUpdate, authorization: str | None = auth_header()):
        db, user = administrator(club_id, authorization)
        if payload.submit_for_review and not payload.support_email:
            raise HTTPException(422, "Add a contact email before submitting your club.")
        try:
            saved = db.rpc("pcs_save_club_settings", {
                "p_actor_id": user.user_id, "p_actor_email": user.email,
                "p_club_id": club_id,
                "p_expected_updated_at": payload.expected_updated_at.isoformat(),
                "p_name": payload.name, "p_tagline": payload.tagline,
                "p_support_email": payload.support_email,
                "p_submit": payload.submit_for_review,
            }).execute().data
        except Exception as exc:
            code = getattr(exc, "code", "")
            if code == "40001":
                raise HTTPException(409, "Club settings changed since you opened this page. Reload before saving.") from exc
            if code == "42501":
                raise HTTPException(403, "Club administrator access required.") from exc
            if code == "22023":
                raise HTTPException(422, "Check the club details and setup status before submitting.") from exc
            if code == "P0002":
                raise HTTPException(404, "Club not found.") from exc
            raise HTTPException(503, "Could not confirm the save. Reload club settings before retrying.") from exc
        return settings_response(saved)
