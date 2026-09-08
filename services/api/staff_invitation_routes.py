"""Email-bound staff invitations. No assignment exists until verified acceptance."""
from datetime import datetime, timezone
from html import escape
from typing import Any, Literal
from urllib.parse import urlencode, urlsplit
from uuid import UUID

from fastapi import HTTPException
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from jupr_app.config import EMAIL_MODE_LIVE, get_email_mode, get_next_web_base_url
from jupr_app.domain.admin.staff_policy import ADMIN_ROLES, validate_scopes
from jupr_app.domain.notifications.smtp_mailer import send_email_with_inline_chart
from services.api.admin_auth_routes import require_admin_assignments
from services.api.auth import authenticate_bearer, auth_header


INVITATION_FIELDS = "id,club_id,email,role,scopes,access_expires_at,expires_at,status,created_at,accepted_at,cancelled_at"


class InvitedEmail(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    email: str = Field(min_length=3, max_length=254, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

    @field_validator("email")
    @classmethod
    def normalized(cls, value: str) -> str:
        return value.lower()


class StaffInvitationCreate(InvitedEmail):
    invitation_id: UUID
    role: Literal["administrator", "operator"]
    scopes: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    access_expires_at: AwareDatetime | None = None


def invitation_response(row):
    result = {key: row.get(key) for key in INVITATION_FIELDS.split(",")}
    if result["status"] == "pending" and any(
        value and datetime.fromisoformat(str(value).replace("Z", "+00:00")) <= datetime.now(timezone.utc)
        for value in (row.get("expires_at"), row.get("access_expires_at"))
    ):
        result["status"] = "expired"
    return result


def invitation_rpc(db, **params):
    try:
        return db.rpc("pcs_staff_invitation", params).execute().data
    except Exception as exc:
        code = getattr(exc, "code", "")
        if code == "42501":
            raise HTTPException(403, "Use an authorized account with the invited, verified email.") from exc
        if code == "40001":
            raise HTTPException(409, "The invitation or staff access changed. Refresh the list or ask a club administrator for a new invitation.") from exc
        if code == "P0002":
            raise HTTPException(404, "Invitation unavailable.") from exc
        if code == "22023":
            raise HTTPException(422, "Check the invitation email, role, scopes and expiration.") from exc
        raise HTTPException(503, "Could not confirm the invitation update. Reload before retrying.") from exc


def send_invitation_sign_in(db, row):
    # Authentication links must never be redirected to a staging mailbox. Stop
    # before creating an Auth user/token in every non-live email mode.
    if get_email_mode() != EMAIL_MODE_LIVE:
        return False
    origin = get_next_web_base_url(default="")
    parsed = urlsplit(origin)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("A trusted HTTPS web origin is required")
    generated = db.auth.admin.generate_link({"type": "magiclink", "email": row["email"]})
    token_hash = generated.properties.hashed_token
    if not token_hash:
        raise ValueError("No authentication token generated")
    # The fragment stays out of request/referrer logs. Only the recipient gets
    # this credential; the staff list/API response contains just invitation IDs.
    link = f"{origin.rstrip('/')}/admin/accept-invitation?invitation={row['id']}#" + urlencode({"staff_token_hash": token_hash})
    send_email_with_inline_chart(
        to_email=row["email"], subject="Sign in to review your club staff invitation",
        html_body=f'<p>You requested a sign-in link to review a club staff invitation.</p><p><a href="{escape(link, quote=True)}">Sign in and review invitation</a></p><p>You will review the club and access before accepting. If you did not request this, ignore this email.</p>',
        text_body=f"Sign in to review your club staff invitation:\n{link}\nYou will review the club and access before accepting. If you did not request this, ignore this email.",
    )
    return True


def install_staff_invitation_routes(app, *, get_supabase_client):
    def administrator(club_id, authorization):
        db = get_supabase_client()
        user, assignments = require_admin_assignments(get_supabase_client=lambda: db, authorization=authorization, requested_club_id=club_id)
        if not any(row["role"] in ADMIN_ROLES for row in assignments):
            raise HTTPException(403, "Club administrator access required.")
        return db, user

    @app.get("/admin/clubs/{club_id}/staff/invitations")
    def invitations(club_id: str, authorization: str | None = auth_header()):
        db, _ = administrator(club_id, authorization)
        rows = db.table("club_staff_invitations").select(INVITATION_FIELDS).eq("club_id", club_id).order("created_at", desc=True).limit(500).execute().data or []
        return {"invitations": [invitation_response(row) for row in rows]}

    @app.post("/admin/clubs/{club_id}/staff/invitations")
    def create(club_id: str, payload: StaffInvitationCreate, authorization: str | None = auth_header()):
        db, user = administrator(club_id, authorization)
        try:
            scopes = validate_scopes(payload.scopes) if payload.role == "operator" else []
            if payload.access_expires_at and (payload.role == "administrator" or payload.access_expires_at <= datetime.now(timezone.utc)):
                raise ValueError("Only operators can have a future access expiration.")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        row = invitation_rpc(db, p_action="create", p_id=str(payload.invitation_id), p_club_id=club_id,
                             p_actor_id=user.user_id, p_actor_email=user.email, p_email=payload.email,
                             p_role=payload.role, p_scopes=scopes,
                             p_access_expires_at=payload.access_expires_at.isoformat() if payload.access_expires_at else None)
        return {"invitation": invitation_response(row)}

    @app.post("/admin/clubs/{club_id}/staff/invitations/{invitation_id}/cancel")
    def cancel(club_id: str, invitation_id: UUID, authorization: str | None = auth_header()):
        db, user = administrator(club_id, authorization)
        row = invitation_rpc(db, p_action="cancel", p_id=str(invitation_id), p_club_id=club_id, p_actor_id=user.user_id, p_actor_email=user.email)
        return {"invitation": invitation_response(row)}

    @app.get("/staff-invitations/{invitation_id}")
    def review(invitation_id: UUID, authorization: str | None = auth_header()):
        user = authenticate_bearer(authorization)
        db = get_supabase_client()
        rows = db.table("club_staff_invitations").select(INVITATION_FIELDS).eq("id", str(invitation_id)).eq("email", user.email).limit(1).execute().data or []
        if not rows:
            raise HTTPException(404, "Invitation unavailable for this account. Sign in with the invited email.")
        row = rows[0]
        clubs = db.table("clubs").select("id,slug,name").eq("id", row["club_id"]).limit(1).execute().data or []
        if not clubs:
            raise HTTPException(404, "Club unavailable.")
        return {"invitation": invitation_response(row), "club": clubs[0]}

    @app.post("/staff-invitations/{invitation_id}/accept")
    def accept(invitation_id: UUID, authorization: str | None = auth_header()):
        user = authenticate_bearer(authorization)
        row = invitation_rpc(get_supabase_client(), p_action="accept", p_id=str(invitation_id), p_actor_id=user.user_id, p_actor_email=user.email)
        return {"invitation": invitation_response(row)}

    @app.post("/staff-invitations/{invitation_id}/sign-in")
    def sign_in(invitation_id: UUID, payload: InvitedEmail):
        if get_email_mode() != EMAIL_MODE_LIVE:
            return {"email_enabled": False, "message": "Sign-in email is disabled in this test environment. Use an existing test account."}
        db = get_supabase_client()
        try:
            row = invitation_rpc(db, p_action="email_claim", p_id=str(invitation_id), p_email=payload.email)
            if row:
                send_invitation_sign_in(db, row)
        except HTTPException as exc:
            if exc.status_code not in (403, 404, 409):
                raise
        except Exception as exc:
            raise HTTPException(503, "Unable to send a sign-in email. Wait a minute before retrying.") from exc
        return {"email_enabled": True, "message": "If the email matches an available invitation, a sign-in link will arrive shortly. Wait a minute before requesting another; each invitation allows up to five emails."}
