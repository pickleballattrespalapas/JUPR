"""Invite new clubs directly from an organizer's saved interclub setup."""
import re
import unicodedata
from typing import Literal
from uuid import UUID

from fastapi import HTTPException
from pydantic import Field

from jupr_app.domain.admin.staff_policy import ADMIN_ROLES
from services.api.admin_auth_routes import require_admin_assignments
from services.api.auth import authenticate_bearer, auth_header
from services.api.interclub_models import PlanningDraft
from services.api.staff_invitation_routes import (
    InvitedEmail, InvitationSignIn, invitation_response, send_invitation_sign_in,
    invitation_sign_in_options, get_email_mode, EMAIL_MODE_LIVE, EMAIL_DISABLED_MESSAGE,
)

FIELDS = "id,organizer_club_id,season_id,club_id,club_name,email,status,revision,expires_at,created_at,accepted_at"


class ClubInviteCreate(InvitedEmail):
    invitation_id: UUID
    expected_revision: int = Field(ge=1)
    name: str = Field(min_length=1, max_length=120)
    draft: PlanningDraft


class ClubInviteUpdate(InvitedEmail):
    expected_revision: int = Field(ge=1)
    action: Literal["cancel", "renew"]


def response(row):
    # Share the recipient's existing sign-in/review flow without exposing grant
    # snapshots, inviter identity, email throttles or commercial settings.
    result = invitation_response({**row, "role": "administrator", "scopes": []})
    result.update({key: row.get(key) for key in FIELDS.split(",") if key != "status"})
    return result


def rpc(db, name="pcs_interclub_club_invitation", **params):
    try:
        return db.rpc(name, params).execute().data
    except Exception as exc:
        code = getattr(exc, "code", "")
        if code == "23505": raise HTTPException(409, "A club with this name or address already exists. Refresh the list and select that club.") from exc
        if code == "42501": raise HTTPException(403, "Use an authorized organizer account or the invited, verified email.") from exc
        if code == "40001": raise HTTPException(409, "The setup, invitation or club access changed. Reload before continuing.") from exc
        if code == "P0002": raise HTTPException(404, "Club invitation or season unavailable.") from exc
        if code == "22023": raise HTTPException(422, "Check the club name, administrator email and season's club limit.") from exc
        raise HTTPException(503, "Could not confirm the invitation. Reload the setup before retrying.") from exc


def install_club_join_invitation_routes(app, *, get_supabase_client):
    def organizer(club_id, season_id, authorization):
        db = get_supabase_client()
        user, assignments = require_admin_assignments(get_supabase_client=lambda: db, authorization=authorization, requested_club_id=club_id)
        if not any(row["role"] in ADMIN_ROLES for row in assignments):
            raise HTTPException(403, "Organizer administrator access required.")
        rows = db.table("pcs_interclub_drafts").select("id,draft").eq("id", str(season_id)).eq("organizer_club_id", club_id).limit(1).execute().data or []
        if not rows: raise HTTPException(404, "Season unavailable for this organizer.")
        return db, user

    @app.get("/admin/clubs/{club_id}/interclub/setup/{season_id}/club-invitations")
    def invitations(club_id: str, season_id: UUID, authorization: str | None = auth_header()):
        db, _ = organizer(club_id, season_id, authorization)
        rows = db.table("pcs_club_join_invitations").select(FIELDS).eq("organizer_club_id", club_id).eq("season_id", str(season_id)).order("created_at").limit(32).execute().data or []
        clubs = db.table("clubs").select("id,name,slug").in_("id", [row["club_id"] for row in rows]).execute().data or [] if rows else []
        return {"invitations": [response(row) for row in rows], "clubs": clubs}

    @app.post("/admin/clubs/{club_id}/interclub/setup/{season_id}/club-invitations")
    def create(club_id: str, season_id: UUID, payload: ClubInviteCreate, authorization: str | None = auth_header()):
        db, user = organizer(club_id, season_id, authorization)
        normalized = unicodedata.normalize("NFKD", payload.name).encode("ascii", "ignore").decode().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")[:60].rstrip("-")
        if len(slug) < 3: raise HTTPException(422, "Use a club name with at least three letters or numbers.")
        result = rpc(db, "pcs_create_interclub_club_invitation", p_actor_id=user.user_id, p_actor_email=user.email,
                     p_club_id=club_id, p_season_id=str(season_id), p_id=str(payload.invitation_id),
                     p_revision=payload.expected_revision, p_draft=payload.draft.model_dump(mode="json"),
                     p_name=payload.name, p_slug=slug, p_email=payload.email)
        return {"invitation": response(result["invitation"]), "club": result["club"],
                "season": {key: result["season"].get(key) for key in ("id", "revision", "draft", "updated_at")}}

    @app.post("/admin/clubs/{club_id}/interclub/setup/{season_id}/club-invitations/{invitation_id}")
    def update(club_id: str, season_id: UUID, invitation_id: UUID, payload: ClubInviteUpdate, authorization: str | None = auth_header()):
        db, user = organizer(club_id, season_id, authorization)
        row = rpc(db, p_action=payload.action, p_id=str(invitation_id), p_club_id=club_id, p_season_id=str(season_id),
                  p_actor_id=user.user_id, p_actor_email=user.email, p_revision=payload.expected_revision, p_email=payload.email)
        return {"invitation": response(row)}

    @app.get("/club-invitations/{invitation_id}")
    def review(invitation_id: UUID, authorization: str | None = auth_header()):
        user = authenticate_bearer(authorization); db = get_supabase_client()
        rows = db.table("pcs_club_join_invitations").select(FIELDS).eq("id", str(invitation_id)).eq("email", user.email).limit(1).execute().data or []
        if not rows: raise HTTPException(404, "Invitation unavailable for this account. Sign in with the invited email.")
        row = rows[0]
        clubs = db.table("clubs").select("id,name,slug").in_("id", [row["club_id"], row["organizer_club_id"]]).execute().data or []
        target = next((club for club in clubs if club["id"] == row["club_id"]), None)
        if not target: raise HTTPException(404, "Club unavailable.")
        return {"invitation": response(row), "club": target,
                "organizer": next((club for club in clubs if club["id"] == row["organizer_club_id"]), None)}

    @app.post("/club-invitations/{invitation_id}/accept")
    def accept(invitation_id: UUID, authorization: str | None = auth_header()):
        user = authenticate_bearer(authorization)
        row = rpc(get_supabase_client(), p_action="accept", p_id=str(invitation_id), p_actor_id=user.user_id, p_actor_email=user.email)
        return {"invitation": response(row)}

    @app.get("/club-invitations/{invitation_id}/sign-in")
    def sign_in_options(invitation_id: UUID):
        return invitation_sign_in_options(get_email_mode())

    @app.post("/club-invitations/{invitation_id}/sign-in")
    def sign_in(invitation_id: UUID, payload: InvitationSignIn):
        if get_email_mode() != EMAIL_MODE_LIVE:
            return {"email_enabled": False, "message": EMAIL_DISABLED_MESSAGE}
        db = get_supabase_client()
        try:
            row = rpc(db, p_action="email_claim", p_id=str(invitation_id), p_email=payload.email)
            if row: send_invitation_sign_in(db, row, club_join=True, setup_password=payload.setup_password)
        except HTTPException as exc:
            if exc.status_code not in (403, 404, 409): raise
        except Exception as exc:
            raise HTTPException(503, "Unable to send a sign-in email. Wait a minute before retrying.") from exc
        return {"email_enabled": True, "message": "If the email matches an available invitation, a sign-in link will arrive shortly. Wait a minute before requesting another; each invitation allows up to five emails."}
