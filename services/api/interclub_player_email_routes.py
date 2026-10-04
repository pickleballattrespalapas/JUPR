"""Club administrator preview and delivery of interclub player invitations."""
from typing import Literal
from uuid import UUID

from fastapi import HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field

from services.api.auth import auth_header
from jupr_app.services.staging_write_guard import require_staging_communications_mutations
from jupr_app.services import interclub_player_email_service as emails
from jupr_app.services.interclub_registration_phase import RegistrationPhaseError


def _require_communications_mutations():
    try:
        require_staging_communications_mutations()
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc


class EmailPreview(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: Literal["season", "meet"]
    meet_id: UUID | None = None
    recipient_ids: list[str] = Field(min_length=1, max_length=emails.MAX_RECIPIENTS)
    subject: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=10000)


class EmailCreate(EmailPreview):
    operation_key: UUID
    preview_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


def install_interclub_player_email_routes(app, *, get_supabase_client):
    from services.api.interclub_player_pool_routes import pool_admin_context

    base = "/admin/clubs/{club_id}/interclub/player-pools/{season_id}/emails"

    def context(club_id, season_id, authorization, response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        return pool_admin_context(get_supabase_client, authorization, club_id, str(season_id))

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except emails.EmailNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except RegistrationPhaseError as exc:
            raise HTTPException(423, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get(base + "/audience")
    def audience(club_id: str, season_id: UUID, response: Response,
                 kind: Literal["season", "meet"] = Query(...), meet_id: UUID | None = Query(None),
                 authorization: str | None = auth_header()):
        db, _, _, _ = context(club_id, season_id, authorization, response)
        result = call(emails.build_audience, db, club_id=club_id, season_id=str(season_id), kind=kind,
            meet_id=str(meet_id) if meet_id else None)
        return {key: value for key, value in result.items() if not key.startswith("_")}

    @app.post("/admin/clubs/{club_id}/interclub/player-pools/{season_id}/emails/preview")
    def preview(club_id: str, season_id: UUID, payload: EmailPreview, response: Response,
                authorization: str | None = auth_header()):
        db, _, _, _ = context(club_id, season_id, authorization, response)
        _require_communications_mutations()
        return call(emails.preview_email, db, club_id=club_id, season_id=str(season_id), **payload.model_dump(mode="json"))

    @app.post("/admin/clubs/{club_id}/interclub/player-pools/{season_id}/emails")
    def create(club_id: str, season_id: UUID, payload: EmailCreate, response: Response,
               authorization: str | None = auth_header()):
        db, user, _, _ = context(club_id, season_id, authorization, response)
        _require_communications_mutations()
        return call(emails.create_email, db, club_id=club_id, season_id=str(season_id), user=user, **payload.model_dump(mode="json"))

    @app.get(base + "/{operation_key}")
    def batch(club_id: str, season_id: UUID, operation_key: UUID, response: Response,
              authorization: str | None = auth_header()):
        db, _, _, _ = context(club_id, season_id, authorization, response)
        return call(emails.get_email, db, club_id=club_id, season_id=str(season_id), operation_key=str(operation_key))

    @app.post("/admin/clubs/{club_id}/interclub/player-pools/{season_id}/emails/{operation_key}/recipients/{recipient_index}/send")
    def send(club_id: str, season_id: UUID, operation_key: UUID, recipient_index: int, response: Response,
             authorization: str | None = auth_header()):
        db, user, _, _ = context(club_id, season_id, authorization, response)
        _require_communications_mutations()
        return call(emails.send_recipient, db, club_id=club_id, season_id=str(season_id), user=user,
            operation_key=str(operation_key), recipient_index=recipient_index)
