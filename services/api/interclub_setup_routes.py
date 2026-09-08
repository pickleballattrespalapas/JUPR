"""Saved interclub planning; no roster, score, email or rating mutations."""
from uuid import UUID
from fastapi import HTTPException
from pydantic import BaseModel, Field
from services.api.interclub_models import PlanningDraft, SeasonDraft
from services.api.auth import authenticate_bearer, auth_header
from jupr_app.domain.admin.roles import resolve_admin_role
from jupr_app.domain.admin.staff_policy import ADMIN_ROLES

class SaveDraft(BaseModel):
    season_id: UUID
    expected_revision: int = Field(ge=0)
    draft: PlanningDraft


def install_interclub_setup_routes(app, *, get_supabase_client):
    def authorize(club_id, authorization):
        user=authenticate_bearer(authorization);db=get_supabase_client()
        role=resolve_admin_role(supabase=db,club_id=club_id,email=user.email,user_id=user.user_id,allowlist=set())
        if not role.assigned or role.role not in ADMIN_ROLES: raise HTTPException(403,'Organizer administrator access required.')
        return db,user

    @app.get('/admin/clubs/{club_id}/interclub/setup')
    def setup(club_id: str, authorization: str | None = auth_header()):
        db,_=authorize(club_id,authorization)
        return {'seasons':db.table('pcs_interclub_drafts').select('id,revision,draft,updated_at').eq('organizer_club_id',club_id).order('updated_at',desc=True).limit(100).execute().data or []}

    @app.get('/admin/clubs/{club_id}/interclub/club-choices')
    def choices(club_id: str, offset: int = 0, authorization: str | None = auth_header()):
        db,_=authorize(club_id,authorization)
        if offset<0: raise HTTPException(422,'Invalid page.')
        rows=db.table('clubs').select('id,name,slug').order('id').range(offset,offset+99).execute().data or []
        return {'clubs':rows,'next_offset':offset+100 if len(rows)==100 else None}

    @app.put('/admin/clubs/{club_id}/interclub/setup')
    def save(club_id: str, payload: SaveDraft, authorization: str | None = auth_header()):
        db,user=authorize(club_id,authorization)
        try:
            result=db.rpc('pcs_save_interclub_draft',{'p_actor_id':user.user_id,'p_actor_email':user.email,'p_club_id':club_id,'p_id':str(payload.season_id),'p_revision':payload.expected_revision,'p_draft':payload.draft.model_dump(mode='json')}).execute().data
        except Exception as exc:
            code=getattr(exc,'code','')
            if code=='40001': raise HTTPException(409,'This setup changed or invitations are already open. Reload before continuing.') from exc
            if code=='42501': raise HTTPException(403,'Organizer administrator access required.') from exc
            if code=='22023': raise HTTPException(422,'One of the selected clubs is no longer available.') from exc
            raise HTTPException(503,'Could not confirm the save. Reload the season list before retrying.') from exc
        return {'season':result}
