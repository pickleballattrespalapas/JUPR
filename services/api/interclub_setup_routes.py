"""Saved interclub planning; no roster, score, email or rating mutations."""
from datetime import date, datetime, timedelta
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator
from services.api.auth import authenticate_bearer, auth_header
from jupr_app.domain.admin.roles import resolve_admin_role
from jupr_app.domain.admin.staff_policy import ADMIN_ROLES

class MeetDraft(BaseModel):
    host_club_id: str = Field(min_length=1, max_length=100)
    club_ids: list[str] = Field(min_length=2, max_length=4)
    starts_at: datetime
    duration_minutes: int = Field(default=180, ge=30, le=180)
    courts: int = Field(default=4, ge=1, le=100)

class SeasonDraft(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    start_date: date
    end_date: date
    timezone: str = 'America/Mazatlan'
    divisions: list[Literal['3.5','4.0','4.5/Open']] = Field(min_length=1, max_length=3)
    club_ids: list[str] = Field(default_factory=list, max_length=32)
    meets: list[MeetDraft] = Field(default_factory=list, max_length=100)

    @model_validator(mode='after')
    def consistent(self):
        self.name = self.name.strip()
        if not self.name: raise ValueError('Enter a season name.')
        if self.end_date < self.start_date: raise ValueError('End date must follow start date.')
        try: zone=ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError): raise ValueError('Choose a valid timezone.')
        if len(set(self.club_ids)) != len(self.club_ids) or len(set(self.divisions)) != len(self.divisions):
            raise ValueError('Remove duplicate clubs or divisions.')
        intervals=[]
        for meet in self.meets:
            if len(set(meet.club_ids)) != len(meet.club_ids): raise ValueError('A club can appear only once in a meet.')
            if not set(meet.club_ids).issubset(self.club_ids): raise ValueError('Meet clubs must be selected for this season.')
            if meet.host_club_id not in meet.club_ids: raise ValueError('The host must play in its meet.')
            if meet.starts_at.tzinfo is None: raise ValueError('Meet time must include a timezone offset.')
            start=meet.starts_at; end=start+timedelta(minutes=meet.duration_minutes)
            if not self.start_date <= start.astimezone(zone).date() <= self.end_date or end.astimezone(zone).date()>self.end_date:
                raise ValueError('Meet time must fall within the season dates.')
            for old_start,old_end,old_clubs in intervals:
                if start<old_end and end>old_start and set(meet.club_ids)&old_clubs:
                    raise ValueError('A club cannot attend overlapping meets.')
            intervals.append((start,end,set(meet.club_ids)))
        return self

class SaveDraft(BaseModel):
    season_id: UUID
    expected_revision: int = Field(ge=0)
    draft: SeasonDraft


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
            if code=='40001': raise HTTPException(409,'This season changed in another tab. Reload it before saving.') from exc
            if code=='42501': raise HTTPException(403,'Organizer administrator access required.') from exc
            if code=='22023': raise HTTPException(422,'One of the selected clubs is no longer available.') from exc
            raise HTTPException(503,'Could not confirm the save. Reload the season list before retrying.') from exc
        return {'season':result}
