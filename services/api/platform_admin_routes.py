"""Platform onboarding, independently authorized from club memberships."""
from typing import Literal
from fastapi import HTTPException
from pydantic import BaseModel, Field
from services.api.auth import authenticate_bearer, auth_header

class ClubCreate(BaseModel):
    slug: str = Field(min_length=3, max_length=60, pattern=r'^[a-z0-9]+(-[a-z0-9]+)*$')
    name: str = Field(min_length=1, max_length=120)
    administrator_email: str = Field(max_length=254, pattern=r'^[^\s@]+@[^\s@]+\.[^\s@]+$')

class OnboardingUpdate(BaseModel):
    status: Literal['draft', 'in_progress', 'ready_for_review']


def require_platform_admin(db, user):
    rows = db.table('pcs_platform_admins').select('user_id,revoked_at').eq('user_id', user.user_id).execute().data or []
    if not any(row.get('user_id') == user.user_id and not row.get('revoked_at') for row in rows):
        raise HTTPException(403, 'PCS Super Admin access required.')


def install_platform_admin_routes(app, *, get_supabase_client):
    def authorize(authorization):
        user = authenticate_bearer(authorization)
        db = get_supabase_client()
        require_platform_admin(db, user)
        return db, user

    def rpc(db, name, params):
        try:
            return db.rpc(name, params).execute().data
        except Exception as exc:
            code = getattr(exc, 'code', '')
            if code == '23505': raise HTTPException(409, 'That club address is already in use.') from exc
            if code == '42501': raise HTTPException(403, 'PCS Super Admin access required.') from exc
            if code == '22023': raise HTTPException(422, 'Check the club details and administrator assignment.') from exc
            if code == 'P0002': raise HTTPException(404, 'Club not found.') from exc
            raise HTTPException(503, 'Could not save. Refresh the club list before retrying.') from exc

    @app.get('/admin/platform/clubs')
    def clubs(authorization: str | None = auth_header(), offset: int = 0):
        db, _ = authorize(authorization)
        if offset < 0: raise HTTPException(422, 'Invalid page.')
        rows = db.table('clubs').select('id,slug,name,is_active,plan_status,onboarding_status').order('id').range(offset, offset+49).execute().data or []
        return {'clubs': rows, 'next_offset': offset+50 if len(rows)==50 else None}

    @app.post('/admin/platform/clubs')
    def create(payload: ClubCreate, authorization: str | None = auth_header()):
        db, user = authorize(authorization)
        return {'club': rpc(db, 'pcs_onboard_club', {'p_actor_id': user.user_id, 'p_slug': payload.slug, 'p_name': payload.name, 'p_admin_email': payload.administrator_email.lower()})}

    @app.patch('/admin/platform/clubs/{club_id}/onboarding')
    def update(club_id: str, payload: OnboardingUpdate, authorization: str | None = auth_header()):
        db, user = authorize(authorization)
        return {'club': rpc(db, 'pcs_review_onboarding', {'p_actor_id': user.user_id, 'p_club_id': club_id, 'p_status': payload.status})}
