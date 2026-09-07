from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api import admin_badge_management_routes as routes


class Database:
    def __init__(self, role='administrator', **overrides):
        self.assignment=dict(user_id=None,role=role,revoked_at=None,expires_at=None) | overrides
    def table(self, _): return self
    def select(self, _): return self
    def eq(self, *_): return self
    def execute(self): return SimpleNamespace(data=[self.assignment])


def client(monkeypatch, role='administrator', assigned=True, **overrides):
    monkeypatch.setenv('JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS','1')
    db=Database(role,**overrides)
    monkeypatch.setattr(routes,'authenticate_bearer',lambda _: SimpleNamespace(email='admin@example.com',user_id=str(uuid4())))
    monkeypatch.setattr(routes,'resolve_admin_role',lambda **_: SimpleNamespace(role=role,assigned=assigned))
    app=FastAPI();routes.install_admin_badge_management_routes(app,get_supabase_client=lambda: db)
    return TestClient(app)


def payload():
    return dict(operation_id=str(uuid4()),player_id=1,badge_id='good_sport',criteria=['honest_calls'],note='Honest line call.',contribution_date='2026-01-01')


@pytest.mark.parametrize('role',['operator','organizer','scorekeeper','read_only'])
def test_operator_roles_cannot_issue_or_read_admin_recognition(monkeypatch,role):
    c=client(monkeypatch,role)
    assert c.get('/admin/clubs/club/badge-management').status_code==403
    assert c.post('/admin/clubs/club/badge-management/awards',json=payload()).status_code==403


def test_unassigned_revoked_and_expired_admins_are_denied(monkeypatch):
    assert client(monkeypatch,assigned=False).get('/admin/clubs/club/badge-management').status_code==403
    for override in [{'revoked_at':'2026-01-01'},{'expires_at':'2026-01-01T00:00:00Z'}]:
        c=client(monkeypatch,**override)
        assert c.post('/admin/clubs/club/badge-management/awards',json=payload()).status_code==403


def test_admin_request_preserves_stable_retry_identity(monkeypatch):
    c=client(monkeypatch)
    seen=[]
    def save(_supabase,**kwargs):
        seen.append(kwargs)
        return {'ok':True}
    monkeypatch.setattr(routes,'save_badge_management',save)
    p=payload()
    for _ in range(2):
        assert c.post('/admin/clubs/club/badge-management/awards',json=p).status_code==200
    assert seen[0]['operation_id']==seen[1]['operation_id']==p['operation_id']
    assert seen[0]['payload']['criteria']==['honest_calls']
    assert seen[0]['actor_role']=='administrator'


def test_new_season_request_carries_admin_dates_and_revision(monkeypatch):
    c=client(monkeypatch)
    checks=[]
    monkeypatch.setattr(routes, 'check_saved_season', lambda _db, club: checks.append(club))
    def save(_supabase,**kwargs):
        assert kwargs['action']=='save_season'
        assert kwargs['payload']['start_date']=='2026-09-15'
        assert kwargs['payload']['end_date']=='2027-09-14'
        assert kwargs['payload']['expected_revision']==0
        return {'ok':True}
    monkeypatch.setattr(routes,'save_badge_management',save)
    response=c.post('/admin/clubs/club/badge-management/seasons',json=dict(operation_id=str(uuid4()),id=str(uuid4()),name='Club season',start_date='2026-09-15',end_date='2027-09-14',timezone='America/Mazatlan',expected_revision=0))
    assert response.status_code==200
    assert checks==['club']
