from types import SimpleNamespace
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from services.api import platform_admin_routes as routes

class Query:
    def __init__(self, rows): self.rows=rows
    def select(self,*a): return self
    def eq(self,key,value): self.rows=[r for r in self.rows if r.get(key)==value]; return self
    def order(self,*a): return self
    def range(self,start,end): self.rows=self.rows[start:end+1]; return self
    def execute(self): return SimpleNamespace(data=self.rows)

@pytest.fixture
def client(monkeypatch):
    db=SimpleNamespace(admins=[], calls=[])
    db.table=lambda name: Query(db.admins if name=='pcs_platform_admins' else [{'id':'tres'}])
    def rpc(name,params):
        db.calls.append((name,params))
        return SimpleNamespace(execute=lambda:SimpleNamespace(data={'id':'new-club'}))
    db.rpc=rpc
    monkeypatch.setattr(routes,'authenticate_bearer',lambda token:SimpleNamespace(user_id='trusted-user',email='owner@example.com'))
    app=FastAPI();routes.install_platform_admin_routes(app,get_supabase_client=lambda:db)
    return TestClient(app),db

@pytest.mark.parametrize('membership',[[],[{'user_id':'someone-else'}],[{'user_id':'trusted-user','revoked_at':'2026-01-01'}]])
def test_platform_access_not_inherited_from_club_roles(client,membership):
    c,db=client;db.admins=membership
    assert c.get('/admin/platform/clubs').status_code==403
    assert c.post('/admin/platform/clubs',json={'slug':'new-club','name':'New Club','administrator_email':'admin@example.com'}).status_code==403
    assert not db.calls

def test_creation_uses_verified_actor_not_payload(client):
    c,db=client;db.admins=[{'user_id':'trusted-user','revoked_at':None}]
    r=c.post('/admin/platform/clubs',json={'slug':'new-club','name':'New Club','administrator_email':'ADMIN@example.com','p_actor_id':'forged'})
    assert r.status_code==200
    assert db.calls[0][1]['p_actor_id']=='trusted-user'
    assert db.calls[0][1]['p_admin_email']=='admin@example.com'

def test_status_does_not_allow_launch_or_billing(client):
    c,db=client;db.admins=[{'user_id':'trusted-user'}]
    assert c.patch('/admin/platform/clubs/tres/onboarding',json={'status':'active'}).status_code==422
    assert c.patch('/admin/platform/clubs/tres/onboarding',json={'status':'ready_for_review'}).status_code==200
    assert c.get('/admin/platform/clubs?offset=-1').status_code==422

@pytest.mark.parametrize('path,method,payload',[
    ('details','get',None),
    ('profile','patch',{'name':'Tres','support_email':'staff@example.com'}),
])
def test_profile_and_details_require_platform_access(client,path,method,payload):
    c,db=client
    kwargs={'json':payload} if payload else {}
    assert getattr(c,method)(f'/admin/platform/clubs/tres/{path}',**kwargs).status_code==403
    assert not db.calls

def test_profile_passes_verified_actor_and_route_club(client):
    c,db=client;db.admins=[{'user_id':'trusted-user'}]
    assert c.patch('/admin/platform/clubs/tres/profile',json={
        'name':'Tres','support_email':'staff@example.com','p_actor_id':'forged','p_club_id':'other'
    }).status_code==200
    assert db.calls==[('pcs_update_club_profile',{
        'p_actor_id':'trusted-user','p_club_id':'tres','p_name':'Tres','p_support_email':'staff@example.com'
    })]
