from copy import deepcopy
from types import SimpleNamespace
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from services.api import interclub_setup_routes as routes

def draft():
 return dict(name='Southern BCS',start_date='2027-01-01',end_date='2027-03-31',divisions=['3.5','4.0'],club_ids=['a','b','c','d'],meets=[dict(host_club_id='a',club_ids=['a','b'],starts_at='2027-01-10T09:00:00-07:00',courts=4)])

@pytest.mark.parametrize('case',['host','unknown_club','duplicate','overlap','naive_time','outside','duration','blank_name','timezone'])
def test_invalid_season_is_rejected(case):
 d=draft()
 if case=='host':d['meets'][0]['host_club_id']='c'
 if case=='unknown_club':d['meets'][0]['club_ids']=['a','unknown']
 if case=='duplicate':d['club_ids'].append('a')
 if case=='overlap':d['meets'].append(deepcopy(d['meets'][0]))
 if case=='naive_time':d['meets'][0]['starts_at']='2027-01-10T09:00:00'
 if case=='outside':d['meets'][0]['starts_at']='2027-04-01T09:00:00-07:00'
 if case=='duration':d['meets'][0]['duration_minutes']=181
 if case=='blank_name':d['name']='  '
 if case=='timezone':d['timezone']='not-a-zone'
 with pytest.raises(ValidationError):routes.SeasonDraft(**d)

def test_simultaneous_meets_allowed_for_different_clubs():
 d=draft();d['meets'].append(dict(host_club_id='c',club_ids=['c','d'],starts_at='2027-01-10T09:00:00-07:00',courts=4))
 assert len(routes.SeasonDraft(**d).meets)==2

@pytest.mark.parametrize('role',['operator','read_only','__unassigned__'])
def test_non_admin_cannot_save_draft(monkeypatch,role):
 monkeypatch.setattr(routes,'authenticate_bearer',lambda _:SimpleNamespace(user_id='verified',email='test@example.com'))
 monkeypatch.setattr(routes,'resolve_admin_role',lambda **_:SimpleNamespace(role=role,assigned=role!='__unassigned__'))
 app=FastAPI();routes.install_interclub_setup_routes(app,get_supabase_client=lambda:None)
 c=TestClient(app)
 assert c.put('/admin/clubs/a/interclub/setup',json={'season_id':'00000000-0000-0000-0000-000000000001','expected_revision':0,'draft':draft()}).status_code==403

def test_rpc_uses_route_club_and_verified_actor(monkeypatch):
 calls=[]
 monkeypatch.setattr(routes,'authenticate_bearer',lambda _:SimpleNamespace(user_id='verified',email='test@example.com'))
 monkeypatch.setattr(routes,'resolve_admin_role',lambda **_:SimpleNamespace(role='administrator',assigned=True))
 db=SimpleNamespace(rpc=lambda name,args:(calls.append(args) or SimpleNamespace(execute=lambda:SimpleNamespace(data={'revision':1}))))
 app=FastAPI();routes.install_interclub_setup_routes(app,get_supabase_client=lambda:db)
 r=TestClient(app).put('/admin/clubs/real-club/interclub/setup',json={'season_id':'00000000-0000-0000-0000-000000000001','expected_revision':0,'draft':draft(),'p_club_id':'forged','p_actor_id':'forged'})
 assert r.status_code==200
 assert calls[0]['p_club_id']=='real-club' and calls[0]['p_actor_id']=='verified'
