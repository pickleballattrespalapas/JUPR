from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from services.api import admin_auth_routes, club_join_invitation_routes as routes, staff_invitation_routes as staff
from tests.test_api_contract_club_settings import Query


class InvitationQuery(Query):
    def order(self, *_args, **_kwargs): return self
    def in_(self, key, values):
        self.rows = [r for r in self.rows if r.get(key) in values]
        return self


@pytest.fixture
def setup(monkeypatch):
    user = SimpleNamespace(user_id=str(uuid4()), email="admin@example.test")
    season_id, invitation_id = str(uuid4()), str(uuid4())
    assignment = dict(email=user.email, user_id=user.user_id, club_id="alpha", role="administrator")
    draft = dict(name="Southern BCS", club_ids=["alpha"], setup_step=1)
    season = dict(id=season_id, organizer_club_id="alpha", revision=2, draft=draft)
    invite = dict(id=invitation_id, club_id="la-ribera", club_name="La Ribera", organizer_club_id="alpha", season_id=season_id,
                  email="invited@example.test", status="pending", revision=1, expires_at="2099-01-01T00:00:00Z", target_after={"private":True})
    club = dict(id="la-ribera", slug="la-ribera", name="La Ribera")
    state = dict(user=user, invite=invite, assignment=assignment, season=season, calls=[], error=None, sent=[])
    monkeypatch.setattr(admin_auth_routes, "authenticate_bearer", lambda _: user)
    monkeypatch.setattr(routes, "authenticate_bearer", lambda _: user)
    monkeypatch.setattr(routes, "get_email_mode", lambda: "dry_run")
    monkeypatch.setattr(routes, "send_invitation_sign_in", lambda *a, **kw: state["sent"].append((a,kw)))
    def table(name):
        return InvitationQuery({"admin_role_assignments":[assignment], "pcs_interclub_drafts":[season],
                                "pcs_club_join_invitations":[invite], "clubs":[club,dict(id="alpha",name="Organizer",slug="alpha")]}[name])
    def rpc(name, params):
        state["calls"].append((name, params))
        def execute():
            if state["error"]:
                exc = RuntimeError("private sql details"); exc.code = state["error"]; raise exc
            return SimpleNamespace(data=dict(invitation=invite,club=club,season=season) if name == "pcs_create_interclub_club_invitation" else invite)
        return SimpleNamespace(execute=execute)
    app = FastAPI(); routes.install_club_join_invitation_routes(app, get_supabase_client=lambda: SimpleNamespace(table=table,rpc=rpc))
    state["root"] = f"/admin/clubs/alpha/interclub/setup/{season_id}/club-invitations"
    state["payload"] = dict(invitation_id=invitation_id,expected_revision=2,name=" La Ríbera ",email=" INVITED@EXAMPLE.TEST ",draft=draft)
    return TestClient(app), state


def test_create_saves_selection_with_verified_organizer_and_no_access_or_email(setup):
    c,s=setup; r=c.post(s["root"],json=s["payload"])
    assert r.status_code==200
    name,args=s["calls"][0]
    assert name=="pcs_create_interclub_club_invitation"
    assert args["p_club_id"]=="alpha" and args["p_actor_id"]==s["user"].user_id
    assert args["p_season_id"]==s["season"]["id"] and args["p_revision"]==2
    assert args["p_slug"]=="la-ribera" and args["p_email"]=="invited@example.test"
    assert args["p_draft"]["club_ids"]==["alpha"]
    assert "private" not in r.text and "organizer_club_id" not in r.json()["season"]
    assert not s["sent"]


@pytest.mark.parametrize("change",[dict(role="operator"),dict(club_id="other"),dict(revoked_at="2020-01-01"),dict(user_id="other"),dict(expires_at="2020-01-01T00:00:00Z")])
def test_current_organizer_admin_required(setup,change):
    c,s=setup; s["assignment"].update(change)
    assert c.get(s["root"]).status_code==403
    assert c.post(s["root"],json=s["payload"]).status_code==403
    assert not s["calls"]


def test_cannot_invite_from_another_organizers_draft(setup):
    c,s=setup; s["season"]["organizer_club_id"]="other"
    assert c.get(s["root"]).status_code==404
    assert c.post(s["root"],json=s["payload"]).status_code==404
    assert not s["calls"]


@pytest.mark.parametrize("field,value",[("name"," "),("email","bad"),("actor_id","forged"),("club_id","existing"),("role","super_admin"),("expected_revision",0),("invitation_id","invalid")])
def test_invalid_and_privileged_inputs_rejected(setup,field,value):
    c,s=setup
    assert c.post(s["root"],json={**s["payload"],field:value}).status_code==422
    assert not s["calls"]


def test_list_expiry_and_safe_recipient_review(setup):
    c,s=setup; s["invite"]["expires_at"]="2020-01-01T00:00:00Z"
    r=c.get(s["root"])
    assert r.json()["invitations"][0]["status"]=="expired" and r.json()["clubs"][0]["id"]=="la-ribera"
    assert "private" not in r.text
    path=f"/club-invitations/{s['invite']['id']}"
    assert c.get(path).status_code==404
    s["user"].email="invited@example.test"
    r=c.get(path)
    assert r.status_code==200 and r.json()["organizer"]["name"]=="Organizer"
    assert r.json()["invitation"]["role"]=="administrator" and "private" not in r.text
    assert c.post(path+"/accept").status_code==200
    assert s["calls"][-1][1]==dict(p_action="accept",p_id=s["invite"]["id"],p_actor_id=s["user"].user_id,p_actor_email="invited@example.test")


@pytest.mark.parametrize("action",["cancel","renew"])
def test_update_scopes_club_season_revision_and_email(setup,action):
    c,s=setup
    r=c.post(s["root"]+"/"+s["invite"]["id"],json=dict(action=action,expected_revision=1,email="NEW@example.test"))
    assert r.status_code==200
    assert s["calls"][-1][1]==dict(p_action=action,p_id=s["invite"]["id"],p_club_id="alpha",p_season_id=s["season"]["id"],
      p_actor_id=s["user"].user_id,p_actor_email=s["user"].email,p_revision=1,p_email="new@example.test")


@pytest.mark.parametrize("code,status",[("23505",409),("42501",403),("40001",409),("P0002",404),("22023",422),("other",503)])
def test_safe_database_rejections(setup,code,status):
    c,s=setup; s["error"]=code
    r=c.post(s["root"],json=s["payload"])
    assert r.status_code==status and "private" not in r.text


@pytest.mark.parametrize("mode",["dry_run","staging_redirect"])
def test_staging_sign_in_creates_no_auth_token_or_mail(setup,monkeypatch,mode):
    c,s=setup; monkeypatch.setattr(routes,"get_email_mode",lambda:mode)
    r=c.post(f"/club-invitations/{s['invite']['id']}/sign-in",json={"email":"invited@example.test"})
    assert not r.json()["email_enabled"] and not s["calls"] and not s["sent"]


def test_live_sign_in_claim_uses_club_invitation_callback(setup,monkeypatch):
    c,s=setup; monkeypatch.setattr(routes,"get_email_mode",lambda:"live")
    assert c.post(f"/club-invitations/{s['invite']['id']}/sign-in",json={"email":"invited@example.test"}).status_code==200
    assert s["calls"][-1][1]["p_action"]=="email_claim" and s["sent"][-1][1]=={"club_join":True}
    generated=[]; sent=[]
    monkeypatch.setattr(staff,"get_email_mode",lambda:"live")
    monkeypatch.setattr(staff,"get_next_web_base_url",lambda **_:"https://web.example.test")
    monkeypatch.setattr(staff,"send_email_with_inline_chart",lambda **kw:sent.append(kw))
    db=SimpleNamespace(auth=SimpleNamespace(admin=SimpleNamespace(generate_link=lambda p: generated.append(p) or SimpleNamespace(properties=SimpleNamespace(hashed_token="test-secret")))))
    staff.send_invitation_sign_in(db,s["invite"],club_join=True)
    assert "&kind=club#staff_token_hash=test-secret" in sent[0]["text_body"]
    assert sent[0]["to_email"]==s["invite"]["email"] and generated==[{"type":"magiclink","email":s["invite"]["email"]}]


def test_unauthenticated_recipient_denied(setup,monkeypatch):
    c,s=setup
    def deny(_): raise HTTPException(401,"Sign in required")
    monkeypatch.setattr(routes,"authenticate_bearer",deny)
    assert c.post(f"/club-invitations/{s['invite']['id']}/accept").status_code==401
    assert not s["calls"]
