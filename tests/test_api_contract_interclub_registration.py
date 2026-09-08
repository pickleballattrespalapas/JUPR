from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api import admin_auth_routes, interclub_registration_routes as routes


class Query:
    def __init__(self, rows):
        self.rows, self.filters, self.columns, self.bounds = rows, [], "*", (0, 100000)
    def select(self, fields): self.columns = fields; return self
    def eq(self, key, value): self.filters.append(lambda row: row.get(key) == value); return self
    def in_(self, key, values): self.filters.append(lambda row: row.get(key) in values); return self
    def ilike(self, key, value): self.filters.append(lambda row: value.strip("%").lower() in row[key].lower()); return self
    def order(self, *_args, **_kwargs): return self
    def limit(self, limit): self.bounds = (0, limit); return self
    def range(self, start, end): self.bounds = (start, end+1); return self
    def execute(self):
        rows = [r for r in self.rows if all(check(r) for check in self.filters)][self.bounds[0]:self.bounds[1]]
        return SimpleNamespace(data=[dict(r) if self.columns == "*" else {k: r.get(k) for k in self.columns.split(",")} for r in rows])


@pytest.fixture
def setup(monkeypatch):
    uid, sid, tid = str(uuid4()), str(uuid4()), str(uuid4())
    user = SimpleNamespace(user_id=uid, email="admin@example.test")
    assignment = dict(club_id="alpha", email=user.email, user_id=uid, role="administrator")
    season = dict(id=sid, organizer_club_id="alpha", source_revision=2, opened_at="2026-09-08T00:00:00Z", roster_deadline="2099-01-01T00:00:00Z",
                  details=dict(name="Coastal League",club_ids=["beta","gamma"],divisions=["3.5"],timezone="America/Mazatlan",meets=[]), rules={"3.5":dict(min_rating=None,max_rating=3.75,women_required=2)})
    participation = dict(season_id=sid,club_id="beta",status="accepted",revision=2)
    version = dict(team_id=tid,revision=1,name="Beta Blue",status="needs_exception",issues=[dict(code="rating_above_maximum",message="Player exceeds limit",private="secret")],late_change=False,
                   roster=[dict(entry_id=str(uuid4()),player_id=str(i),name=f"Player {i}",starting_rating=3.6,gender="female",email="private@example.test",phone="secret") for i in range(1,5)])
    team = dict(id=tid,season_id=sid,club_id="beta",division="3.5",withdrawn=False,**version)
    tables = {"admin_role_assignments":[assignment],"pcs_interclub_seasons":[season],"pcs_interclub_participations":[participation],
              "pcs_interclub_current_rosters":[team],"pcs_interclub_teams":[team],"pcs_interclub_roster_versions":[version],
              "clubs":[dict(id=c,name=c.title(),slug=c) for c in ("alpha","beta","gamma")],
              "players":[dict(id=i,club_id="beta",name=f"Player {i}",rating=1600,gender="female",active=True,email="private@example.test") for i in range(1,5)] + [dict(id=99,club_id="gamma",name="Private other club player",active=True,rating=1800)],
              "pcs_interclub_entries":[dict(season_id=sid,club_id="beta",player_id=1,starting_rating=3.2)]}
    state = dict(user=user,assignment=assignment,season=season,participation=participation,team=team,tables=tables,calls=[],reads=[],error="")
    def table(name): state["reads"].append(name); return Query(tables[name])
    def rpc(name, params):
        state["calls"].append((name,params))
        def execute():
            if state["error"]:
                e = RuntimeError("private database failure"); e.code=state["error"]; raise e
            data = season if name == "pcs_open_interclub_registration" else participation if name == "pcs_interclub_participation" else {"team":team,"roster":version}
            return SimpleNamespace(data=data)
        return SimpleNamespace(execute=execute)
    monkeypatch.setattr(admin_auth_routes,"authenticate_bearer",lambda _: user)
    app = FastAPI(); routes.install_interclub_registration_routes(app,get_supabase_client=lambda:SimpleNamespace(table=table,rpc=rpc))
    return TestClient(app),state


def base(s, club="alpha"):
    return f"/admin/clubs/{club}/interclub/registrations/{s['season']['id']}"


def test_organizer_sees_submitted_facts_without_other_club_directories_or_contacts(setup):
    c,s=setup
    response=c.get(base(s)); assert response.status_code==200
    roster=response.json()["teams"][0]["roster"]
    assert roster[0]["name"]=="Player 1" and "player_id" not in roster[0]
    assert "private" not in response.text and "phone" not in response.text
    assert "players" not in s["reads"] and "pcs_interclub_entries" not in s["reads"]
    assert c.get(base(s)+"/players").status_code==403


def test_participating_club_sees_only_own_teams_and_directory_with_frozen_seed(setup):
    c,s=setup; s["assignment"]["club_id"]="beta"
    s["tables"]["pcs_interclub_current_rosters"].append({**s["team"],"id":str(uuid4()),"club_id":"gamma","name":"Other club private team"})
    result=c.get(base(s,"beta")).json()
    assert not result["is_organizer"] and len(result["teams"])==1 and result["teams"][0]["roster"][0]["player_id"]=="1"
    assert [p["club_id"] for p in result["participations"]]==["beta"]
    response=c.get(base(s,"beta")+"/players")
    assert response.status_code==200
    assert len(response.json()["players"])==4 and response.json()["players"][0]["starting_rating"]==3.2
    assert response.json()["players"][1]["starting_rating"]==4.0
    assert "private" not in response.text and "email" not in response.text


def test_uninvited_club_cannot_read_season_history_or_players(setup):
    c,s=setup; s["assignment"]["club_id"]="gamma"
    for suffix in ["","/players",f"/teams/{s['team']['id']}/history"]:
        assert c.get(base(s,"gamma")+suffix).status_code==404


@pytest.mark.parametrize("change",[dict(role="operator"),dict(user_id="someone-else"),dict(revoked_at="2020-01-01"),dict(expires_at="2020-01-01T00:00:00Z")])
def test_revoked_expired_wrong_identity_or_operator_cannot_access_registration(setup, change):
    c,s=setup; s["assignment"].update(change)
    assert c.get("/admin/clubs/alpha/interclub/registrations").status_code==403
    assert c.get(base(s)).status_code==403
    assert c.post(base(s)+"/participations/beta",json={"action":"cancel","expected_revision":2}).status_code==403
    assert not s["calls"]


def test_club_list_only_unions_owned_and_invited_seasons(setup):
    c,s=setup
    s["tables"]["pcs_interclub_seasons"].append({**s["season"],"id":str(uuid4()),"organizer_club_id":"gamma"})
    assert len(c.get("/admin/clubs/alpha/interclub/registrations").json()["seasons"])==1
    s["assignment"]["club_id"]="beta"
    assert len(c.get("/admin/clubs/beta/interclub/registrations").json()["seasons"])==1


def test_open_uses_exact_saved_revision_and_verified_organizer(setup):
    c,s=setup
    response=c.post(base(s)+"/open",json={"expected_revision":2,"roster_deadline":"2099-01-01T00:00:00Z","rules":s["season"]["rules"]})
    assert response.status_code==200
    name,args=s["calls"][0]
    assert name=="pcs_open_interclub_registration" and args["p_club_id"]=="alpha" and args["p_actor_id"]==s["user"].user_id
    assert args["p_revision"]==2 and args["p_rules"]==s["season"]["rules"]


@pytest.mark.parametrize("patch",[dict(min_rating=4,max_rating=3),dict(max_rating=9),dict(women_required=5),dict(private="injected")])
def test_invalid_eligibility_settings_are_rejected_before_writing(setup,patch):
    c,s=setup
    response=c.post(base(s)+"/open",json={"expected_revision":2,"roster_deadline":"2099-01-01T00:00:00Z","rules":{"3.5":patch}})
    assert response.status_code==422 and not s["calls"]


def test_roster_and_decision_pass_scope_and_revision_to_atomic_transactions(setup):
    c,s=setup; s["assignment"]["club_id"]="beta"
    url=base(s,"beta")+f"/teams/{s['team']['id']}"
    response=c.put(url,json={"expected_revision":1,"name":"Beta Blue","division":"3.5","player_ids":[1,2,3,4]})
    assert response.status_code==200
    name,args=s["calls"][-1]
    assert name=="pcs_save_interclub_roster" and args["p_club_id"]=="beta" and args["p_revision"]==1 and args["p_player_ids"]==[1,2,3,4]
    assert "private" not in response.text
    s["assignment"]["club_id"]="alpha"
    response=c.post(base(s)+f"/teams/{s['team']['id']}/eligibility",json={"expected_revision":1,"approve":True,"reason":" Approved for this roster "})
    assert response.status_code==200 and s["calls"][-1][1]["p_reason"]=="Approved for this roster"
    assert "player_id" not in response.json()["team"]["roster"][0]


@pytest.mark.parametrize("patch",[dict(player_ids=[1,2,3]),dict(player_ids=[1,1,2,3]),dict(player_ids=[-1,2,3,4]),dict(player_ids=[1,2,3,4,5]),dict(club_id="gamma"),dict(actor_id="forged"),dict(name=" ")])
def test_bad_or_forged_roster_inputs_never_write(setup,patch):
    c,s=setup
    payload=dict(expected_revision=0,name="Team",division="3.5",player_ids=[1,2,3,4]); payload.update(patch)
    assert c.put(base(s)+f"/teams/{s['team']['id']}",json=payload).status_code==422
    assert not s["calls"]


def test_history_is_private_to_owner_and_organizer_with_contact_redaction(setup):
    c,s=setup
    response=c.get(base(s)+f"/teams/{s['team']['id']}/history")
    assert response.status_code==200 and "private" not in response.text
    assert "player_id" not in response.json()["history"][0]["roster"][0]
    s["assignment"]["club_id"]="gamma"
    s["tables"]["pcs_interclub_participations"].append({**s["participation"],"club_id":"gamma"})
    assert c.get(base(s,"gamma")+f"/teams/{s['team']['id']}/history").status_code==404


@pytest.mark.parametrize("code,status",[("42501",403),("40001",409),("23505",409),("22023",422),("P0002",404),("unknown",503)])
def test_database_failures_are_actionable_without_internal_details(setup,code,status):
    c,s=setup; s["error"]=code
    r=c.post(base(s)+"/participations/beta",json={"expected_revision":2,"action":"cancel"})
    assert r.status_code==status and "private" not in r.text


def test_player_and_team_lists_have_explicit_pagination(setup):
    c,s=setup; s["assignment"]["club_id"]="beta"
    s["tables"]["players"]=[dict(id=i,club_id="beta",name=f"Player {i}",rating=1400,active=True) for i in range(1,103)]
    s["tables"]["pcs_interclub_current_rosters"]=[{**s["team"],"id":str(uuid4())} for _ in range(102)]
    assert len(c.get(base(s,"beta")+"/players").json()["players"])==100
    assert c.get(base(s,"beta")+"/players").json()["next_offset"]==100
    assert len(c.get(base(s,"beta")+"/players?offset=100").json()["players"])==2
    assert c.get(base(s,"beta")).json()["next_team_offset"]==100
    assert len(c.get(base(s,"beta")+"?team_offset=100").json()["teams"])==2
