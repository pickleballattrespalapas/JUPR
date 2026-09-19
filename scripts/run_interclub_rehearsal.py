#!/usr/bin/env python3
"""Full functional rehearsal against the deployed, isolated staging API.

Only canonical staging Actions may run this. All directory players, staff,
seasons and rating history belong to newly created synthetic clubs. The private
state file never becomes an artifact. Cleanup disables links/publications and
revokes/deletes the synthetic identities, retaining isolated sporting evidence.
Test-only time travel changes only this run's fixtures, never the server clock.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit, parse_qs
from urllib.request import Request, build_opener, HTTPRedirectHandler
from uuid import uuid4

API = "https://juprleagues-api-staging.fly.dev"
AUTH = "https://sijpxjxvdtrehmqvirfi.supabase.co"
MARKER = "interclub-functional-rehearsal-v1"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Rehearsal requests must not redirect credentials.")


def now():
    return datetime.now(timezone.utc)


def iso(value):
    return value.isoformat()


def validate_environment(env):
    if (env.get("GITHUB_ACTIONS") != "true" or env.get("GITHUB_REF") != "refs/heads/staging"
            or env.get("STAGING_API_BASE_URL") != API or env.get("STAGING_SUPABASE_URL") != AUTH
            or not re.fullmatch(r"[0-9a-f]{40}", env.get("GITHUB_SHA", ""))):
        raise RuntimeError("Rehearsal requires the canonical isolated staging workflow.")
    for key in ("STAGING_SUPABASE_SERVICE_ROLE_KEY", "STAGING_SUPABASE_ANON_KEY"):
        if not env.get(key):
            raise RuntimeError("A required staging credential is missing.")


class Rehearsal:
    def __init__(self, directory):
        validate_environment(os.environ)
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.private = directory.parent / "interclub-rehearsal-private.json"
        self.state = {"marker": MARKER, "sha": os.environ["GITHUB_SHA"], "run": uuid4().hex[:12],
                      "clubs": [], "users": [], "seasons": [], "players": [], "checks": []}
        self.stage = "environment"
        self.service = os.environ["STAGING_SUPABASE_SERVICE_ROLE_KEY"]
        self.anon = os.environ["STAGING_SUPABASE_ANON_KEY"]
        self.secrets = [self.service, self.anon]
        self.persist()

    def persist(self):
        self.private.write_text(json.dumps(self.state))
        self.private.chmod(0o600)

    def redact(self, value):
        value = str(value)
        for secret in self.secrets:
            value = value.replace(secret, "[private]")
        return re.sub(r"(token[=\" :]+)[^\s\"&]+", r"\1[private]", value, flags=re.I)[:1200]

    def request(self, origin, method, path, payload=None, *, token=None, service=False, expected=(200, 201, 204)):
        if origin not in (API, AUTH) or not path.startswith("/") or path.startswith("//"):
            raise RuntimeError("Refusing an unknown rehearsal destination.")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if origin == AUTH:
            headers["apikey"] = self.service if service else self.anon
        if service:
            headers.update(Authorization=f"Bearer {self.service}", Prefer="return=representation")
        elif token:
            headers["Authorization"] = f"Bearer {token}"
        req = Request(origin + path, data=json.dumps(payload).encode() if payload is not None else None, headers=headers, method=method)
        started=time.monotonic()
        self.last_request={"method":method,"path":path.split('?')[0]}
        try:
            with build_opener(NoRedirect).open(req, timeout=120) as response:
                status, data = response.status, response.read()
        except HTTPError as exc:
            status, data = exc.code, exc.read()
        except TimeoutError as exc:
            raise RuntimeError(f"{method} {path.split('?')[0]} timed out") from exc
        try:
            result = json.loads(data) if data else None
        except ValueError:
            result = {"detail": "Non-JSON response"}
        if status not in expected:
            detail = (result or {}).get("detail", (result or {}).get("message", "Unexpected response")) if isinstance(result, dict) else "Unexpected response"
            raise RuntimeError(self.redact(f"{method} {path.split('?')[0]}: HTTP {status}; {detail}"))
        if status == 409 and time.monotonic()-started>15:
            raise AssertionError("An application conflict must return promptly without transaction retries")
        return result

    def db(self, method, table, payload=None, **filters):
        return self.request(AUTH, method, "/rest/v1/" + table + ("?" + urlencode(filters) if filters else ""), payload, service=True)

    def api(self, method, path, payload=None, actor=0, expected=(200, 201, 204)):
        token = None if actor is None else self.state["users"][actor]["token"]
        return self.request(API, method, path, payload, token=token, expected=expected)

    def check(self, condition, name):
        if not condition:
            raise AssertionError(name)
        self.state["checks"].append(name)
        print("PASS " + name, flush=True)
        self.persist()

    def phase(self, name):
        self.stage = name
        print("REHEARSAL " + name, flush=True)

    def setup(self):
        self.phase("isolated fixtures and authenticated staff")
        health = self.api("GET", "/health", actor=None)
        self.check(health["environment"] == "staging" and health["git_commit_sha"] == self.state["sha"]
                   and health["jwt_verification_project_ref"] == "sijpxjxvdtrehmqvirfi"
                   and health["write_wave"] == "open"
                   and health["write_prerequisites"]["email_mode"] == "dry_run", "exact deployed SHA and staging isolation")
        clubs = [f"qa-ic-{self.state['run']}-{i}" for i in range(4)]
        self.state["clubs"] = clubs
        self.persist()
        self.db("POST", "clubs", [{"id": club, "slug": club, "name": f"Rehearsal {self.state['run']} Club {i+1}",
                                  "is_active": False, "status": "draft", "onboarding_status": "draft"} for i, club in enumerate(clubs)])
        for label, assigned, role, scopes in [("organizer", clubs, "administrator", []),
                                               ("participant", [clubs[1]], "administrator", []),
                                               ("operator", [clubs[1]], "operator", [{"kind": "program_type", "program_type": "leagues", "resource_id": ""}]),
                                               ("nonhost", [clubs[2]], "administrator", [])]:
            email = f"qa-ic-{self.state['run']}-{label}@example.invalid"
            user = self.request(AUTH, "POST", "/auth/v1/admin/users", {"email": email, "email_confirm": True,
                "app_metadata": {"pcs_rehearsal": MARKER, "run": self.state["run"]}}, service=True)
            row = {"id": user["id"], "email": email, "token": "", "role": role}
            self.state["users"].append(row)
            self.persist()
            self.db("POST", "admin_role_assignments", [{"club_id": club, "user_id": user["id"], "email": email,
                    "role": role, "scopes": scopes, "expires_at":iso(now()+timedelta(hours=2))} for club in assigned])
            generated = self.request(AUTH, "POST", "/auth/v1/admin/generate_link", {"type": "magiclink", "email": email}, service=True)
            props = generated.get("properties", generated)
            token_hash = props.get("hashed_token")
            if not token_hash:
                raise RuntimeError("Staging Auth did not provide a private verification token.")
            self.secrets.append(token_hash)
            session = self.request(AUTH, "POST", "/auth/v1/verify", {"token_hash": token_hash, "type": "magiclink"})
            row["token"] = session["access_token"]
            self.secrets.append(row["token"])
            print(f"::add-mask::{row['token']}", flush=True)
            self.persist()
        for ci, club in enumerate(clubs):
            for di, division in enumerate(["3.5", "4.0", "4.5"]):
                for pi in range(6):
                    pid = 7000000000000 + int(uuid4().hex[:11], 16)
                    self.state["players"].append({"id": pid, "club_id": club,
                        "name": f"Rehearsal {ci+1} {division} {'F' if pi<3 else 'M'}{pi}",
                        "normalized_name": f"rehearsal {ci+1} {division} {pi}", "rating": float(division)*400+100,
                        "starting_rating": float(division)*400+100, "active": True,
                        "gender": "female" if pi<3 else "male", "division": division})
        self.persist()
        self.check(len(self.api("GET", "/admin/auth/workspaces")["workspaces"]) == 4, "synthetic administrator limited to four isolated clubs")
        self.api("GET", f"/admin/clubs/{clubs[0]}/interclub/setup", actor=1, expected=(403,))
        self.check(True, "participant cannot enter another club workspace")

    def registration(self, club, sid):
        return f"/admin/clubs/{club}/interclub/registrations/{sid}"

    def competition(self, club, sid, meet=None, phase="regular"):
        root = f"/admin/clubs/{club}/interclub/competition/{sid}"
        return root + (f"/meets/{meet}/{phase}" if meet else "")

    def season(self, label, club_count=4, divisions=None, *, public_signups=False):
        clubs = self.state["clubs"][:club_count]
        divisions = divisions or ["3.5"]
        sid = str(uuid4())
        record = {"id": sid, "label": label, "clubs": clubs, "divisions": divisions, "meets": []}
        record["players"] = [{**p, "id": 7000000000000 + int(uuid4().hex[:11],16),
                              "normalized_name": p["normalized_name"]+" "+sid}
                             for p in self.state["players"] if p["club_id"] in clubs and p["division"] in divisions]
        self.state["seasons"].append(record)
        self.persist()
        self.db("POST", "players", [{k:v for k,v in p.items() if k!="division"} for p in record["players"]])
        start = now()+timedelta(days=2)
        draft = {"name": f"Rehearsal {self.state['run']} {label}", "start_date": start.date().isoformat(),
                 "end_date": (start+timedelta(days=60)).date().isoformat(), "timezone": "America/Mazatlan",
                 "divisions": divisions, "club_ids": clubs, "meets": [{"host_club_id": clubs[1], "club_ids": clubs,
                 "starts_at": iso(start+timedelta(days=3)), "duration_minutes": 180, "courts": 12}]}
        saved = self.api("PUT", f"/admin/clubs/{clubs[0]}/interclub/setup", {"season_id": sid, "expected_revision": 0, "draft": draft})
        self.api("POST", self.registration(clubs[0], sid)+"/open", {"expected_revision": saved["season"]["revision"],
            "rules": {division: {"min_rating": float(division), "max_rating": float(division)+.499, "women_required":2} for division in divisions}})
        for club in clubs:
            root = self.registration(club, sid)
            self.api("GET", root+"/pool", expected=(404,))
            self.api("POST", root+f"/participations/{club}", {"expected_revision":1, "action":"accept"})
            pool = self.api("PUT", root+"/pool", {"expected_revision":0,"open":True})
            self.check(pool["email_mode"] == "dry_run", label+" dry-run email confirmed")
            record.setdefault("signup", {})[club] = pool["signup"]
            selected = [p for p in record["players"] if p["club_id"]==club]
            for index, player in enumerate(selected):
                email = f"p-{player['id']}@example.invalid"
                details = {"name":player["name"],"email":email,"divisions":[player["division"]],"notes":"Private rehearsal note",
                           "email_consent":True,"request_id":str(uuid4())}
                if public_signups and index == 0:
                    shared = "/public/interclub-signups/"+pool["signup"]["share_id"]
                    result = self.api("POST", shared, details, actor=None)
                    self.check(result["status"]=="registered" and "#token=" in result["manage_url"], "account-free player signup "+club)
                    self.check(self.api("POST", shared, details, actor=None)["manage_url"]==result["manage_url"], "signup network retry keeps one identity "+club)
                    duplicate = self.api("POST", shared, {**details,"request_id":str(uuid4())},actor=None)
                    self.check(duplicate["status"]=="already_registered" and "manage_url" not in duplicate, "duplicate signup does not expose private link "+club)
                    members = self.api("GET",root+"/pool")["members"]
                    member = next(m for m in members if m["email"]==email)
                else:
                    # Bulk fixture preparation; the first signup per club exercises the actual public API.
                    member = self.db("POST","pcs_interclub_pool_members", {"season_id":sid,"club_id":club,"name":player["name"],
                        "email":email,"divisions":[player["division"]],"notes":"Private rehearsal note","request_id":str(uuid4()),"request_fingerprint":"rehearsal"})[0]
                linked = self.api("PATCH",root+"/pool/members/"+member["id"], {"expected_revision":member["revision"],"player_id":player["id"],"status":"active"})["member"]
                if linked["approval_status"] != "approved":
                    raise AssertionError("Preseason player failed automatic approval")
            record.setdefault("pools", {})[club] = self.api("GET", root+"/pool")
        detail = self.api("GET",self.registration(clubs[0],sid))
        record["meets"] = detail["meets"]
        self.persist()
        return record

    def roster(self, season, meet, club, division, *, partial=False, alternate=False):
        players=[p for p in season["players"] if p["club_id"]==club and p["division"]==division]
        ids=[players[i]["id"] for i in ([0,1] if partial else [0,2,3,5] if alternate else [0,1,3,4])]
        mid=meet["id"]
        root=self.registration(club,season["id"])+"/meets/"+mid
        current=self.api("GET",root)
        existing=next((t for t in current["teams"] if t["club_id"]==club and t["division"]==division and not t["withdrawn"]),None)
        tid=existing["id"] if existing else str(uuid4())
        data={"expected_meet_revision":current["meet"]["revision"],"expected_revision":existing["revision"] if existing else 0,
              "name":f"Rehearsal {division}","division":division,"player_ids":ids,"missing_pairing_forfeit":partial}
        result=self.api("PUT",root+"/teams/"+tid,data)
        return result

    def open_dates(self, season):
        """Move only this synthetic season's enrollment history before its rehearsed meet."""
        sid=season["id"]
        row=self.db("GET","pcs_interclub_seasons",select="details",id="eq."+sid)[0]
        details={**row["details"],"start_date":(now()-timedelta(days=30)).date().isoformat()}
        self.db("PATCH","pcs_interclub_seasons",{"details":details},id="eq."+sid)
        self.db("PATCH","pcs_interclub_entries",{"entered_at":iso(now()-timedelta(days=29))},season_id="eq."+sid)
        self.db("PATCH","pcs_interclub_pool_members",{"approved_at":iso(now()-timedelta(days=29))},season_id="eq."+sid)

    def move_meet(self, season, meet, when, deadline=None):
        self.db("PATCH","pcs_interclub_meets",{"starts_at":iso(when),"roster_deadline":iso(deadline or when-timedelta(hours=1))},id="eq."+meet["id"],season_id="eq."+season["id"])
        return self.api("GET",self.registration(season["clubs"][0],season["id"])+"/meets/"+meet["id"])["meet"]

    def generated(self, season, meet, *, phase="regular", division=None, pair=None, format="gender", revision=0):
        data={"expected_revision":revision,"format":format}
        if pair:
            data.update(division=division,club_a=pair[0],club_b=pair[1])
        return self.api("POST",self.competition(season["clubs"][0],season["id"],meet["id"],phase)+"/generate",data)["batch"]

    def complete(self, document, played_at, winner=None, *, cycle=False, final_tie=False):
        doc=deepcopy(document)
        clubs=self.state["clubs"]
        for encounter in doc["encounters"]:
            a,b=encounter["club_a"],encounter["club_b"]
            best=winner or min([a,b],key=clubs.index)
            if cycle:
                best=a if (clubs.index(a)-clubs.index(b))%3==1 else b
            for pi,pairing in enumerate(encounter["pairings"]):
                for gi,game in enumerate(pairing["games"]):
                    if game["status"] in {"forfeit","double_forfeit"}:continue
                    win=best if not final_tie or pi<2 else (b if best==a else a)
                    game.update(status="completed",a=11 if win==a else 5,b=11 if win==b else 5,winner="a" if win==a else "b",
                                played_at=iso(played_at+timedelta(minutes=encounter["rotation"]*20+gi*4+pi)))
            if final_tie:
                encounter["tiebreak"]={"status":"completed","a":23 if best==a else 21,"b":23 if best==b else 21,
                    "order_a":list(dict.fromkeys(e for p in encounter["pairings"] for e in p["players_a"])),
                    "order_b":list(dict.fromkeys(e for p in encounter["pairings"] for e in p["players_b"]))}
        return doc

    def save(self, season, meet, batch, document):
        return self.api("PUT",self.competition(season["clubs"][0],season["id"],meet["id"],document["phase"]),
                        {"expected_revision":batch["revision"],"document":document})["batch"]

    def approve(self, season, meet, batch):
        root=self.competition(season["clubs"][0],season["id"],meet["id"],batch["phase"])
        hostroot=self.competition(season["clubs"][1],season["id"],meet["id"],batch["phase"])
        correction=bool(batch.get("approved_document"))
        submitted=self.api("POST",(root if correction else hostroot)+"/submit",{"expected_revision":batch["revision"]},actor=0 if correction else 2)["batch"]
        self.api("POST",hostroot+"/approve",{"expected_revision":submitted["revision"]},actor=2,expected=(403,))
        self.api("POST",self.competition(season["clubs"][1],season["id"],meet["id"],batch["phase"])+"/approve",
                 {"expected_revision":submitted["revision"]},actor=1,expected=(403,))
        approved=self.api("POST",root+"/approve",{"expected_revision":submitted["revision"]})
        self.check(approved["ratings"]["status"]=="completed" and approved["batch"]["ratings_status"]=="completed",
                   season["label"]+" organizer approval completes both rating streams")
        return approved["batch"]

    def report(self, status, error=None):
        report={"status":status,"candidate_sha":self.state["sha"],"run":self.state["run"],"stage":self.stage,
                "checks":self.state["checks"],"checks_passed":len(self.state["checks"]),
                "error":self.redact(error) if error else None,"email_mode":"dry_run",
                "fixture_clubs":self.state["clubs"],"fixture_seasons":[{"id":s["id"],"label":s["label"]} for s in self.state["seasons"]]}
        (self.directory/"interclub-rehearsal.json").write_text(json.dumps(report,indent=2))
        print(json.dumps({k:report[k] for k in ["status","candidate_sha","stage","checks_passed","error"]}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir",type=Path,required=True)
    parser.add_argument("--cleanup",action="store_true")
    args=parser.parse_args()
    if args.cleanup:
        return cleanup(args.report_dir)
    rehearsal=Rehearsal(args.report_dir)
    try:
        rehearsal.setup()
        run_scenarios(rehearsal)
        rehearsal.report("passed")
        return 0
    except Exception as exc:
        rehearsal.report("failed",f"{type(exc).__name__}: {exc}")
        return 1


def cleanup(directory):
    validate_environment(os.environ)
    path=directory.parent/"interclub-rehearsal-private.json"
    if not path.exists():return 0
    state=json.loads(path.read_text())
    if state.get("marker")!=MARKER or state.get("sha")!=os.environ["GITHUB_SHA"]:
        raise RuntimeError("Unrecognized rehearsal cleanup state")
    r=Rehearsal.__new__(Rehearsal)
    r.directory,r.private,r.state=directory,path,state
    r.service=os.environ["STAGING_SUPABASE_SERVICE_ROLE_KEY"]
    r.anon=os.environ["STAGING_SUPABASE_ANON_KEY"]
    r.secrets=[r.service,r.anon]+[u["token"] for u in state["users"] if u.get("token")]
    errors=[]
    for season in state["seasons"]:
        try:
            r.db("PATCH","pcs_interclub_pool_settings",{"open":False},season_id="eq."+season["id"])
            r.db("PATCH","pcs_interclub_availability_settings",{"open":False},season_id="eq."+season["id"])
            r.db("PATCH","pcs_interclub_publications",{"published":None,"published_at":None},season_id="eq."+season["id"])
        except Exception as exc:errors.append(r.redact(exc))
    for club in state["clubs"]:
        if not club.startswith("qa-ic-"+state["run"]+"-"):raise RuntimeError("Cleanup club boundary failed")
        try:
            r.db("PATCH","clubs",{"is_active":False,"status":"draft"},id="eq."+club)
            r.db("PATCH","admin_role_assignments",{"revoked_at":iso(now())},club_id="eq."+club)
        except Exception as exc:errors.append(r.redact(exc))
    for user in state["users"]:
        try:
            if user.get("token"):
                r.request(AUTH,"POST","/auth/v1/logout?scope=global",token=user["token"],expected=(200,204,401,403))
            saved=r.request(AUTH,"GET","/auth/v1/admin/users/"+user["id"],service=True)
            if saved.get("app_metadata",{}).get("run")!=state["run"]:raise RuntimeError("Cleanup identity boundary failed")
            r.request(AUTH,"DELETE","/auth/v1/admin/users/"+user["id"],service=True)
        except Exception as exc:errors.append(r.redact(exc))
    if not errors:
        path.unlink(missing_ok=True)
    (directory/"interclub-rehearsal-cleanup.json").write_text(json.dumps({"status":"failed" if errors else "passed",
        "links_closed":not errors,"publications_hidden":not errors,"synthetic_staff_revoked":not errors,"identities_removed":not errors,
        "retained":"Isolated synthetic sporting records only","errors":errors},indent=2))
    print("Rehearsal cleanup "+("failed" if errors else "passed"),flush=True)
    return 1 if errors else 0


def run_scenarios(r):
    from scripts.interclub_rehearsal_scenarios import run
    run(r)


if __name__=="__main__":
    raise SystemExit(main())
