"""Real API scenarios; service access is restricted to synthetic fixture setup."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import combinations
import json
import time
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from scripts.run_interclub_rehearsal import iso, now


def token(url):
    return parse_qs(urlsplit(url).fragment)["token"][0]


def games(document):
    return [g for e in document["encounters"] for p in e["pairings"] for g in p["games"]]


def workspace(r, s):
    return r.api("GET", r.competition(s["clubs"][0], s["id"]))


def new_meet(r, s, phase="regular", clubs=None):
    clubs = clubs or s["clubs"]
    result = r.api("POST", r.competition(s["clubs"][0], s["id"])+"/meets", {
        "host_club_id": clubs[1], "club_ids": clubs, "starts_at": iso(now()+timedelta(days=6+len(s["meets"]))),
        "roster_deadline": iso(now()+timedelta(days=5+len(s["meets"]))),
        "courts": 12, "duration_minutes": 180, "competition_phase": phase})["meet"]
    s["meets"].append(result)
    r.persist()
    return result


def prepare(r, s, m, *, phase="regular", partial=(), mixed=False):
    r.open_dates(s)
    # Keep successive rehearsed meets two days apart. Real scheduling guards
    # also apply to fixture time travel; finals must follow regular-season play.
    index = next(i for i, meet in enumerate(s["meets"]) if meet["id"] == m["id"])
    first_days_ago = 14 if s["label"] == "incidents" else 4
    when = now()-timedelta(days=first_days_ago-2*index)
    deadline = when-timedelta(hours=1)
    r.move_meet(s,m,now()+timedelta(days=3),deadline)
    for division in s["divisions"]:
        for club in m["club_ids"]:
            r.roster(s, m, club, division, partial=club in partial)
    r.move_meet(s, m, when, deadline)
    return r.generated(s, m, phase=phase, format="mixed" if mixed else "gender") if phase == "regular" else None


def rating_evidence(r, s, b, expected_games):
    source = r.db("GET", "pcs_interclub_rating_sources", batch_id="eq."+b["id"])[0]
    effects = r.db("GET", "pcs_interclub_rating_effects", batch_id="eq."+b["id"], generation_id="eq."+source["generation_id"], limit="1000")
    r.check(len(effects) == expected_games*8 and Counter(e["stream"] for e in effects) == Counter({"league":expected_games*4,"overall":expected_games*4}),
            s["label"]+f" {expected_games} completed doubles games update four players in both streams")
    return effects


def register_late_player(r, s):
    # Season start controls late-player eligibility independently of the signup
    # window. Enroll while the commissioner window is open; review after close.
    r.open_dates(s)
    club = s["clubs"][0]
    late_id = 7000000000000 + int(uuid4().hex[:11],16)
    r.db("POST","players",{"id":late_id,"club_id":club,"name":"Late Rehearsal Player",
        "normalized_name":"late "+s["id"],"rating":1500,"starting_rating":1500,"active":True,"gender":"female"})
    late_email = f"late-{r.state['run']}@example.invalid"
    r.api("POST","/public/interclub-signups/"+s["signup"][club]["share_id"],
        {"name":"Late Rehearsal Player","email":late_email,"divisions":["3.5"],"notes":"Late addition rehearsal","request_id":str(uuid4()),"email_consent":True},actor=None)
    poolroot = r.registration(club,s["id"])+"/pool"
    late = next(x for x in r.api("GET",poolroot)["members"] if x["email"] == late_email)
    late = r.api("PATCH",poolroot+"/members/"+late["id"],{"expected_revision":late["revision"],"player_id":late_id,"status":"active"})["member"]
    r.check(late["approval_status"] == "pending", "linked late addition still needs organizer approval")
    return late


def email_and_responses(r, s, m):
    r.phase("signup, private edits, dry-run invitations and availability")
    club = s["clubs"][0]
    root = r.registration(club, s["id"])
    member = s["pools"][club]["members"][0]
    secret = token(member["manage_url"])
    r.secrets.append(secret)
    review = r.api("POST", "/public/interclub-player-response/review", {"token":secret}, actor=None)
    body = {"token":secret,"action":"update_season","expected_revision":review["member"]["revision"],
            "name":member["name"],"email":member["email"],"divisions":member["divisions"],"notes":"Private revised note","status":"active"}
    edited = r.api("POST", "/public/interclub-player-response/respond", body, actor=None)
    r.api("POST", "/public/interclub-player-response/respond", body, actor=None, expected=(409,))
    r.check(edited["member"]["notes"] == "Private revised note", "personal link edit persists; stale edits rejected")
    r.api("GET", root+"/pool", actor=1, expected=(403,))
    # Two consenting directory players share one household address. The preview
    # must group the email while preserving the selected player count.
    members = r.api("GET",root+"/pool")["members"][:2]
    address = f"household-{r.state['run']}@example.invalid"
    r.db("POST","player_profile_update_subscriptions",[{"club_id":club,"player_id":m["player_id"],
        "email":address,"email_normalized":address,"request_status":"active","verified_at":iso(now())} for m in members])
    eroot = f"/admin/clubs/{club}/interclub/player-pools/{s['id']}/emails"
    audience = r.api("GET",eroot+"/audience?kind=season")
    payload = {"kind":"season","recipient_ids":[str(m["player_id"]) for m in members],**audience["defaults"]}
    preview = r.api("POST",eroot+"/preview",payload)
    r.check(preview["recipient_count"] == 1 and preview["player_count"] == 2,"season invitation groups a household email without losing players")
    operation = str(uuid4())
    r.api("POST",eroot,{**payload,"operation_key":operation,"preview_fingerprint":preview["preview_fingerprint"]})
    season_send = r.api("POST",eroot+f"/{operation}/recipients/0/send")
    r.check(season_send["status"] == "dry_run" and len(season_send["links"]) == 1,"season invitation dry-run supplies the actual signup link")
    late = register_late_player(r, s)
    r.close_registration(s)
    review = r.api("POST", "/public/interclub-player-response/review", {"token":secret}, actor=None)
    r.check(not review["can_respond"] and review["can_withdraw"], "closed season personal link preserves review and withdrawal only")
    r.api("POST", "/public/interclub-player-response/respond", {**body, "expected_revision": review["member"]["revision"]}, actor=None, expected=(423,))
    r.check(True, "closed season rejects active signup edits before meet availability opens")
    r.phase("closed registration, dry-run meet invitations and availability")
    availability = root+"/meets/"+m["id"]+"/availability"
    r.api("PUT", availability, {"expected_revision":0,"open":True,"deadline":iso(now()+timedelta(days=1))})
    eroot = f"/admin/clubs/{club}/interclub/player-pools/{s['id']}/emails"
    audience = r.api("GET", eroot+"/audience?kind=meet&meet_id="+m["id"])
    r.check(audience["delivery_mode"] == "dry_run" and audience["send_available"], "meet invitation audience ready in dry-run")
    selected = [c["id"] for c in audience["candidates"] if c["available"]][:3]
    payload = {"kind":"meet","meet_id":m["id"],"recipient_ids":selected,**audience["defaults"]}
    preview = r.api("POST", eroot+"/preview", payload)
    operation = str(uuid4())
    create = {**payload,"operation_key":operation,"preview_fingerprint":preview["preview_fingerprint"]}
    created = r.api("POST", eroot, create)
    r.check(r.api("POST", eroot, create)["operation_key"] == operation, "invitation creation retry is idempotent")
    for i in range(created["recipient_count"]):
        route = eroot+f"/{operation}/recipients/{i}/send"
        sent = r.api("POST", route)
        r.check(sent["status"] == "dry_run" and bool(sent["links"]), "dry-run invitation creates usable private links")
        r.check(r.api("POST", route) == sent, "invitation send retry returns the original result")
    response_rows = r.api("GET", availability)["responses"]
    for row, status in zip(response_rows, ["available","maybe","unavailable"]):
        secret = token(row["response_url"])
        r.secrets.append(secret)
        body = {"token":secret,"action":"respond_meet","expected_revision":row["revision"],"status":status}
        result = r.api("POST", "/public/interclub-player-response/respond", body, actor=None)
        r.api("POST", "/public/interclub-player-response/respond", body, actor=None, expected=(409,))
        r.check(result["availability"]["status"] == status, "player RSVP persists: "+status)
    r.check({row["status"] for row in r.api("GET", availability)["responses"]} == {"available","maybe","unavailable"}, "administrator sees the three RSVP states")
    state = r.api("GET", availability)["settings"]
    r.api("PUT", availability, {"expected_revision":state["revision"],"open":False,"deadline":state["deadline"]})
    review = r.api("POST", "/public/interclub-player-response/review", {"token":secret}, actor=None)
    r.check(not review["can_respond"], "closed RSVP link is read-only")
    return late


def publication(r, s):
    root = f"/admin/clubs/{s['clubs'][0]}/interclub/{s['id']}/publication"
    current = r.api("GET", root)
    if not current["publication"]["revision"]:
        r.api("PUT", root, {"revision":0,"document":{"results":[]}})
    current = r.api("GET", root)
    r.api("POST", root+"/publish", {"revision":current["publication"]["revision"],"preview_fingerprint":current["preview_fingerprint"]})
    return r.api("GET", "/public/interclub/"+s["id"], actor=None)


def full_season(r):
    r.phase("four-club season and mixed participation by skill level")
    s = r.season("full-season", divisions=["3.5","4.0","4.5"], public_signups=True)
    m = s["meets"][0]
    late = email_and_responses(r, s, m)
    for division, count in [("3.5",4),("4.0",3),("4.5",2)]:
        for club in s["clubs"][:count]:
            r.roster(s, m, club, division)
    ownroot = r.registration(s["clubs"][0],s["id"])+"/meets/"+m["id"]
    current = r.api("GET",ownroot)
    team = next(t for t in current["teams"] if t["club_id"] == s["clubs"][0] and t["division"] == "3.5")
    baseline = {"expected_meet_revision":current["meet"]["revision"],"expected_revision":team["revision"],
                "name":"Invalid roster rehearsal","division":"3.5"}
    own = [p for p in s["players"] if p["club_id"] == s["clubs"][0] and p["division"] == "3.5"]
    invalid_sets = [[own[i]["id"] for i in [0,3,4,5]],
                   [p["id"] for p in s["players"] if p["club_id"] == s["clubs"][0] and p["division"] == "4.5"][:4],
                   [p["id"] for p in s["players"] if p["club_id"] == s["clubs"][1] and p["division"] == "3.5"][:4]]
    for ids in invalid_sets:
        r.api("PUT",ownroot+"/teams/"+team["id"],{**baseline,"player_ids":ids},expected=(422,))
    r.check(True,"wrong gender balance, skill band and represented-club rosters rejected")
    nonhostroot = r.competition(s["clubs"][2],s["id"],m["id"])
    hidden = r.api("GET",nonhostroot,actor=3)
    r.check(hidden["lineups_hidden"] and all(t["club_id"] == s["clubs"][2] for t in hidden["teams"]),"opponent lineups stay hidden from non-host before roster deadline")
    r.api("POST",nonhostroot+"/generate",{"expected_revision":0,"format":"gender"},actor=3,expected=(403,))
    # The host participant identity can manage this meet but cannot approve it.
    detail = r.api("GET", r.competition(s["clubs"][1],s["id"],m["id"]), actor=1)
    r.check(detail["can_manage"] and not detail["is_organizer"], "host can manage its meet without organizer approval authority")
    # Enrollment happened before close. Approval remains available after close,
    # without adding this player to the earlier fixture meet's locked lineup.
    poolroot = r.registration(s["clubs"][0],s["id"])+"/pool"
    request = {"member_id":late["id"],"expected_revision":late["revision"],"approve":True,"reason":"Synthetic late player approval"}
    r.api("POST",r.registration(s["clubs"][1],s["id"])+"/pool/approvals",request,actor=1,expected=(403,))
    approved = r.api("POST",poolroot+"/approvals",request)
    r.check(approved["member"]["approval_status"] == "approved","only organizer approves the late addition")
    r.move_meet(s, m, now()-timedelta(days=3))
    b = r.generated(s, m)
    r.check(Counter(e["division"] for e in b["document"]["encounters"]) == {"3.5":6,"4.0":3,"4.5":1}, "four, three and two clubs generate six, three and one matchups")
    r.check(len(games(b["document"])) == 60, "three, six and nine games per player across participation sizes")
    root = r.competition(s["clubs"][0],s["id"],m["id"])
    r.api("POST", root+"/submit", {"expected_revision":b["revision"]}, expected=(422,))
    invalid = deepcopy(b["document"])
    games(invalid)[0].update(status="completed",a=11,b=10,played_at=iso(now()-timedelta(days=3)))
    r.api("PUT", root, {"expected_revision":b["revision"],"document":invalid}, expected=(422,))
    r.api("GET", "/public/interclub/"+s["id"], actor=None, expected=(404,))
    complete = r.complete(b["document"], datetime.fromisoformat(m["starts_at"]))
    b = r.save(s, m, b, complete)
    r.api("PUT", root, {"expected_revision":b["revision"]-1,"document":complete}, expected=(409,))
    r.check(not workspace(r,s)["standings"]["divisions"], "saved drafts do not become official standings")
    b = r.approve(s,m,b)
    rating_evidence(r,s,b,60)
    current = workspace(r,s)
    for div, points in [("3.5",[9,6,3,0]),("4.0",[6,3,0,0]),("4.5",[3,0,0,0])]:
        rows = {row["club_id"]:row for row in current["standings"]["divisions"][div]}
        r.check([rows[c]["points"] for c in s["clubs"]] == points, "correct 3/1/0 standings for "+div)
    source_before = r.db("GET","pcs_interclub_rating_sources",batch_id="eq."+b["id"])[0]
    r.api("POST",root+"/retry-ratings",{"expected_revision":b["revision"]})
    source_after = r.db("GET","pcs_interclub_rating_sources",batch_id="eq."+b["id"])[0]
    r.check(source_before["generation_id"] == source_after["generation_id"], "rating retry does not duplicate a completed generation")
    public = publication(r,s)
    encoded = json.dumps(public)
    r.check(all(secret not in encoded for secret in ["Private revised note","@example.invalid","token_nonce","injury_reason"]), "publication excludes private contact and injury information")
    pubroot = f"/admin/clubs/{s['clubs'][0]}/interclub/{s['id']}/publication"
    old_preview = r.api("GET",pubroot)
    reopened = r.api("POST",root+"/reopen",{"expected_revision":b["revision"],"reason":"Rehearsal paper score correction"})["batch"]
    correction = deepcopy(reopened["document"])
    game = games(correction)[0]
    game["b" if game["a"]>game["b"] else "a"] = 6
    b = r.save(s,m,reopened,correction)
    r.check(r.api("GET","/public/interclub/"+s["id"],actor=None) == public, "correction draft preserves the published results")
    b = r.approve(s,m,b)
    rating_evidence(r,s,b,60)
    r.api("POST",pubroot+"/publish",{"revision":old_preview["publication"]["revision"],"preview_fingerprint":old_preview["preview_fingerprint"]},expected=(409,))
    r.check(r.api("GET","/public/interclub/"+s["id"],actor=None) == public, "approval requires a fresh publication review")
    r.check(publication(r,s) != public, "reviewed correction replaces the public snapshot")
    r.phase("MLP finals, rotating singles and overall Club Cup")
    final = new_meet(r,s,"final",s["clubs"][:2])
    prepare(r,s,final,phase="final")
    fb = None
    for division in s["divisions"]:
        fb = r.generated(s,final,phase="final",division=division,pair=s["clubs"][:2],format="mlp",revision=fb["revision"] if fb else 0)
    fd = r.complete(fb["document"],datetime.fromisoformat(final["starts_at"]),final_tie=True)
    fb = r.approve(s,final,r.save(s,final,fb,fd))
    rating_evidence(r,s,fb,12)
    cup = workspace(r,s)["club_cup"]
    r.check(cup["status"] == "complete" and cup["champions"] == [s["clubs"][0]], "three MLP finals settle the Club Cup")
    top = cup["standings"][0]
    r.check(top["regular_points"] == 18 and top["championship_points"] == 18 and top["points"] == 36, "Cup adds regular points and six championship points per skill level")
    publication(r,s)
    s["browser_meet"] = m["id"]
    r.persist()
    return s


def incidents(r):
    r.phase("injury, substitution, partial teams and weather")
    s = r.season("incidents",club_count=2)
    r.close_registration(s)
    m = s["meets"][0]
    b = prepare(r,s,m)
    doc = r.complete(b["document"],datetime.fromisoformat(m["starts_at"]))
    encounter = doc["encounters"][0]
    women = encounter["pairings"][0]
    women["games"][1].update(status="retired",a=8,b=2,winner="b",injury_reason="Synthetic injury rehearsal")
    eligible = r.api("GET",r.competition(s["clubs"][0],s["id"],m["id"]))["eligible_players"][encounter["club_a"]]
    substitute = next(p["entry_id"] for p in eligible if p["gender"] in {"female","F","f"} and p["entry_id"] not in women["players_a"])
    women["games"][2].update(players_a=[women["players_a"][1],substitute],injury_reason="Eligible replacement after injury")
    b = r.approve(s,m,r.save(s,m,b,doc))
    rating_evidence(r,s,b,5)
    r.check(True,"injury uses the actual score and a following-game eligible replacement")
    for label,partial,mode in [("missing-pairing",[s["clubs"][1]],"normal"),("double-forfeit",s["clubs"],"normal"),
                                ("weather-partial",[],"partial"),("weather-cancelled",[],"cancelled")]:
        m = new_meet(r,s)
        b = prepare(r,s,m,partial=partial)
        # Balance incident results so the fixture squad stays in this skill band
        # while real league ratings carry forward to successive roster cutoffs.
        winner = s["clubs"][1 if label in {"missing-pairing", "weather-partial"} else 0]
        doc = r.complete(b["document"],datetime.fromisoformat(m["starts_at"]),winner=winner)
        if mode != "normal":
            doc["weather"] = "finalized_partial"
            for pairing in doc["encounters"][0]["pairings"][(1 if mode=="partial" else 0):]:
                for game in pairing["games"]:
                    game.update(status="unplayed",a=None,b=None,winner=None,played_at=None)
        expected = sum(g["status"]=="completed" for g in games(doc))
        b = r.approve(s,m,r.save(s,m,b,doc))
        rating_evidence(r,s,b,expected)
        r.check(all(g["a"] is None and g["b"] is None for g in games(doc) if g["status"] in {"forfeit","double_forfeit","unplayed"}),label+" never fabricates scores or rating games")
    # A delay can be saved without prematurely making any score official.
    m = new_meet(r,s)
    b = prepare(r,s,m,mixed=True)
    doc = r.complete(b["document"],datetime.fromisoformat(m["starts_at"]))
    doc["weather"] = "delay"
    for game in doc["encounters"][0]["pairings"][1]["games"][1:]:
        game.update(status="pending",a=None,b=None,winner=None,played_at=None)
    b = r.save(s,m,b,doc)
    old = deepcopy(b["document"]["encounters"][0]["pairings"][0])
    root = r.competition(s["clubs"][0],s["id"],m["id"])
    replay_deadline = now()+timedelta(seconds=20)
    b = r.api("POST",root+"/reschedule",{"expected_revision":b["revision"],"reason":"Synthetic weather replay",
        "starts_at":iso(now()+timedelta(minutes=5)),"roster_deadline":iso(replay_deadline)})["batch"]
    pairs = b["document"]["encounters"][0]["pairings"]
    r.check(pairs[0] == old and all(g["status"]=="pending" and g["a"] is None for g in pairs[1]["games"]),"weather replay preserves completed pairing and resets all three games of unfinished pairing")
    for club in s["clubs"]:
        r.roster(s,m,club,"3.5",alternate=True)
    b = r.api("POST",root+"/refresh-lineups",{"expected_revision":b["revision"]})["batch"]
    r.check(b["document"]["encounters"][0]["pairings"][0] == old,"replay roster refresh leaves completed pairing unchanged")
    # Let the real new cutoff pass: the replay now locks the current league
    # ratings, while the retained pairing still uses its original snapshot.
    while now() <= replay_deadline:
        time.sleep(min(1,max(0.01,(replay_deadline-now()).total_seconds())))
    r.move_meet(s,m,now(),replay_deadline)
    replay = deepcopy(b["document"])
    for game in games(replay):
        if game["status"] == "pending":
            game.update(status="completed",a=11,b=9,winner="a",played_at=iso(now()))
    b = r.approve(s,m,r.save(s,m,b,replay))
    rating_evidence(r,s,b,6)
    r.check(b["document"]["encounters"][0]["pairings"][0] == old,"completed weather replay keeps prior pairing and rates six games exactly once")
    ui_meet = new_meet(r,s)
    ui_batch = prepare(r,s,ui_meet)
    s["browser_meet"] = ui_meet["id"]
    s["browser_batch"] = ui_batch
    r.persist()
    return s


def qualifier(r):
    r.phase("three-way cutoff tie and MLP qualifying round robin")
    s = r.season("qualifying-playoff",club_count=3)
    r.close_registration(s)
    m = s["meets"][0]
    b = prepare(r,s,m)
    b = r.approve(s,m,r.save(s,m,b,r.complete(b["document"],datetime.fromisoformat(m["starts_at"]),cycle=True)))
    q = workspace(r,s)["standings"]["qualification"]["3.5"]
    r.check(q["status"] == "playoff_required" and set(q["playoff_required"]) == set(s["clubs"]),"unresolved three-way tie requires a playoff rather than arbitrary finalists")
    m = new_meet(r,s,"qualifier")
    prepare(r,s,m,phase="qualifier")
    b = None
    for pair in combinations(s["clubs"],2):
        b = r.generated(s,m,phase="qualifier",division="3.5",pair=pair,format="mlp",revision=b["revision"] if b else 0)
    r.check(len(b["document"]["encounters"]) == 3,"qualifying packet contains all three pairwise matchups")
    b = r.approve(s,m,r.save(s,m,b,r.complete(b["document"],datetime.fromisoformat(m["starts_at"]))))
    rating_evidence(r,s,b,12)
    result = workspace(r,s)
    q = result["standings"]["qualification"]["3.5"]
    r.check(q["status"] == "ready" and set(q["qualifiers"]) == set(s["clubs"][:2]),"qualifying results resolve the two championship places")
    r.check(all(row["championship_points"] == 0 for row in result["club_cup"]["standings"]),"qualifying playoff gives no championship bonus")
    return s


def season_adjustments(r):
    """Commissioner exceptions and calendar edits use isolated future fixtures."""
    r.phase("closed registration late requests and upcoming meet changes")
    s = r.season("season-adjustments", club_count=2)
    club, organizer = s["clubs"][1], s["clubs"][0]
    root = r.registration(club, s["id"])
    review_root = r.registration(organizer, s["id"])+"/pool/approvals"
    requests = []
    for label in ("approved", "rejected"):
        player_id = 7000000000000 + int(uuid4().hex[:11], 16)
        r.db("POST", "players", {"id": player_id, "club_id": club,
            "name": "Late request rehearsal "+label, "normalized_name": "late "+s["id"]+" "+label,
            "rating": 1500, "starting_rating": 1500, "active": True, "gender": "female"})
        requests.append({"player_id": player_id, "divisions": ["3.5"], "reason": "Verbal commitment after registration closed: "+label})
    r.api("POST", root+"/pool/late-requests", requests[0], actor=1, expected=(423,))
    r.close_registration(s)
    window = dict(s["registration"])
    pending = r.api("POST", root+"/pool/late-requests", requests[0], actor=1)["member"]
    r.check(pending["approval_status"] == "pending" and pending["late_join"]
            and pending["late_request_reason"] == requests[0]["reason"],
            "post-close request remains pending even before season starts and preserves its explanation")
    r.check(not r.db("GET", "pcs_interclub_entries", season_id="eq."+s["id"], player_id="eq."+str(requests[0]["player_id"])),
            "pending late request cannot seed league participation")
    candidates = r.api("GET", r.competition(club, s["id"], s["meets"][0]["id"]), actor=1)["eligible_players"]
    r.check(sum(len(rows) for rows in candidates.values()) == len(s["players"])
            and all(row["eligibility_rating"] == 3.75 and not row["rating_locked"]
                    and "player_id" not in row and "email" not in row
                    for rows in candidates.values() for row in rows),
            "host meet context batches current ratings and excludes pending requests and private player identifiers")
    r.api("POST", root+"/pool/late-requests", requests[0], actor=1, expected=(409,))
    r.api("POST", r.registration(organizer, s["id"])+"/pool/late-requests", requests[0], actor=1, expected=(403,))
    decision = {"member_id": pending["id"], "expected_revision": pending["revision"],
                "approve": True, "reason": "Commissioner accepts the late commitment"}
    r.api("POST", root+"/pool/approvals", decision, actor=1, expected=(403,))
    approved = r.api("POST", review_root, decision)["member"]
    r.api("POST", review_root, decision, expected=(409,))
    r.check(approved["approval_status"] == "approved" and approved["late_request_reason"] == requests[0]["reason"]
            and approved["approval_reason"] == decision["reason"], "commissioner approval retains separate request and decision reasons")
    rejected = r.api("POST", root+"/pool/late-requests", requests[1], actor=1)["member"]
    rejected = r.api("POST", review_root, {"member_id": rejected["id"], "expected_revision": rejected["revision"],
        "approve": False, "reason": "Commissioner declines this rehearsal request"})["member"]
    r.check(rejected["approval_status"] == "rejected" and not r.db("GET", "pcs_interclub_entries",
        season_id="eq."+s["id"], player_id="eq."+str(requests[1]["player_id"])), "rejected request remains out of league participation")

    meet = s["meets"][0]
    team = r.roster(s, meet, club, "3.5")["team"]
    original_players = [p["player_id"] for p in team["roster"]]
    availability = root+"/meets/"+meet["id"]+"/availability"
    r.api("PUT", availability, {"expected_revision": 0, "open": True, "deadline": iso(now()+timedelta(days=1))}, actor=1)
    eroot = f"/admin/clubs/{club}/interclub/player-pools/{s['id']}/emails"
    audience = r.api("GET", eroot+"/audience?kind=meet&meet_id="+meet["id"], actor=1)
    member_id = next(c["id"] for c in audience["candidates"] if c["available"])
    invitation = {"kind": "meet", "meet_id": meet["id"], "recipient_ids": [member_id], **audience["defaults"]}
    preview = r.api("POST", eroot+"/preview", invitation, actor=1)
    operation = str(uuid4())
    r.api("POST", eroot, {**invitation, "operation_key": operation, "preview_fingerprint": preview["preview_fingerprint"]}, actor=1)
    sent = r.api("POST", eroot+"/"+operation+"/recipients/0/send", actor=1)
    r.check(sent["status"] == "dry_run", "reschedule rehearsal prepares a private RSVP without sending email")
    response = next(row for row in r.api("GET", availability, actor=1)["responses"] if row["member_id"] == member_id)
    old_token = token(response["response_url"])
    r.secrets.append(old_token)
    r.api("POST", "/public/interclub-player-response/respond", {"token": old_token, "action": "respond_meet",
        "expected_revision": response["revision"], "status": "available"}, actor=None)
    earlier = datetime.fromisoformat(meet["starts_at"].replace("Z", "+00:00"))-timedelta(hours=12)
    update = {"expected_revision": meet["revision"], "starts_at": iso(earlier),
        "roster_deadline": iso(earlier-timedelta(hours=1)), "duration_minutes": 180, "courts": meet["courts"]}
    schedule = r.competition(organizer, s["id"])+"/meets/"+meet["id"]+"/schedule"
    r.api("PUT", r.competition(club, s["id"])+"/meets/"+meet["id"]+"/schedule", update, actor=1, expected=(403,))
    changed = r.api("PUT", schedule, update)
    r.api("PUT", schedule, update, expected=(409,))
    meet.update(changed["meet"])
    saved = next(t for t in r.api("GET", root+"/meets/"+meet["id"], actor=1)["teams"] if t["id"] == team["id"])
    r.check([p["player_id"] for p in saved["roster"]] == original_players and saved["revision"] == team["revision"]+1
            and changed["rosters_refreshed"] == 1 and all(not p["rating_locked"] for p in saved["roster"]),
            "moving an upcoming meet earlier preserves selected players in an audited roster revision")
    state = r.api("GET", availability, actor=1)
    r.check(not state["settings"]["open"] and all(row["status"] == "invited" and not row["responded_at"] for row in state["responses"])
            and changed["availability_reset_count"] == 1, "changed meet closes collection and requires fresh availability confirmation")
    r.api("POST", "/public/interclub-player-response/review", {"token": old_token}, actor=None, expected=(404,))
    r.api("PUT", availability, {"expected_revision": state["settings"]["revision"], "open": True,
        "deadline": iso(now()+timedelta(days=1))}, actor=1)
    refreshed = r.api("GET", availability, actor=1)["responses"][0]
    fresh_token = token(refreshed["response_url"])
    r.secrets.append(fresh_token)
    reconfirmed = r.api("POST", "/public/interclub-player-response/respond", {"token": fresh_token, "action": "respond_meet",
        "expected_revision": refreshed["revision"], "status": "available"}, actor=None)
    r.check(reconfirmed["availability"]["status"] == "available", "fresh private response works for the rescheduled meet")

    create = {"request_id": str(uuid4()), "host_club_id": club, "club_ids": s["clubs"],
        "starts_at": iso(now()+timedelta(days=9)), "roster_deadline": iso(now()+timedelta(days=8)),
        "duration_minutes": 180, "courts": 12, "competition_phase": "regular"}
    add_root = r.competition(organizer, s["id"])+"/meets"
    created = r.api("POST", add_root, create)["meet"]
    s["meets"].append(created)
    r.check(r.api("POST", add_root, create)["meet"]["id"] == created["id"], "retrying Add meet creates one scheduled meet")
    r.api("POST", add_root, {**create, "courts": 10}, expected=(409,))
    season = r.api("GET", r.registration(organizer, s["id"]))["season"]
    projected = r.db("GET", "pcs_interclub_seasons", select="details", id="eq."+s["id"])[0]["details"]["meets"]
    r.check(season["registration"] == window and len(projected) == 2
            and datetime.fromisoformat(projected[0]["starts_at"].replace("Z", "+00:00")) == earlier,
            "late approvals and calendar changes preserve closed registration and keep the season schedule current")
    s["browser_late_player"] = "Browser late commitment"
    r.db("POST", "players", {"id": 7000000000000 + int(uuid4().hex[:11], 16), "club_id": organizer,
        "name": s["browser_late_player"], "normalized_name": "browser late "+s["id"],
        "rating": 1500, "starting_rating": 1500, "active": True, "gender": "female"})
    r.persist()


def run(r):
    failures = []
    # Every scenario uses fresh season players, so an assertion in one should
    # not prevent the remaining independent rehearsals from producing evidence.
    for scenario in (full_season, incidents, qualifier, season_adjustments):
        try:
            scenario(r)
        except Exception as exc:
            failure = r.redact(f"{scenario.__name__}: {type(exc).__name__}: {exc}")
            failures.append(failure)
            print("REHEARSAL FAILURE "+failure, flush=True)
    if failures:
        r.phase("scenario failures")
        raise RuntimeError("; ".join(failures))
    # The browser completes score/approval flows on closed seasons, then joins
    # this separate open season without reopening any completed competition.
    r.phase("open registration fixture for anonymous browser signup")
    r.season("browser-signup", club_count=2)
    r.phase("browser handoff")
    r.check(True,"API rehearsal complete; all synthetic sessions reserved for browser verification and cleanup")
