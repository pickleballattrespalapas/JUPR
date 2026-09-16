"""Read-only acceptance checks against the three synthetic staging clubs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

from prepare_multiclub_isolation_fixture import CLUBS, LEAGUE, PROJECT_REF

ORIGIN = "https://juprleagues-api-staging.fly.dev"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def verify() -> dict:
    opener = build_opener(NoRedirect)
    checks = 0

    def get(path: str, expected: int = 200):
        nonlocal checks
        try:
            with opener.open(Request(ORIGIN+path, headers={"Accept": "application/json"}), timeout=30) as response:
                status, raw = response.status, response.read()
        except HTTPError as error:
            status, raw = error.code, error.read()
        if status != expected:
            raise AssertionError(f"{path}: expected {expected}, got {status}")
        checks += 1
        value = json.loads(raw)
        # Club support contacts are public; player-profile contact fixtures are not.
        for prefix in ("shared-", "lr-", "cb-", "lp-"):
            for i in range(1, 25):
                assert f"{prefix}{i:02}@multiclub.example.invalid" not in raw.decode()
        return value

    health = get("/health")
    assert health["environment"] == "staging" and health["supabase_project_ref"] == PROJECT_REF
    assert health["ok"] is True
    states = {}
    for cid, _name, code, _color in CLUBS:
        path = "/clubs/"+cid
        players = get(path+"/players?status=all&limit=100")["players"]
        matches = get(path+"/matches?limit=100")["matches"]
        ids = {int(p["id"]) for p in players}
        assert len(ids) == 24 and len(matches) == 60
        assert sum(p["is_active"] for p in players) == 23
        assert sum(p["matches_played"] for p in players) == 240
        assert all(p["name"].startswith(code+" Test") or p["name"].endswith("[TEST]") for p in players)
        for match in matches:
            assert {int(p["id"]) for p in match["team_1"]+match["team_2"]}.issubset(ids)
        alex = next(p for p in players if p["name"] == "Alex Rivera [TEST]")
        assert get(path+f"/players/{alex['id']}")["player"]["id"] == alex["id"]
        assert len(get(path+f"/players/{alex['id']}/matches")["matches"]) == 12
        get(path+f"/matches/{matches[0]['id']}")
        league = get(path+"/league-results?league_name="+quote(LEAGUE))
        assert league["selected_league"] == LEAGUE
        assert {int(p["player_id"]) for p in league["players"]}.issubset(ids)
        assert len(league["players"]) >= 20
        states[cid] = {"ids": ids, "alex": alex, "match_id": matches[0]["id"]}
    for cid, own in states.items():
        for other_id, other in states.items():
            if cid == other_id:
                continue
            assert own["ids"].isdisjoint(other["ids"])
            get(f"/clubs/{cid}/players/{other['alex']['id']}", 404)
            get(f"/clubs/{cid}/matches/{other['match_id']}", 404)
            assert get(f"/clubs/{cid}/players/{other['alex']['id']}/matches")["matches"] == []
    final_health = get("/health")
    assert final_health["git_commit_sha"] == health["git_commit_sha"], "Staging changed during verification"
    return {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(), "api_origin": ORIGIN,
            "candidate_sha": health["git_commit_sha"], "http_checks": checks, "foreign_object_checks": 18,
            "clubs": [{"id": cid, "players": 24, "active_players": 23, "matches": 60,
                       "shared_player_id": row["alex"]["id"], "shared_player_rating": row["alex"]["rating_jupr"]}
                      for cid, row in states.items()]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify()
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))
