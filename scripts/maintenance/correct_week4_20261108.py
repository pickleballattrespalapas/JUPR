"""Owner-approved, idempotent correction of the November 8 Week 4 ladder.

Run only from the protected production maintenance workflow. Uses the existing
atomic match edit and leased replay services; never publishes or sends a recap.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from supabase import create_client

from jupr_app.domain.gamification.badge_policy import badges_enabled
from jupr_app.services.match_edit_durability_service import apply_atomic_match_edits, recover_atomic_match_edit
from jupr_app.services.admin_weekly_recap_service import generate_admin_weekly_recap

CLUB = "tres_palapas"
LEAGUE = "Fall Ladder League 2026"
SESSION = "a50a9f42-7d68-5694-a4f5-d62672020878"
KEY = "owner-approved-week4-date-20261108-v1"
OLD = "2026-11-01"
NEW = "2026-11-08"
# Midnight in the club's America/Mazatlan timezone, not midnight UTC.
STAMP = "2026-11-08T07:00:00+00:00"
ACTOR = {"actor_email": "baumannjoe@yahoo.com", "actor_role": "club_owner", "source": KEY}
IDS = set(range(3131, 3231))
SCORE_FIELDS = ("t1_p1", "t1_p2", "t2_p1", "t2_p2", "score_t1", "score_t2")


def rows(db, table, **filters):
    q = db.table(table).select("*").eq("club_id", CLUB)
    for key, value in filters.items():
        q = q.eq(key, value)
    return q.execute().data or []


def validate(session, rounds, matches):
    assert session["id"] == SESSION and session["league_name"] == LEAGUE
    assert session["week_tag"] == "Week 4" and session["status"] == "complete"
    assert len(rounds) == 2 and {r["round_number"] for r in rounds} == {1, 2}
    assert len(matches) == 100 and {int(m["id"]) for m in matches} == IDS
    by_id = {int(m["id"]): m for m in matches}
    submitted = []
    for r in rounds:
        assert r["session_id"] == SESSION and r["status"] == "submitted"
        assert r["match_date"] in {OLD, NEW}
        ids, source = r["submitted_match_ids"], r["matches_json"]
        assert len(ids) == len(source) == r["submitted_match_count"] == 50
        for mid, saved in zip(ids, source):
            m = by_id[int(mid)]
            assert m["club_id"] == CLUB and m["league"] == LEAGUE and m["week_tag"] == "Week 4"
            assert not m.get("deleted_at") and m["context_type"] == "league_live_session"
            assert m["context_id"] == saved["context_id"]
            assert all(m[k] == saved[k] for k in SCORE_FIELDS)
            assert saved["date"] in {OLD, NEW} and m["date"][:10] in {OLD, NEW}
            submitted.append(int(mid))
    assert len(set(submitted)) == 100 and set(submitted) == IDS


def audit(db, phase, before=None, after=None):
    result = db.table("admin_activity_log").insert({
        "club_id": CLUB, "actor_email": ACTOR["actor_email"], "actor_role": ACTOR["actor_role"],
        "action_type": "correct_league_live_date", "entity_type": "league_live_session",
        "entity_id": SESSION, "source_page": KEY, "before_json": before,
        "after_json": {"phase": phase, **(after or {})},
        "note": "Joe authorized changing Week 4 from November 1 to November 8, 2026, and repairing recap award lookups.",
        "flagged_for_review": True,
    }).execute()
    assert result.data, "Required audit did not persist"


def main():
    assert os.environ.get("OWNER_APPROVAL") == "CORRECT WEEK 4 TO 2026-11-08"
    assert os.environ.get("SUPABASE_URL", "").rstrip("/") == "https://dnoockbwfenunhcibwfn.supabase.co"
    assert not badges_enabled(), "Badges must remain retired"
    db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    sessions = rows(db, "league_live_sessions", id=SESSION)
    assert len(sessions) == 1
    rounds = rows(db, "league_live_rounds", session_id=SESSION)
    matches = rows(db, "matches", league=LEAGUE, week_tag="Week 4")
    validate(sessions[0], rounds, matches)
    original_scores = {m["id"]: [m[k] for k in SCORE_FIELDS] for m in matches}
    operations = rows(db, "match_edit_operations", idempotency_key=KEY)
    assert len(operations) <= 1
    if not operations:
        assert all(m["date"][:10] == OLD and m["row_version"] == 1 for m in matches)
        audit(db, "intent", before={"rounds": rounds, "match_ids": sorted(IDS)})
        result = apply_atomic_match_edits(db, club_id=CLUB,
            patches=[{"id": mid, "date": STAMP, "week_tag": "Week 4"} for mid in sorted(IDS)],
            correction_note="Owner-approved date correction: Week 4 was played November 8, 2026.",
            idempotency_key=KEY, **ACTOR)
        assert result["operation_status"] == "succeeded", result
    elif operations[0]["status"] != "succeeded":
        result = recover_atomic_match_edit(db, club_id=CLUB, operation_id=operations[0]["id"], **ACTOR)
        assert result["operation_status"] == "succeeded", result
    else:
        result = {"operation_id": operations[0]["id"], "replay_job_id": operations[0].get("replay_job_id")}
    for r in rounds:
        if r["match_date"] == NEW and all(m["date"] == NEW for m in r["matches_json"]):
            continue
        saved = [{**m, "date": NEW} for m in r["matches_json"]]
        updated = db.table("league_live_rounds").update({
            "match_date": NEW, "matches_json": saved, "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("club_id", CLUB).eq("id", r["id"]).eq("updated_at", r["updated_at"]).execute().data
        assert updated and len(updated) == 1, "Saved round changed concurrently; rerun to reconcile"
    final_matches = rows(db, "matches", league=LEAGUE, week_tag="Week 4")
    final_rounds = rows(db, "league_live_rounds", session_id=SESSION)
    validate(sessions[0], final_rounds, final_matches)
    assert all(datetime.fromisoformat(m["date"].replace("Z", "+00:00")) == datetime.fromisoformat(STAMP) for m in final_matches)
    assert all(r["match_date"] == NEW and all(m["date"] == NEW for m in r["matches_json"]) for r in final_rounds)
    assert original_scores == {m["id"]: [m[k] for k in SCORE_FIELDS] for m in final_matches}
    audit(db, "date_verified", after={"match_count": 100, "round_count": 2, "date": NEW,
                                      "operation_id": result["operation_id"], "replay_job_id": result.get("replay_job_id")})
    print(json.dumps({"date_corrected": NEW, "verified_games": 100, "verified_rounds": 2, "operation_id": result["operation_id"]}), flush=True)
    drafts = rows(db, "weekly_recaps", week_start="2026-11-02")
    assert len(drafts) == 1 and drafts[0]["week_end"] == "2026-11-08"
    draft = drafts[0]
    assert draft["status"] == "draft", "Do not replace an already published recap"
    refreshed = generate_admin_weekly_recap(db, club_id=CLUB, week_start="2026-11-02", week_end=NEW,
        confirmation_text="GENERATE RECAP", expected_recap_id=draft["id"], expected_row_version=draft["row_version"],
        preserve_edits=True, **ACTOR)
    after = rows(db, "weekly_recaps", week_start="2026-11-02")[0]
    recap = after["generated_json"]
    assert recap["numbers"]["matches"] >= 234
    assert any(LEAGUE in race["title"] for race in recap["report"]["season_races"])
    assert after["edits_json"].get("editorial") == draft["edits_json"].get("editorial")
    assert after["status"] == "draft" and after.get("published_json") == draft.get("published_json")
    print(json.dumps({"recap_refreshed": True, "matches": recap["numbers"]["matches"],
                      "season_races": len(recap["report"]["season_races"]), "published": False}), flush=True)


if __name__ == "__main__":
    main()
