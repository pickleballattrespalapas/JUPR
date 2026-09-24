"""Preview or append the reviewed historical badge awards after deployment."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json

from jupr_app.data.client import make_supabase
from jupr_app.data.paged_reads import read_all_rows
from jupr_app.domain.gamification.award_identity import award_key
from jupr_app.domain.gamification.badge_engine import compute_candidates_for_club
from jupr_app.domain.gamification.badge_worker import _resolve_context
from jupr_app.domain.gamification.badges_repo import upsert_player_badges
from jupr_app.domain.gamification.program_badge_service import reconcile_program_badges
from jupr_app.services.production_feature_policy import production_feature_enabled
from scripts.preview_badge_reactivation import RESTORED_AUTOMATIC

ACTIVATION = "tres_operations_badges_20260924"


def select_additions(candidates, existing):
    known = {award_key(row) for row in existing}  # Revocations remain authoritative.
    additions = []
    for candidate in candidates:
        key = award_key(asdict(candidate))
        if candidate.badge_id in RESTORED_AUTOMATIC and key not in known:
            known.add(key)
            additions.append(candidate)
    return additions


def activate(db, *, apply=False):
    club_id = "tres_palapas"
    if not production_feature_enabled("badges", club_id):
        raise PermissionError("The reviewed production badge policy is not active.")
    completed = db.table("worker_run_log").select("id").eq("club_id", club_id).eq("worker_name", ACTIVATION).eq("status", "success").limit(1).execute().data or []
    if apply and completed:
        return {"ok": True, "already_applied": True, "activation": ACTIVATION}
    existing = read_all_rows(lambda: db.table("player_badges").select("*").eq("club_id", club_id), order="id")
    context = _resolve_context(None, db, club_id, 5000)
    candidates = compute_candidates_for_club(club_id, ctx=context, strict=True)
    additions = select_additions(candidates, existing)
    program = reconcile_program_badges(db, club_id, dry_run=True)
    known = {award_key(row) for row in existing}
    program_additions = [row for row in program["awards"] if award_key(row) not in known]
    summary = {"activation": ACTIVATION, "club_id": club_id, "apply": apply,
               "existing_awards": len(existing), "classic_additions": len(additions),
               "classic_by_badge": dict(Counter(c.badge_id for c in additions)),
               "program_additions": len(program_additions), "program_review_items": len(program["review"])}
    if not apply:
        return {"ok": True, **summary}
    log = db.table("worker_run_log").insert({"worker_name": ACTIVATION, "club_id": club_id,
        "status": "started", "summary_json": summary}).execute().data
    if not log:
        raise RuntimeError("Activation audit could not be recorded. No awards were changed.")
    run_id = log[0]["id"]
    try:
        inserted = upsert_player_badges(db, club_id, additions, awarded_by=ACTIVATION, rule_version=ACTIVATION)
        program_result = reconcile_program_badges(db, club_id)
        after = read_all_rows(lambda: db.table("player_badges").select("*").eq("club_id", club_id), order="id")
        by_id = {row["id"]: row for row in after}
        if any(by_id.get(row["id"]) != row for row in existing):
            raise RuntimeError("Existing badge records changed during activation; review the recorded run.")
        summary.update(ok=True, classic_inserted=len(inserted), program_result=program_result,
                       existing_awards_unchanged=True, total_awards_after=len(after))
        db.table("worker_run_log").update({"status": "success", "finished_at": datetime.now(timezone.utc).isoformat(),
            "summary_json": summary}).eq("id", run_id).execute()
        return summary
    except Exception:
        db.table("worker_run_log").update({"status": "failed", "finished_at": datetime.now(timezone.utc).isoformat(),
            "error_text": "Badge activation did not complete; inspect the existing run before retrying."}).eq("id", run_id).execute()
        raise


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    db = make_supabase(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    print(json.dumps(activate(db, apply=args.apply), sort_keys=True))
