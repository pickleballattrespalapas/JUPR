"""Pure, conservative historical repair planning. No database client or writes.

Existing awards are evidence, not a cache to delete and rebuild. Only proven
losing-side Hall of Fame awards and redundant lifetime participation awards are
proposed for revocation. Ambiguous/old-rule awards stay in the review list.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from types import SimpleNamespace
from typing import Any

import pandas as pd
from jupr_app.domain.gamification.award_identity import award_key

from jupr_app.domain.gamification.badge_engine import compute_candidates_for_club

LIFETIME_PARTICIPATION = {"participant", "dedicated_participant_50", "lifetime_participant_200"}
SAFE_BACKFILLS = LIFETIME_PARTICIPATION | {"first_win", "social_butterfly", "network_builder", "high_roller", "legendary_upset", "level_up"}
SINGLE_LIFETIME = SAFE_BACKFILLS - {"legendary_upset", "level_up"}


def fingerprint(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()




def build_reconciliation_plan(snapshot: dict, *, club_id: str, as_of: str) -> dict:
    """Plan from a complete, tenant-scoped snapshot, without mutating it."""
    for table, rows in snapshot.items():
        if table == "badges":
            continue
        if any(row.get("club_id") not in (None, club_id) for row in rows):
            raise ValueError(f"Mixed club snapshot: {table}")
    awards = snapshot["player_badges"]
    active = [row for row in awards if not row.get("revoked_at")]
    ctx = SimpleNamespace(club_id=club_id, df_players_all=pd.DataFrame(snapshot["players"]),
        df_leagues=pd.DataFrame(snapshot["league_ratings"]), df_matches=pd.DataFrame(snapshot["matches"]),
        df_meta=pd.DataFrame(snapshot["leagues_metadata"]), df_badges=pd.DataFrame(snapshot["badges"]),
        df_player_badges=pd.DataFrame(awards))
    catalog_ids = {row["badge_id"] for row in snapshot["badges"]}
    candidates = [asdict(c) for c in compute_candidates_for_club(club_id, ctx=ctx, as_of=pd.Timestamp(as_of).to_pydatetime(), strict=True) if c.badge_id in catalog_ids]
    matches = {str(row["id"]): row for row in snapshot["matches"]}
    revocations, review = [], []
    grouped = defaultdict(list)
    for row in active:
        grouped[award_key(row)].append(row)
        if row["badge_id"] == "hall_of_fame_night":
            match = matches.get(str(row.get("match_id")))
            if match:
                pid = int(row["player_id"])
                team = 1 if pid in (match.get("t1_p1"), match.get("t1_p2")) else 2 if pid in (match.get("t2_p1"), match.get("t2_p2")) else None
                try:
                    s1, s2 = int(match["score_t1"]), int(match["score_t2"])
                    lost = (team == 1 and s1 < s2) or (team == 2 and s2 < s1)
                except (TypeError, ValueError, KeyError):
                    lost = False
                if lost:
                    revocations.append({"row": row, "reason": "Hall of Fame requires a win; recorded match proves this player lost."})
    for key, rows in grouped.items():
        if key[1] in LIFETIME_PARTICIPATION and len(rows) > 1:
            # Prefer a dated award, then the earliest date. Preserve its ID/time.
            ordered = sorted(rows, key=lambda r: (not bool(r.get("earned_at")), str(r.get("earned_at") or ""), str(r.get("id") or fingerprint(r))))
            for row in ordered[1:]:
                revocations.append({"row": row, "reason": "Duplicate lifetime participation milestone; earliest award retained.", "retained_row": ordered[0]})
    # Include revoked keys too: a backfill must never resurrect a staff revocation.
    known_keys = {award_key(row) for row in awards}
    candidate_keys = {award_key(row) for row in candidates}
    additions = []
    for candidate in candidates:
        key = award_key(candidate)
        if candidate["badge_id"] not in SAFE_BACKFILLS or key in known_keys:
            continue
        known_keys.add(key)
        match = matches.get(str(candidate.get("match_id")))
        # Aggregate counters do not establish an exact historical earning date.
        basis = "recorded_match" if match else "eligibility_verified_at"
        candidate["earned_at"] = str(match["date"]) if match else as_of
        candidate["value_json"] = dict(candidate.get("value_json") or {}, reconciliation_version="badge-repair-2026-09-v1", earned_time_basis=basis)
        additions.append(candidate)
    revoked_fingerprints = {fingerprint(item["row"]) for item in revocations}
    for row in active:
        if fingerprint(row) in revoked_fingerprints:
            continue
        if row["badge_id"] in {"level_up", "high_roller", "most_improved_monthly", "upset_champion", "clean_sweep_week"} and award_key(row) not in candidate_keys:
            review.append({"row": row, "reason": "Current rules do not reproduce this award. Historical peak, earlier rule, or period result needs review; keep awarded."})
    def counts(items, nested=False):
        return dict(sorted(Counter((item["row"] if nested else item)["badge_id"] for item in items).items()))
    return {"version": "badge-repair-2026-09-v1", "club_id": club_id, "as_of": as_of,
        "snapshot_sha256": fingerprint(snapshot), "additions": additions, "revocations": revocations, "review": review,
        "summary": {"existing_rows": len(awards), "active_rows": len(active), "additions": counts(additions),
                    "revocations": counts(revocations, True), "review": counts(review, True), "trophy_changes": 0}}


def simulate_reconciliation(snapshot: dict, plan: dict) -> dict:
    """Check exact source state and simulate soft revocations/additions for review."""
    if fingerprint(snapshot) != plan["snapshot_sha256"]:
        raise ValueError("Snapshot changed; regenerate and review the plan.")
    result = deepcopy(snapshot)
    rows = result["player_badges"]
    for index, change in enumerate(plan["revocations"]):
        matches = [row for row in rows if fingerprint(row) == fingerprint(change["row"])]
        if len(matches) != 1:
            raise ValueError("Revocation does not identify exactly one award")
        matches[0].update(revoked_at=plan["as_of"], revoke_reason=change["reason"])
    for index, candidate in enumerate(plan["additions"]):
        rows.append(dict(candidate, id=f"simulation-{index}", revoked_at=None, rule_version=plan["version"], awarded_by="engine"))
    return result
