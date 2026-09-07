"""Chronological opponent records shared by the four rivalry badges."""
from __future__ import annotations

import pandas as pd

from jupr_app.domain.gamification.badge_types import BadgeCandidate, BadgeEvaluationContext


def rivalry_candidates(ctx: BadgeEvaluationContext, badge_id: str) -> list[BadgeCandidate]:
    if ctx.facts.empty:
        return []
    facts = ctx.facts.dropna(subset=["player_id", "match_id", "date_dt"]).copy()
    cutoff = pd.to_datetime(ctx.as_of, utc=True) if ctx.as_of else pd.Timestamp.now(tz="UTC")
    facts = facts[facts["date_dt"] <= cutoff].sort_values(["date_dt", "match_id"])
    facts = facts.drop_duplicates(["player_id", "match_id"])
    candidates = []
    for player_id, games in facts.groupby("player_id"):
        records: dict[int, dict] = {}
        found_first = False
        for row in games.itertuples(index=False):
            opponents = getattr(row, "opponent_ids", None)
            if not isinstance(opponents, (list, tuple, set)):
                continue
            opponent_ids = {int(pid) for pid in opponents if str(pid).isdigit() and int(pid) > 0 and int(pid) != int(player_id)}
            for opponent_id in sorted(opponent_ids):
                record = records.setdefault(opponent_id, {"wins": 0, "games": 0, "nemesis": False, "streak": 0})
                was_nemesis = record["nemesis"]
                won = row.win == True
                record["games"] += 1
                record["wins"] += int(won)
                record["streak"] = record["streak"] + 1 if won and was_nemesis else 0
                evidence = {"opponent_id": opponent_id, "head_to_head_matches": record["games"],
                            "head_to_head_wins": record["wins"], "head_to_head_losses": record["games"] - record["wins"]}
                unlocked = []
                if was_nemesis and won:
                    unlocked.append("rivalry_win")
                    if record["streak"] == 3:
                        unlocked.append("rivalry_streak")
                    if record["wins"] * 2 == record["games"]:
                        unlocked.append("settled_the_score")
                if not was_nemesis and record["games"] >= 6 and record["wins"] * 5 <= record["games"] * 2:
                    record["nemesis"] = True
                    if not found_first:
                        unlocked.append("nemesis_found")
                        found_first = True
                if badge_id in unlocked:
                    lifetime = badge_id == "nemesis_found"
                    candidates.append(BadgeCandidate(
                        badge_id=badge_id, player_id=int(player_id), club_id=ctx.club_id,
                        context_type="overall" if lifetime else "opponent",
                        context_id="nemesis_found:lifetime" if lifetime else f"{badge_id}:{opponent_id}:{row.match_id}",
                        match_id=str(row.match_id), value_json=evidence,
                    ))
    return candidates
