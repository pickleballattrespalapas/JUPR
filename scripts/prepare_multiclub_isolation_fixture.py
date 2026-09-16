"""Build an additive, staging-only SQL fixture; never connect or send email.

Apply the generated transaction only to the named staging project. A completed
fixture is a no-op on rerun so manual testing changes are never reset.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import random
import sys
from uuid import NAMESPACE_URL, uuid5

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jupr_app.domain.ratings import calculate_hybrid_elo

PROJECT_REF = "sijpxjxvdtrehmqvirfi"
MARKER = "multiclub-isolation-20260916"
LEAGUE = "Isolation Test Ladder"
CLUBS = (
    ("la-ribera-pickelball-club", "La Ribera Pickelball Club", "LR", "#2563eb"),
    ("cabo-test-club", "Cabo Test Club", "CB", "#d97706"),
    ("la-paz-test-club", "La Paz Test Club", "LP", "#7c3aed"),
)
SHARED_NAMES = ("Alex Rivera", "Sam Torres", "Morgan Cruz", "Jamie Flores", "Taylor Vega", "Casey Luna")


def fixture_id(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"pcs-staging:{MARKER}:{label}"))


def build_fixture(anchor: date = date(2026, 9, 16)) -> dict:
    clubs = []
    for club_number, (club_id, name, code, color) in enumerate(CLUBS):
        players = []
        for i in range(24):
            initial = (3.15 + .05 * (i // 2) + .05 * club_number) if i < 12 else (4.02 + .03 * ((i - 12) // 2) + .03 * club_number)
            players.append({
                "name": f"{SHARED_NAMES[i]} [TEST]" if i < 6 else f"{code} Test Player {i + 1:02}",
                "email": f"shared-{i + 1:02}@multiclub.example.invalid" if i < 6 else f"{code.lower()}-{i + 1:02}@multiclub.example.invalid",
                "gender": "female" if i % 2 == 0 else "male",
                "starting_rating": round(initial * 400, 4),
                "rating": round(initial * 400, 4),
                "wins": 0, "losses": 0, "matches_played": 0,
                "active": i != 23, "last_game_at": None,
            })
        rng = random.Random(20260916 + club_number)
        matches = []
        for round_index in range(12):
            played = anchor - timedelta(days=36 - 7 * (round_index // 2))
            for court in range(5):
                slots = list(range(court * 4, court * 4 + 4))
                rng.shuffle(slots)
                before = [players[i]["rating"] for i in slots]
                losing_score = rng.randint(5, 9)
                scores = [11, losing_score] if rng.randrange(2) else [losing_score, 11]
                d1, d2 = calculate_hybrid_elo(sum(before[:2]) / 2, sum(before[2:]) / 2, *scores)
                at = (datetime.combine(played, time(15), timezone.utc) + timedelta(minutes=round_index % 2 * 30 + court)).isoformat()
                after = []
                for slot, i in enumerate(slots):
                    p = players[i]
                    p["rating"] = round(p["rating"] + (d1 if slot < 2 else d2), 4)
                    won = (scores[0] > scores[1]) == (slot < 2)
                    p["wins" if won else "losses"] += 1
                    p["matches_played"] += 1
                    p["last_game_at"] = at
                    after.append(p["rating"])
                # Compact arrays keep the generated SQL reviewable and small.
                matches.append([at, *slots, *scores, *before, *after, d1, d2, f"{MARKER}:{code}:{round_index}:{court}"])
        clubs.append({
            "id": club_id, "name": name, "code": code, "color": color,
            "invitation_id": fixture_id(f"invite:{club_id}"),
            "tournament_id": fixture_id(f"tournament:{club_id}"),
            "draw_id": fixture_id(f"draw:{club_id}"),
            "team_ids": [fixture_id(f"tournament-team:{club_id}:{i}") for i in range(4)],
            "players": players, "matches": matches,
            "rosters": [
                {"id": fixture_id(f"roster:{club_id}:{meet}:{division}"), "meet": meet, "division": division,
                 "slots": ([0, 1, 2, 3] if meet == 0 else [0, 1, 4, 5]) if division == "3.5" else ([12, 13, 14, 15] if meet == 0 else [12, 13, 16, 17])}
                for meet in range(2) for division in ("3.5", "4.0")
            ],
        })
    club_ids = [c[0] for c in CLUBS]
    start = anchor + timedelta(days=17)
    rules = {"3.5": {"min_rating": None, "max_rating": 3.999, "women_required": 2}, "4.0": {"min_rating": None, "max_rating": 4.499, "women_required": 2}}
    draft = {
        "name": "Three Club Isolation Test", "start_date": start.isoformat(),
        "end_date": (start + timedelta(days=28)).isoformat(), "timezone": "America/Mazatlan",
        "divisions": ["3.5", "4.0"], "club_ids": club_ids, "registration_rules": rules, "setup_step": 4,
        "meets": [{"host_club_id": club_ids[i], "club_ids": club_ids,
                   "starts_at": datetime.combine(start + timedelta(days=i * 14), time(15), timezone.utc).isoformat(),
                   "duration_minutes": 180, "courts": 4} for i in range(2)],
    }
    return {"marker": MARKER, "project_ref": PROJECT_REF, "league": LEAGUE,
            "season_id": fixture_id("season"), "draft": draft, "clubs": clubs}


def render_sql(project_ref: str, anchor: date = date(2026, 9, 16)) -> str:
    if project_ref != PROJECT_REF:
        raise ValueError("Only the isolated staging project is permitted.")
    template = (ROOT / "scripts/fixtures/multiclub_isolation.sql").read_text()
    return template.replace("__FIXTURE_PAYLOAD__", json.dumps(build_fixture(anchor), separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render_sql(args.project_ref))
    print(json.dumps({"output": str(args.output), "project_ref": PROJECT_REF, "clubs": 3, "players": 72, "matches": 180}))


if __name__ == "__main__":
    main()
