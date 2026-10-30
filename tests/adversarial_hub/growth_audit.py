"""Replay two growing Hub RRs against the real domain, using synthetic data only.

Run with PYTHONPATH=. python tests/adversarial_hub/growth_audit.py --scenario burst
The output separates actual completed-round byes from physical court waiting.
No network, database, authentication, or production writes are involved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random
import time

from jupr_app.domain import adaptive_play_engine as generator
from jupr_app.domain.round_robin_hub import apply_hub_command, build_hub_snapshot

START = datetime(2026, 10, 27, 17, tzinfo=timezone.utc)


def lineup(match):
    return tuple(match.get("sideA") or match.get("teamA") or []), tuple(match.get("sideB") or match.get("teamB") or [])


def finished(match):
    return match.get("scoreA") is not None and match.get("scoreB") is not None


class GrowthAudit:
    def __init__(self, *, courts=10, preview_rounds=3, seed=7, slow_group=False):
        self.rng = random.Random(seed)
        self.minute = 0.0
        self.commands = 0
        self.scored = Counter()
        self.first_start = {}
        self.joined = {}
        self.due = {}
        self.slow_group = slow_group
        self.protected_checks = 0
        self.queue_checks = 0
        self.trace = []
        self.events = {}
        self.hub = {"id": "synthetic-growth-audit", "groups": [], "players": {}, "courts": [
            {"id": f"c{i}", "label": f"Court {i}", "open": True,
             "current": None, "next": None, "dedicated_session_key": None}
            for i in range(1, courts + 1)]}
        for group, count, offset in [("A", 4, 1000), ("B", 6, 2000)]:
            self.events[group] = generator.start_generator_event(generator.create_generator_preview(
                generator_kind="round_robin", play_format="doubles", title=f"Group {group}",
                participant_names=[f"{group} Player {i}" for i in range(1, count + 1)],
                player_ids=list(range(offset + 1, offset + count + 1)), court_count=0,
                total_rounds=preview_rounds))
            self.joined.update({(group, p["id"]): 0 for p in self.events[group]["participants"]})
        for group in self.events:
            self.command("add_group", {"session_key": group})

    @property
    def now(self):
        return (START + timedelta(minutes=self.minute, microseconds=self.commands)).isoformat()

    def matches(self):
        return {(key, m["id"]): m for key, event in self.events.items()
                for row in event["rounds"] for m in row["matches"]}

    def snapshot(self):
        return build_hub_snapshot(self.hub, self.events, now=self.now)

    def command(self, action, payload=None):
        payload = payload or {}
        before = deepcopy(self.matches())
        self.commands += 1
        self.trace.append({"minute": round(self.minute, 3), "action": action, "payload": payload})
        self.hub, self.events = apply_hub_command(self.hub, self.events, action=action, payload=payload, now=self.now)
        # Force persistence/reload serialization; no scheduler memory carries between commands.
        self.hub, self.events = json.loads(json.dumps([self.hub, self.events]))
        after = self.matches()
        for key, old in before.items():
            if old.get("hubStartedAt") or finished(old):
                assert key in after and lineup(old) == lineup(after[key]), ("changed committed lineup", key)
                self.protected_checks += 1
            if finished(old):
                assert (old["scoreA"], old["scoreB"]) == (after[key]["scoreA"], after[key]["scoreB"]), ("changed saved score", key)
        view = self.snapshot()
        commitments = []
        for court in view["courts"]:
            for slot in ("current", "next"):
                match = court.get(slot)
                if match:
                    commitments.extend(p["identity"] for p in match["side_a"] + match["side_b"])
        assert len(commitments) == len(set(commitments)), "double court commitment"
        for group, event in self.events.items():
            assert all(p["name"].startswith(group + " Player ") for p in event["participants"]), "cross-group roster bleed"
            for row in event["rounds"]:
                seated = [p for m in row["matches"] for side in lineup(m) for p in side]
                assert len(seated) == len(set(seated)), ("duplicate seat", group, row["number"])
                assert not set(seated) & set(row["byeParticipantIds"]), ("playing and bye", group, row["number"])
        return view

    def fill(self):
        before = self.snapshot()
        free = {c["id"] for c in before["courts"] if c["open"] and not c["current"] and not c["next"]}
        best = next((m for m in before["eligible_queue"] if free & set(m["allowed_court_ids"])), None)
        view = self.command("auto_fill_courts")
        live = {(c["current"]["session_key"], c["current"]["match_id"]) for c in view["courts"] if c["current"]}
        if best:
            assert (best["session_key"], best["match_id"]) in live, "longest-waiting eligible game was bypassed"
            self.queue_checks += 1
        for court in view["courts"]:
            match = court.get("current")
            if not match:
                continue
            key = (match["session_key"], match["match_id"])
            if key not in self.due:
                duration = self.rng.uniform(7, 13)
                if self.slow_group and key[0] == "B":
                    duration *= 1.8
                self.due[key] = (self.minute + duration, court["id"])
                for person in match["side_a"] + match["side_b"]:
                    self.first_start.setdefault((key[0], person["id"]), self.minute)
        return view

    def arrive(self, group):
        number = len(self.events[group]["participants"]) + 1
        self.command("add_player", {"session_key": group, "name": f"{group} Player {number}",
                                   "player_id": (1000 if group == "A" else 2000) + number})
        person = self.events[group]["participants"][-1]
        self.joined[(group, person["id"])] = self.minute

    def score_next(self):
        assert self.due, "No game can progress even though both groups have available players"
        (group, match), (at, court) = min(self.due.items(), key=lambda entry: entry[1])
        self.minute = max(self.minute, at)
        self.command("score_and_release", {"session_key": group, "match_id": match,
                     "court_id": court, "score_a": 11, "score_b": 7})
        self.due.pop((group, match))
        self.scored[group] += 1

    def report(self, name, elapsed):
        groups = {}
        for key, event in self.events.items():
            completed = [r for r in event["rounds"] if r["status"] in {"saved", "played"}]
            totals = Counter()
            streak = Counter()
            longest = Counter()
            first_repeated = []
            for row in completed:
                byes = set(row["byeParticipantIds"])
                active = {p for m in row["matches"] for side in lineup(m) for p in side} | byes
                totals.update(byes)
                for person in event["participants"]:
                    pid = person["id"]
                    streak[pid] = streak[pid] + 1 if pid in byes else 0
                    longest[pid] = max(longest[pid], streak[pid])
                    if streak[pid] == 2 and len(first_repeated) < 5:
                        first_repeated.append({"name": person["name"], "rounds": [row["number"] - 1, row["number"]],
                                               "roster_count": len(active), "bye_count": len(byes)})
            people = [{"name": p["name"], "id": p["id"], "byes": totals[p["id"]],
                       "max_consecutive_byes": longest[p["id"]],
                       "first_game_wait_minutes": round(self.first_start[(key,p["id"])] - self.joined[(key,p["id"])], 2)
                       if (key,p["id"]) in self.first_start else None} for p in event["participants"]]
            groups[key] = {"started_with": 4 if key == "A" else 6, "final_roster": len(people),
                           "completed_rounds": len(completed), "scored_games": self.scored[key],
                           "max_consecutive_byes": max(longest.values(), default=0),
                           "players_with_repeated_byes": sum(p["max_consecutive_byes"] > 1 for p in people),
                           "first_repeated_byes": first_repeated, "players": people}
        return {"scenario": name, "shared_courts": len(self.hub["courts"]), "commands": self.commands,
                "simulated_minutes": round(self.minute,2), "runtime_seconds": round(elapsed,2),
                "protected_match_checks": self.protected_checks, "queue_priority_checks": self.queue_checks,
                "groups": groups, "passed_bye_fairness": all(g["max_consecutive_byes"] <= 1 for g in groups.values())}


def run_scenario(name, *, target=25, tail_rounds=8):
    started = time.monotonic()
    audit = GrowthAudit(courts=4 if name == "limited_courts" else 10,
                        preview_rounds=1 if name == "single_preview" else 3,
                        slow_group=name in {"limited_courts", "pause_return"})
    arrivals = []
    for group, initial in [("A",4),("B",6)]:
        for n in range(target-initial):
            at = 1 + n*.01 if name in {"burst","single_preview"} else 2 + n*2
            arrivals.append((at,group))
    arrivals.sort()
    availability = [(55,"pause_player","player:1001"),(55,"pause_player","player:2001"),
                    (85,"resume_player","player:1001"),(85,"resume_player","player:2001")] if name == "pause_return" else []
    goals = None
    audit.fill()
    for _ in range(1600):
        due = min((at for at,_ in audit.due.values()), default=float("inf"))
        arrival_at = arrivals[0][0] if arrivals else float("inf")
        availability_at = availability[0][0] if availability else float("inf")
        if arrivals and arrival_at <= min(due, availability_at):
            audit.minute, group = arrivals.pop(0)
            audit.arrive(group)
        elif availability and availability_at <= due:
            audit.minute, action, identity = availability.pop(0)
            audit.command(action,{"identity":identity})
        else:
            audit.score_next()
        audit.fill()
        if not arrivals and not availability:
            if goals is None:
                goals = {key:event["currentRoundNumber"] + tail_rounds for key,event in audit.events.items()}
            if all(audit.events[key]["currentRoundNumber"] >= goal for key,goal in goals.items()) and all(
                (key,p["id"]) in audit.first_start for key,event in audit.events.items() for p in event["participants"]):
                return audit.report(name,time.monotonic()-started)
    raise AssertionError("Growth audit did not finish within 1600 operations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["burst","gradual","limited_courts","pause_return","single_preview"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target", type=int, default=25)
    parser.add_argument("--tail-rounds", type=int, default=8)
    args = parser.parse_args()
    result = run_scenario(args.scenario,target=args.target,tail_rounds=args.tail_rounds)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({**result,"groups":{k:{f:v for f,v in g.items() if f!="players"} for k,g in result["groups"].items()}},sort_keys=True),flush=True)
    raise SystemExit(0 if result["passed_bye_fairness"] else 1)
