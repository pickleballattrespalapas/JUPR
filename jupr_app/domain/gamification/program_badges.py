"""Pure program award calculation over one transactionally consistent club snapshot.

The database revision guards both snapshot application and admin tie decisions.
Unknown identities/dates never turn into guessed credits. The same evaluator is
used for previews, historical credit, normal play, and result corrections.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from copy import deepcopy
import json
from typing import Any

from jupr_app.domain.gamification.program_badge_catalog import PROGRAM_BADGES, RULE_VERSION
from jupr_app.domain.live_beta_engine import round_robin_matches, round_robin_standings, round_robin_result_fingerprint, round_robin_players_of_record


def timestamp(value: Any) -> str | None:
    try:
        text = str(value)
        # A recorded calendar date is valid evidence at date precision (UTC).
        result = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError):
        return None


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def positive_id(value: Any) -> int | None:
    try:
        if isinstance(value, bool) or str(int(value)) != str(value):
            return None
        result = int(value)
        return result if result > 0 else None
    except (ValueError, TypeError):
        return None


def played(a: Any, b: Any, result_type: Any = None) -> bool:
    if str(result_type or "PLAYED").upper() not in {"PLAYED", "NORMAL", "STANDARD"}:
        return False
    return (isinstance(a, int) and not isinstance(a, bool) and isinstance(b, int) and not isinstance(b, bool)
            and a >= 0 and b >= 0 and a != b)


def event_from_state(state: dict) -> dict:
    return state.get("event") or (state.get("page_state") or {}).get("event") or {}


def evaluate_program_badges(snapshot: dict) -> dict:
    club = str(snapshot["club_id"])
    now = timestamp(snapshot.get("as_of")) or datetime.now(timezone.utc).isoformat()
    players = {int(p["id"]): p for p in snapshot.get("players", []) if str(p.get("club_id", club)) == club}
    awards: list[dict] = []
    review: list[dict] = []
    ties: list[dict] = []
    definitions = {b.id: b for b in PROGRAM_BADGES}
    # source -> earned date, stored separately for each player and achievement.
    credits: dict[str, dict[int, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    pair_results: dict[tuple[int, int], dict[str, tuple[str, bool]]] = defaultdict(dict)
    finalizations = {(r["source_type"], str(r["source_id"])): r for r in snapshot.get("finalizations", [])}

    def hold(source: str, reason: str):
        review.append({"source": source, "reason": reason})

    def date(value: Any) -> str | None:
        result = timestamp(value)
        return result if result and result <= now else None

    def award(bid: str, pid: int, context: str, when: str, evidence: dict):
        if pid not in players:
            return
        definition = definitions[bid]
        detail = evidence.get("detail") or definition.requirement
        awards.append({"player_id": pid, "badge_id": bid, "context_type": "overall", "context_id": context,
                       "earned_at": when, "value_num": definition.threshold,
                       "value_json": {"rule_version": RULE_VERSION, "requirement": definition.requirement,
                                      "title": definition.name, "tape_excerpt": detail, **evidence}})

    def result(key: str, when: Any, team_a: list, team_b: list, a: Any, b: Any, result_type=None):
        earned = date(when)
        if not earned or not played(a, b, result_type):
            return
        members = [positive_id(p) for p in team_a + team_b]
        if None in members or len(members) != len(set(members)) or not all(p in players for p in members):
            return
        if len(team_a) != 2 or len(team_b) != 2:
            return
        for team, won in ((team_a, a > b), (team_b, b > a)):
            pair = tuple(sorted(int(p) for p in team))
            pair_results[pair][key] = (earned, won)

    tournaments = {str(t["id"]): t for t in snapshot.get("tournaments", [])}
    games = {str(g["id"]): g for g in snapshot.get("tournament_games", [])}
    teams = {str(t["id"]): t for t in snapshot.get("tournament_teams", [])}
    draws = {str(d["id"]): d for d in snapshot.get("tournament_event_draws", [])}
    canonical = [m for m in snapshot.get("matches", []) if not m.get("deleted_at")]
    excluded_game_ids = {str(m["tournament_game_id"]) for m in snapshot.get("matches", [])
                         if m.get("deleted_at") and m.get("tournament_game_id")}
    # The competition parent owns partnership credit, even when several rating
    # game rows have been published. A missing parent is held, never guessed.
    for m in canonical:
        if m.get("tournament_game_id"):
            if str(m["tournament_game_id"]) not in games:
                hold(f"match:{m['id']}", "Tournament competition result is missing.")
            continue
        if m.get("tournament_id"):
            hold(f"match:{m['id']}", "Tournament result has no competition-match linkage for deduplication.")
            continue
        result(f"match:{m['id']}", m.get("date"), [m.get("t1_p1"), m.get("t1_p2")],
               [m.get("t2_p1"), m.get("t2_p2")], m.get("score_t1"), m.get("score_t2"))

    played_events: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for gid, g in games.items():
        if g.get("series_parent_game_id") or str(g.get("stage", "")).upper() == "SERIES_GAME":
            continue
        if gid in excluded_game_ids or not date(g.get("finalized_at")) or not played(g.get("score_a"), g.get("score_b"), g.get("result_type")):
            continue
        ta, tb = teams.get(str(g.get("team_a_id")), {}), teams.get(str(g.get("team_b_id")), {})
        tid = str(g.get("tournament_id"))
        if not ta or not tb or any(str(t.get("tournament_id")) != tid for t in (ta, tb)):
            hold(f"tournament_game:{gid}", "Missing or inconsistent team identities.")
            continue
        a = [p for p in (ta.get("player1_id"), ta.get("player2_id")) if p]
        b = [p for p in (tb.get("player1_id"), tb.get("player2_id")) if p]
        winner = str(g.get("winner_team_id") or "")
        expected = str(g.get("team_a_id") if g["score_a"] > g["score_b"] else g.get("team_b_id"))
        if winner != expected:
            hold(f"tournament_game:{gid}", "Winner and final scores disagree.")
            continue
        children = [c for c in games.values() if str(c.get("series_parent_game_id")) == gid]
        if any(str(c["id"]) in excluded_game_ids for c in children):
            hold(f"tournament_game:{gid}", "A rating game was excluded; result needs review.")
            continue
        result(f"tournament_game:{gid}", g["finalized_at"], a, b, g["score_a"], g["score_b"], g.get("result_type"))
        draw = draws.get(str(g.get("draw_id")), {})
        event_id = str(g.get("event_option_id") or draw.get("event_option_id") or "")
        if not event_id:  # legacy single-event tournaments can count participation
            event_id = "legacy"
        for team in (ta, tb):
            if team.get("retired_at") or str(team.get("competition_status", "")).upper() in {"WITHDRAWN", "RETIRED"}:
                continue
            for p in (team.get("player1_id"), team.get("player2_id")):
                if p in players:
                    played_events[tid][p].add(event_id)

    for tid, t in tournaments.items():
        if str(t.get("status", "")).upper() not in {"COMPLETED", "ARCHIVED"}:
            continue
        closure = finalizations.get(("tournament", tid), {})
        earned = date(closure.get("completed_at"))
        if not earned:
            hold(f"tournament:{tid}", "No verified tournament completion date.")
            continue
        for pid in played_events[tid]:
            credits["tournaments"][pid][f"tournament:{tid}"] = earned
        medals: dict[int, dict[str, dict]] = defaultdict(dict)
        for podium in snapshot.get("tournament_podium", []):
            if str(podium.get("tournament_id")) != tid or podium.get("placement") not in {1, 2, 3}:
                continue
            team = teams.get(str(podium.get("team_id")), {})
            draw = draws.get(str(podium.get("draw_id")), {})
            event_id = str(team.get("event_option_id") or draw.get("event_option_id") or "")
            if not event_id or str(team.get("tournament_id")) != tid:
                hold(f"podium:{podium['id']}", "Medal lacks a distinct competition event identity.")
                continue
            for pid in (team.get("player1_id"), team.get("player2_id")):
                if pid in players and event_id in played_events[tid].get(pid, set()):
                    old = medals[pid].get(event_id)
                    if old is None or podium["placement"] < old["placement"]:
                        medals[pid][event_id] = {"event_id": event_id, "event_name": draw.get("name") or event_id,
                                                  "placement": podium["placement"], "podium_id": str(podium["id"])}
        for pid, events in medals.items():
            if len(events) >= 3:
                award("triple_crown", pid, f"tournament:{tid}", earned,
                      {"tournament_id": tid, "tournament_name": t.get("name"), "medals": list(events.values()),
                       "detail": f"Medals in {len(events)} events at {t.get('name') or 'this tournament'}."})

    for league in snapshot.get("leagues_metadata", []):
        lid = str(league["id"])
        if str(league.get("status", "")).lower() != "ended":
            continue
        closure = finalizations.get(("league", lid), {})
        earned, minimum = date(closure.get("completed_at")), closure.get("min_games")
        if not earned or minimum is None or not isinstance(minimum, int) or minimum < 0:
            hold(f"league:{lid}", "Missing completion date or required minimum games at closure.")
            continue
        counts: dict[int, int] = defaultdict(int)
        for match in canonical:
            match_date = date(match.get("date"))
            if (str(match.get("league")) != str(league.get("league_name")) or not match_date or match_date > earned
                    or not played(match.get("score_t1"), match.get("score_t2"))):
                continue
            for pid in set(match.get(key) for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")):
                if pid in players:
                    counts[pid] += 1
        for pid, count in counts.items():
            if count >= max(1, minimum):
                credits["leagues"][pid][f"league:{lid}"] = earned

    # Saved, moderated social events are the durable source. Their transient live
    # session is not a second event. Public quick/private sessions never qualify.
    saved = {str(e.get("source_event_uid")): e for e in snapshot.get("live_events", [])}
    sessions = {str(event_from_state(s.get("state") or {}).get("sourceEventUid")): s for s in snapshot.get("live_sessions", [])}
    decisions = {d["source_key"]: d for d in snapshot.get("decisions", [])}
    for uid in sorted(set(saved) | set(sessions)):
        social, session = saved.get(uid), sessions.get(uid)
        identity = "live:" + uid
        if uid in {"", "None"}:
            continue
        if social:
            if social.get("status") != "saved":
                continue
            event = social.get("raw_event_json") or {}
            earned = date(social.get("event_date"))
            # Use the normalized, persisted mapping; never match names here.
            links = {str(p["participant_key"]): p.get("linked_player_id") for p in snapshot.get("live_event_participants", [])
                     if str(p.get("event_id")) == str(social["id"])}
        elif session and not str(session.get("source", "")).startswith("public"):
            if session.get("status") != "completed":
                continue
            event = event_from_state(session.get("state") or {})
            earned = date(session.get("completed_at") or finalizations.get(("round_robin", str(session["id"])), {}).get("completed_at"))
            links = {str(p["id"]): p.get("player_id") for p in event.get("participants", [])}
        else:
            continue
        if event.get("type") != "round_robin":
            continue
        display_proof = round_robin_result_fingerprint(event)
        event = deepcopy(event)
        # Inject persisted identity links before materializing recorded substitutes.
        for participant in event.get("participants", []):
            participant["player_id"] = links.get(str(participant["id"]))
        event = round_robin_players_of_record(event)
        links = {str(p["id"]): p.get("player_id") for p in event.get("participants", [])}
        if not earned:
            hold(identity, "Missing recorded round-robin completion date.")
            continue
        matches = [m for m in round_robin_matches({**event, "rounds": [r for r in event.get("rounds", []) if str(r.get("status", "")).lower() not in {"skipped", "cancelled", "canceled"}]}) if str(m.get("status", "")).lower() not in {"cancelled", "canceled", "bye"}]
        if not matches or len({m.get("id") for m in matches}) != len(matches) or any(not m.get("id") or not played(m.get("scoreA"), m.get("scoreB")) for m in matches):
            hold(identity, "Round robin has missing, invalid, or unfinished assigned results.")
            continue
        # Check all participants including guests for the winner. A linked runner
        # up must never inherit an unlinked guest's victory.
        participants = {str(p["id"]): p for p in event.get("participants", [])}
        linked = {key: positive_id(value) for key, value in links.items()}
        linked_values = [v for v in linked.values() if v is not None]
        if len(set(linked_values)) != len(linked_values):
            hold(identity, "The same player is linked to multiple participants.")
            continue
        if any(len(m.get("teamA", [])) not in (1, 2) or len(m.get("teamB", [])) != len(m.get("teamA", []))
               or len(set(m["teamA"] + m["teamB"])) != len(m["teamA"] + m["teamB"])
               or not all(str(p) in participants for p in m["teamA"] + m["teamB"]) for m in matches):
            hold(identity, "Round robin has invalid participant assignments.")
            continue
        if social:
            normalized = [m for m in snapshot.get("live_event_matches", []) if str(m.get("event_id")) == str(social["id"])]
            recorded = {str(m.get("match_key")): m for m in normalized}
            if len(recorded) != len(matches) or any(str(m["id"]) not in recorded or
                    (m["scoreA"], m["scoreB"]) != (recorded[str(m["id"])].get("score_t1"), recorded[str(m["id"])].get("score_t2")) for m in matches):
                hold(identity, "Saved match records and the final round-robin result disagree or are incomplete.")
                continue
        official = (session.get("state") or {}).get("official_publish") or {} if session else {}
        published = set(official.get("published_live_match_ids") or official.get("published_match_ids") or [])
        if published:
            mapping = official.get("match_context_by_live_id") or {}
            by_context = {str(m.get("context_id")): m for m in canonical}
            conflict = False
            for m in matches:
                if str(m["id"]) not in published:
                    continue
                record = by_context.get(str(mapping.get(str(m["id"]))))
                if not record or (m["scoreA"], m["scoreB"]) != (record.get("score_t1"), record.get("score_t2")):
                    conflict = True
                    break
            if conflict:
                hold(identity, "Published match records need to be reconciled with the final round-robin results.")
                continue
        partners: dict[int, set[int]] = defaultdict(set)
        for m in matches:
            a, b = [linked.get(str(p)) for p in m["teamA"]], [linked.get(str(p)) for p in m["teamB"]]
            for pid in a + b:
                if pid in players:
                    credits["round_robins"][pid][identity] = earned
            # Social records are not rating projections. Official live sessions
            # use published canonical matches for lifetime pair totals.
            if (social and not event.get("official_context")) or (not social and str(m["id"]) not in published):
                result(f"{identity}:{m['id']}", earned, a, b, m["scoreA"], m["scoreB"])
            winning = a if m["scoreA"] > m["scoreB"] else b
            if len(winning) == 2 and all(p in players for p in winning):
                partners[winning[0]].add(winning[1])
                partners[winning[1]].add(winning[0])
        for pid, partner_ids in partners.items():
            if len(partner_ids) >= 5:
                award("five_winning_partners", pid, identity, earned,
                      {"event_name": event.get("name"), "partner_ids": sorted(partner_ids),
                       "detail": f"Won with {len(partner_ids)} different partners in {event.get('name') or 'this round robin'}."})
        standings = round_robin_standings({**event, "rounds": [{"matches": matches}], "round_robin_winner": None})
        leaders = [s for s in standings if s["rank"] == 1]
        evidence = {"matches": matches, "links": linked, "standings": standings, "earned_at": earned}
        proof = fingerprint(evidence)
        winner = linked.get(str(leaders[0]["participantId"])) if len(leaders) == 1 else None
        if len(leaders) > 1:
            eligible = [{"participant_id": str(s["participantId"]), "player_id": linked.get(str(s["participantId"])), "name": s["name"],
                         "wins": s["wins"], "differential": s["differential"], "points": s["pointsFor"]} for s in leaders]
            decision = decisions.get(identity, {})
            if decision.get("source_fingerprint") == proof and decision.get("winner_player_id") in {p["player_id"] for p in eligible}:
                winner = decision["winner_player_id"]
            else:
                ties.append({"source_key": identity, "source_fingerprint": proof, "event_name": event.get("name") or "Round robin",
                             "completed_at": earned, "leaders": eligible, "display_fingerprint": display_proof})
        if winner in players:
            credits["round_robin_wins"][winner][identity] = earned
        elif len(leaders) == 1:
            hold(identity, "Winner is not linked to a player at this club.")

    for series, people in credits.items():
        thresholds = (1, 5, 10, 25, 50) if series == "round_robin_wins" else (5, 10, 25)
        for pid, sources in people.items():
            ordered = sorted(sources.items(), key=lambda row: (row[1], row[0]))
            for n in thresholds:
                if len(ordered) >= n:
                    bid = f"{series}_{n}" if series == "round_robin_wins" else f"{series}_completed_{n}"
                    award(bid, pid, "club_lifetime", ordered[n-1][1], {"sources": [k for k, _ in ordered[:n]], "milestone": n})
    for pair, sources in pair_results.items():
        for series, wins_only in (("matches", False), ("wins", True)):
            ordered = sorted(((k, t) for k, (t, win) in sources.items() if win or not wins_only), key=lambda row: (row[1], row[0]))
            for n in (10, 25, 50):
                if len(ordered) < n:
                    continue
                for pid, partner in (pair, pair[::-1]):
                    award(f"{series}_together_{n}", pid, f"pair:{pair[0]}:{pair[1]}", ordered[n-1][1],
                          {"partner_id": partner, "partner_name": players[partner].get("name"), "milestone": n,
                           "sources": [k for k, _ in ordered[:n]],
                           "detail": f"{n} {series} with {players[partner].get('name') or f'Player {partner}'}."})
    awards.sort(key=lambda a: (a["player_id"], a["badge_id"], a["context_id"]))
    return {"club_id": club, "revision": snapshot["revision"], "rule_version": RULE_VERSION,
            "awards": awards, "pending_ties": ties, "review": review,
            "counts": {"awards": len(awards), "players": len({a['player_id'] for a in awards}),
                       "pending_ties": len(ties), "review": len(review)}}
