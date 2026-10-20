"""Pure court allocation for existing generator sessions.

The session owns all match results and standings.  The hub owns only court
assignments and player availability.  Callers persist both together atomically.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone, timedelta
from typing import Any

from jupr_app.domain.adaptive_play_engine import (
    advance_generator_event, mark_generator_round_played, save_generator_round,
)
from jupr_app.domain.moneyball_expectations import is_moneyball, validate_score


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def player_identity(player: dict[str, Any], session_key: str = "", hub: dict[str, Any] | None = None) -> str:
    """Names are labels; only a profile or an immutable roster entry identifies a person."""
    if hub is not None:
        identity = (hub.get("participant_identities", {}).get(session_key) or {}).get(str(player["id"]))
        if identity:
            return identity
    if player.get("player_id") is not None:
        return f"player:{int(player['player_id'])}"
    if not session_key or not player.get("id"):
        raise ValueError("A guest needs an immutable session participant ID.")
    return f"guest:{session_key}:{player['id']}"


def _name_key(name: Any) -> str:
    # Match the database exactly, independently of its Unicode locale/version.
    whitespace = "\t\n\v\f\r\x1c\x1d\x1e\x1f\x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "".join(chr(n) for n in (*range(0xC0, 0xD7), *range(0xD8, 0xDF)))
    lowercase = "abcdefghijklmnopqrstuvwxyz" + "".join(chr(n + 32) for n in (*range(0xC0, 0xD7), *range(0xD8, 0xDF)))
    translated = str(name or "").translate(str.maketrans(uppercase + whitespace, lowercase + " " * len(whitespace)))
    return " ".join(part for part in translated.split(" ") if part)


def _ref(value: dict[str, Any]) -> tuple[str, str]:
    return str(value.get("session_key") or ""), str(value.get("participant_id") or "")


def _ref_json(ref: tuple[str, str]) -> dict[str, str]:
    return {"session_key": ref[0], "participant_id": ref[1]}


def _identity_data(hub: dict[str, Any], events: dict[str, dict[str, Any]]) -> dict[str, Any]:
    nodes = {}
    titles = [str(events.get(str(g["session_key"]), {}).get("name") or g["session_key"]) for g in hub.get("groups") or []]
    for index, group in enumerate(hub.get("groups") or []):
        key = str(group["session_key"])
        if key not in events:
            raise ValueError("An attached round robin is unavailable.")
        title = titles[index]
        qualifier = f"{title} ({index + 1})" if titles.count(title) > 1 else title
        for player in events[key].get("participants") or []:
            ref = (key, str(player.get("id") or ""))
            if not ref[1] or not _name_key(player.get("name")) or ref in nodes:
                raise ValueError("Every hub participant needs a unique session-scoped ID and name.")
            nodes[ref] = {**_ref_json(ref), "name": player["name"], "group_title": title,
                "qualifier": qualifier, "profile_id": int(player["player_id"]) if player.get("player_id") is not None else None,
                "base": player_identity(player, key), "name_key": _name_key(player["name"])}
    parent = {ref: ref for ref in nodes}

    def find(ref):
        while parent[ref] != ref:
            parent[ref] = parent[parent[ref]]
            ref = parent[ref]
        return ref

    def union(a, b):
        parent[find(a)] = find(b)

    profiles = {}
    for ref, node in nodes.items():
        if node["profile_id"] is not None:
            previous = profiles.setdefault(node["profile_id"], ref)
            union(ref, previous)
    decisions = hub.get("identity_decisions") or []
    for decision in decisions:
        left, right = _ref(decision.get("left") or {}), _ref(decision.get("right") or {})
        if left not in nodes or right not in nodes or decision.get("resolution") not in {"same", "different"}:
            raise ValueError("An identity decision references an unavailable participant.")
        if decision["resolution"] == "same":
            union(left, right)
    components = {}
    for ref in nodes:
        components.setdefault(find(ref), []).append(ref)
    identities = {}
    for members in components.values():
        profile_ids = {nodes[ref]["profile_id"] for ref in members if nodes[ref]["profile_id"] is not None}
        if len(profile_ids) > 1:
            raise ValueError("Different club profiles identify different people and cannot be merged.")
        identity = f"player:{next(iter(profile_ids))}" if profile_ids else min(nodes[ref]["base"] for ref in members)
        for ref in members:
            identities[ref] = identity
    different = set()
    for decision in decisions:
        if decision["resolution"] == "different":
            pair = frozenset((identities[_ref(decision["left"])], identities[_ref(decision["right"])]))
            if len(pair) != 2:
                raise ValueError("Identity decisions conflict with a club profile or a same-person confirmation.")
            different.add(pair)
    pending_pairs = []
    by_name = {}
    for ref, node in nodes.items():
        node["identity"] = identities[ref]
        by_name.setdefault(node["name_key"], []).append(ref)
    for refs in by_name.values():
        for index, left in enumerate(refs):
            for right in refs[index + 1:]:
                if identities[left] != identities[right] and (nodes[left]["profile_id"] is None or nodes[right]["profile_id"] is None) and frozenset((identities[left], identities[right])) not in different:
                    pending_pairs.append((left, right))
    pending = {identities[ref] for pair in pending_pairs for ref in pair}
    for refs in by_name.values():
        distinct = len({identities[ref] for ref in refs}) > 1
        for ref in refs:
            node = nodes[ref]
            node["display_name"] = f"{node['name']} · {node['qualifier']}" if distinct else node["name"]
    return {"nodes": nodes, "identities": identities, "different": different,
        "pending_pairs": pending_pairs, "pending": pending, "by_name": by_name}


def _matches(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row.get("matches") or []) or [m for c in row.get("courts") or [] for m in c.get("matches") or []]


def _round(event: dict[str, Any], number: int | None = None) -> dict[str, Any]:
    number = int(number if number is not None else event.get("currentRoundNumber") or 1)
    row = next((r for r in event.get("rounds") or [] if int(r["number"]) == number), None)
    if row is None:
        raise ValueError("The current round is unavailable. Reload the round robin.")
    return row


def _complete(match: dict[str, Any], event: dict[str, Any]) -> bool:
    if event.get("scoringMode") == "unscored":
        return match.get("status") == "played"
    return match.get("scoreA") is not None and match.get("scoreB") is not None


def _key(value: dict[str, Any]) -> tuple[str, str]:
    return str(value.get("session_key") or ""), str(value.get("match_id") or "")


def _group_keys(hub: dict[str, Any]) -> set[str]:
    return {str(g["session_key"]) for g in hub.get("groups") or []}


def _event(hub: dict[str, Any], events: dict[str, dict[str, Any]], session_key: str) -> dict[str, Any]:
    if session_key not in _group_keys(hub) or session_key not in events:
        raise ValueError("This round robin is not attached to this hub.")
    return events[session_key]


def _participants(event: dict[str, Any], match: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {str(p["id"]): p for p in event.get("participants") or []}
    ids = [str(x) for x in (match.get("sideA") or match.get("teamA") or []) + (match.get("sideB") or match.get("teamB") or [])]
    if any(pid not in by_id for pid in ids) or len(ids) not in (2, 4) or len(set(ids)) != len(ids):
        raise ValueError("This matchup has an incomplete or duplicate lineup.")
    players = [by_id[pid] for pid in ids]
    return players


def _find_match(hub: dict[str, Any], events: dict[str, dict[str, Any]], assignment: dict[str, Any]) -> tuple[dict, dict, dict]:
    event = _event(hub, events, str(assignment.get("session_key") or ""))
    row = _round(event, int(assignment.get("round_number") or event.get("currentRoundNumber") or 1))
    match = next((m for m in _matches(row) if str(m["id"]) == str(assignment.get("match_id") or "")), None)
    if match is None:
        raise ValueError("The assigned matchup changed. Reload before using this court.")
    return event, row, match


def _court(hub: dict[str, Any], court_id: str) -> dict[str, Any]:
    result = next((c for c in hub.get("courts") or [] if str(c["id"]) == court_id), None)
    if result is None:
        raise ValueError("Choose a court in this hub.")
    return result


def _allowed(hub: dict[str, Any], session_key: str) -> list[str]:
    dedicated = [c for c in hub.get("courts") or [] if c.get("dedicated_session_key") == session_key]
    return [str(c["id"]) for c in (dedicated or [c for c in hub.get("courts") or [] if not c.get("dedicated_session_key")]) if c.get("open", True)]


def _wait(player: dict[str, Any], now: str) -> float:
    end = player.get("pause_started_at") if player.get("paused") else now
    return max(0.0, (_time(end or now) - _time(player.get("wait_since") or now)).total_seconds())


def _ensure_players(hub: dict[str, Any], events: dict[str, dict[str, Any]], now: str) -> None:
    old_map, old_players = hub.get("participant_identities") or {}, hub.get("players") or {}
    legacy = hub.get("identity_version") != 2
    existing = {(str(g["session_key"]), str(p["id"])) for g in hub.get("groups") or [] for p in events.get(str(g["session_key"]), {}).get("participants") or []}
    hub["identity_decisions"] = [d for d in hub.get("identity_decisions") or [] if _ref(d.get("left") or {}) in existing and _ref(d.get("right") or {}) in existing]
    data = _identity_data(hub, events)
    mapping, sources = {}, {}
    for ref, identity in data["identities"].items():
        mapping.setdefault(ref[0], {})[ref[1]] = identity
        node = data["nodes"][ref]
        old_identity = (old_map.get(ref[0]) or {}).get(ref[1])
        if old_identity is None:
            legacy_name = " ".join(str(node["name"]).lower().split())
            old_identity = (f"player:{node['profile_id']}" if node["profile_id"] is not None else f"name:{legacy_name}") if legacy else node["base"]
        row = old_players.get(old_identity) or {"paused": False, "wait_since": now, "pause_started_at": None}
        sources.setdefault(identity, {})[old_identity] = row
    players = {}
    for identity, rows in sources.items():
        values = list(rows.values())
        if len(values) == 1:
            players[identity] = deepcopy(values[0])
        else:
            # A shared person gets no extra idle credit from an older alias; pause is conservative.
            paused = any(row.get("paused") for row in values)
            players[identity] = {"paused": paused, "wait_since": (_time(now) - timedelta(seconds=min(_wait(row, now) for row in values))).isoformat(), "pause_started_at": now if paused else None}
    hub.update(identity_version=2, participant_identities=mapping, players=players)


def _identity_review(hub: dict[str, Any], data: dict[str, Any], claims: dict[str, Any]) -> list[dict[str, Any]]:
    # Keep confirmed relationships reviewable even after someone changes a label.
    buckets = [{ref} for ref in data["nodes"]]

    def connect(refs):
        refs = set(refs)
        touched = [bucket for bucket in buckets if bucket & refs]
        merged = set().union(*touched)
        for bucket in touched:
            buckets.remove(bucket)
        buckets.append(merged)

    for refs in data["by_name"].values():
        connect(refs)
    same_people = {}
    for ref, identity in data["identities"].items():
        same_people.setdefault(identity, []).append(ref)
    for refs in same_people.values():
        connect(refs)
    for decision in hub.get("identity_decisions") or []:
        connect((_ref(decision["left"]), _ref(decision["right"])))
    review = []
    for refs in sorted((bucket for bucket in buckets if len(bucket) > 1), key=lambda bucket: min(bucket)):
        nodes = [data["nodes"][ref] for ref in sorted(refs)]
        identities = {node["identity"] for node in nodes}
        has_pending = any(left in refs and right in refs for left, right in data["pending_pairs"])
        status = "pending" if has_pending else "same" if len(identities) == 1 else "different" if len(identities) == len(nodes) else "mixed"
        review.append({"id": "guest:" + ":".join(min(refs)), "name": nodes[0]["name"], "status": status,
            "can_merge": len({node["profile_id"] for node in nodes if node["profile_id"] is not None}) <= 1,
            "assigned": any(node["identity"] in claims for node in nodes),
            "participants": [{key: node[key] for key in ("name", "display_name", "session_key", "group_title", "participant_id", "identity", "profile_id")} for node in nodes]})
    return review


def _resolve_identities(hub: dict[str, Any], events: dict[str, dict[str, Any]], payload: dict[str, Any], now: str) -> None:
    data = _identity_data(hub, events)
    resolution = str(payload.get("resolution") or "")
    decisions = deepcopy(hub.get("identity_decisions") or [])

    def decide(left, right, kind):
        pair = frozenset((left, right))
        decisions[:] = [d for d in decisions if frozenset((_ref(d["left"]), _ref(d["right"]))) != pair]
        decisions.append({"left": _ref_json(left), "right": _ref_json(right), "resolution": kind})

    if resolution == "all_different":
        # Resolve only unanswered pairs. Existing SAME components stay intact.
        for left, right in data["pending_pairs"]:
            decide(left, right, "different")
    elif resolution in {"same", "different"}:
        refs = sorted({_ref(ref) for ref in payload.get("participants") or [] if isinstance(ref, dict)})
        if len(refs) < 2 or any(ref not in data["nodes"] for ref in refs):
            raise ValueError("Choose at least two current participant entries to review.")
        selected_identities = {data["identities"][ref] for ref in refs}
        if resolution == "same":
            profiles = {node["profile_id"] for node in data["nodes"].values() if node["identity"] in selected_identities and node["profile_id"] is not None}
            if len(profiles) > 1:
                raise ValueError("Different club profiles identify different people and cannot be merged.")
            if len(selected_identities) > 1 and selected_identities & _claims(hub, events).keys():
                raise ValueError("Finish or release affected assignments before marking these as the same person.")
            decisions[:] = [d for d in decisions if not (d["resolution"] == "different" and data["identities"][_ref(d["left"])] in selected_identities and data["identities"][_ref(d["right"])] in selected_identities)]
            for right in refs[1:]:
                decide(refs[0], right, "same")
        else:
            splitting = set()
            for identity in selected_identities:
                members = {ref for ref, value in data["identities"].items() if value == identity}
                selected_members = members & set(refs)
                # Multiple occurrences of one club profile remain the same person.
                if len(selected_members) > 1 and any(data["nodes"][ref]["profile_id"] is None for ref in selected_members):
                    if not members <= set(refs):
                        raise ValueError("Select every entry in the confirmed same-person group before separating them.")
                    splitting.update(members)
            decisions[:] = [d for d in decisions if not (d["resolution"] == "same" and (_ref(d["left"]) in splitting or _ref(d["right"]) in splitting))]
            temporary = {**hub, "identity_decisions": decisions}
            separated = _identity_data(temporary, events)["identities"]
            for index, left in enumerate(refs):
                for right in refs[index + 1:]:
                    if separated[left] != separated[right]:
                        decide(left, right, "different")
    else:
        raise ValueError("Choose Same person, Different people, or All different.")
    hub["identity_decisions"] = decisions
    _ensure_players(hub, events, now)


def _claims(hub: dict[str, Any], events: dict[str, dict[str, Any]]) -> dict[str, tuple[str, str]]:
    claims: dict[str, tuple[str, str]] = {}
    seen_matches: set[tuple[str, str]] = set()
    for court in hub.get("courts") or []:
        for slot in ("current", "next"):
            assignment = court.get(slot)
            if not assignment:
                continue
            key = _key(assignment)
            if key in seen_matches:
                raise ValueError("A matchup cannot occupy or reserve two courts.")
            seen_matches.add(key)
            event, row, match = _find_match(hub, events, assignment)
            if int(row["number"]) != int(event.get("currentRoundNumber") or 1):
                raise ValueError("A court assignment points to a different round. Resolve it before advancing.")
            if _complete(match, event):
                raise ValueError("A completed matchup still has a court assignment.")
            if str(court["id"]) not in _allowed(hub, key[0]):
                raise ValueError("This court is not available to the assigned round robin.")
            for player in _participants(event, match):
                identity = player_identity(player, key[0], hub)
                if identity in claims:
                    raise ValueError("A player cannot be assigned to two courts.")
                claims[identity] = key
    return claims


def build_hub_snapshot(hub: dict[str, Any], events: dict[str, dict[str, Any]], *, now: str) -> dict[str, Any]:
    state = deepcopy(hub)
    _ensure_players(state, events, now)
    claims = _claims(state, events)
    identity_data = _identity_data(state, events)
    assignments: dict[tuple[str, str], tuple[dict, str]] = {}
    playing_identities: set[str] = set()
    for court in state.get("courts") or []:
        for slot in ("current", "next"):
            if court.get(slot):
                assignments[_key(court[slot])] = (court, slot)
                if slot == "current":
                    current_event, _, current_match = _find_match(state, events, court[slot])
                    playing_identities.update(player_identity(p, court[slot]["session_key"], state) for p in _participants(current_event, current_match))
    ready, blocked, reserved, groups, all_matches = [], [], [], [], {}
    public_players: dict[str, dict[str, Any]] = {}
    for group in state.get("groups") or []:
        key = str(group["session_key"])
        event = events[key]
        row = _round(event)
        completed = sum(_complete(m, event) for m in _matches(row))
        groups.append({"session_key": key, "title": event.get("name") or key,
            "current_round_number": int(row["number"]), "round_status": row.get("status"),
            "total_rounds": int(event.get("totalRounds") or len(event.get("rounds") or [])),
            "completed_matches": completed, "total_matches": len(_matches(row)),
            "can_advance": row.get("status") in {"saved", "played", "skipped"} and event.get("status") == "active",
            "advance_label": "Finish session" if ((is_moneyball(event) and all(r.get("status") == "saved" for r in event.get("rounds") or [])) or (event.get("playoff") and int(row["number"]) >= int(event.get("totalRounds") or 0))) else "Next round",
            "event_format": event.get("eventFormat") or "standard", "scoring_mode": event.get("scoringMode") or "scored",
            "rating_mode": event.get("ratingMode") or "unrated", "status": event.get("status"),
            "dedicated_court_ids": [c["id"] for c in state.get("courts") or [] if c.get("dedicated_session_key") == key]})
        by_id = {str(p["id"]): p for p in event.get("participants") or []}
        for participant in event.get("participants") or []:
            identity = player_identity(participant, key, state)
            info = public_players.setdefault(identity, {"identity": identity, "name": participant.get("name"),
                "display_name": identity_data["nodes"][(key, str(participant["id"]))]["display_name"],
                "paused": bool(state["players"][identity].get("paused")),
                "wait_seconds": 0 if identity in playing_identities else int(_wait(state["players"][identity], now)), "session_keys": [], "participant_ids": {},
                "on_court": identity in playing_identities, "reserved": identity in claims and identity not in playing_identities})
            info["session_keys"].append(key)
            info["participant_ids"][key] = str(participant["id"])
        for match in _matches(row):
            players = _participants(event, match)
            identities = [player_identity(p, key, state) for p in players]
            waits = sorted((_wait(state["players"][i], now) for i in identities), reverse=True)
            side = lambda ids: [{"id": str(pid), "name": by_id[str(pid)].get("name"), "display_name": identity_data["nodes"][(key, str(pid))]["display_name"], "identity": player_identity(by_id[str(pid)], key, state)} for pid in ids]
            side_a = side(match.get("sideA") or match.get("teamA") or [])
            side_b = side(match.get("sideB") or match.get("teamB") or [])
            entry = {"session_key": key, "match_id": str(match["id"]), "round_number": int(row["number"]),
                "group_title": event.get("name") or key, "side_a": side_a, "side_b": side_b,
                "label": " / ".join(str(p["display_name"]) for p in side_a) + " vs " + " / ".join(str(p["display_name"]) for p in side_b),
                "wait_seconds": int(waits[0]) if waits else 0, "wait_order": waits,
                "allowed_court_ids": _allowed(state, key), "blockers": [], "status": "waiting",
                "score_a": match.get("scoreA"), "score_b": match.get("scoreB")}
            all_matches[(key, str(match["id"]))] = entry
            if _complete(match, event):
                entry["status"] = "completed"
                continue
            assignment = assignments.get((key, str(match["id"])))
            if assignment:
                court, slot = assignment
                entry["status"] = "on_court" if slot == "current" else "reserved"
                entry["court_id" if slot == "current" else "reserved_court_id"] = court["id"]
                entry["assigned_at"] = court[slot].get("assigned_at")
                if slot == "next":
                    reserved.append(entry)
                continue
            if event.get("status") != "active" or row.get("status") not in {"active", "preview"}:
                entry["blockers"].append("This round is complete.")
            if any(state["players"][i].get("paused") for i in identities):
                entry["blockers"].append("A player is paused.")
            if any(i in identity_data["pending"] for i in identities):
                entry["blockers"].append("Confirm matching player names in Players before assigning this match.")
            if len(set(identities)) != len(identities):
                entry["blockers"].append("The same person appears twice in this match. Update the group roster.")
            if any(i in claims for i in identities):
                entry["blockers"].append("A player is already assigned to a court.")
            if not entry["allowed_court_ids"]:
                entry["blockers"].append("No open court is available to this group.")
            if entry["blockers"]:
                entry["status"] = "blocked"
                blocked.append(entry)
            else:
                ready.append(entry)
    ordering = lambda m: (tuple(-x for x in m["wait_order"]), m["session_key"], m["match_id"])
    ready.sort(key=ordering)
    blocked.sort(key=ordering)
    reserved.sort(key=ordering)
    courts = []
    for court in state.get("courts") or []:
        item = deepcopy(court)
        for slot in ("current", "next"):
            item[slot] = deepcopy(all_matches.get(_key(court[slot]))) if court.get(slot) else None
        courts.append(item)
    review = _identity_review(state, identity_data, claims)
    return {"courts": courts, "groups": groups, "eligible_queue": ready, "blocked_queue": blocked,
        "reserved_queue": reserved, "players": list(public_players.values()), "identity_review": review,
        "identity_pending_count": sum(row["status"] == "pending" for row in review)}


def _candidate(hub: dict[str, Any], events: dict[str, dict[str, Any]], payload: dict[str, Any], now: str) -> dict[str, Any]:
    snapshot = build_hub_snapshot(hub, events, now=now)
    key = _key(payload)
    candidate = next((m for m in snapshot["eligible_queue"] if _key(m) == key), None)
    if candidate is None:
        raise ValueError("This match is not ready. Reload to check its players, round, and courts.")
    return {"session_key": key[0], "match_id": key[1], "round_number": candidate["round_number"], "assigned_at": now}


def _reset_wait(hub: dict[str, Any], events: dict[str, dict[str, Any]], assignment: dict[str, Any], now: str) -> None:
    event, _, match = _find_match(hub, events, assignment)
    for player in _participants(event, match):
        row = hub["players"][player_identity(player, assignment["session_key"], hub)]
        row["wait_since"] = now
        if row.get("paused"):
            row["pause_started_at"] = now


def _sync_assignment_projection(hub: dict[str, Any], events: dict[str, dict[str, Any]]) -> None:
    for key in _group_keys(hub):
        events[key]["hubAssignedMatchIds"] = []
        for row in events[key].get("rounds") or []:
            for match in _matches(row):
                match.pop("reservedCourtLabel", None)
                if not _complete(match, events[key]):
                    match.pop("courtLabel", None)
    for court in hub.get("courts") or []:
        for slot in ("current", "next"):
            assignment = court.get(slot)
            if assignment:
                event, _, match = _find_match(hub, events, assignment)
                event["hubAssignedMatchIds"].append(str(match["id"]))
                match["courtLabel" if slot == "current" else "reservedCourtLabel"] = court["label"]


def _promote(hub: dict[str, Any], events: dict[str, dict[str, Any]], now: str) -> None:
    for court in hub.get("courts") or []:
        if court.get("current") or not court.get("next"):
            continue
        assignment = court.pop("next")
        court["next"] = None
        try:
            candidate = _candidate(hub, events, assignment, now)
            if court["id"] not in _allowed(hub, candidate["session_key"]):
                continue
            court["current"] = candidate
        except ValueError:
            # A paused/departed player does not make releasing the prior game fail.
            continue


def save_hub_match(event: dict[str, Any], *, match_id: str, score_a: Any = None, score_b: Any = None,
                   played: bool = False, now: str) -> dict[str, Any]:
    result = deepcopy(event)
    row = _round(result)
    if result.get("status") != "active" or row.get("status") not in {"active", "preview"}:
        raise ValueError("Only a match in the current incomplete round can finish.")
    match = next((m for m in _matches(row) if str(m["id"]) == match_id), None)
    if match is None or _complete(match, result):
        raise ValueError("This match is unavailable or already complete.")
    if played:
        if result.get("scoringMode") != "unscored":
            raise ValueError("Enter a score for this scored round robin.")
        match["status"] = "played"
    else:
        if result.get("scoringMode") == "unscored":
            raise ValueError("Use Match played for this unscored round robin.")
        if type(score_a) is not int or type(score_b) is not int or not (0 <= score_a <= 99 and 0 <= score_b <= 99) or score_a == score_b:
            raise ValueError("Enter complete, non-tied scores between 0 and 99.")
        if is_moneyball(result):
            validate_score(score_a, score_b, (result.get("moneyball") or {}).get("targetScore", 11))
        match.update(scoreA=score_a, scoreB=score_b, status="scored")
    match["hubCompletedAt"] = now
    if all(_complete(m, result) for m in _matches(row)):
        if played:
            result = mark_generator_round_played(result, round_number=int(row["number"]))
        else:
            result = save_generator_round(result, round_number=int(row["number"]), scores=[
                {"match_id": m["id"], "score_a": m.get("scoreA"), "score_b": m.get("scoreB")} for m in _matches(row)])
    return result


def validate_hub_session_change(hub: dict[str, Any], before_events: dict[str, dict[str, Any]],
                                session_key: str, event: dict[str, Any]) -> None:
    """Guard existing generator controls before their joint hub/session commit."""
    before = before_events[session_key]
    prior_round = int(before.get("currentRoundNumber") or 1)
    next_round = int(event.get("currentRoundNumber") or 1)
    if next_round != prior_round:
        if _round(event, prior_round).get("status") not in {"saved", "played", "skipped"} or next_round != prior_round + 1:
            raise ValueError("Finish the current round before moving to the next round.")
    for new_round in event.get("rounds") or []:
        old_round = next((r for r in before.get("rounds") or [] if r["number"] == new_round["number"]), {})
        if int(new_round["number"]) > prior_round and new_round.get("status") in {"saved", "played", "skipped"} and old_round.get("status") != new_round.get("status"):
            raise ValueError("Rounds in a hub must finish in order.")
        if int(new_round["number"]) < prior_round and old_round.get("status") == "skipped" and new_round.get("status") != "skipped":
            raise ValueError("An earlier skipped round cannot reopen after the hub moves to the next round.")
    for court in hub.get("courts") or []:
        for slot in ("current", "next"):
            assignment = court.get(slot)
            if not assignment or assignment.get("session_key") != session_key:
                continue
            _, old_round, old_match = _find_match(hub, before_events, assignment)
            replacement_events = {**before_events, session_key: event}
            _, new_round, new_match = _find_match(hub, replacement_events, assignment)
            if new_round.get("status") == "skipped":
                raise ValueError("Release this group's courts and reservations before skipping the round.")
            identity_fields = ("id", "sideA", "sideB", "teamA", "teamB", "playFormat")
            old_people = [player_identity(p, session_key) for p in _participants(before, old_match)]
            new_people = [player_identity(p, session_key) for p in _participants(event, new_match)]
            if any(old_match.get(k) != new_match.get(k) for k in identity_fields) or old_people != new_people:
                raise ValueError("Move this assigned matchup back to the queue before changing its players or score.")
    if event.get("status") == "completed" and _round(event, prior_round).get("status") not in {"saved", "played", "skipped"}:
        raise ValueError("Resolve every match in the current round before finishing this round robin.")


def reconcile_hub_events(hub: dict[str, Any], before_events: dict[str, dict[str, Any]],
                         after_events: dict[str, dict[str, Any]], *, now: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Reconcile original generator controls with the same court authority."""
    state, updated = deepcopy(hub), deepcopy(after_events)
    _ensure_players(state, before_events, now)
    _claims(state, before_events)
    for key in _group_keys(state):
        if key not in updated:
            raise ValueError("An attached session cannot disappear while its hub is active.")
        validate_hub_session_change(state, before_events, key, updated[key])
        prior = before_events[key]
        for row in prior.get("rounds") or []:
            for match in _matches(row):
                if _complete(match, prior):
                    continue
                new_row = next((r for r in updated[key].get("rounds") or [] if r["number"] == row["number"]), None)
                new_match = next((m for m in _matches(new_row or {}) if m["id"] == match["id"]), None)
                if new_match and _complete(new_match, updated[key]):
                    _reset_wait(state, before_events, {"session_key": key, "round_number": row["number"], "match_id": match["id"]}, now)
    for court in state.get("courts") or []:
        for slot in ("current", "next"):
            assignment = court.get(slot)
            if assignment:
                event, _, match = _find_match(state, updated, assignment)
                if _complete(match, event):
                    court[slot] = None
    _ensure_players(state, updated, now)
    _promote(state, updated, now)
    _sync_assignment_projection(state, updated)
    _claims(state, updated)
    return state, updated


def mutate_hub_roster(event: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Apply standard roster controls to waiting games, preserving live history.

    The original generator treats displaying a round as starting all its games.
    Hub sessions have explicit assignments, so only assigned or completed games
    are immutable; the remaining roster can be scheduled again safely.
    """
    from jupr_app.domain.adaptive_play_engine import (
        mutate_generator_roster, active_participant_ids, history_before_round,
        _apply_match_history, _doubles_round, _singles_round, _mixed_round,
        _regenerate_round_robin_from, generator_match_play_format,
    )
    original = deepcopy(event)
    hub_id = original.pop("roundRobinHubId", None)
    result = mutate_generator_roster(original, **kwargs)
    result["roundRobinHubId"] = hub_id
    current = int(event.get("currentRoundNumber") or 1)
    original_row = _round(event)
    if original_row.get("status") not in {"active", "preview"}:
        return result
    assigned = set(event.get("hubAssignedMatchIds") or [])
    protected = [deepcopy(m) for m in _matches(original_row) if str(m["id"]) in assigned or _complete(m, event)]
    protected_people = {str(pid) for m in protected for pid in (m.get("sideA") or m.get("teamA") or []) + (m.get("sideB") or m.get("teamB") or [])}
    original_ids = active_participant_ids(event, current)
    active = list(original_ids)
    action = str(kwargs.get("action") or "")
    participant_id = str(kwargs.get("participant_id") or "")
    old_people = {str(p["id"]) for p in event.get("participants") or []}
    added = next((p for p in result.get("participants") or [] if str(p["id"]) not in old_people), None)
    if action == "add" and added:
        added["active_from_round"] = current
        active.append(str(added["id"]))
    elif action == "remove":
        if participant_id not in protected_people:
            active = [pid for pid in active if pid != participant_id]
            next(p for p in result["participants"] if str(p["id"]) == participant_id)["inactive_from_round"] = current
    elif action == "substitute" and added and participant_id not in protected_people:
        active = [str(added["id"]) if pid == participant_id else pid for pid in active]
        added["active_from_round"] = current
        source = next(p for p in result["participants"] if str(p["id"]) == participant_id)
        if kwargs.get("substitute_scope") == "round":
            source["inactive_rounds"] = [n for n in source.get("inactive_rounds") or [] if int(n) != current + 1]
            source["inactive_rounds"].append(current)
            added["inactive_from_round"] = current + 1
            for row in event.get("rounds") or []:
                if int(row["number"]) > current:
                    result.setdefault("rosterByRound", {})[str(row["number"])] = active_participant_ids(event, int(row["number"]))
        else:
            source["inactive_from_round"] = current
    else:
        # An on-court player's substitution/removal takes effect next round.
        # Reorder also changes only future matchups once a round is underway.
        return result
    active = list(dict.fromkeys(active))
    result.setdefault("rosterByRound", {})[str(current)] = active
    waiting = [pid for pid in active if pid not in protected_people]
    history = history_before_round(result, current)
    for match in protected:
        _apply_match_history(history, match, str(result.get("playFormat") or "doubles"))
    roster_pos = {str(p["id"]): int(p.get("roster_order") or 0) for p in result.get("participants") or []}
    play_format = result.get("playFormat") or "doubles"
    if play_format == "doubles_singles":
        counts = {kind: sum(generator_match_play_format(m, play_format) == kind for m in protected) for kind in ("singles", "doubles")}
        generated, byes, warnings, _ = _mixed_round(active_ids=waiting,
            doubles_court_count=max(0, int(result.get("doublesCourtCount") or 0) - counts["doubles"]),
            singles_court_count=max(0, int(result.get("singlesCourtCount") or 0) - counts["singles"]),
            history=history, roster_pos=roster_pos, round_number=current)
    else:
        generate = _singles_round if play_format == "singles" else _doubles_round
        generated, byes, warnings = generate(active_ids=waiting, court_count=0, history=history,
            roster_pos=roster_pos, round_number=current)
    revision = int(result.get("hubRosterRevision") or 0) + 1
    result["hubRosterRevision"] = revision
    offset = max((int(m.get("court") or 0) for m in protected), default=0)
    for index, match in enumerate(generated, 1):
        match["id"] = f"hub-r{current}-v{revision}-m{index}"
        match["court"] = offset + index
    row = _round(result)
    row["matches"] = [*protected, *generated]
    row.pop("courts", None)
    row["byeParticipantIds"] = list(byes)
    row["warnings"] = warnings
    row["formatCounts"] = {kind: sum(generator_match_play_format(m, play_format) == kind for m in row["matches"]) for kind in ("singles", "doubles")}
    result["adaptiveFromRound"] = min(int(result.get("adaptiveFromRound") or current), current)
    result["perfectRotation"] = False
    if all(_complete(m, result) for m in row["matches"]):
        if result.get("scoringMode") == "unscored":
            result = mark_generator_round_played(result, round_number=current)
        else:
            result = save_generator_round(result, round_number=current, scores=[
                {"match_id": m["id"], "score_a": m.get("scoreA"), "score_b": m.get("scoreB")} for m in row["matches"]])
    _regenerate_round_robin_from(result, current + 1)
    return result


def apply_hub_command(hub: dict[str, Any], events: dict[str, dict[str, Any]], *, action: str,
                      payload: dict[str, Any], now: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    state, updated = deepcopy(hub), deepcopy(events)
    state.setdefault("courts", [])
    state.setdefault("groups", [])
    state.setdefault("players", {})
    _ensure_players(state, updated, now)
    _claims(state, updated)
    session_key = str(payload.get("session_key") or "")
    if action == "add_group":
        event = updated.get(session_key)
        if not event or event.get("generatorKind") != "round_robin" or event.get("status") != "active":
            raise ValueError("Choose an active round robin.")
        if session_key in _group_keys(state):
            raise ValueError("This round robin is already in the hub.")
        current = int(event.get("currentRoundNumber") or 1)
        if any(int(r["number"]) < current and r.get("status") not in {"saved", "played", "skipped"} for r in event.get("rounds") or []):
            raise ValueError("Finish earlier incomplete rounds before attaching this round robin.")
        if any(int(r["number"]) > current and r.get("status") in {"saved", "played"} for r in event.get("rounds") or []):
            raise ValueError("This session has out-of-order results. Resolve them before attaching it.")
        state["groups"].append({"session_key": session_key, "joined_at": now})
    elif action == "remove_group":
        event = _event(state, updated, session_key)
        if any(d["resolution"] == "same" and ((_ref(d["left"])[0] == session_key) != (_ref(d["right"])[0] == session_key)) for d in state.get("identity_decisions") or []):
            raise ValueError("Review and separate shared guest identities in Players before removing this group.")
        if any(c.get(s, {}).get("session_key") == session_key for c in state["courts"] for s in ("current", "next") if c.get(s)):
            raise ValueError("Release this group's courts and reservations before removing it.")
        if any(not _complete(m, event) for m in _matches(_round(event))) and any(_complete(m, event) for m in _matches(_round(event))):
            raise ValueError("Finish the partially scored round before removing this group.")
        state["groups"] = [g for g in state["groups"] if g["session_key"] != session_key]
        for court in state["courts"]:
            if court.get("dedicated_session_key") == session_key:
                court["dedicated_session_key"] = None
        event.pop("roundRobinHubId", None)
        event.pop("hubAssignedMatchIds", None)
    elif action == "configure_courts":
        courts = payload.get("courts")
        if not isinstance(courts, list) or not 1 <= len(courts) <= 40:
            raise ValueError("Configure between 1 and 40 courts.")
        previous = {str(c["id"]): c for c in state["courts"]}
        result, seen_ids, seen_labels = [], set(), set()
        for data in courts:
            cid, label = str(data.get("id") or "").strip(), " ".join(str(data.get("label") or "").split())
            if not cid or len(cid) > 160 or not label or len(label) > 80 or cid in seen_ids or label.casefold() in seen_labels:
                raise ValueError("Court names and IDs must be present and unique.")
            dedicated = str(data.get("dedicated_session_key") or "") or None
            if dedicated and dedicated not in _group_keys(state):
                raise ValueError("Dedicate courts only to a group in this hub.")
            old = previous.get(cid, {})
            result.append({"id": cid, "label": label, "dedicated_session_key": dedicated,
                "open": bool(data.get("open", True)), "current": deepcopy(old.get("current")), "next": deepcopy(old.get("next"))})
            seen_ids.add(cid)
            seen_labels.add(label.casefold())
        if any((c.get("current") or c.get("next")) and cid not in seen_ids for cid, c in previous.items()):
            raise ValueError("Release a court and its reservation before removing it.")
        state["courts"] = result
    elif action == "resolve_identities":
        _resolve_identities(state, updated, payload, now)
    elif action in {"pause_player", "resume_player"}:
        identity = str(payload.get("identity") or "")
        if identity not in state["players"]:
            raise ValueError("Choose a player in this hub.")
        player = state["players"][identity]
        if action == "pause_player" and not player.get("paused"):
            player.update(paused=True, pause_started_at=now)
            for court in state["courts"]:
                if court.get("next"):
                    event, _, match = _find_match(state, updated, court["next"])
                    if identity in {player_identity(p, court["next"]["session_key"], state) for p in _participants(event, match)}:
                        court["next"] = None
        elif action == "resume_player" and player.get("paused"):
            elapsed = max(0.0, (_time(now) - _time(player.get("pause_started_at") or now)).total_seconds())
            player["wait_since"] = (_time(player.get("wait_since") or now) + timedelta(seconds=elapsed)).isoformat()
            player.update(paused=False, pause_started_at=None)
    elif action == "auto_fill_courts":
        while True:
            ready = build_hub_snapshot(state, updated, now=now)["eligible_queue"]
            chosen = None
            for candidate in ready:
                court = next((c for c in state["courts"] if c["id"] in candidate["allowed_court_ids"] and not c.get("current") and not c.get("next")), None)
                if court is not None:
                    chosen = (candidate, court)
                    break
            if chosen is None:
                break
            candidate, court = chosen
            court["current"] = _candidate(state, updated, candidate, now)
    elif action in {"assign_next_court", "assign_game", "reserve_game_for_court"}:
        assignment = _candidate(state, updated, payload, now)
        allowed = _allowed(state, session_key)
        if action == "assign_next_court":
            court = next((c for c in state["courts"] if c["id"] in allowed and not c.get("current") and not c.get("next")), None)
            if court is None:
                raise ValueError("No permitted court is currently open.")
        else:
            court = _court(state, str(payload.get("court_id") or ""))
        if court["id"] not in allowed:
            raise ValueError("This court is not available to this round robin.")
        if action == "reserve_game_for_court":
            if not court.get("current") or court.get("next"):
                raise ValueError("Choose an occupied court without a next match.")
            court["next"] = assignment
        else:
            if court.get("current") or court.get("next"):
                raise ValueError("This court is occupied or reserved.")
            court["current"] = assignment
    elif action in {"cancel_reservation", "release_game", "move_game"}:
        court = _court(state, str(payload.get("court_id") or ""))
        slot = "next" if action == "cancel_reservation" else "current"
        assignment = court.get(slot)
        if not assignment:
            raise ValueError("This court assignment is no longer available.")
        if payload.get("match_id") and _key(payload) != _key(assignment):
            raise ValueError("The court assignment changed. Reload before continuing.")
        if action == "move_game":
            target = _court(state, str(payload.get("target_court_id") or ""))
            if target["id"] not in _allowed(state, assignment["session_key"]) or target.get("current") or target.get("next"):
                raise ValueError("Choose an empty court available to this group.")
            target["current"] = assignment
        court[slot] = None
        _promote(state, updated, now)
    elif action in {"score_and_release", "mark_played"}:
        court = _court(state, str(payload.get("court_id") or ""))
        assignment = court.get("current")
        if not assignment or _key(assignment) != _key(payload):
            raise ValueError("The court assignment changed. Reload before saving this result.")
        event, _, _ = _find_match(state, updated, assignment)
        _reset_wait(state, updated, assignment, now)
        updated[session_key] = save_hub_match(event, match_id=str(payload.get("match_id") or ""),
            score_a=payload.get("score_a"), score_b=payload.get("score_b"), played=action == "mark_played", now=now)
        court["current"] = None
        _promote(state, updated, now)
    elif action == "advance_round":
        event = _event(state, updated, session_key)
        if any(c.get(s, {}).get("session_key") == session_key for c in state["courts"] for s in ("current", "next") if c.get(s)):
            raise ValueError("Finish or release this group's court assignments before advancing.")
        if _round(event).get("status") not in {"saved", "played", "skipped"}:
            raise ValueError("Finish the current round before generating the next round.")
        updated[session_key] = advance_generator_event(event)
    else:
        raise ValueError("Unsupported round robin hub command.")
    _ensure_players(state, updated, now)
    _sync_assignment_projection(state, updated)
    _claims(state, updated)
    return state, updated
