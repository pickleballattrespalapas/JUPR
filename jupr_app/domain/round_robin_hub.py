"""Pure court allocation for existing generator sessions.

The session owns all match results and standings.  The hub owns only court
assignments and player availability.  Callers persist both together atomically.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone, timedelta
from typing import Any

from jupr_app.domain.adaptive_play_engine import (
    active_participant_ids, advance_generator_event, mark_generator_round_played, save_generator_round,
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
    if match.get("status") == "not_played":
        return True
    if event.get("scoringMode") == "unscored":
        return match.get("status") == "played"
    return match.get("scoreA") is not None and match.get("scoreB") is not None


def _actually_committed_match(event: dict[str, Any], match: dict[str, Any]) -> bool:
    """A court release does not undo a dispatch or erase a recorded lineup."""
    return (str(match.get("id")) in (event.get("hubAssignedMatchIds") or [])
            or bool(match.get("hubStartedAt")) or _complete(match, event)
            or match.get("scoreA") is not None or match.get("scoreB") is not None)


def _committed_match(event: dict[str, Any], match: dict[str, Any]) -> bool:
    return _actually_committed_match(event, match) or bool(match.get("hubPlanLocked"))


def _departure_round(event: dict[str, Any], participant_id: str) -> int:
    committed_rounds = [int(row["number"]) for row in event.get("rounds") or [] for match in _matches(row)
        if _actually_committed_match(event, match) and any(str(p["id"]) == participant_id for p in _participants(event, match))]
    current = int(event.get("currentRoundNumber") or 1)
    return max(current, max(committed_rounds, default=current - 1) + 1)


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
    # A dedication excludes other groups; its owner can still use shared courts.
    courts = [c for c in hub.get("courts") or [] if c.get("open", True)]
    dedicated = [c for c in courts if c.get("dedicated_session_key") == session_key]
    shared = [c for c in courts if not c.get("dedicated_session_key")]
    return [str(c["id"]) for c in dedicated + shared]


def _next_open_court(hub: dict[str, Any], allowed: list[str]) -> dict[str, Any] | None:
    # Prefer the group's dedicated courts, preserving shared capacity for others.
    courts = {str(c["id"]): c for c in hub.get("courts") or []}
    return next((courts[cid] for cid in allowed
                 if not courts[cid].get("current") and not courts[cid].get("next")), None)


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
            "can_merge": len({node["profile_id"] for node in nodes if node["profile_id"] is not None}) <= 1 and len(identities & claims.keys()) <= 1,
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
            if len(selected_identities & _claims(hub, events).keys()) > 1:
                raise ValueError("Finish or resolve affected assignments before marking these as the same person; one person cannot keep two court commitments.")
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
    if resolution == "same":
        _consolidate_identity_memberships(hub, events, now)


def _memberships_overlap(event: dict[str, Any], left: dict[str, Any], right: dict[str, Any]) -> bool:
    current = int(event.get("currentRoundNumber") or 1)
    boundaries = {current}
    for person in (left, right):
        boundaries.add(int(person.get("active_from_round") or 1))
        if person.get("inactive_from_round") is not None:
            boundaries.add(int(person["inactive_from_round"]))
        for number in person.get("inactive_rounds") or []:
            boundaries.update((int(number), int(number) + 1))
    for number in event.get("rosterByRound") or {}:
        boundaries.update((int(number), int(number) + 1))
    ids = {str(left["id"]), str(right["id"])}
    return any(ids <= set(active_participant_ids(event, number)) for number in boundaries if number >= current)


def _consolidate_identity_memberships(hub: dict[str, Any], events: dict[str, Any], now: str) -> None:
    """A confirmed person gets one future seat per group; historical records remain."""
    for key in sorted(_group_keys(hub)):
        event = events[key]
        if _roster_change_unavailable_reason(event):
            continue
        identities: dict[str, list[dict[str, Any]]] = {}
        for person in event.get("participants") or []:
            if _can_remove_participant(event, person):
                identities.setdefault(player_identity(person, key, hub), []).append(person)
        for members in identities.values():
            if len(members) < 2:
                continue
            committed = {str(p["id"]) for row in event.get("rounds") or [] for match in _matches(row)
                if _committed_match(event, match) and not _complete(match, event)
                for p in _participants(event, match)}
            active_committed = [p for p in members if str(p["id"]) in committed]
            if len(active_committed) > 1:
                raise ValueError("Resolve the duplicate person's started games before combining their roster entries.")
            members.sort(key=lambda p: (str(p["id"]) not in committed,
                p.get("inactive_from_round") is not None, int(p.get("roster_order") or 0), str(p["id"])))
            retained = members[0]
            for duplicate in members[1:]:
                if not _memberships_overlap(event, retained, duplicate):
                    continue
                effective = _departure_round(event, str(duplicate["id"]))
                old_event = deepcopy(event)
                old_retained = deepcopy(retained)
                retained["active_from_round"] = min(int(retained.get("active_from_round") or 1),
                    max(effective, int(duplicate.get("active_from_round") or 1)))
                ends = [p.get("inactive_from_round") for p in (retained, duplicate)]
                retained["inactive_from_round"] = None if None in ends else max(map(int, ends))
                exceptional = set(map(int, retained.get("inactive_rounds") or [])) | set(map(int, duplicate.get("inactive_rounds") or []))
                retained["inactive_rounds"] = sorted(number for number in exceptional
                    if (number < effective and number in (old_retained.get("inactive_rounds") or []))
                    or (number >= effective and not {str(retained["id"]), str(duplicate["id"])} & set(active_participant_ids(old_event, number))))
                for number, ids in (event.get("rosterByRound") or {}).items():
                    if int(number) >= effective and str(duplicate["id"]) in ids and str(retained["id"]) not in ids:
                        ids.append(str(retained["id"]))
                event = mutate_hub_roster(event, action="remove", participant_id=str(duplicate["id"]))
                current_by_id = {str(p["id"]): p for p in event["participants"]}
                retained = current_by_id[str(retained["id"])]
                removed = current_by_id[str(duplicate["id"])]
                removed["merged_into"] = str(retained["id"])
                event.setdefault("hubIdentityMembershipChanges", []).append({"at": now,
                    "participantId": str(duplicate["id"]), "retainedParticipantId": str(retained["id"]),
                    "effectiveRound": removed.get("inactive_from_round")})
            events[key] = event
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
            if row.get("status") not in {"active", "preview"}:
                raise ValueError("A court assignment points to a completed round.")
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


def _roster_change_unavailable_reason(event: dict[str, Any]) -> str | None:
    if event.get("status") != "active":
        return "Roster changes require an active round robin."
    if event.get("hubSupplemental"):
        return "An exhibition keeps its selected four players."
    if is_moneyball(event):
        return "Moneyball’s starting lineup is fixed. Use the generator for substitutions."
    if event.get("playoff"):
        return "Playoff teams are fixed after seeding."
    return None


def _add_player_unavailable_reason(event: dict[str, Any]) -> str | None:
    unavailable = _roster_change_unavailable_reason(event)
    if unavailable:
        return unavailable
    if len(event.get("participants") or []) >= 40:
        return "This round robin already has the maximum of 40 players."
    return None


def _can_remove_participant(event: dict[str, Any], participant: dict[str, Any]) -> bool:
    from jupr_app.domain.adaptive_play_engine import participant_has_upcoming_membership
    return participant_has_upcoming_membership(event, participant)


def _can_rejoin_profile(event: dict[str, Any], session_key: str, hub: dict[str, Any],
                        identity: str, claims: dict[str, Any]) -> bool:
    existing = [p for p in event.get("participants") or [] if player_identity(p, session_key, hub) == identity]
    if not existing:
        return True
    current = int(event.get("currentRoundNumber") or 1)
    if any(p.get("inactive_from_round") is None or int(p["inactive_from_round"]) > current for p in existing):
        return False
    claim = claims.get(identity)
    if claim and claim[0] == session_key:
        return False
    remaining_rounds = {current, current + 1}
    remaining_rounds.update(int(r["number"]) for r in event.get("rounds") or [] if int(r["number"]) >= current)
    remaining_rounds.update(int(number) for number in event.get("rosterByRound") or {} if int(number) >= current)
    remaining_ids = {pid for number in remaining_rounds for pid in active_participant_ids(event, number)}
    return all(str(p["id"]) not in remaining_ids for p in existing)


def _finish_details(event: dict[str, Any]) -> dict[str, Any]:
    assigned = set(event.get("hubAssignedMatchIds") or [])
    touched = [int(r["number"]) for r in event.get("rounds") or []
        if r.get("status") in {"saved", "played"} or r.get("hubStartedAt") or any(
            str(m["id"]) in assigned or m.get("hubStartedAt") or _complete(m, event)
            or m.get("scoreA") is not None or m.get("scoreB") is not None for m in _matches(r))]
    boundary = max(touched, default=0)
    if is_moneyball(event) or event.get("playoff"):
        boundary = max((int(r["number"]) for r in event.get("rounds") or []), default=boundary)
    requested = event.get("hubFinishAfterRound")
    if requested is not None:
        boundary = int(requested)
    pending = sum(not _complete(m, event) for r in event.get("rounds") or []
        if int(r["number"]) <= boundary and r.get("status") != "skipped" for m in _matches(r))
    current = int(event.get("currentRoundNumber") or 1)
    options = {boundary}
    if not is_moneyball(event) and not event.get("playoff"):
        options.update(int(r["number"]) for r in event.get("rounds") or []
            if int(r["number"]) in {current, current + 1} and int(r["number"]) >= boundary
            and int(r["number"]) <= int(event.get("hubPreparedThroughRound") or current)
            and r.get("status") != "skipped")
    return {"can_finish": event.get("status") == "active" and requested is None,
        "finish_unavailable_reason": ("This round robin is already finishing." if requested is not None
            else "This round robin is complete." if event.get("status") != "active" else None),
        "finish_requested": requested is not None, "finish_after_round": requested,
        "finish_boundary_round": boundary, "finish_pending_matches": pending,
        "finish_round_options": [{"round_number": number, "pending_matches": sum(not _complete(m, event)
            for r in event.get("rounds") or [] if int(r["number"]) <= number and r.get("status") != "skipped"
            for m in _matches(r))} for number in sorted(options)]}


def _trim_replaceable_hub_previews(event: dict[str, Any]) -> None:
    """Retain history and commitments; an unlocked distant preview is not a promise."""
    horizon = int(event.get("currentRoundNumber") or 1) + 1
    retained = []
    for row in event.get("rounds") or []:
        number = int(row["number"])
        if number <= horizon or row.get("status") != "preview":
            retained.append(row)
            continue
        active = set(active_participant_ids(event, number))
        people = lambda match: set(map(str, (match.get("sideA") or match.get("teamA") or [])
            + (match.get("sideB") or match.get("teamB") or [])))
        matches = _matches(row)
        protected = [m for m in matches if _actually_committed_match(event, m)
            or (m.get("hubPlanLocked") and people(m) <= active)]
        invalid_locks = [m for m in matches if m.get("hubPlanLocked") and m not in protected]
        if invalid_locks:
            event.setdefault("hubPlanAdjustments", []).append({"reason": "player_departure", "round": number,
                "unlockedMatchIds": [str(m["id"]) for m in invalid_locks]})
        if not protected:
            continue
        if len(protected) != len(matches):
            # Keep the actual promise, not the unrelated old cohort templates
            # around it. Fill this round's remainder when it enters the window.
            row["matches"] = protected
            row.pop("courts", None)
            row["hubPlanPartial"] = True
            playing = {pid for match in protected for pid in people(match)}
            row["byeParticipantIds"] = sorted(active - playing)
            from jupr_app.domain.adaptive_play_engine import generator_match_play_format
            row["formatCounts"] = {kind: sum(generator_match_play_format(m, event.get("playFormat") or "doubles") == kind
                for m in protected) for kind in ("doubles", "singles")}
        retained.append(row)
    event["rounds"] = retained


def _plan_hub_round(event: dict[str, Any], number: int, *, preserve_unaffected: bool = False) -> None:
    """Choose a complete group rotation before results or court speed can affect it."""
    from jupr_app.domain.adaptive_play_engine import (
        _apply_match_history, _doubles_round, _singles_round, _mixed_round,
        generator_match_play_format, history_before_round,
    )

    row = next((r for r in event.get("rounds") or [] if int(r["number"]) == number), None)
    if row is not None and row.get("status") in {"saved", "played", "skipped"}:
        return
    active = list(active_participant_ids(event, number))
    active_set = set(active)
    people = lambda match: {str(pid) for pid in
        (match.get("sideA") or match.get("teamA") or []) + (match.get("sideB") or match.get("teamB") or [])}
    unlocked = [m for m in _matches(row or {}) if m.get("hubPlanLocked")
        and not _actually_committed_match(event, m) and not people(m) <= active_set]
    if unlocked:
        event.setdefault("hubPlanAdjustments", []).append({"reason": "player_departure", "round": number,
            "unlockedMatchIds": [str(m["id"]) for m in unlocked]})
    kept = [m for m in _matches(row or {}) if _actually_committed_match(event, m)
        or (m.get("hubPlanLocked") and people(m) <= active_set)
        or (preserve_unaffected and people(m) <= active_set)]
    occupied = {pid for match in kept for pid in people(match)}
    later = {pid for r in event.get("rounds") or [] if int(r["number"]) > number
        for m in _matches(r) if _actually_committed_match(event, m) for pid in people(m)}
    waiting = [pid for pid in active if pid not in occupied and pid not in later]
    deferred = [pid for pid in active if pid not in occupied and pid in later]
    history = history_before_round(event, number, include_preview=True)
    play_format = event.get("playFormat") or "doubles"
    for match in kept:
        _apply_match_history(history, match, play_format)
    roster_pos = {str(p["id"]): int(p.get("roster_order") or 0) for p in event.get("participants") or []}
    if play_format == "doubles_singles":
        counts = {kind: sum(generator_match_play_format(m, play_format) == kind for m in kept)
            for kind in ("doubles", "singles")}
        generated, byes, warnings, _ = _mixed_round(active_ids=waiting,
            doubles_court_count=max(0, int(event.get("doublesCourtCount") or 0) - counts["doubles"]),
            singles_court_count=max(0, int(event.get("singlesCourtCount") or 0) - counts["singles"]),
            history=history, roster_pos=roster_pos, round_number=number)
    else:
        generate = _singles_round if play_format == "singles" else _doubles_round
        options = {} if play_format == "singles" else {"search_attempts": 64, "subset_attempts": 4}
        generated, byes, warnings = generate(active_ids=waiting, court_count=0,
            history=history, roster_pos=roster_pos, round_number=number, **options)
    revision = int(event.get("hubPlanRevision") or 0) + 1
    event["hubPlanRevision"] = revision
    offset = max((int(m.get("court") or 0) for m in kept), default=0)
    for index, match in enumerate(generated, 1):
        match.update(id=f"hub-plan-v{revision}-r{number}-m{index}", court=offset + index)
    if row is None:
        row = {"number": number, "status": "preview", "savedAt": None, "skippedAt": None}
        event.setdefault("rounds", []).append(row)
    row.update(matches=[*kept, *generated], byeParticipantIds=list(dict.fromkeys([*byes, *deferred])),
        warnings=warnings, hubPlanRevision=revision)
    row.pop("hubPlanPartial", None)
    row.pop("courts", None)
    row["formatCounts"] = {kind: sum(generator_match_play_format(m, play_format) == kind for m in row["matches"])
        for kind in ("doubles", "singles")}
    event.setdefault("rosterByRound", {})[str(number)] = active
    event["rounds"].sort(key=lambda item: int(item["number"]))
    event["totalRounds"] = max(int(event.get("totalRounds") or 0), number)


def repair_hub_roster_plan(event: dict[str, Any], *, from_round: int | None = None) -> dict[str, Any]:
    """Repair one roster transaction without freezing old arrival cohorts."""
    result = deepcopy(event)
    if is_moneyball(result) or result.get("playoff") or result.get("generatorKind") != "round_robin":
        return result
    _trim_replaceable_hub_previews(result)
    current = int(result.get("currentRoundNumber") or 1)
    first = current if from_round is None else max(current, from_round)
    boundary = result.get("hubFinishAfterRound")
    numbers = {current, current + 1}
    for number in sorted(numbers):
        if number < first or (boundary is not None and number > int(boundary)):
            continue
        _plan_hub_round(result, number, preserve_unaffected=number == current)
    result["hubPlanVersion"] = 1
    result["hubRollingEnabled"] = True
    result["hubPreparedThroughRound"] = max(int(r["number"]) for r in result["rounds"])
    return result


def _prepare_rolling_rounds(event: dict[str, Any]) -> dict[str, Any]:
    """Keep the current round and one stable full-group successor ready."""
    from jupr_app.domain.adaptive_play_engine import prepare_hub_round

    result = deepcopy(event)
    result["hubRollingEnabled"] = True
    result["hubPreparedThroughRound"] = max((int(r["number"]) for r in result.get("rounds") or []
        if all((m.get("sideA") or m.get("teamA")) and (m.get("sideB") or m.get("teamB")) for m in _matches(r))),
        default=int(result.get("currentRoundNumber") or 1))
    boundary = result.get("hubFinishAfterRound")
    if result.get("status") != "active" or (boundary is not None
            and (is_moneyball(result) or result.get("playoff"))):
        return result
    if result.get("playoff"):
        for row in result.get("rounds") or []:
            if int(row["number"]) > int(result.get("currentRoundNumber") or 1):
                try:
                    result = prepare_hub_round(result, int(row["number"]))
                except ValueError:
                    pass  # Playoff successors require the previous phase's results.
        result["hubPreparedThroughRound"] = max((int(r["number"]) for r in result.get("rounds") or []
            if all((m.get("sideA") or m.get("teamA")) and (m.get("sideB") or m.get("teamB")) for m in _matches(r))),
            default=int(result.get("currentRoundNumber") or 1))
        return result
    if is_moneyball(result):
        return result
    _trim_replaceable_hub_previews(result)
    current = int(result.get("currentRoundNumber") or 1)
    if _round(result, current).get("hubPlanPartial") and (boundary is None or current <= int(boundary)):
        _plan_hub_round(result, current)
    upcoming = current + 1
    next_row = next((r for r in result["rounds"] if int(r["number"]) == upcoming), None)
    if (boundary is None or upcoming <= int(boundary)) and (
            result.get("hubPlanVersion") != 1 or next_row is None or next_row.get("hubPlanPartial")):
        _plan_hub_round(result, upcoming)
    result["hubPlanVersion"] = 1
    result["hubPreparedThroughRound"] = max(int(r["number"]) for r in result["rounds"])
    return result


def _settle_hub_rounds(event: dict[str, Any]) -> dict[str, Any]:
    """Commit standings in order even when later games finished first."""
    from jupr_app.domain.adaptive_play_engine import complete_generator_event

    result = deepcopy(event)
    result["hubRollingEnabled"] = True
    while result.get("status") == "active":
        row = _round(result)
        number = int(row["number"])
        if row.get("status") not in {"saved", "played", "skipped"}:
            matches = _matches(row)
            if not matches:
                if result.get("hubFinishAfterRound") is not None or any(int(r["number"]) > number and _matches(r) for r in result.get("rounds") or []):
                    row.update(status="skipped", skipReason="No matches remain after the roster change.")
                else:
                    break
            elif not all(_complete(m, result) for m in matches):
                break
            if not matches:
                pass
            elif result.get("scoringMode") == "unscored":
                result = mark_generator_round_played(result, round_number=number)
            else:
                result = save_generator_round(result, round_number=number, scores=[
                    {"match_id": m["id"], "score_a": m.get("scoreA"), "score_b": m.get("scoreB")} for m in matches])
        boundary = result.get("hubFinishAfterRound")
        if boundary is not None and number >= int(boundary):
            result = complete_generator_event(result)
            break
        result = _auto_advance_hub_round(result)
    return result


def _repair_paused_waiting_games(hub: dict[str, Any], events: dict[str, dict[str, Any]], *, now: str,
                                 excluded_session_keys: set[str] | None = None) -> None:
    """Repair only paused games at the players' next unstarted opportunity.

    This is a write-path operation. A free bye takes a paused player's place
    without disturbing their teammates. If no such substitution is possible,
    only the affected games may be regrouped; started games and healthy plans
    remain authoritative. Pausing does not change anybody's roster membership.
    """
    from jupr_app.domain.adaptive_play_engine import (
        _apply_match_history, _doubles_round, _singles_round, generator_match_play_format,
        history_before_round,
    )

    claims = _claims(hub, events)
    pending_identities = _identity_data(hub, events)["pending"]
    assigned = {_key(c[slot]) for c in hub.get("courts") or [] for slot in ("current", "next") if c.get(slot)}

    def people(match: dict[str, Any]) -> list[str]:
        return [str(pid) for pid in (match.get("sideA") or match.get("teamA") or [])
            + (match.get("sideB") or match.get("teamB") or [])]

    def protected(key: str, match: dict[str, Any], event: dict[str, Any]) -> bool:
        return (key, str(match["id"])) in assigned or _committed_match(event, match)

    def distinct(ids: list[str], identities: dict[str, str]) -> list[str]:
        seen: set[str] = set()
        result = []
        for pid in ids:
            if identities[pid] not in seen:
                result.append(pid)
                seen.add(identities[pid])
        return result

    for key in sorted(_group_keys(hub) - (excluded_session_keys or set())):
        event = events[key]
        if (event.get("status") != "active" or event.get("generatorKind") != "round_robin"
                or is_moneyball(event) or event.get("playoff")):
            continue
        by_id = {str(p["id"]): p for p in event.get("participants") or []}
        identities = {pid: player_identity(p, key, hub) for pid, p in by_id.items()}
        paused = {pid for pid, identity in identities.items() if hub["players"][identity].get("paused")}
        # A pause is an absence, not extra rotation credit. Returning while
        # this round is still pending restores a bye, never an old matchup.
        current_row = _round(event)
        if current_row.get("status") in {"active", "preview"}:
            active_now = set(active_participant_ids(event, int(current_row["number"])))
            playing_now = {pid for match in _matches(current_row) for pid in people(match)}
            old_byes = {str(pid) for pid in current_row.get("byeParticipantIds") or []}
            old_absent = {str(pid) for pid in current_row.get("hubPausedParticipantIds") or []}
            paused_byes = old_byes & paused
            returning_byes = (old_absent - paused) & (active_now - playing_now)
            if paused_byes or returning_byes:
                current_row["byeParticipantIds"] = sorted((old_byes - paused_byes) | returning_byes)
                current_row["hubPausedParticipantIds"] = sorted((old_absent | paused_byes) - returning_byes)
        if not paused:
            continue
        play_format = event.get("playFormat") or "doubles"
        roster_pos = {pid: int(p.get("roster_order") or 0) for pid, p in by_id.items()}
        priority = lambda pid: (-_wait(hub["players"][identities[pid]], now), roster_pos[pid], pid)
        changed = False
        for row in sorted(event.get("rounds") or [], key=lambda r: int(r["number"])):
            number = int(row["number"])
            if (number < int(event.get("currentRoundNumber") or 1) or row.get("status") not in {"active", "preview"}
                    or (event.get("hubFinishAfterRound") is not None and number > int(event["hubFinishAfterRound"]))):
                continue
            matches = _matches(row)
            active = {str(pid) for pid in active_participant_ids(event, number) if str(pid) in by_id}
            # In particular, an all-paused group must not skip its entire plan
            # or keep manufacturing empty continuation rounds on every write.
            minimum = 2 if play_format == "singles" or (play_format == "doubles_singles" and int(event.get("singlesCourtCount") or 0)) else 4
            if len({identities[pid] for pid in active - paused if identities[pid] not in pending_identities}) < minimum:
                continue
            earliest: dict[str, int] = {}
            later_protected: set[str] = set()
            for planned in sorted(event.get("rounds") or [], key=lambda r: int(r["number"])):
                if planned.get("status") in {"saved", "played", "skipped"}:
                    continue
                for match in _matches(planned):
                    ids = [identities[pid] for pid in people(match) if pid in identities]
                    if int(planned["number"]) > number and protected(key, match, event):
                        later_protected.update(ids)
                    if not _complete(match, event):
                        for identity in ids:
                            earliest.setdefault(identity, int(planned["number"]))

            def available(pid: str) -> bool:
                identity = identities[pid]
                return (pid in active and pid not in paused and identity not in pending_identities
                    and identity not in claims and identity not in later_protected and earliest.get(identity, number) >= number)

            affected = []
            for match in matches:
                ids = people(match)
                if not paused.intersection(ids) or protected(key, match, event) or any(pid not in identities for pid in ids):
                    continue
                # Future previews wait until their participants actually reach
                # them; repairing a pause must not replan the whole schedule.
                if (any(earliest.get(identities[pid], number) < number for pid in ids)
                        or len({identities[pid] for pid in ids}) != len(ids)):
                    continue
                affected.append(match)
            if not affected:
                continue
            affected_ids = {str(m["id"]) for m in affected}
            kept = [m for m in matches if str(m["id"]) not in affected_ids]
            used_identities = {identities[pid] for m in matches for pid in people(m) if pid in identities}
            donors = sorted((pid for pid in active if available(pid) and identities[pid] not in used_identities), key=priority)
            # Distinct local roster entries can resolve to the same person.
            donors = distinct(donors, identities)
            replacements: list[dict[str, Any]] = []
            regroup = []
            for match in affected:
                needed = sum(pid in paused for pid in people(match))
                if len(donors) >= needed and all(available(pid) for pid in people(match) if pid not in paused):
                    donor_iter = iter(donors[:needed])
                    donors = donors[needed:]
                    replacement = deepcopy(match)
                    for side, alias in (("sideA", "teamA"), ("sideB", "teamB")):
                        ids = [next(donor_iter) if str(pid) in paused else str(pid) for pid in match.get(side) or match.get(alias) or []]
                        replacement[side] = ids
                        replacement[alias] = list(ids)
                    replacements.append(replacement)
                else:
                    regroup.append(match)
            direct_by_id = {str(m["id"]): m for m in replacements}
            if regroup:
                # A no-bye round may have several paused games. Pooling only
                # their survivors avoids stealing from an unaffected matchup
                # and moving the same blockage elsewhere.
                pool = sorted({pid for match in regroup for pid in people(match) if available(pid)} | set(donors), key=priority)
                pool = distinct(pool, identities)
                history = history_before_round(event, number, include_preview=True)
                for match in [*kept, *replacements]:
                    _apply_match_history(history, match, play_format)
                revision = int(event.get("hubPauseRevision") or 0) + 1
                for template in regroup:
                    kind = generator_match_play_format(template, play_format)
                    size = 2 if kind == "singles" else 4
                    if len(pool) < size:
                        continue
                    selected, pool = pool[:size], pool[size:]
                    generate = _singles_round if kind == "singles" else _doubles_round
                    generated, _, _ = generate(active_ids=selected, court_count=1,
                        history=history, roster_pos=roster_pos, round_number=number)
                    for match in generated:
                        match.update(id=f"hub-r{number}-pause{revision}-m{len(replacements) + 1}", court=template.get("court", 1))
                        replacements.append(match)
                        _apply_match_history(history, match, play_format)
                if not replacements and not kept:
                    continue  # No legal game yet: retain the explicit pause blocker.
                if (not replacements and number > int(event.get("currentRoundNumber") or 1)
                        and not any(protected(key, match, event) for match in kept)):
                    # Do not erase underfilled games through untouched future
                    # previews merely to ask preparation for another template.
                    continue
                event["hubPauseRevision"] = revision
            regroup_ids = {str(m["id"]) for m in regroup}
            row["matches"] = [direct_by_id.get(str(m["id"]), m) for m in matches if str(m["id"]) not in regroup_ids]
            row["matches"].extend(m for m in replacements if str(m["id"]) not in direct_by_id)
            row.pop("courts", None)
            playing = {pid for m in row["matches"] for pid in people(m)}
            row["byeParticipantIds"] = sorted(active - playing - paused, key=lambda pid: (roster_pos[pid], pid))
            row["hubPausedParticipantIds"] = sorted((active & paused) - playing)
            row["formatCounts"] = {kind: sum(generator_match_play_format(m, play_format) == kind for m in row["matches"])
                for kind in ("singles", "doubles")}
            row.setdefault("hubPauseAdjustments", []).append({"at": now, "originalMatches": deepcopy(affected),
                "replacementMatchIds": [str(m["id"]) for m in replacements]})
            event["perfectRotation"] = False
            event["adaptiveFromRound"] = min(int(event.get("adaptiveFromRound") or number), number)
            changed = True
        if changed:
            events[key] = _prepare_rolling_rounds(_settle_hub_rounds(event))


def _seat_resumed_byes(hub: dict[str, Any], events: dict[str, dict[str, Any]], *, identity: str | None,
                       now: str, excluded_session_keys: set[str]) -> None:
    """Use free players in the current round without undoing any promised game."""
    from jupr_app.domain.adaptive_play_engine import (
        _apply_match_history, _doubles_round, _singles_round, _mixed_round,
        generator_match_play_format, history_before_round,
    )

    claims = _claims(hub, events)
    unresolved = _identity_data(hub, events)["pending"]
    for key in sorted(_group_keys(hub) - excluded_session_keys):
        event = events[key]
        if _roster_change_unavailable_reason(event) or event.get("hubFinishAfterRound") is not None:
            continue
        row = _round(event)
        if row.get("status") not in {"active", "preview"}:
            continue
        if identity is None and not row.get("hubPauseAdjustments") and "hubPausedParticipantIds" not in row:
            continue  # Fill repairs a prior pause; ordinary bye plans stay intact.
        number = int(row["number"])
        by_id = {str(p["id"]): p for p in event.get("participants") or []}
        identities = {pid: player_identity(p, key, hub) for pid, p in by_id.items()}
        playing = {str(p["id"]) for m in _matches(row) for p in _participants(event, m)}
        committed = {str(p["id"]) for r in event.get("rounds") or [] if int(r["number"]) > number
            for m in _matches(r) if _complete(m, event) or m.get("hubStartedAt")
            or str(m["id"]) in (event.get("hubAssignedMatchIds") or [])
            or m.get("scoreA") is not None or m.get("scoreB") is not None
            for p in _participants(event, m)}
        occupied_identities = {identities[pid] for pid in playing | committed}
        free = [pid for pid in active_participant_ids(event, number) if identities[pid] not in occupied_identities
            and identities[pid] not in claims and identities[pid] not in unresolved
            and not hub["players"][identities[pid]].get("paused")]
        if identity is not None and identity not in {identities[pid] for pid in free}:
            continue
        # A shared identity may have multiple roster entries, but only one seat.
        free = list({identities[pid]: pid for pid in free}.values())
        roster_pos = {pid: i for i, pid in enumerate(sorted(free,
            key=lambda pid: (-_wait(hub["players"][identities[pid]], now), int(by_id[pid].get("roster_order") or 0), pid)))}
        history = history_before_round(event, number, include_preview=True)
        play_format = event.get("playFormat") or "doubles"
        for match in _matches(row):
            _apply_match_history(history, match, play_format)
        if play_format == "doubles_singles":
            counts = {kind: sum(generator_match_play_format(m, play_format) == kind for m in _matches(row)) for kind in ("singles", "doubles")}
            games, _, _, _ = _mixed_round(active_ids=free,
                doubles_court_count=max(0, int(event.get("doublesCourtCount") or 0) - counts["doubles"]),
                singles_court_count=max(0, int(event.get("singlesCourtCount") or 0) - counts["singles"]),
                history=history, roster_pos=roster_pos, round_number=number)
        else:
            generate = _singles_round if play_format == "singles" else _doubles_round
            games, _, _ = generate(active_ids=free, court_count=0,
                history=history, roster_pos=roster_pos, round_number=number)
        if not games:
            continue
        revision = int(event.get("hubResumeRevision") or 0) + 1
        offset = max((int(m.get("court") or 0) for m in _matches(row)), default=0)
        for i, match in enumerate(games, 1):
            match.update(id=f"hub-r{number}-resume{revision}-m{i}", court=offset + i)
        row["matches"] = [*_matches(row), *games]
        row.pop("courts", None)
        playing.update(str(p["id"]) for m in games for p in _participants(event, m))
        active = set(active_participant_ids(event, number))
        paused = {pid for pid in active if hub["players"][identities[pid]].get("paused")}
        row["byeParticipantIds"] = sorted(active - playing - paused)
        row["hubPausedParticipantIds"] = sorted(paused - playing)
        row["formatCounts"] = {kind: sum(generator_match_play_format(m, play_format) == kind for m in row["matches"]) for kind in ("singles", "doubles")}
        event.update(hubResumeRevision=revision, perfectRotation=False,
            adaptiveFromRound=min(int(event.get("adaptiveFromRound") or number), number))
        events[key] = _prepare_rolling_rounds(event)


def build_hub_snapshot(hub: dict[str, Any], events: dict[str, dict[str, Any]], *, now: str) -> dict[str, Any]:
    state = deepcopy(hub)
    _ensure_players(state, events, now)
    claims = _claims(state, events)
    identity_data = _identity_data(state, events)
    assignments: dict[tuple[str, str], tuple[dict, str]] = {}
    playing_identities: set[str] = set()
    person_courts: dict[str, dict[str, dict[str, Any]]] = {}
    for court in state.get("courts") or []:
        for slot in ("current", "next"):
            if court.get(slot):
                assignments[_key(court[slot])] = (court, slot)
                assigned_key = str(court[slot]["session_key"])
                assigned_event, _, assigned_match = _find_match(state, events, court[slot])
                for person in _participants(assigned_event, assigned_match):
                    person_courts.setdefault(player_identity(person, assigned_key, state), {})[slot] = {
                        "session_key": assigned_key, "group_title": assigned_event.get("name") or assigned_key,
                        "court_id": str(court["id"]), "court_label": court["label"]}
                if slot == "current":
                    current_event, _, current_match = _find_match(state, events, court[slot])
                    playing_identities.update(player_identity(p, court[slot]["session_key"], state) for p in _participants(current_event, current_match))
    ready, blocked, reserved, groups, all_matches = [], [], [], [], {}
    public_players: dict[str, dict[str, Any]] = {}
    for group in state.get("groups") or []:
        key = str(group["session_key"])
        event = events[key]
        row = _round(event)
        active = set(active_participant_ids(event, int(row["number"])))
        add_unavailable = _add_player_unavailable_reason(event)
        remove_unavailable = _roster_change_unavailable_reason(event)
        completed = sum(_complete(m, event) for m in _matches(row))
        groups.append({"session_key": key, "title": event.get("name") or key,
            "registration_driven": event.get("hubRegistrationDriven") is True,
            "minimum_players": 4 if event.get("playFormat") == "doubles" else 2,
            "active_player_count": len(active),
            "upcoming_player_count": 0, "departed_player_count": 0,
            "roster_entry_count": len(event.get("participants") or []),
            "add_player_effective_round": int(row["number"]),
            "current_round_number": int(row["number"]), "round_status": row.get("status"),
            "planned_through_round": int(event.get("hubPreparedThroughRound") or row["number"]),
            "next_round_number": next((int(r["number"]) for r in event.get("rounds") or []
                if int(r["number"]) == int(row["number"]) + 1 and r.get("status") != "skipped"), None),
            "plan_revision": int(event.get("hubPlanRevision") or 0),
            "next_round_matches": [],
            "supplemental": bool(event.get("hubSupplemental")),
            "standings_policy": "exhibition" if event.get("hubSupplemental") else "group",
            "total_rounds": int(event.get("totalRounds") or len(event.get("rounds") or [])),
            "completed_matches": completed, "total_matches": len(_matches(row)),
            "played_matches": sum(_complete(m, event) and m.get("status") != "not_played" for m in _matches(row)),
            "not_played_matches": sum(m.get("status") == "not_played" for m in _matches(row)),
            "pending_matches": len(_matches(row)) - completed,
            "can_advance": row.get("status") in {"saved", "played", "skipped"} and event.get("status") == "active",
            "advance_label": "Finish session" if ((is_moneyball(event) and all(r.get("status") == "saved" for r in event.get("rounds") or [])) or (event.get("playoff") and int(row["number"]) >= int(event.get("totalRounds") or 0))) else "Next round",
            "event_format": event.get("eventFormat") or "standard", "scoring_mode": event.get("scoringMode") or "scored",
            "rating_mode": event.get("ratingMode") or "unrated", "status": event.get("status"),
            **_finish_details(event),
            "can_add_player": add_unavailable is None, "add_player_unavailable_reason": add_unavailable,
            "can_remove_player": remove_unavailable is None, "remove_player_unavailable_reason": remove_unavailable,
            "players": [],
            "dedicated_court_ids": [c["id"] for c in state.get("courts") or [] if c.get("dedicated_session_key") == key]})
        by_id = {str(p["id"]): p for p in event.get("participants") or []}
        for participant in event.get("participants") or []:
            identity = player_identity(participant, key, state)
            upcoming = _can_remove_participant(event, participant)
            can_remove = remove_unavailable is None and upcoming
            is_active = str(participant["id"]) in active
            membership_status = "active" if is_active else ("upcoming" if upcoming else "departed")
            if not is_active:
                groups[-1]["upcoming_player_count" if upcoming else "departed_player_count"] += 1
            courts = person_courts.get(identity, {})
            current_court, reserved_court = courts.get("current"), courts.get("next")
            on_court = bool(current_court and current_court["session_key"] == key and is_active)
            reserved_here = bool(reserved_court and reserved_court["session_key"] == key and upcoming)
            info = public_players.setdefault(identity, {"identity": identity, "name": participant.get("name"),
                "display_name": identity_data["nodes"][(key, str(participant["id"]))]["display_name"],
                "paused": bool(state["players"][identity].get("paused")),
                "wait_seconds": 0 if identity in playing_identities else int(_wait(state["players"][identity], now)), "session_keys": [], "participant_ids": {},
                "on_court": identity in playing_identities, "reserved": identity in claims and identity not in playing_identities})
            if key not in info["session_keys"]:
                info["session_keys"].append(key)
            if key not in info["participant_ids"] or upcoming:
                info["participant_ids"][key] = str(participant["id"])
            groups[-1]["players"].append({
                "participant_id": str(participant["id"]), "name": participant.get("name"),
                "display_name": identity_data["nodes"][(key, str(participant["id"]))]["display_name"],
                "identity": identity, "profile_id": participant.get("player_id"),
                "can_rejoin": add_unavailable is None and _can_rejoin_profile(event, key, state, identity, claims),
                "can_remove": can_remove,
                "remove_effective_round": _departure_round(event, str(participant["id"])),
                "remove_affected_match_ids": [str(m["id"]) for r in event.get("rounds") or []
                    if int(r["number"]) >= _departure_round(event, str(participant["id"]))
                    for m in _matches(r) if not _actually_committed_match(event, m)
                    and str(participant["id"]) in (m.get("sideA") or m.get("teamA") or []) + (m.get("sideB") or m.get("teamB") or [])],
                "membership_status": membership_status,
                "current_court": current_court, "reserved_court": reserved_court,
                "on_court": on_court, "reserved": reserved_here,
                "playing_elsewhere": bool(current_court and current_court["session_key"] != key),
                "reserved_elsewhere": bool(reserved_court and reserved_court["session_key"] != key),
                "remove_unavailable_reason": None if can_remove else remove_unavailable or "This player has already left this round robin.",
                "active": str(participant["id"]) in active,
                "on_bye": not on_court and str(participant["id"]) in active and str(participant["id"]) in {str(pid) for pid in row.get("byeParticipantIds") or []},
                "active_from_round": int(participant.get("active_from_round") or 1),
                "inactive_from_round": participant.get("inactive_from_round"),
                "inactive_rounds": list(participant.get("inactive_rounds") or []),
                **{field: info[field] for field in ("paused", "wait_seconds")},
            })
        next_round = next((r for r in event.get("rounds") or []
            if int(r["number"]) == groups[-1]["next_round_number"]), {})
        next_byes = set(map(str, next_round.get("byeParticipantIds") or []))
        groups[-1]["next_round_byes"] = [p for p in groups[-1]["players"] if p["participant_id"] in next_byes]
        planned = sorted((r for r in event.get("rounds") or []
            if int(r["number"]) <= int(event.get("hubPreparedThroughRound") or event.get("currentRoundNumber") or 1)),
            key=lambda r: int(r["number"]))
        earlier_pending: dict[str, int] = {}
        for planned_row in planned:
            if planned_row.get("status") in {"saved", "played", "skipped"}:
                continue
            for planned_match in _matches(planned_row):
                if _complete(planned_match, event):
                    continue
                for pid in (planned_match.get("sideA") or planned_match.get("teamA") or []) + (planned_match.get("sideB") or planned_match.get("teamB") or []):
                    if str(pid) in by_id:
                        identity = player_identity(by_id[str(pid)], key, state)
                        earlier_pending.setdefault(identity, int(planned_row["number"]))
        for match_row, match in ((r, m) for r in planned for m in _matches(r)):
            if match_row.get("status") == "skipped" or event.get("status") != "active":
                continue
            if not (match.get("sideA") or match.get("teamA")) or not (match.get("sideB") or match.get("teamB")):
                continue  # A playoff final has no lineup before its prerequisites finish.
            players = _participants(event, match)
            identities = [player_identity(p, key, state) for p in players]
            waits = sorted((_wait(state["players"][i], now) for i in identities), reverse=True)
            def side(ids):
                result = []
                for pid in ids:
                    identity = player_identity(by_id[str(pid)], key, state)
                    courts = person_courts.get(identity, {})
                    earlier_round = earlier_pending.get(identity, int(match_row["number"]))
                    readiness = {
                        "current_court": courts.get("current"), "reserved_court": courts.get("next"),
                        "paused": bool(state["players"][identity].get("paused")),
                        "waiting_round_number": earlier_round if earlier_round < int(match_row["number"]) else None,
                        "identity_review": identity in identity_data["pending"],
                        "duplicate_in_match": identities.count(identity) > 1,
                    }
                    readiness["ready"] = not any(readiness.values())
                    result.append({"id": str(pid), "name": by_id[str(pid)].get("name"),
                        "display_name": identity_data["nodes"][(key, str(pid))]["display_name"],
                        "identity": identity, "readiness": readiness})
                return result
            side_a = side(match.get("sideA") or match.get("teamA") or [])
            side_b = side(match.get("sideB") or match.get("teamB") or [])
            entry = {"session_key": key, "match_id": str(match["id"]), "round_number": int(match_row["number"]),
                "group_title": event.get("name") or key, "side_a": side_a, "side_b": side_b,
                "label": " / ".join(str(p["display_name"]) for p in side_a) + " vs " + " / ".join(str(p["display_name"]) for p in side_b),
                "wait_seconds": int(waits[0]) if waits else 0, "wait_order": waits,
                "allowed_court_ids": _allowed(state, key), "blockers": [], "status": "waiting",
                "score_a": match.get("scoreA"), "score_b": match.get("scoreB"),
                "plan_locked": bool(match.get("hubPlanLocked")),
                "result_status": "not_played" if match.get("status") == "not_played" else "played" if _complete(match, event) else "pending",
                "not_played_reason": match.get("notPlayedReason")}
            edit_reason = _roster_change_unavailable_reason(event)
            if _complete(match, event) or match.get("scoreA") is not None or match.get("scoreB") is not None or match_row.get("status") not in {"active", "preview"}:
                edit_reason = "Recorded games keep their players."
            entry.update(can_edit_players=edit_reason is None, edit_players_unavailable_reason=edit_reason)
            all_matches[(key, str(match["id"]))] = entry
            if int(match_row["number"]) == groups[-1]["next_round_number"]:
                groups[-1]["next_round_matches"].append(entry)
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
            if event.get("status") != "active" or match_row.get("status") not in {"active", "preview"}:
                entry["blockers"].append("This round is complete.")
            if any(earlier_pending.get(i, int(match_row["number"])) < int(match_row["number"]) for i in identities):
                entry["blockers"].append("A player has an unfinished match in an earlier round.")
            if event.get("hubFinishAfterRound") is not None and int(match_row["number"]) > int(event["hubFinishAfterRound"]):
                entry["blockers"].append("This round robin is finishing before this round.")
            if any(state["players"][i].get("paused") for i in identities):
                entry["blockers"].append("A player is paused.")
                if is_moneyball(event) or event.get("playoff"):
                    entry["blockers"].append("This format keeps its teams fixed, so automatic replacements are unavailable. Resume the player to continue.")
                elif match.get("hubStartedAt"):
                    entry["blockers"].append("This game has already started. Resume the player to keep its recorded lineup.")
                else:
                    entry["blockers"].append("Waiting for enough available players to replace paused players without changing another queued game.")
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
    ordering = lambda m: (tuple(-x for x in m["wait_order"]), m["round_number"], m["session_key"], m["match_id"])
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
                event, row, match = _find_match(hub, events, assignment)
                event["hubAssignedMatchIds"].append(str(match["id"]))
                if slot == "current":
                    match.setdefault("hubStartedAt", assignment.get("assigned_at"))
                    row.setdefault("hubStartedAt", assignment.get("assigned_at"))
                match["courtLabel" if slot == "current" else "reservedCourtLabel"] = court["label"]
                # A dispatched round is no longer an untouched automatic continuation.
                event.pop("hubAutoAdvancedFromRound", None)


def _auto_advance_hub_round(event: dict[str, Any]) -> dict[str, Any]:
    """Use the generator's existing rolling, Moneyball and playoff progression."""
    current = int(event.get("currentRoundNumber") or 1)
    advanced = advance_generator_event(event)
    advanced.pop("hubAutoAdvancedFromRound", None)
    if advanced.get("status") == "active" and int(advanced.get("currentRoundNumber") or 1) == current + 1:
        # Retain an explicit Finish/playoff boundary until the new round is dispatched.
        advanced["hubAutoAdvancedFromRound"] = current
    return advanced


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
                   played: bool = False, not_played: bool = False, reason: str | None = None, now: str) -> dict[str, Any]:
    result = deepcopy(event)
    row = next((r for r in result.get("rounds") or [] if any(str(m["id"]) == match_id for m in _matches(r))), None)
    if row is None or result.get("status") != "active" or row.get("status") not in {"active", "preview"}:
        raise ValueError("Only a match in an incomplete round can finish.")
    match = next((m for m in _matches(row) if str(m["id"]) == match_id), None)
    if match is None or _complete(match, result):
        raise ValueError("This match is unavailable or already complete.")
    if not_played or (not played and score_a is None and score_b is None):
        if is_moneyball(result):
            raise ValueError("Moneyball games require a complete score.")
        if any(str(match_id) in {str(mid) for mid in m.get("sourceMatchIds") or []}
            for future in result.get("rounds") or [] for m in _matches(future)):
            raise ValueError("This playoff game needs a winner before the next game can be played.")
        match.update(scoreA=None, scoreB=None, status="not_played", completedAt=now)
        if reason:
            match["notPlayedReason"] = reason.strip()
    elif played:
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
    return _settle_hub_rounds(result)


def validate_hub_session_change(hub: dict[str, Any], before_events: dict[str, dict[str, Any]],
                                session_key: str, event: dict[str, Any]) -> None:
    """Guard existing generator controls before their joint hub/session commit."""
    from jupr_app.domain.adaptive_play_engine import hub_auto_advance_boundary

    before = before_events[session_key]
    prior_round = int(before.get("currentRoundNumber") or 1)
    next_round = int(event.get("currentRoundNumber") or 1)
    if event.get("playoff") and not before.get("playoff") and next_round == prior_round:
        boundary = hub_auto_advance_boundary(before)
        if boundary is None or (event.get("playoff") or {}).get("sourceRound") != boundary:
            raise ValueError("Finish the current round before starting a playoff.")
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
                         after_events: dict[str, dict[str, Any]], *, now: str,
                         excluded_session_keys: set[str] | None = None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
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
                if new_match and _complete(new_match, updated[key]) and new_match.get("status") != "not_played":
                    _reset_wait(state, before_events, {"session_key": key, "round_number": row["number"], "match_id": match["id"]}, now)
    for court in state.get("courts") or []:
        for slot in ("current", "next"):
            assignment = court.get(slot)
            if assignment:
                event, _, match = _find_match(state, updated, assignment)
                if _complete(match, event):
                    court[slot] = None
    for key in _group_keys(state) - (excluded_session_keys or set()):
        number = int(before_events[key].get("currentRoundNumber") or 1)
        if (int(updated[key].get("currentRoundNumber") or 1) == number + 1
                and _round(before_events[key], number).get("status") not in {"saved", "played"}
                and _round(updated[key], number).get("status") in {"saved", "played"}):
            updated[key]["hubAutoAdvancedFromRound"] = number
        updated[key] = _prepare_rolling_rounds(_settle_hub_rounds(updated[key]))
    _ensure_players(state, updated, now)
    _repair_paused_waiting_games(state, updated, now=now, excluded_session_keys=excluded_session_keys)
    _promote(state, updated, now)
    _sync_assignment_projection(state, updated)
    _claims(state, updated)
    return state, updated


def _sync_future_roster_membership(event: dict[str, Any], participant_ids: set[str], current: int) -> None:
    """Carry a roster edit into overrides whose future round objects were trimmed."""
    from jupr_app.domain.adaptive_play_engine import _participant_active_in_round
    changed = [p for p in event.get("participants") or [] if str(p["id"]) in participant_ids]
    for number, roster in (event.get("rosterByRound") or {}).items():
        if int(number) <= current:
            continue
        active = [str(p["id"]) for p in changed if _participant_active_in_round(p, int(number))]
        retained = [str(pid) for pid in roster if str(pid) not in participant_ids or str(pid) in active]
        event["rosterByRound"][number] = list(dict.fromkeys([*retained, *active]))


def repair_unstarted_hub_round(event: dict[str, Any]) -> dict[str, Any]:
    """Repair an untouched automatic continuation through an explicit joint commit.

    This never runs on a read. Callers must reconcile and persist the result with
    the hub's current court read set, so a concurrent dispatch prevents repair.
    """
    from jupr_app.domain.adaptive_play_engine import (
        hub_auto_advance_boundary, _participant_active_in_round,
    )
    if hub_auto_advance_boundary(event) is None:
        raise ValueError("Only an untouched automatically generated hub round can be repaired.")
    result = deepcopy(event)
    current = int(result["currentRoundNumber"])
    ids = {str(p["id"]) for p in result.get("participants") or []}
    result.setdefault("rosterByRound", {})[str(current)] = [str(p["id"]) for p in result.get("participants") or [] if _participant_active_in_round(p, current)]
    _sync_future_roster_membership(result, ids, current)
    result.update(perfectRotation=False, adaptiveFromRound=min(int(result.get("adaptiveFromRound") or current), current))
    _plan_hub_round(result, current)
    return repair_hub_roster_plan(result, from_round=current + 1)


def mutate_hub_roster(event: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Keep current commitments and rebuild one full-group successor after arrivals."""
    from jupr_app.domain.adaptive_play_engine import (
        mutate_generator_roster, history_before_round, _participant_active_in_round,
        _apply_match_history, _doubles_round, _singles_round, _mixed_round, generator_match_play_format,
    )
    result = mutate_generator_roster(event, _hub_managed=True, **kwargs)
    result["rounds"] = deepcopy(event.get("rounds") or [])
    result["rosterByRound"] = deepcopy(event.get("rosterByRound") or {})
    _trim_replaceable_hub_previews(result)
    current = int(event.get("currentRoundNumber") or 1)
    action = str(kwargs.get("action") or "")
    target = str(kwargs.get("participant_id") or "")
    protected = {int(r["number"]): [deepcopy(m) for m in _matches(r)
        if _committed_match(event, m)]
        for r in event.get("rounds") or []}
    people = lambda matches: {str(pid) for m in matches for pid in
        (m.get("sideA") or m.get("teamA") or []) + (m.get("sideB") or m.get("teamB") or [])}
    target_rounds = [number for number, games in protected.items()
        if target in people([m for m in games if _actually_committed_match(event, m)])]
    effective = max(current, max(target_rounds, default=current - 1) + 1)
    old_ids = {str(p["id"]) for p in event.get("participants") or []}
    added = next((p for p in result.get("participants") or [] if str(p["id"]) not in old_ids), None)
    source = next((p for p in result.get("participants") or [] if str(p["id"]) == target), None)
    changed = ({target} if action in {"remove", "substitute"} else set()) | ({str(added["id"])} if added else set())
    if action == "add" and added:
        added["active_from_round"] = current
        effective = current
    elif action == "remove" and source:
        source["inactive_from_round"] = effective
    elif action == "substitute" and added and source:
        old_source = next(p for p in event["participants"] if str(p["id"]) == target)
        added["active_from_round"] = effective
        if kwargs.get("substitute_scope") == "round":
            source["inactive_rounds"] = list(dict.fromkeys([*old_source.get("inactive_rounds", []), effective]))
            added["inactive_from_round"] = effective + 1
        else:
            source["inactive_from_round"] = effective
    if result.get("rosterRevisions"):
        result["rosterRevisions"][-1]["effectiveRound"] = effective
    by_id = {str(p["id"]): p for p in result.get("participants") or []}
    for key, ids in list(result["rosterByRound"].items()):
        if int(key) < current:
            continue
        filtered = [str(pid) for pid in ids if str(pid) not in changed or _participant_active_in_round(by_id[str(pid)], int(key))]
        filtered.extend(pid for pid in by_id if pid in changed and pid not in filtered and _participant_active_in_round(by_id[pid], int(key)))
        result["rosterByRound"][key] = filtered
    revision = int(result.get("hubRosterRevision") or 0) + 1
    result["hubRosterRevision"] = revision
    roster_pos = {pid: int(p.get("roster_order") or 0) for pid, p in by_id.items()}
    play_format = result.get("playFormat") or "doubles"
    for row in sorted(result["rounds"], key=lambda r: int(r["number"])):
        number = int(row["number"])
        if number > current:
            continue  # The full future pool is repaired once below.
        if number < current or row.get("status") not in {"active", "preview"}:
            continue
        if action == "reorder" and number == current:
            continue
        active = list(active_participant_ids(result, number))
        active = [pid for pid in active if pid not in changed or _participant_active_in_round(by_id[pid], number)]
        active.extend(pid for pid in by_id if pid in changed and pid not in active and _participant_active_in_round(by_id[pid], number))
        result["rosterByRound"][str(number)] = active
        protected_ids = {str(m["id"]) for m in protected[number]}
        kept, affected = [], []
        for match in _matches(row):
            ids = people([match])
            if str(match["id"]) in protected_ids and (_actually_committed_match(result, match) or ids <= set(active)):
                kept.append(match)
            elif action == "substitute" and target in ids and added and target not in active and str(added["id"]) in active:
                replacement = deepcopy(match)
                for side, alias in (("sideA", "teamA"), ("sideB", "teamB")):
                    replacement[side] = [str(added["id"]) if str(pid) == target else str(pid)
                        for pid in match.get(side) or match.get(alias) or []]
                    replacement[alias] = list(replacement[side])
                kept.append(replacement)
                affected.append(match)
            elif ids <= set(active) and action != "reorder":
                kept.append(match)
            else:
                affected.append(match)
        kept_people = people(kept)
        # Someone who already reached a later round cannot acquire a new
        # earlier obligation because another player changed the roster.
        later_people = set().union(*(people(games) for n, games in protected.items() if n > number))
        deferred_byes = [pid for pid in active if pid in later_people and pid not in kept_people]
        waiting = [pid for pid in active if pid not in kept_people and pid not in later_people]
        history = history_before_round(result, number, include_preview=True)
        for match in kept:
            _apply_match_history(history, match, play_format)
        if play_format == "doubles_singles":
            counts = {kind: sum(generator_match_play_format(m, play_format) == kind for m in kept) for kind in ("singles", "doubles")}
            generated, byes, warnings, _ = _mixed_round(active_ids=waiting,
                doubles_court_count=max(0, int(result.get("doublesCourtCount") or 0) - counts["doubles"]),
                singles_court_count=max(0, int(result.get("singlesCourtCount") or 0) - counts["singles"]),
                history=history, roster_pos=roster_pos, round_number=number)
        else:
            generate = _singles_round if play_format == "singles" else _doubles_round
            generated, byes, warnings = generate(active_ids=waiting, court_count=0,
                history=history, roster_pos=roster_pos, round_number=number)
        offset = max((int(m.get("court") or 0) for m in kept), default=0)
        for index, match in enumerate(generated, 1):
            match.update(id=f"hub-r{number}-v{revision}-m{index}", court=offset + index)
        row["matches"] = [*kept, *generated]
        row.pop("courts", None)
        row["byeParticipantIds"] = list(dict.fromkeys([*byes, *deferred_byes]))
        row["warnings"] = warnings
        row["formatCounts"] = {kind: sum(generator_match_play_format(m, play_format) == kind for m in row["matches"]) for kind in ("singles", "doubles")}

        if affected or generated:
            row.setdefault("hubRosterAdjustments", []).append({"action": action, "revision": revision,
                "originalMatches": deepcopy(affected), "replacementMatchIds": [str(m["id"]) for m in generated]})

    result["adaptiveFromRound"] = min(int(result.get("adaptiveFromRound") or current), current)
    result["perfectRotation"] = False
    return repair_hub_roster_plan(result, from_round=current + 1 if action == "reorder" else current)


def _edit_match_players(hub: dict[str, Any], events: dict[str, dict[str, Any]], payload: dict[str, Any], now: str,
                        *, _planning: bool = False) -> None:
    """Explicit lineup correction, committed with the court and roster read set."""
    from jupr_app.domain.adaptive_play_engine import _clean_name, _next_participant_id

    key = str(payload.get("session_key") or "")
    event, row, match = _find_match(hub, events, payload)
    unavailable = _roster_change_unavailable_reason(event)
    if unavailable:
        raise ValueError(unavailable)
    if row.get("status") not in {"active", "preview"} or _complete(match, event) or match.get("scoreA") is not None or match.get("scoreB") is not None:
        raise ValueError("Recorded games keep their players.")
    assignment = next(((c, slot) for c in hub.get("courts") or [] for slot in ("current", "next")
        if c.get(slot) and _key(c[slot]) == _key(payload)), None)
    if payload.get("court_id"):
        if not assignment or assignment[0]["id"] != payload["court_id"] or assignment[1] != "current":
            raise ValueError("The court assignment changed. Reload before editing players.")
    elif assignment:
        raise ValueError("This game now has a court assignment. Open Edit players from its court.")
    target = str(payload.get("participant_id") or "")
    original_ids = [str(p["id"]) for p in _participants(event, match)]
    if target not in original_ids:
        raise ValueError("Choose a player in this match.")
    incoming = str(payload.get("replacement_participant_id") or "")
    number = int(row["number"])
    planning = _planning and assignment is None and number > int(event.get("currentRoundNumber") or 1)
    by_id = {str(p["id"]): p for p in event.get("participants") or []}
    added = not incoming
    if incoming and (payload.get("name") is not None or payload.get("player_id") is not None):
        raise ValueError("Choose an existing player or add a new player, not both.")
    if added:
        unavailable = _add_player_unavailable_reason(event)
        if unavailable:
            raise ValueError(unavailable)
        name, profile = payload.get("name"), payload.get("player_id")
        if not isinstance(name, str) or not name.strip() or len(name) > 160:
            raise ValueError("Enter a player name of up to 160 characters.")
        if profile is not None and (type(profile) is not int or profile <= 0):
            raise ValueError("Choose a valid club player profile.")
        if any((profile is not None and p.get("player_id") == profile)
                or (profile is None and _name_key(p.get("name")) == _name_key(name)) for p in by_id.values()):
            raise ValueError("This player is already on the roster. Choose them from Round robin players.")
        incoming = _next_participant_id(event)
        player = {"id": incoming, "name": _clean_name(name), "seed": len(by_id) + 1,
            "roster_order": len(by_id) + 1, "active_from_round": number,
            "inactive_from_round": None, "inactive_rounds": []}
        if profile is not None:
            player["player_id"] = profile
        event.setdefault("participants", []).append(player)
        by_id[incoming] = player
        event.setdefault("rosterByRound", {})[str(number)] = [*active_participant_ids(event, number)]
        if incoming not in event["rosterByRound"][str(number)]:
            event["rosterByRound"][str(number)].append(incoming)
        _sync_future_roster_membership(event, {incoming}, number)
        _ensure_players(hub, events, now)
    elif incoming not in by_id or incoming not in active_participant_ids(event, number):
        raise ValueError("Choose a player active in this round.")
    if incoming == target:
        raise ValueError("Choose a different player.")
    identity = player_identity(by_id[incoming], key, hub)
    if hub["players"][identity].get("paused"):
        raise ValueError("Resume this player before adding them to a game.")
    if identity in _identity_data(hub, events)["pending"]:
        if added:
            raise ValueError("Use Add player first and review the matching names in Players before assigning this person to a match.")
        raise ValueError("Confirm matching player names in Players before editing this match.")
    claimed = _claims(hub, events).get(identity)
    if claimed and claimed != _key(payload):
        claim_round = next((int(r["number"]) for r in events[claimed[0]].get("rounds") or []
            if any(str(m["id"]) == claimed[1] for m in _matches(r))), number)
        if not planning or (claimed[0] == key and claim_round >= number):
            raise ValueError("This player is already assigned or reserved on another court.")
    others = []
    for planned in event.get("rounds") or []:
        for other in _matches(planned):
            if incoming in original_ids or other is match or incoming not in {str(p["id"]) for p in _participants(event, other)}:
                continue
            n = int(planned["number"])
            recorded = _complete(other, event) or other.get("scoreA") is not None or other.get("scoreB") is not None
            if not planning and n < number and planned.get("status") in {"active", "preview"} and not _complete(other, event):
                raise ValueError("This player has an unfinished game in an earlier round.")
            if n >= number and (recorded or other.get("hubStartedAt") or str(other["id"]) in (event.get("hubAssignedMatchIds") or [])):
                raise ValueError("This player has already committed to another game. Choose a free player.")
            if n == number:
                if other.get("hubPlanLocked"):
                    raise ValueError("This player belongs to another manually locked game. Edit that game first.")
                others.append(other)
    if len(others) > 1:
        raise ValueError("This player appears in multiple games in this round. Review the round robin.")
    affected = [match, *others]
    before = deepcopy(affected)
    for game in affected:
        for side, alias in (("sideA", "teamA"), ("sideB", "teamB")):
            ids = [incoming if str(pid) == target else target if str(pid) == incoming else str(pid)
                for pid in game.get(side) or game.get(alias) or []]
            game[side], game[alias] = ids, list(ids)
        game["hubPlanLocked"] = True
    people = [player_identity(p, key, hub) for p in _participants(event, match)]
    if len(people) != len(set(people)):
        raise ValueError("The same person cannot appear twice in a game.")
    active = set(active_participant_ids(event, number))
    playing = {str(p["id"]) for game in _matches(row) for p in _participants(event, game)}
    paused = {pid for pid in active if hub["players"][player_identity(by_id[pid], key, hub)].get("paused")}
    row["byeParticipantIds"] = sorted(active - playing - paused)
    row["hubPausedParticipantIds"] = sorted(paused - playing)
    event.setdefault("hubLineupEdits", []).append({"at": now, "round": number, "matchId": str(match["id"]),
        "participantId": target, "replacementParticipantId": incoming, "added": added, "originalMatches": before})
    event.update(perfectRotation=False, adaptiveFromRound=min(int(event.get("adaptiveFromRound") or number), number))
    if added:
        # New arrivals join future rotations. Keep this explicit lineup, all
        # dispatched games (including ones returned to the queue), and results.
        assigned = list(event.get("hubAssignedMatchIds") or [])
        event["hubAssignedMatchIds"] = list(set(assigned) | {str(match["id"])} | {
            str(m["id"]) for r in event.get("rounds") or [] for m in _matches(r)
            if m.get("hubStartedAt") or m.get("scoreA") is not None or m.get("scoreB") is not None})
        event = mutate_hub_roster(event, action="reorder", roster_order=list(by_id))
        event["hubAssignedMatchIds"] = assigned
        events[key] = event


def apply_hub_command(hub: dict[str, Any], events: dict[str, dict[str, Any]], *, action: str,
                      payload: dict[str, Any], now: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    state, updated = deepcopy(hub), deepcopy(events)
    state.setdefault("courts", [])
    state.setdefault("groups", [])
    state.setdefault("players", {})
    excluded = set(payload.get("excluded_session_keys") or [])
    if action in {"assign_next_court", "assign_game", "reserve_game_for_court", "edit_match_players", "set_match_lineup"}:
        # A user explicitly selecting a legacy preview has announced that game.
        # Preserve it during the first-write upgrade, so its ID cannot disappear
        # before this very command can validate and assign/edit it.
        target_event = updated.get(str(payload.get("session_key") or ""))
        if target_event is not None and target_event.get("hubPlanVersion") != 1:
            target = next((m for r in target_event.get("rounds") or [] for m in _matches(r)
                if str(m["id"]) == str(payload.get("match_id") or "")), None)
            if target is not None:
                target["hubPlanLocked"] = True
    for key in _group_keys(state) - excluded:
        updated[key] = _prepare_rolling_rounds(updated[key])
    _ensure_players(state, updated, now)
    _claims(state, updated)
    if action in {"auto_fill_courts", "assign_next_court", "assign_game", "reserve_game_for_court"}:
        _repair_paused_waiting_games(state, updated, now=now, excluded_session_keys=excluded)
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
        event["roundRobinHubId"] = state.get("id") or event.get("roundRobinHubId") or "hub"
    elif action == "add_player":
        from jupr_app.domain.adaptive_play_engine import mutate_generator_roster
        event = _event(state, updated, session_key)
        unavailable = _add_player_unavailable_reason(event)
        if unavailable:
            raise ValueError(unavailable)
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 160:
            raise ValueError("Enter a player name of up to 160 characters.")
        player_id = payload.get("player_id")
        if player_id is not None:
            if type(player_id) is not int or player_id <= 0:
                raise ValueError("Choose a valid club player profile.")
            identity = f"player:{player_id}"
            if not _can_rejoin_profile(event, session_key, state, identity, _claims(state, updated)):
                raise ValueError("This club player is already in this round robin's active or upcoming roster.")
        changed = {**updated, session_key: mutate_generator_roster(event, action="add", name=name, player_id=player_id)}
        state, updated = reconcile_hub_events(state, updated, changed, now=now, excluded_session_keys=excluded)
    elif action == "remove_player":
        from jupr_app.domain.adaptive_play_engine import mutate_generator_roster
        event = _event(state, updated, session_key)
        unavailable = _roster_change_unavailable_reason(event)
        if unavailable:
            raise ValueError(unavailable)
        participant_id = payload.get("participant_id")
        participant = next((p for p in event.get("participants") or [] if isinstance(participant_id, str) and str(p["id"]) == participant_id), None)
        if participant is None:
            raise ValueError("Choose a player in this round robin.")
        if not _can_remove_participant(event, participant):
            raise ValueError("This player has already left this round robin.")
        changed = {**updated, session_key: mutate_generator_roster(event, action="remove", participant_id=participant_id)}
        state, updated = reconcile_hub_events(state, updated, changed, now=now, excluded_session_keys=excluded)
    elif action in {"rejoin_player", "transfer_player"}:
        source_key = str(payload.get("source_session_key") or session_key)
        destination_key = str(payload.get("destination_session_key") or session_key)
        if action == "transfer_player" and source_key == destination_key:
            raise ValueError("Choose a different destination round robin.")
        source = _event(state, updated, source_key)
        destination = _event(state, updated, destination_key)
        person = next((p for p in source.get("participants") or []
            if str(p["id"]) == payload.get("participant_id")), None)
        if person is None:
            raise ValueError("Choose an existing player in the source round robin.")
        identity = player_identity(person, source_key, state)
        availability = deepcopy(state["players"][identity])
        if action == "rejoin_player":
            if not _can_rejoin_profile(source, source_key, state, identity, _claims(state, updated)):
                raise ValueError("This person already has an active or upcoming membership in this round robin.")
        else:
            unavailable = _roster_change_unavailable_reason(source)
            if unavailable:
                raise ValueError(unavailable)
            if not _can_remove_participant(source, person):
                raise ValueError("This player has already left the source round robin.")
            if any(_committed_match(source, match) and not _complete(match, source)
                and any(player_identity(p, source_key, state) == identity for p in _participants(source, match))
                for row in source.get("rounds") or [] for match in _matches(row)):
                raise ValueError("Finish or resolve this player's started games and cancel reservations before moving groups.")
        existing = [p for p in destination.get("participants") or []
            if player_identity(p, destination_key, state) == identity and _can_remove_participant(destination, p)]
        if not existing:
            unavailable = _add_player_unavailable_reason(destination)
            if unavailable:
                raise ValueError(unavailable)
        elif _roster_change_unavailable_reason(destination):
            raise ValueError(_roster_change_unavailable_reason(destination))
        changed = deepcopy(updated)
        if action == "transfer_player":
            for member in source.get("participants") or []:
                if player_identity(member, source_key, state) == identity and _can_remove_participant(changed[source_key], member):
                    changed[source_key] = mutate_hub_roster(changed[source_key], action="remove", participant_id=str(member["id"]))
        if not existing:
            name = payload.get("name", person.get("name"))
            if not isinstance(name, str) or not name.strip() or len(name) > 160:
                raise ValueError("Enter a player name of up to 160 characters.")
            changed[destination_key] = mutate_hub_roster(changed[destination_key], action="add", name=name, player_id=person.get("player_id"))
            new_person = changed[destination_key]["participants"][-1]
        else:
            new_person = existing[0]
        state, updated = reconcile_hub_events(state, updated, changed, now=now, excluded_session_keys=excluded)
        if not existing:
            new_identity = player_identity(new_person, destination_key, state)
            state["players"][new_identity].update(availability)
            _resolve_identities(state, updated, {"resolution": "same", "participants": [
                {"session_key": source_key, "participant_id": str(person["id"])},
                {"session_key": destination_key, "participant_id": str(new_person["id"])}]}, now)
        state.setdefault("membership_changes", []).append({"action": action, "at": now,
            "source_session_key": source_key, "source_participant_id": str(person["id"]),
            "destination_session_key": destination_key, "destination_participant_id": str(new_person["id"]),
            "source_effective_round": int(source.get("currentRoundNumber") or 1),
            "destination_effective_round": int(destination.get("currentRoundNumber") or 1)})
    elif action == "edit_match_players":
        _edit_match_players(state, updated, payload, now)
    elif action == "set_match_lineup":
        from jupr_app.domain.hub_lineup_controls import set_hub_match_lineup
        set_hub_match_lineup(state, updated, payload, now=now)
    elif action == "remove_players":
        from jupr_app.domain.hub_lineup_controls import remove_hub_players
        event = _event(state, updated, session_key)
        changed = {**updated, session_key: remove_hub_players(event, payload.get("participant_ids"), now=now)}
        state, updated = reconcile_hub_events(state, updated, changed, now=now, excluded_session_keys=excluded)
    elif action == "finish_group":
        event = _event(state, updated, session_key)
        details = _finish_details(event)
        if not details["can_finish"]:
            raise ValueError(details["finish_unavailable_reason"])
        boundary = int(details["finish_boundary_round"])
        if "finish_after_round" in payload:
            requested = payload["finish_after_round"]
            if type(requested) is not int or requested not in {option["round_number"] for option in details["finish_round_options"]}:
                raise ValueError("Choose a prepared final round that includes every started or reserved game.")
            boundary = requested
        event["hubFinishAfterRound"] = boundary
        event.pop("hubAutoAdvancedFromRound", None)
        for row in event.get("rounds") or []:
            if int(row["number"]) > boundary:
                row.update(status="skipped", skippedAt=now, skipReason="Round robin finished before this round was played.")
        if boundary == 0:
            event.update(status="completed", completedAt=now)
        else:
            updated[session_key] = _settle_hub_rounds(event)
    elif action == "remove_group":
        event = _event(state, updated, session_key)
        if any(d["resolution"] == "same" and ((_ref(d["left"])[0] == session_key) != (_ref(d["right"])[0] == session_key)) for d in state.get("identity_decisions") or []):
            raise ValueError("Review and separate shared guest identities in Players before removing this group.")
        if any(c.get(s, {}).get("session_key") == session_key for c in state["courts"] for s in ("current", "next") if c.get(s)):
            raise ValueError("Release this group's courts and reservations before removing it.")
        if any(r.get("status") not in {"saved", "played", "skipped"} and any(_complete(m, event) for m in _matches(r)) for r in event.get("rounds") or []):
            raise ValueError("Finish the partially scored round before removing this group.")
        state["groups"] = [g for g in state["groups"] if g["session_key"] != session_key]
        for court in state["courts"]:
            if court.get("dedicated_session_key") == session_key:
                court["dedicated_session_key"] = None
        event.pop("roundRobinHubId", None)
        event.pop("hubAssignedMatchIds", None)
        for field in ("hubRollingEnabled", "hubPreparedThroughRound", "hubFinishAfterRound", "hubAutoAdvancedFromRound"):
            event.pop(field, None)
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
        excluded = set(payload.get("excluded_session_keys") or [])
        # Explicit Fill also resumes legacy groups left at a completed round.
        for key in _group_keys(state) - excluded:
            event = updated[key]
            if event.get("status") == "active" and _round(event).get("status") in {"saved", "played", "skipped"}:
                    updated[key] = _settle_hub_rounds(event)
        # Recover a player already resumed by an older client/server, without
        # requiring staff to toggle availability again or change healthy plans.
        _seat_resumed_byes(state, updated, identity=None, now=now, excluded_session_keys=excluded)
        while True:
            ready = [entry for entry in build_hub_snapshot(state, updated, now=now)["eligible_queue"] if entry["session_key"] not in excluded]
            chosen = None
            for candidate in ready:
                court = _next_open_court(state, candidate["allowed_court_ids"])
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
            court = _next_open_court(state, allowed)
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
        _repair_paused_waiting_games(state, updated, now=now, excluded_session_keys=excluded)
        _promote(state, updated, now)
    elif action == "resolve_game":
        if payload.get("resolution") != "not_played":
            raise ValueError("Choose Not played to resolve a game without a result.")
        assignment = {"session_key": session_key, "match_id": str(payload.get("match_id") or "")}
        source_event = _event(state, updated, session_key)
        source_row = next((r for r in source_event.get("rounds") or []
            if any(str(m["id"]) == assignment["match_id"] for m in _matches(r))), None)
        if source_row is None:
            raise ValueError("This game is no longer available.")
        assignment["round_number"] = int(source_row["number"])
        event, _, match = _find_match(state, updated, assignment)
        if payload.get("court_id"):
            court = _court(state, str(payload["court_id"]))
            if not any(court.get(slot) and _key(court[slot]) == _key(assignment) for slot in ("current", "next")):
                raise ValueError("The court assignment changed. Reload before resolving this game.")
        updated[session_key] = save_hub_match(event, match_id=str(match["id"]), not_played=True,
            reason=payload.get("reason"), now=now)
        for court in state["courts"]:
            for slot in ("current", "next"):
                if court.get(slot) and _key(court[slot]) == _key(assignment):
                    court[slot] = None
        _repair_paused_waiting_games(state, updated, now=now, excluded_session_keys=excluded)
        _promote(state, updated, now)
    elif action in {"score_and_release", "mark_played"}:
        court = _court(state, str(payload.get("court_id") or ""))
        assignment = court.get("current")
        if not assignment or _key(assignment) != _key(payload):
            raise ValueError("The court assignment changed. Reload before saving this result.")
        event, _, _ = _find_match(state, updated, assignment)
        if action == "mark_played" or payload.get("score_a") is not None or payload.get("score_b") is not None:
            _reset_wait(state, updated, assignment, now)
        updated[session_key] = save_hub_match(event, match_id=str(payload.get("match_id") or ""),
            score_a=payload.get("score_a"), score_b=payload.get("score_b"), played=action == "mark_played", now=now)
        court["current"] = None
        _repair_paused_waiting_games(state, updated, now=now, excluded_session_keys=excluded)
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
    for key in _group_keys(state) - excluded:
        updated[key] = _prepare_rolling_rounds(updated[key])
    _ensure_players(state, updated, now)
    _repair_paused_waiting_games(state, updated, now=now, excluded_session_keys=excluded)
    if action == "resume_player":
        _seat_resumed_byes(state, updated, identity=str(payload.get("identity") or ""), now=now,
            excluded_session_keys=excluded)
    _sync_assignment_projection(state, updated)
    _claims(state, updated)
    return state, updated
