"""Season event titles from approved results, independent of badge processing.

Only linked player identities receive credit. All participants still compete
for first place, so an unlinked winner never promotes a linked runner-up.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from itertools import combinations
from typing import Any
from zoneinfo import ZoneInfo

from jupr_app.data.paged_reads import read_all_rows
from jupr_app.data.public_read_cache import PublicReadCache, public_source_key
from jupr_app.domain.leaderboard_metrics import EVENT_WIN_CARD_KEYS
from jupr_app.domain.live_beta_engine import (
    round_robin_players_of_record, round_robin_result_fingerprint, round_robin_standings,
)
from jupr_app.domain.moneyball_expectations import is_moneyball, settlement, standings as moneyball_standings
from jupr_app.domain.player_visibility import is_merged_player


_snapshots = PublicReadCache(ttl=30, capacity=8)
_SLOTS = ("t1_p1", "t1_p2", "t2_p1", "t2_p2")


def _id(value):
    if isinstance(value, bool):
        return ""
    try:
        number = Decimal(str(value))
        return str(int(number)) if number.is_finite() and number == number.to_integral_value() and number > 0 else ""
    except (ValueError, TypeError, OverflowError, InvalidOperation):
        return ""


def _day(value, zone):
    raw = str(value or "").strip()
    try:
        if len(raw) == 10:
            return datetime.strptime(raw, "%Y-%m-%d").date().isoformat()
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return stamp.replace(tzinfo=timezone.utc).astimezone(zone).date().isoformat() if stamp.tzinfo is None else stamp.astimezone(zone).date().isoformat()
    except (ValueError, TypeError):
        return ""


def _in_period(value, period, now):
    zone = ZoneInfo(period["timezone"])
    day = _day(value, zone)
    return bool(day and (not period.get("start_date") or period["start_date"] <= day)
                and (not period.get("end_date") or day <= period["end_date"])
                and day <= now.astimezone(zone).date().isoformat())


def _date_query(query, column, period, now):
    end = min(period.get("end_date") or "9999-12-31", now.astimezone(ZoneInfo(period["timezone"])).date().isoformat())
    query = query.lte(column, end)
    return query.gte(column, period["start_date"]) if period.get("start_date") else query


def _batches(values):
    values = sorted(set(str(value) for value in values))
    for offset in range(0, len(values), 100):
        yield values[offset:offset + 100]


def _session_rows(db, club_id):
    # Read small version headers afresh. Large frozen Moneyball snapshots are
    # fetched only after a version change, and never for ordinary stat pages.
    def query(columns):
        return db.table("live_sessions").select(columns).eq("club_id", club_id).in_("status", ["active", "completed"]).in_("state->>mode", ["admin_play_generator", "public_play_generator"])
    headers = read_all_rows(lambda: query("id,version,status,updated_at"))
    key = (public_source_key(db), club_id, tuple((str(h["id"]), h.get("version"), h.get("updated_at"), h["status"]) for h in headers))

    def load():
        rows = []
        for batch in _batches(h["id"] for h in headers):
            rows.extend(read_all_rows(lambda: query("id,session_key,title,state,status,version,updated_at").in_("id", batch)))
        return rows

    return _snapshots.get(key, load)


def _raw_matches(round_row):
    games = round_row.get("matches")
    if games is None:
        games = [game for court in round_row.get("courts") or []
                 for game in court.get("matches", []) or [g for mini in court.get("miniRounds") or [] for g in mini.get("matches") or []]]
    return games


def _matches(round_row):
    return [{**game, "teamA": [str(pid) for pid in game.get("sideA", game.get("teamA", []))],
             "teamB": [str(pid) for pid in game.get("sideB", game.get("teamB", []))]}
            for game in _raw_matches(round_row) if str(game.get("status") or "").lower() not in {"not_played", "skipped", "cancelled", "canceled", "bye"}]


def _valid(games, participants):
    if not games:
        return False
    for game in games:
        a, b = game.get("scoreA"), game.get("scoreB")
        seats = game["teamA"] + game["teamB"]
        if (type(a) is not int or type(b) is not int or min(a, b) < 0 or a == b
                or len(game["teamA"]) not in {1, 2} or len(game["teamB"]) != len(game["teamA"])
                or len(set(seats)) != len(seats) or not set(seats) <= participants):
            return False
    return True


def _court_complete(games):
    counts = Counter(pid for game in games for side in ("teamA", "teamB") for pid in game[side])
    if not ((len(counts) == 4 and len(games) == 3 and set(counts.values()) == {3})
            or (len(counts) == 5 and len(games) == 5 and set(counts.values()) == {4})):
        return False
    pairs = [tuple(sorted(game[side])) for game in games for side in ("teamA", "teamB")]
    return len(set(pairs)) == len(list(combinations(counts, 2)))


def _event_credits(event, keys):
    """Return (card, local result identity, participant ID) without private data."""
    participants = {str(p["id"]) for p in event.get("participants") or []}
    rounds = [row for row in event.get("rounds") or [] if row.get("status") == "saved"] if event.get("generatorKind") else [row for row in event.get("rounds") or [] if row.get("status") not in {"skipped", "cancelled", "canceled"}]
    ladder = event.get("generatorKind") == "ladder" or event.get("type") == "league"
    if ladder:
        if "ladder_court_wins" not in keys:
            return
        for row in rounds:
            for court in row.get("courts") or []:
                games = _matches({"courts": [court]})
                if not _valid(games, participants) or not _court_complete(games):
                    continue
                standings = round_robin_standings({"participants": event["participants"], "rounds": [{"matches": games}]})
                for player in standings:
                    if player["rank"] == 1:
                        yield "ladder_court_wins", f"{row['number']}:{court['courtNumber']}", str(player["participantId"])
        return
    if event.get("status") != "completed":
        return
    games = [game for row in rounds if row.get("stage") != "playoff" for game in _matches(row)]
    if not _valid(games, participants):
        return
    if any(not g.get("id") for g in games) or len({str(g["id"]) for g in games}) != len(games):
        return
    if is_moneyball(event):
        if "moneyball_wins" not in keys:
            return
        # The frozen event model owns scoring, eligibility, and split prizes.
        for player in settlement(event, moneyball_standings(event)).get("winners", []):
            yield "moneyball_wins", "title", str(player["participantId"])
        return
    if event.get("type") != "round_robin" or "round_robin_wins" not in keys:
        return
    normalized = {**event, "rounds": [{"matches": games}]}
    decision = event.get("round_robin_winner") or {}
    # The decision fingerprint belongs to the original saved display shape.
    if decision.get("display_fingerprint") == round_robin_result_fingerprint(event):
        normalized["round_robin_winner"] = {**decision, "display_fingerprint": round_robin_result_fingerprint(normalized)}
    else:
        normalized.pop("round_robin_winner", None)
    normalized = round_robin_players_of_record(normalized)
    rows = round_robin_standings(normalized)
    mode = event.get("standingsSort")
    if mode in {"points", "differential"}:
        fields = ("pointsFor", "wins", "differential") if mode == "points" else ("differential", "wins", "pointsFor")
        best = max(tuple(row[key] for key in fields) for row in rows)
        leaders = [row for row in rows if tuple(row[key] for key in fields) == best]
    else:
        leaders = [row for row in rows if row["rank"] == 1]
    if len(leaders) == 1 and not leaders[0].get("winnerNeedsAdmin"):
        yield "round_robin_wins", "title", str(leaders[0]["participantId"])


def _social_sources(db, club_id, period, now):
    events = read_all_rows(lambda: _date_query(db.table("live_events")
        .select("id,club_id,source_event_uid,event_type,event_date,status,raw_event_json")
        .eq("club_id", club_id), "event_date", period, now))
    saved_ids = [row["id"] for row in events if row.get("status") == "saved"]
    people, results = [], []
    for batch in _batches(saved_ids):
        people.extend(read_all_rows(lambda: db.table("live_event_participants").select("id,event_id,participant_key,linked_player_id,display_name_snapshot").in_("event_id", batch)))
        results.extend(read_all_rows(lambda: db.table("live_event_matches").select("id,event_id,match_key,score_t1,score_t2,t1_p1_participant_id,t1_p2_participant_id,t2_p1_participant_id,t2_p2_participant_id").in_("event_id", batch)))
    sources = []
    for row in events:
        if row.get("status") != "saved" or not _in_period(row.get("event_date"), period, now):
            continue
        event = deepcopy(row.get("raw_event_json") or {})
        persons = [p for p in people if str(p["event_id"]) == str(row["id"])]
        records = [g for g in results if str(g["event_id"]) == str(row["id"])]
        links = {str(p["participant_key"]): _id(p.get("linked_player_id")) for p in persons}
        if not event.get("rounds"):
            links = {str(p["id"]): _id(p.get("linked_player_id")) for p in persons}
            event = {"type": row["event_type"], "participants": [{"id": str(p["id"]), "name": p.get("display_name_snapshot") or "Player"} for p in persons],
                     "rounds": [{"matches": [{"id": str(g["id"]), "teamA": [str(g[k]) for k in ("t1_p1_participant_id", "t1_p2_participant_id") if g.get(k)],
                                               "teamB": [str(g[k]) for k in ("t2_p1_participant_id", "t2_p2_participant_id") if g.get(k)],
                                               "scoreA": g["score_t1"], "scoreB": g["score_t2"]} for g in records]}]}
        else:
            recorded = {str(g.get("match_key")): g for g in records}
            games = [g for r in event.get("rounds") or []
                     if (r.get("status") == "saved" if event.get("generatorKind") else r.get("status") not in {"skipped", "cancelled", "canceled"})
                     for g in _matches(r)]
            participant_keys = {str(p["id"]): str(p["participant_key"]) for p in persons}
            if not games or any(str(g.get("id")) not in recorded or
                g["teamA"] + g["teamB"] != [participant_keys.get(str(recorded[str(g["id"])].get(k))) for k in
                    ("t1_p1_participant_id", "t1_p2_participant_id", "t2_p1_participant_id", "t2_p2_participant_id") if recorded[str(g["id"])].get(k)] for g in games):
                continue
            corrected = False
            for r in event.get("rounds") or []:
                for g in _raw_matches(r):
                    record = recorded.get(str(g.get("id")))
                    if record:
                        corrected |= (g.get("scoreA"), g.get("scoreB")) != (record["score_t1"], record["score_t2"])
                        g.update(scoreA=record["score_t1"], scoreB=record["score_t2"])
            if corrected:
                event.pop("round_robin_winner", None)
        event["status"] = "completed"
        sources.append((str(row.get("source_event_uid") or row["id"]), event, links, {}))
    return events, sources


def _league_credits(db, club_id, period, now):
    rounds = read_all_rows(lambda: _date_query(db.table("league_live_rounds")
        .select("id,club_id,session_id,round_number,match_date,matches_json,submitted_match_ids")
        .eq("club_id", club_id).eq("status", "submitted"), "match_date", period, now))
    sessions, canonical = {}, {}
    for batch in _batches(row["session_id"] for row in rounds):
        sessions.update({str(s["id"]): s for s in read_all_rows(lambda: db.table("league_live_sessions").select("id,league_name").eq("club_id", club_id).in_("id", batch))})
    for batch in _batches(mid for row in rounds for mid in row.get("submitted_match_ids") or []):
        canonical.update({_id(g["id"]): g for g in read_all_rows(lambda: db.table("matches").select("id,league,date,deleted_at,score_t1,score_t2," + ",".join(_SLOTS)).eq("club_id", club_id).in_("id", batch))})
    for row in rounds:
        session = sessions.get(str(row["session_id"]))
        ids, saved = row.get("submitted_match_ids") or [], row.get("matches_json") or []
        if not session or not _in_period(row.get("match_date"), period, now) or not ids or len(ids) != len(saved) or len({_id(mid) for mid in ids}) != len(ids):
            continue
        courts = defaultdict(list)
        for mid, snapshot in zip(ids, saved):
            game = canonical.get(_id(mid))
            court = snapshot.get("court", snapshot.get("court_number"))
            if (not game or game.get("deleted_at") or not court or game.get("league") != session["league_name"]
                    or not _in_period(game.get("date"), period, now)
                    or _day(game.get("date"), ZoneInfo(period["timezone"])) != str(row["match_date"])[:10]
                    or any(_id(game.get(slot)) != _id(snapshot.get(slot)) for slot in _SLOTS)):
                break
            courts[str(court)].append({"teamA": [_id(game[k]) for k in _SLOTS[:2]], "teamB": [_id(game[k]) for k in _SLOTS[2:]],
                                       "scoreA": game.get("score_t1"), "scoreB": game.get("score_t2")})
        else:
            for court, games in courts.items():
                pids = {pid for g in games for side in ("teamA", "teamB") for pid in g[side]}
                if "" in pids or not _valid(games, pids) or not _court_complete(games):
                    continue
                standings = round_robin_standings({"participants": [{"id": pid, "name": pid} for pid in pids], "rounds": [{"matches": games}]})
                for winner in standings:
                    if winner["rank"] == 1:
                        yield f"league:{row['session_id']}:{row['round_number']}:{court}", str(winner["participantId"])


def event_winner_highlights(db, *, club_id, period, players, keys, status, card_options, now=None):
    """Full rankings; the caller searches and paginates after ranking."""
    now = now or datetime.now(timezone.utc)
    credits = defaultdict(lambda: defaultdict(set))
    events, sources = _social_sources(db, club_id, period, now)
    saved_uids = {str(row.get("source_event_uid") or row["id"]) for row in events}
    seen_uids = set()
    for row in sorted(_session_rows(db, club_id), key=lambda r: (str(r.get("updated_at") or ""), int(r.get("version") or 0), str(r["id"])), reverse=True):
        state = row.get("state") or {}
        event = (state.get("page_state") or {}).get("event") or {}
        uid = str(event.get("sourceEventUid") or row["id"])
        if uid in saved_uids or uid in seen_uids:
            continue
        seen_uids.add(uid)
        submission, official = state.get("generator_submission") or {}, state.get("official_publish") or {}
        if submission.get("status") != "approved" and not official.get("published_match_ids"):
            continue
        review = submission.get("review") or {}
        request = review.get("request") or {}
        day = request.get("match_date") or submission.get("match_date") or official.get("match_date")
        if not day and official.get("match_context_by_live_id"):
            dates = []
            for batch in _batches(official["match_context_by_live_id"].values()):
                dates.extend(g["date"] for g in read_all_rows(lambda: db.table("matches").select("id,date").eq("club_id", club_id).in_("context_id", batch)))
            day = min(dates) if dates else None
        if _in_period(day, period, now):
            mapping = official.get("match_context_by_live_id") or {
                str(p["live_match_id"]): str(p["context_id"]) for p in review.get("payloads") or [] if p.get("live_match_id") and p.get("context_id")}
            links = {str(p["id"]): _id((request.get("player_ids") or {}).get(str(p["id"]), p.get("player_id"))) for p in event.get("participants") or []}
            sources.append((uid, event, links, mapping))
    canonical = {}
    for batch in _batches(context for _, _, _, mapping in sources for context in mapping.values()):
        canonical.update({str(g["context_id"]): g for g in read_all_rows(lambda: db.table("matches")
            .select("id,context_id,deleted_at,date,score_t1,score_t2," + ",".join(_SLOTS)).eq("club_id", club_id).in_("context_id", batch))})
    for uid, event, links, mapping in sources:
        linked = [pid for pid in links.values() if pid]
        if len(set(linked)) != len(linked):
            continue
        # Preserve recorded substitutes' identities before calculating standings.
        event = deepcopy(event)
        decision = event.get("round_robin_winner") or {}
        valid_decision = decision.get("display_fingerprint") == round_robin_result_fingerprint(event)
        for p in event.get("participants") or []:
            p["player_id"] = int(links[str(p["id"])]) if links.get(str(p["id"])) else None
        if not is_moneyball(event):
            event = round_robin_players_of_record(event)
        if valid_decision:
            event["round_robin_winner"] = {**decision, "display_fingerprint": round_robin_result_fingerprint(event)}
        else:
            event.pop("round_robin_winner", None)
        links = {str(p["id"]): _id(p.get("player_id")) for p in event.get("participants") or []}
        if mapping:
            games = {str(g.get("id")): g for r in event.get("rounds") or [] for g in _raw_matches(r)}
            valid = True
            for mid, context in mapping.items():
                g, record = games.get(mid), canonical.get(context)
                if (not g or not record or record.get("deleted_at") or not _in_period(record.get("date"), period, now)
                        or [links.get(str(pid), "") for side in ("A", "B") for pid in g.get(f"side{side}", g.get(f"team{side}", []))]
                        != [_id(record.get(slot)) for slot in _SLOTS if record.get(slot) is not None]):
                    valid = False
                    break
                if (g.get("scoreA"), g.get("scoreB")) != (record.get("score_t1"), record.get("score_t2")):
                    event.pop("round_robin_winner", None)
                    g.update(scoreA=record.get("score_t1"), scoreB=record.get("score_t2"))
            if not valid:
                continue
        try:
            for key, result_id, participant_id in _event_credits(event, keys):
                if key in keys and links.get(participant_id):
                    credits[key][links[participant_id]].add(f"event:{uid}:{result_id}")
        except (ValueError, KeyError, TypeError):
            # Invalid or unfinished frozen Moneyball results cannot claim a title.
            continue
    if "ladder_court_wins" in keys:
        for identity, pid in _league_credits(db, club_id, period, now):
            credits["ladder_court_wins"][pid].add(identity)
    from jupr_app.services.leaderboard_service import _player_is_active
    output = {}
    for key in keys & EVENT_WIN_CARD_KEYS:
        minimum = max(1, int((card_options.get(key) or {}).get("minimum", 0)))
        rows = []
        for player in players:
            count = len(credits[key][_id(player.get("id"))])
            active = _player_is_active(player)
            if (is_merged_player(player) or count < minimum or (status == "active" and not active)
                    or (status == "inactive" and active)):
                continue
            unit = "court win" if key == "ladder_court_wins" else "title"
            rows.append({"player_id": player["id"], "player_name": str(player.get("name") or "Player"),
                         "club_id": club_id, "league_name": "OVERALL", "is_active": active,
                         "metric_value": count, "metric_sample": count, "metric_display": f"{count} {unit}{'' if count == 1 else 's'}"})
        rows.sort(key=lambda r: (-r["metric_value"], r["player_name"].casefold(), str(r["player_id"])))
        rank = 0
        previous = None
        for index, row in enumerate(rows, 1):
            if row["metric_value"] != previous:
                rank = index
            row.update(rank=rank, rank_position=rank)
            previous = row["metric_value"]
        output[key] = rows
    return output
