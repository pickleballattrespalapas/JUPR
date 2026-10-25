"""Date-scoped, verifiable choices for the recap editor. No source writes."""
from collections import Counter
from decimal import Decimal, InvalidOperation
from math import isfinite

import pandas as pd

from jupr_app.data.paged_reads import read_all_rows
from jupr_app.domain.leaderboard_metrics import compute_overall_metrics, overall_highlights


def identity_key(value):
    """Unify integral database IDs after dataframe coercion; retain opaque IDs."""
    text = str(value)
    try:
        number = Decimal(text)
        if number.is_finite() and number == number.to_integral_value():
            return str(int(number))
    except InvalidOperation:
        pass
    return text


METRIC_LABELS = {
    "highest_rating": "Highest recorded rating",
    "best_win_pct": "Best win %",
    "most_wins": "Most wins",
    "point_differential": "Best point differential",
    "average_margin": "Best average margin",
    "hot_hand": "Current win streak",
    "longest_win_streak": "Longest win streak",
    "close_game_record": "Best close-game record",
    "most_upsets": "Most upset wins",
    "opponent_strength": "Strongest opposition",
    "best_partnership": "Best partnership",
    "partner_variety": "Most partners",
    "playing_days": "Most playing days",
}
METRIC_KEYS = {f"METRIC_{key}": label for key, label in METRIC_LABELS.items()}


def metric_candidates(frame, names, tz_name):
    """Reuse Overall's metrics, restricted to this recap's recorded doubles games.

    The rating card uses the final recorded snapshot in this period, never a
    player's present-day rating. New categories are opt-in in the editor.
    """
    output = {key: [] for key in METRIC_KEYS}
    if frame.empty:
        return output
    games, people = [], {}
    slots = ("t1_p1", "t1_p2", "t2_p1", "t2_p2")
    for _, series in frame.sort_values(["date_dt", "id"]).iterrows():
        row = series.to_dict()
        if row.get("rating_scope") == "unrated":
            continue
        try:
            ids = [int(row[slot]) for slot in slots]
            a, b = int(row["score_t1"]), int(row["score_t2"])
        except (ValueError, TypeError, KeyError):
            continue
        if len(set(ids)) != 4 or min(ids) <= 0 or a == b or min(a, b) < 0:
            continue
        row.update({slot: pid for slot, pid in zip(slots, ids)})
        if pd.isna(row["date_dt"]):
            raise ValueError("Recap metric evidence requires a valid match timestamp.")
        row["date"] = row["date_dt"].isoformat()
        games.append(row)
        for index, pid in enumerate(ids):
            player = people.setdefault(pid, {"player_id": pid, "player_name": names.get(pid, f"Player #{pid}"),
                                             "matches_played": 0, "wins": 0, "rating_jupr": None})
            player["matches_played"] += 1
            player["wins"] += int(a > b if index < 2 else b > a)
            try:
                value = float(row.get(f"{slots[index]}_r_end"))
                player["rating_jupr"] = value / 400 if isfinite(value) else None
            except (ValueError, TypeError):
                player["rating_jupr"] = None
    for row in people.values():
        row["win_pct"] = 100 * row["wins"] / row["matches_played"]
    metrics = compute_overall_metrics(games, names={str(k): v for k, v in names.items()}, timezone=tz_name, partnership_minimum=3)
    ranked = overall_highlights(list(people.values()), metrics=metrics, min_games=3,
                                card_options={key: {"depth": 10, "minimum": 3 if key in {"highest_rating", "close_game_record", "best_partnership"} else 1}
                                              for key in METRIC_LABELS})
    for metric, label in METRIC_LABELS.items():
        key = f"METRIC_{metric}"
        output[key] = [{"candidate_id": f"player:{row['player_id']}", "key": key, "label": label,
                        "display": f"{row['player_name']} — {row['metric_display']} · {row['matches_played']} games",
                        "player_ids": [row["player_id"]], "event_key": None,
                        "value_json": {"metric": metric, "value": row["metric_value"], "sample": row["metric_sample"], "games": row["matches_played"]}}
                       for row in ranked.get(metric, [])]
    return output


def court_result(*, identity, title, day, round_number, court_number, games, names, complete, href=""):
    from jupr_app.domain.live_beta_engine import compute_standings

    names = {identity_key(pid): name for pid, name in names.items()}
    valid = []
    for game in games:
        try:
            values = [float(game.get(key)) for key in ("scoreA", "scoreB")]
            if not all(isfinite(value) and value.is_integer() for value in values):
                continue
            a, b = map(int, values)
        except (TypeError, ValueError):
            continue
        seats = [identity_key(pid) for side in ("teamA", "teamB") for pid in game.get(side, [])]
        if a == b or min(a, b) < 0 or len(set(seats)) != len(seats) or any(pid in {"", "None", "nan"} for pid in seats):
            continue
        if len(seats) not in {2, 4} or len(game.get("teamA", [])) != len(game.get("teamB", [])):
            continue
        valid.append({**game, "scoreA": a, "scoreB": b, "teamA": [identity_key(pid) for pid in game["teamA"]], "teamB": [identity_key(pid) for pid in game["teamB"]]})
    if not valid:
        return None
    pids = sorted({pid for game in valid for side in ("teamA", "teamB") for pid in game[side]})
    rows = compute_standings(valid, [{"id": pid, "name": names.get(pid, f"Player #{pid}")} for pid in pids])
    # Alphabetical display order must never turn a tied result into one winner.
    score = lambda row: (row["wins"], row["differential"], row["pointsFor"])
    best = max(score(row) for row in rows)
    leaders = [row for row in rows if score(row) == best]
    final = complete and len(valid) == len(games)
    label = ("Winner" if len(leaders) == 1 else "Tied for first") if final else "Leader from recorded games"
    return {"id": identity, "kind": "Ladder court", "title": f"{title} · Round {round_number} · Court {court_number}",
            "date": day, "summary": f"{label}: " + ", ".join(row["name"] for row in leaders),
            "court_players": [{"id": f"{'league' if identity.startswith('league:') else identity.rsplit(':', 2)[0]}:{row['participantId']}", "name": row["name"]} for row in leaders],
            "details": [f"{row['name']} — {row['wins']}–{row['losses']} · point difference {row['differential']:+d}" for row in leaders]
                       + [f"{len(valid)} recorded games" + ("" if final else " · incomplete court results")], "href": href}


def generator_courts(row, result):
    """Called only after generator_result has verified approval and play date."""
    event = ((row.get("state") or {}).get("page_state") or {}).get("event") or {}
    if event.get("generatorKind") != "ladder":
        return []
    names = {str(p["id"]): p.get("name") or "Player" for p in event.get("participants", [])}
    output = []
    for round_row in event.get("rounds", []):
        if round_row.get("status") != "saved":
            continue
        for court in round_row.get("courts", []):
            games = [{**game, "teamA": game.get("sideA", game.get("teamA", [])), "teamB": game.get("sideB", game.get("teamB", []))}
                     for game in court.get("matches", [])]
            item = court_result(identity=f"generator:{result['id']}:{round_row['number']}:{court['courtNumber']}",
                                title=result["title"], day=result["date"], round_number=round_row["number"], court_number=court["courtNumber"],
                                games=games, names=names, complete=True, href=result.get("href", ""))
            if item:
                output.append(item)
    return output


def social_courts(source, result):
    event = source.get("raw_event_json") or {}
    if source.get("event_type") == "round_robin" or event.get("type") == "round_robin":
        return []
    names = {str(p["id"]): p.get("name") or "Player" for p in event.get("participants", [])}
    output = []
    for round_row in event.get("rounds", []):
        for court in round_row.get("courts", []):
            games = [game for mini in court.get("miniRounds", []) for game in mini.get("matches", [])]
            item = court_result(identity=f"social:{result['id']}:{round_row['number']}:{court['courtNumber']}",
                                title=result["title"], day=result["date"], round_number=round_row["number"], court_number=court["courtNumber"],
                                games=games, names=names, complete=True)
            if item:
                output.append(item)
    return output


def league_courts(ctx, matches, start, end):
    """Court assignments come from submitted rounds; scores come from live records.

    Deleted/out-of-range/changed-seat matches invalidate the snapshot rather
    than letting stale private round JSON masquerade as a verified result.
    """
    from urllib.parse import quote
    from jupr_app.domain.recaps.weekly_report import played_day

    db, club = ctx.supabase, str(ctx.club_id)
    rounds = read_all_rows(lambda: db.table("league_live_rounds")
                          .select("id,session_id,round_number,match_date,matches_json,submitted_match_ids")
                          .eq("club_id", club).eq("status", "submitted").gte("match_date", start).lte("match_date", end))
    session_ids = sorted({str(row["session_id"]) for row in rounds})
    sessions = {}
    for offset in range(0, len(session_ids), 100):
        batch = session_ids[offset:offset + 100]
        sessions.update({str(row["id"]): row for row in read_all_rows(lambda: db.table("league_live_sessions")
                         .select("id,league_name,week_tag").eq("club_id", club).in_("id", batch))})
    canonical = {identity_key(row["id"]): row for row in matches if row.get("id") is not None and
                 (not row.get("deleted_at") or (isinstance(row.get("deleted_at"), float) and not isfinite(row["deleted_at"])))}
    names = {identity_key(pid): name for pid, name in (getattr(ctx, "id_to_name", {}) or {}).items()}
    scope = getattr(ctx, "recap_leagues", []) or []
    output, warnings = [], []
    slots = ("t1_p1", "t1_p2", "t2_p1", "t2_p2")
    for row in rounds:
        session = sessions.get(str(row["session_id"]))
        if not session or (scope and session["league_name"] not in scope):
            continue
        ids, saved = row.get("submitted_match_ids") or [], row.get("matches_json") or []
        if not ids or len(ids) != len(saved) or len(set(identity_key(v) for v in ids)) != len(ids):
            warnings.append(f"Court results for {session['league_name']} could not be verified. Refresh after checking recorded scores.")
            continue
        by_court = {}
        verified = True
        for mid, source in zip(ids, saved):
            game = canonical.get(identity_key(mid))
            try:
                same_seats = game is not None and all(int(game.get(slot)) == int(source.get(slot)) for slot in slots)
            except (ValueError, TypeError):
                same_seats = False
            if (not game or game.get("league") != session["league_name"] or not start <= played_day(game.get("date")) <= end
                    or not same_seats):
                verified = False
                break
            number = source.get("court", source.get("court_number"))
            if not number:
                verified = False
                break
            by_court.setdefault(int(number), []).append({"teamA": [game[slots[0]], game[slots[1]]], "teamB": [game[slots[2]], game[slots[3]]],
                                                       "scoreA": game.get("score_t1"), "scoreB": game.get("score_t2")})
        if not verified:
            warnings.append(f"Court results for {session['league_name']} changed or are incomplete. Check recorded scores before highlighting a winner.")
            continue
        for number, games in sorted(by_court.items()):
            counts = Counter(identity_key(pid) for game in games for side in ("teamA", "teamB") for pid in game[side])
            complete = (len(counts) == 4 and len(games) == 3 and set(counts.values()) == {3}) or (len(counts) == 5 and len(games) == 5 and set(counts.values()) == {4})
            item = court_result(identity=f"league:{row['session_id']}:{row['round_number']}:{number}", title=session["league_name"],
                                day=str(row["match_date"]), round_number=row["round_number"], court_number=number, games=games, names=names,
                                complete=complete, href=f"leagues/{quote(session['league_name'], safe='')}/standings")
            if item:
                output.append(item)
    return output, list(dict.fromkeys(warnings))
