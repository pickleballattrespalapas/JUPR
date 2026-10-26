"""Week-scoped editorial choices backed by current, canonical match evidence."""
from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Any

from jupr_app.data.paged_reads import read_all_rows

MAX_HIGHLIGHTS = 6
ADMIN_ROLES = {"super_admin", "club_owner", "administrator"}
CATEGORIES = [
    ("unbeaten", "Unbeaten week", "At least 4 games with no losses."),
    ("streak", "Winning streak", "At least 3 consecutive wins in recorded match order this week."),
    ("rating_gain", "Rating gain", "Positive JUPR gain, with rating snapshots available for every game."),
    ("point_difference", "Point difference", "Positive points scored minus points conceded over at least 3 games."),
    ("close_wins", "Close-game wins", "At least 2 wins decided by 1 or 2 points."),
    ("win_pct", "Win percentage", "At least 4 games and a winning record."),
    ("wins", "Weekly wins", "At least 3 wins this week."),
    ("games", "Most active", "Ranked by games played this week."),
    ("court", "Ladder court winners", "Completed courts verified against submitted rounds and current scores."),
]


class HighlightsConflict(ValueError):
    pass


def _number(value):
    try:
        result = float(value)
        return result if isfinite(result) else None
    except (TypeError, ValueError):
        return None


def stat_candidates(matches, names):
    """Compute only this week's valid games; never use current season ratings."""
    stats = {}
    slots = ("t1_p1", "t1_p2", "t2_p1", "t2_p2")
    for game in sorted(matches, key=lambda row: (str(row.get("date") or ""), int(row.get("id") or 0))):
        a, b = _number(game.get("score_t1")), _number(game.get("score_t2"))
        if a is None or b is None or min(a, b) < 0 or a == b or not a.is_integer() or not b.is_integer():
            continue
        ids = [game.get(slot) for slot in slots if game.get(slot) is not None]
        if len(ids) not in {2, 4} or len(set(ids)) != len(ids):
            continue
        for index, slot in enumerate(slots):
            pid = game.get(slot)
            if pid not in names:
                continue
            own, against = (a, b) if index < 2 else (b, a)
            won = own > against
            row = stats.setdefault(pid, {"games": 0, "wins": 0, "points": 0, "against": 0,
                                         "run": 0, "streak": 0, "close_wins": 0, "delta": 0, "rated": 0})
            row["games"] += 1
            row["wins"] += int(won)
            row["points"] += int(own)
            row["against"] += int(against)
            row["run"] = row["run"] + 1 if won else 0
            row["streak"] = max(row["streak"], row["run"])
            row["close_wins"] += int(won and own - against <= 2)
            start, end = _number(game.get(f"{slot}_r")), _number(game.get(f"{slot}_r_end"))
            if start is not None and end is not None:
                row["rated"] += 1
                row["delta"] += (end - start) / 400
    output = []
    labels = {key: title for key, title, _ in CATEGORIES}
    for pid, row in stats.items():
        games, wins = row["games"], row["wins"]
        record = f"{wins}–{games - wins} in {games} games"
        difference = row["points"] - row["against"]
        metrics = [
            ("unbeaten", games >= 4 and wins == games, wins, f"{wins}–0", record),
            ("streak", row["streak"] >= 3, row["streak"], f"{row['streak']} straight wins", f"Longest run this week · {record}"),
            ("rating_gain", row["rated"] == games and row["delta"] >= 0.0005, row["delta"], f"+{row['delta']:.3f} JUPR", record),
            ("point_difference", games >= 3 and difference > 0, difference, f"+{difference} points", f"{row['points']} scored · {row['against']} conceded · {record}"),
            ("close_wins", row["close_wins"] >= 2, row["close_wins"], f"{row['close_wins']} close wins", f"Decided by 1–2 points · {record}"),
            ("win_pct", games >= 4 and wins > games / 2, 100 * wins / games, f"{100 * wins / games:.1f}% wins", record),
            ("wins", wins >= 3, wins, f"{wins} wins", record),
            ("games", True, games, f"{games} games", record),
        ]
        for key, qualifies, value, metric, detail in metrics:
            if qualifies:
                output.append({"id": f"{key}:{pid}", "category": key, "title": labels[key],
                               "name": names[pid], "metric": metric, "detail": detail,
                               "player_ids": [pid], "value": round(value, 6), "sample": games})
    return sorted(output, key=lambda row: (-row["value"], -row["sample"], row["name"].casefold(), row["id"]))


def _court_candidates(db, *, club_id, league_name, week_num, matches, names):
    from jupr_app.domain.recaps.recap_evidence import court_result, identity_key
    from jupr_app.services.public_league_results_service import _parse_week_num

    sessions = read_all_rows(lambda: db.table("league_live_sessions").select("id,week_tag")
                             .eq("club_id", club_id).eq("league_name", league_name))
    canonical = {identity_key(row["id"]): row for row in matches}
    output, warnings = [], []
    slots = ("t1_p1", "t1_p2", "t2_p1", "t2_p2")
    for session in sessions:
        if _parse_week_num(session.get("week_tag")) != week_num:
            continue
        rounds = read_all_rows(lambda: db.table("league_live_rounds")
                              .select("id,session_id,round_number,match_date,matches_json,submitted_match_ids")
                              .eq("club_id", club_id).eq("session_id", session["id"]).eq("status", "submitted"))
        for round_row in rounds:
            ids, saved = round_row.get("submitted_match_ids") or [], round_row.get("matches_json") or []
            by_court = {}
            valid = bool(ids) and len(ids) == len(saved) and len({identity_key(mid) for mid in ids}) == len(ids)
            for mid, snapshot in zip(ids, saved):
                game = canonical.get(identity_key(mid))
                number = snapshot.get("court", snapshot.get("court_number"))
                same_seats = game and all(identity_key(game.get(slot)) == identity_key(snapshot.get(slot)) for slot in slots)
                if not same_seats or not number:
                    valid = False
                    break
                by_court.setdefault(str(number), []).append({"teamA": [game[slots[0]], game[slots[1]]],
                                                           "teamB": [game[slots[2]], game[slots[3]]],
                                                           "scoreA": game.get("score_t1"), "scoreB": game.get("score_t2")})
            if not valid:
                warnings.append(f"Round {round_row['round_number']} court results need verification after score or roster changes.")
                continue
            for number, games in sorted(by_court.items()):
                counts = Counter(pid for game in games for side in ("teamA", "teamB") for pid in game[side])
                complete = (len(counts) == 4 and len(games) == 3 and set(counts.values()) == {3}) or (len(counts) == 5 and len(games) == 5 and set(counts.values()) == {4})
                if not complete or any(pid not in names for pid in counts):
                    continue
                item = court_result(identity=f"league:{session['id']}:{round_row['round_number']}:{number}",
                                    title=league_name, day=str(round_row.get("match_date") or ""),
                                    round_number=round_row["round_number"], court_number=number,
                                    games=games, names=names, complete=True)
                if not item or item["summary"].startswith("Leader from"):
                    continue
                output.append({"id": f"court:{session['id']}:{round_row['round_number']}:{number}",
                               "category": "court", "title": "Ladder court winners",
                               "name": ", ".join(player["name"] for player in item["court_players"]),
                               "metric": f"Round {round_row['round_number']} · Court {number}",
                               "detail": item["summary"] + " · " + " · ".join(item["details"]) + f" · {item['date']}",
                               "player_ids": [int(player["id"].rsplit(":", 1)[-1]) for player in item["court_players"]],
                               "value": 0, "sample": len(games)})
    return output, list(dict.fromkeys(warnings))


def _config(db, club_id, league_name, week_num):
    try:
        rows = db.table("league_weekly_highlights").select("mode,selected_ids,revision,updated_at")\
            .eq("club_id", club_id).eq("league_name", league_name).eq("week_num", week_num).limit(1).execute().data or []
    except Exception as exc:
        # Rolling deployment compatibility only; connection/permission failures surface.
        if getattr(exc, "code", None) not in {"42P01", "PGRST205"} or "league_weekly_highlights" not in str(exc):
            raise
        return {"mode": "automatic", "selected_ids": [], "revision": 0}, False
    return (rows[0] if rows else {"mode": "automatic", "selected_ids": [], "revision": 0}), True


def build_weekly_highlights(db, *, club_id, league_name, week_num, matches, players, include_candidates=False):
    config, available = _config(db, club_id, league_name, week_num) if week_num else ({"mode": "automatic", "selected_ids": [], "revision": 0}, True)
    games = [row for row in matches if row.get("week_num") == week_num and not row.get("deleted_at")]
    names = {pid: str(row.get("name") or f"Player {pid}") for pid, row in players.items()}
    candidates = stat_candidates(games, names)
    warnings = []
    if include_candidates or any(key.startswith("court:") for key in config["selected_ids"]):
        courts, warnings = _court_candidates(db, club_id=club_id, league_name=league_name, week_num=week_num, matches=games, names=names)
        candidates.extend(courts)
    lookup = {row["id"]: row for row in candidates}
    automatic, seen = [], set()
    for category in ("unbeaten", "streak", "rating_gain", "point_difference", "close_wins", "win_pct", "wins", "games"):
        candidate = next((row for row in candidates if row["category"] == category and not seen.intersection(row["player_ids"])), None)
        if candidate:
            automatic.append(candidate)
            seen.update(candidate["player_ids"])
        if len(automatic) == 3:
            break
    cards = automatic if config["mode"] == "automatic" else [lookup[key] for key in config["selected_ids"] if key in lookup]
    result = {"week_num": week_num, "mode": config["mode"], "cards": cards}
    if include_candidates:
        result.update({"revision": config["revision"], "updated_at": config.get("updated_at"),
                       "selected_ids": config["selected_ids"], "automatic_cards": automatic, "max_highlights": MAX_HIGHLIGHTS,
                       "available": available, "warnings": warnings,
                       "unavailable_ids": [key for key in config["selected_ids"] if key not in lookup],
                       "categories": [{"id": key, "title": title, "qualification": qualification,
                                       "candidates": [row for row in candidates if row["category"] == key]}
                                      for key, title, qualification in CATEGORIES]})
    return result


def save_weekly_highlights(db, *, club_id, league_name, week_num, mode, selected_ids, expected_revision, actor_email, actor_role):
    from jupr_app.services.admin_league_results_service import build_admin_league_results

    if actor_role not in ADMIN_ROLES:
        raise PermissionError("Only club administrators can select weekly highlights.")
    results = build_admin_league_results(db, club_id=club_id, league_name=league_name, week_num=week_num, include_highlight_candidates=True)
    if results["selected_week"] != week_num:
        raise ValueError("Choose a week from this league's results.")
    editor = results["weekly_highlight_selection"]
    if not editor["available"]:
        raise RuntimeError("Weekly highlights storage is not available yet.")
    valid = {row["id"] for group in editor["categories"] for row in group["candidates"]}
    if mode not in {"automatic", "curated"} or len(selected_ids) > MAX_HIGHLIGHTS or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Choose up to 6 distinct highlights.")
    if (mode == "automatic" and selected_ids) or any(key not in valid for key in selected_ids):
        raise ValueError("A selected highlight is no longer available. Reload the suggestions and review your choices.")
    try:
        response = db.rpc("save_league_weekly_highlights", {"p_club_id": club_id, "p_league_name": league_name,
                         "p_week_num": week_num, "p_mode": mode, "p_selected_ids": selected_ids,
                         "p_expected_revision": expected_revision, "p_actor_email": actor_email, "p_actor_role": actor_role}).execute()
    except Exception as exc:
        if getattr(exc, "code", None) == "40001":
            raise HighlightsConflict("Another admin changed this week. Reload the saved highlights before saving again.") from exc
        raise
    saved = response.data
    return {"ok": True, "week_num": week_num, "revision": saved["revision"], "mode": saved["mode"], "selected_ids": saved["selected_ids"]}
