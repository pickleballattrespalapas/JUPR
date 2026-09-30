"""Season honors derived only from the reviewed, public-safe official results."""
from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

PERFORMANCE_BADGES = {
    "matchup_win": {"title": "Win Matchup", "requirement": "Win at least two games you played in one three-game inter-club matchup."},
    "matchup_sweep": {"title": "Sweep Matchup", "requirement": "Play and win all three games in one inter-club matchup."},
    "undefeated_meet": {"title": "Undefeated Day", "requirement": "Play at least one game in a meet and win every game you played, including singles tiebreaks."},
}


def award_title(kind: str, name: str, division: str = "") -> str:
    league = name.strip() or "Inter-Club"
    if kind in PERFORMANCE_BADGES:
        return PERFORMANCE_BADGES[kind]["title"]
    return {"participation": f"{league} Season Participant",
            "division_champion": f"{league} {division} Division Champion",
            "club_cup_champion": f"{league} League Champion"}[kind]


def _award(season_id, document, kind, club, entry=None, division="", occurrence=""):
    player = next((p for p in document.get("players", []) if p["id"] == entry), None) if entry else None
    if entry and (not player or player["club_id"] != club):
        raise ValueError("A result participant is missing their season player record. Review the meet before awarding trophies.")
    recipient = f"player:{entry}" if entry else f"club:{club}"
    identity = f"pcs:interclub:{season_id}:{recipient}:{kind}:{division}"
    if occurrence:
        identity += f":{occurrence}"
    return {"id": str(uuid5(NAMESPACE_URL, identity)), "award_key": kind, "division": division,
            "title": award_title(kind, document.get("name", ""), division),
            "recipient_type": "player" if entry else "club", "club_id": club, "entry_id": entry,
            "recipient_name": player["name"] if player else next(c["name"] for c in document["clubs"] if c["id"] == club)}


def _winner(game):
    if game.get("status") not in {"completed", "retired"}:
        return None
    if game.get("status") == "retired":
        return game.get("winner")
    a, b = game.get("a"), game.get("b")
    return ("a" if a > b else "b") if a is not None and b is not None and a != b else None


def performance_awards(season_id: str, document: dict, *, entry_ids: set[str] | None = None) -> list[dict]:
    """Repeatable badges and participation from actual published appearances.

    A replacement earns only their own wins. Forfeits and unplayed games are
    never appearances; an injury retirement uses the official winner.
    """
    if not document.get("players"):
        return []
    awards, days, participants = {}, {}, {}
    clubs = {club["id"]: club["name"] for club in document["clubs"]}
    meets = {meet["id"]: meet for meet in document.get("meets", [])}

    def add(kind, club, entry, division="", occurrence="", **context):
        row = _award(season_id, document, kind, club, entry, division, occurrence)
        awards[row["id"]] = {**row, **context}

    for result in document.get("competition_results", []):
        if result.get("weather") == "rescheduled":
            continue
        meet_id, division = result["meet_id"], result["division"]
        for index, pairing in enumerate(result.get("pairings", [])):
            wins = {}
            for game in pairing["games"]:
                if game.get("status") not in {"completed", "retired"}:
                    continue
                winner = _winner(game)
                for side in ("a", "b"):
                    club = result[f"club_{side}"]
                    for entry in set(game.get(f"players_{side}", [])):
                        if entry_ids is not None and entry not in entry_ids:
                            continue
                        key = (club, entry)
                        stamp = game.get("played_at") or ""
                        participants.setdefault(key, []).append(stamp)
                        days.setdefault((meet_id, club, entry), []).append((winner == side, stamp))
                        if winner == side:
                            wins.setdefault(key, []).append(stamp)
            # Championship pairings are one game each, not three-game matchups.
            if len(pairing["games"]) == 3:
                occurrence = f"{meet_id}:{result.get('id') or ':'.join(sorted([result['club_a'], result['club_b']]))}:{pairing.get('kind', index)}"
                matchup = f"{division} · {clubs[result['club_a']]} vs {clubs[result['club_b']]}"
                kind = pairing.get("kind", "").replace("_", " ").title()
                for (club, entry), stamps in wins.items():
                    for badge, threshold in (("matchup_win", 2), ("matchup_sweep", 3)):
                        if len(stamps) >= threshold:
                            add(badge, club, entry, division, occurrence, meet_id=meet_id,
                                detail=f"{matchup}{' · ' + kind if kind else ''}", earned_at=max(stamps) or None)
        tie = result.get("tiebreak") or {}
        if tie.get("status") == "completed":
            for side in ("a", "b"):
                club = result[f"club_{side}"]
                for entry in set(tie.get(f"players_{side}", [])):
                    if entry_ids is not None and entry not in entry_ids:
                        continue
                    stamp = tie.get("played_at") or ""
                    participants.setdefault((club, entry), []).append(stamp)
                    days.setdefault((meet_id, club, entry), []).append((_winner(tie) == side, stamp))
    for (meet_id, club, entry), games in days.items():
        if all(won for won, _ in games):
            host = clubs.get(meets.get(meet_id, {}).get("host_club_id"))
            add("undefeated_meet", club, entry, occurrence=meet_id, meet_id=meet_id,
                detail=f"{len(games)} game{'s' if len(games) != 1 else ''} won · Meet{(' at ' + host) if host else ''}",
                earned_at=max(stamp for _, stamp in games) or None)
    for (club, entry), stamps in participants.items():
        add("participation", club, entry, detail="Played in this inter-club season.", earned_at=min(filter(None, stamps), default=None))
    return sorted(awards.values(), key=lambda row: row["id"])


def played_players(result: dict, side: str) -> set[str]:
    games = [game for pairing in result.get("pairings", []) for game in pairing["games"]]
    if result.get("tiebreak"):
        games.append(result["tiebreak"])
    return {entry for game in games if game.get("status") in {"completed", "retired"}
            for entry in game.get(f"players_{side}", [])}


def final_results(document: dict) -> dict:
    results = document.get("competition_results", [])
    cup = document.get("club_cup", {})
    # Older published snapshots have no completion marker. Check every scheduled
    # meet as well as the Cup so an unfinished season never gets final honors.
    all_meets = {meet["id"] for meet in document.get("meets", [])}
    finished = {row["meet_id"] for row in results if row.get("weather") != "rescheduled"}
    complete = bool(all_meets) and all_meets <= finished and cup.get("status") == "complete"
    if document.get("season_complete") is False:
        complete = False
    finals = []
    for result in results:
        side = result.get("outcome", {}).get("winner")
        if result.get("phase") != "final" or side not in {"a", "b"}:
            continue
        other = "b" if side == "a" else "a"
        tiebreak = result.get("tiebreak") or {}
        finals.append({"division": result["division"], "meet_id": result["meet_id"],
                       "winner": result[f"club_{side}"], "runner_up": result[f"club_{other}"],
                       "games_won": result["outcome"][f"games_{side}"],
                       "games_lost": result["outcome"][f"games_{other}"],
                       "tiebreak": ({"winner_score": tiebreak[side], "runner_up_score": tiebreak[other]}
                                    if tiebreak.get("status") == "completed" else None),
                       "players": sorted(played_players(result, side))})
    participants = {entry for result in results for side in ("a", "b") for entry in played_players(result, side)}
    return {"complete": complete, "champions": cup.get("champions", []) if complete else [],
            "divisions": sorted(finals, key=lambda row: row["division"]), "players": len(participants),
            "meets": len(finished), "clubs": len(document.get("clubs", []))}


def season_awards(season_id: str, document: dict) -> list[dict]:
    summary = final_results(document)
    if not summary["complete"]:
        raise ValueError("Approve every scheduled meet and finish the championships before awarding season trophies.")
    participants: dict[str, set[str]] = {}
    division_players: dict[tuple[str, str], set[str]] = {}
    for result in document.get("competition_results", []):
        for side in ("a", "b"):
            participants.setdefault(result[f"club_{side}"], set()).update(played_players(result, side))
            division_players.setdefault((result[f"club_{side}"], result["division"]), set()).update(played_players(result, side))
    awards = []

    def add(kind: str, club: str, entry: str | None = None, division: str = ""):
        awards.append(_award(season_id, document, kind, club, entry, division))

    for club, entries in sorted(participants.items()):
        if not entries:
            continue
        for entry in sorted(entries):
            add("participation", club, entry)
    for final in summary["divisions"]:
        add("division_champion", final["winner"], division=final["division"])
        for entry in sorted(division_players.get((final["winner"], final["division"]), set())):
            add("division_champion", final["winner"], entry, final["division"])
    for club in summary["champions"]:
        add("club_cup_champion", club)
        for entry in sorted(participants.get(club, set())):
            add("club_cup_champion", club, entry)
    return sorted(awards, key=lambda row: row["id"])
