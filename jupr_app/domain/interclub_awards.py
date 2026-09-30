"""Season honors derived only from the reviewed, public-safe official results."""
from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5


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
    clubs = {club["id"]: club["name"] for club in document["clubs"]}
    catalog = {player["id"]: player for player in document.get("players", [])}
    participants: dict[str, set[str]] = {}
    for result in document.get("competition_results", []):
        for side in ("a", "b"):
            participants.setdefault(result[f"club_{side}"], set()).update(played_players(result, side))
    awards = []

    def add(kind: str, club: str, entry: str | None = None, division: str = ""):
        titles = {"participation": "Interclub Season Participant", "division_champion": f"{division} Interclub Champion",
                  "club_cup_champion": "Interclub Club Cup Champion"}
        player = catalog.get(entry) if entry else None
        if entry and (not player or player["club_id"] != club):
            raise ValueError("A result participant is missing their season player record. Review the meet before awarding trophies.")
        recipient = f"player:{entry}" if entry else f"club:{club}"
        awards.append({"id": str(uuid5(NAMESPACE_URL, f"pcs:interclub:{season_id}:{recipient}:{kind}:{division}")),
                       "award_key": kind, "division": division, "title": titles[kind],
                       "recipient_type": "player" if entry else "club", "club_id": club,
                       "entry_id": entry, "recipient_name": player["name"] if player else clubs[club]})

    for club, entries in sorted(participants.items()):
        if not entries:
            continue
        add("participation", club)
        for entry in sorted(entries):
            add("participation", club, entry)
    for final in summary["divisions"]:
        add("division_champion", final["winner"], division=final["division"])
        for entry in final["players"]:
            add("division_champion", final["winner"], entry, final["division"])
    for club in summary["champions"]:
        add("club_cup_champion", club)
        for entry in sorted(participants.get(club, set())):
            add("club_cup_champion", club, entry)
    return sorted(awards, key=lambda row: row["id"])
