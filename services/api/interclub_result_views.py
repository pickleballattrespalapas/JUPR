"""Small, explicit result projections shared by publication and review."""
from __future__ import annotations

from jupr_app.domain import interclub_competition as competition


def result_rows(documents, *, participants=True):
    rows = []
    for document in documents:
        outcomes = {row["id"]: row for row in competition.summarize_document(document)["encounters"]} if participants else {}
        for encounter in document["encounters"]:
            row = {key: encounter[key] for key in ("id", "division", "club_a", "club_b")}
            row.update(meet_id=document["meet_id"], phase=document["phase"], weather=document["weather"])
            row["pairings"] = []
            for pairing in encounter["pairings"]:
                games = []
                for game in pairing["games"]:
                    public_game = {key: game.get(key) for key in ("status", "a", "b", "winner")}
                    if participants:
                        public_game["played_at"] = game.get("played_at")
                        for side in ("a", "b"):
                            # A lineup on an unplayed game is not an appearance.
                            public_game[f"players_{side}"] = list(game.get(f"players_{side}") or pairing.get(f"players_{side}", [])) if game["status"] in {"completed", "retired"} else []
                    games.append(public_game)
                row["pairings"].append({"kind": pairing["kind"], "games": games})
            tie = encounter.get("tiebreak")
            row["tiebreak"] = {key: tie.get(key) for key in ("status", "a", "b")} if tie else None
            if participants:
                row["outcome"] = {key: outcomes[encounter["id"]][key] for key in
                                  ("winner", "points_a", "points_b", "pairings_a", "pairings_b", "games_a", "games_b")}
                if tie:
                    row["tiebreak"]["played_at"] = tie.get("played_at")
                    for side in ("a", "b"):
                        row["tiebreak"][f"players_{side}"] = list(tie.get(f"order_{side}", [])) if tie["status"] == "completed" else []
            rows.append(row)
    return rows


def result_player_catalog(db, season_id, rows):
    """Only names of result participants, never contacts or club-local IDs."""
    represented = {}
    for row in rows:
        games = [g for pairing in row["pairings"] for g in pairing["games"]]
        if row.get("tiebreak"):
            games.append(row["tiebreak"])
        for game in games:
            for side in ("a", "b"):
                for entry_id in game.get(f"players_{side}", []):
                    represented[entry_id] = row[f"club_{side}"]
    if not represented:
        return []
    entries = []
    ids = sorted(represented)
    for offset in range(0, len(ids), 400):
        entries.extend(db.table("pcs_interclub_entries").select("id,club_id,player_id").eq("season_id", str(season_id)).in_("id", ids[offset:offset + 400]).execute().data or [])
    result = []
    for club_id in sorted(set(represented.values())):
        selected = [entry for entry in entries if entry["club_id"] == club_id and represented.get(entry["id"]) == club_id]
        for offset in range(0, len(selected), 400):
            chunk = selected[offset:offset + 400]
            players = db.table("players").select("id,name").eq("club_id", club_id).in_("id", [entry["player_id"] for entry in chunk]).execute().data or []
            names = {str(player["id"]): player["name"] for player in players}
            result.extend({"id": entry["id"], "club_id": club_id, "name": names[str(entry["player_id"])]}
                          for entry in chunk if str(entry["player_id"]) in names)
    return sorted(result, key=lambda player: (player["name"].casefold(), player["club_id"], player["id"]))
