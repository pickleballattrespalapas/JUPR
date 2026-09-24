"""Deterministic Southern BCS scheduling, score validation and standings.

The functions here have no database, network, authentication or rating effects.
Only organizer-approved document revisions should be passed to season standings
or a rating publisher. A draft never becomes official merely by having scores.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations, groupby
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from services.api.interclub_competition_models import CompetitionDocument


KINDS = {"gender": ("women", "men"), "mixed": ("mixed_a", "mixed_b"),
         "mlp": ("women", "men", "mixed_a", "mixed_b")}
TERMINAL = {"completed", "retired", "forfeit", "double_forfeit", "unplayed"}
DECIDED = {"completed", "retired", "forfeit", "double_forfeit"}
PLAYED = {"completed", "retired"}


def _id(*parts: Any) -> str:
    return str(uuid5(NAMESPACE_URL, "pcs-interclub:" + ":".join(map(str, parts))))


def _normalize(document: dict | CompetitionDocument) -> dict:
    model = document if isinstance(document, CompetitionDocument) else CompetitionDocument.model_validate(document)
    return model.model_dump(mode="json")


def _lineup(pairing: dict, game: dict, side: str) -> list[str]:
    return game.get(f"players_{side}") or pairing.get(f"players_{side}") or []


def _cutoff(pairing: dict) -> datetime | None:
    value = pairing.get("eligibility_deadline")
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def _distinct_players(players: list[str], count: int, label: str) -> None:
    if len(players) != count or len(set(players)) != count or any(not str(p).strip() for p in players):
        raise ValueError(f"{label} must contain {count} different players.")


def _score_winner(a: int | None, b: int | None, target: int, label: str) -> str:
    if a is None or b is None:
        raise ValueError(f"Enter both scores for {label}.")
    high, low = max(a, b), min(a, b)
    # The game ends on the first winning score: 11–8 or 15–13, never 15–8.
    if high < target or high - low < 2 or (high > target and high - low != 2):
        raise ValueError(f"{label} must finish at {target}, or win by two beyond {target}; there is no score cap.")
    return "a" if a > b else "b"


def _game_winner(game: dict) -> str | None:
    if game["status"] == "completed":
        return _score_winner(game.get("a"), game.get("b"), 11, "a completed doubles game")
    if game["status"] in {"retired", "forfeit"}:
        return game.get("winner")
    return None


def singles_court(division: str) -> str:
    """Court geometry is a skill-level rule, independent of player ratings."""
    try:
        rating = float(division.split("/")[0])
    except ValueError:
        rating = 4.0  # Open is above the skinny-singles skill bands.
    return "skinny" if rating < 4.0 else "full"


def singles_rotation(rallies_completed: int) -> int:
    """Zero-based active order index for the *next* rally, for both clubs."""
    if isinstance(rallies_completed, bool) or not isinstance(rallies_completed, int) or rallies_completed < 0:
        raise ValueError("Completed rallies must be a nonnegative integer.")
    return (rallies_completed // 4) % 4


def _validate_game(game: dict, pairing: dict, document: dict, *, official: bool) -> None:
    status, a, b = game["status"], game.get("a"), game.get("b")
    if status in {"forfeit", "double_forfeit", "unplayed"} and (a is not None or b is not None):
        raise ValueError("Unplayed forfeits and weather-unplayed games have no numeric score.")
    if status in {"unplayed", "double_forfeit"} and game.get("winner") is not None:
        raise ValueError("A weather-unplayed or double-forfeited game cannot have a winner.")
    if status == "unplayed" and document["weather"] != "finalized_partial":
        raise ValueError("Mark weather-unplayed games only when the meet cannot be rescheduled.")
    if document["phase"] != "regular" and status == "unplayed":
        raise ValueError("A championship or qualifying matchup must be decided on court or by an explicit forfeit.")
    if document["phase"] != "regular" and status == "double_forfeit":
        raise ValueError("Both championship teams need four eligible players; a double forfeit cannot decide their final.")
    if official and status == "pending":
        raise ValueError("Finish every game disposition before submitting the whole meet.")
    if status == "completed" and (official or (a is not None and b is not None)):
        winner = _score_winner(a, b, 11, "a completed doubles game")
        if game.get("winner") not in {None, winner}:
            raise ValueError("The selected game winner does not match the completed score.")
        game["winner"] = winner
    if official and status in {"retired", "forfeit"} and not game.get("winner"):
        raise ValueError("Select the winner of every injury-retired or forfeited game.")
    if official and status == "retired" and (a is None or b is None):
        raise ValueError("Record both actual scores when the injury stopped play.")
    if status == "retired" and a is not None and b is not None and max(a, b) >= 11 and abs(a - b) >= 2:
        raise ValueError("This score already finished the game; record it as completed rather than an injury retirement.")
    if official and status in PLAYED and not game.get("played_at"):
        raise ValueError("Record the actual play date and time for each played game.")
    if official and status in PLAYED:
        cutoff = _cutoff(pairing)
        if cutoff and datetime.fromisoformat(game["played_at"].replace("Z", "+00:00")) < cutoff:
            raise ValueError("A played game cannot precede its roster eligibility deadline.")
        players_a, players_b = _lineup(pairing, game, "a"), _lineup(pairing, game, "b")
        _distinct_players(players_a, 2, "Club A's actual doubles lineup")
        _distinct_players(players_b, 2, "Club B's actual doubles lineup")
        if set(players_a) & set(players_b):
            raise ValueError("A player cannot represent both clubs in a game.")


def _pairing_result(pairing: dict) -> dict:
    games_a = games_b = losses_a = losses_b = pd = 0
    played_a: set[str] = set()
    played_b: set[str] = set()
    def ready(game: dict) -> bool:
        if game["status"] == "completed":
            return game.get("a") is not None and game.get("b") is not None
        if game["status"] == "retired":
            return game.get("a") is not None and game.get("b") is not None and bool(game.get("winner"))
        if game["status"] == "forfeit":
            return bool(game.get("winner"))
        return game["status"] in {"unplayed", "double_forfeit"}

    complete = all(ready(g) for g in pairing["games"])
    for game in pairing["games"]:
        if not ready(game):
            continue
        if game["status"] in DECIDED:
            winner = _game_winner(game)
            games_a += int(winner == "a")
            games_b += int(winner == "b")
            losses_a += int(winner == "b" or game["status"] == "double_forfeit")
            losses_b += int(winner == "a" or game["status"] == "double_forfeit")
        if game["status"] in PLAYED:
            pd += (game.get("a") or 0) - (game.get("b") or 0)
            played_a.update(_lineup(pairing, game, "a"))
            played_b.update(_lineup(pairing, game, "b"))
    winner = ("a" if games_a > games_b else "b" if games_b > games_a else "draw") if complete else None
    if complete and all(game["status"] == "double_forfeit" for game in pairing["games"]):
        winner = "double_forfeit"
    return {"id": pairing["id"], "kind": pairing["kind"], "winner": winner,
            "complete": complete, "games_a": games_a, "games_b": games_b,
            "losses_a": losses_a, "losses_b": losses_b,
            "point_differential": pd, "played_a": sorted(played_a), "played_b": sorted(played_b)}


def matchup_points(pairing_winners: Iterable[str]) -> tuple[int, int]:
    """The agreed W+D exception intentionally awards 3/1, not 3/0."""
    winners = list(pairing_winners)
    if len(winners) != 2 or any(w not in {"a", "b", "draw", "double_forfeit"} for w in winners):
        raise ValueError("A regular matchup needs two decided pairing results.")
    wins_a, wins_b = winners.count("a"), winners.count("b")
    if wins_a == wins_b:
        return (0, 0) if winners == ["double_forfeit", "double_forfeit"] else (1, 1)
    if wins_a > wins_b:
        return 3, int("draw" in winners)
    return int("draw" in winners), 3


def _encounter_result(encounter: dict, phase: str) -> dict:
    pairings = [_pairing_result(pairing) for pairing in encounter["pairings"]]
    complete = all(p["complete"] for p in pairings)
    wins_a = sum(p["winner"] == "a" for p in pairings)
    wins_b = sum(p["winner"] == "b" for p in pairings)
    points_a = points_b = 0
    winner = None
    if complete and phase == "regular":
        points_a, points_b = matchup_points(p["winner"] for p in pairings)
        winner = "a" if wins_a > wins_b else "b" if wins_b > wins_a else "draw"
        if points_a == points_b == 0:
            winner = "double_forfeit"
    elif complete:
        if wins_a != wins_b:
            winner = "a" if wins_a > wins_b else "b"
        else:
            tie = encounter.get("tiebreak")
            if tie and tie.get("status") == "completed":
                winner = _score_winner(tie.get("a"), tie.get("b"), 21, "the rotating-singles tiebreak")
            else:
                complete = False
    return {"id": encounter["id"], "division": encounter["division"],
            "club_a": encounter["club_a"], "club_b": encounter["club_b"],
            "winner": winner, "complete": complete, "points_a": points_a, "points_b": points_b,
            "pairings_a": wins_a, "pairings_b": wins_b,
            "games_a": sum(p["games_a"] for p in pairings),
            "games_b": sum(p["games_b"] for p in pairings),
            "losses_a": sum(p["losses_a"] for p in pairings),
            "losses_b": sum(p["losses_b"] for p in pairings),
            "point_differential": sum(p["point_differential"] for p in pairings),
            "played_a": sorted({x for p in pairings for x in p["played_a"]}),
            "played_b": sorted({x for p in pairings for x in p["played_b"]}), "pairings": pairings}


def validate_document(document: dict | CompetitionDocument, official: bool = False) -> dict:
    doc = _normalize(document)
    if (doc["phase"] == "regular") == (doc["format"] == "mlp"):
        raise ValueError("Regular meets use gender or mixed doubles; finals and qualifiers use MLP.")
    if doc["phase"] != "regular" and doc["schedule_mode"] == "staggered":
        raise ValueError("Staggered scheduling is available for regular-season meets.")
    if official and not doc["encounters"]:
        raise ValueError("Generate at least one matchup before submitting a meet.")
    ids: set[str] = set()
    pairs: set[tuple] = set()
    occupied: set[tuple] = set()
    club_rounds: set[tuple] = set()
    played_in_round: dict[tuple, set[str]] = defaultdict(set)
    current_lineups: dict[tuple, set[str]] = {}
    removed_players: dict[tuple, set[str]] = defaultdict(set)
    for encounter in sorted(doc["encounters"], key=lambda e: (e["rotation"], e["id"])):
        if encounter["club_a"] == encounter["club_b"]:
            raise ValueError("A club cannot play itself.")
        encounter_key = (encounter["division"], *sorted([encounter["club_a"], encounter["club_b"]]))
        if encounter_key in pairs:
            raise ValueError("A round robin has one matchup per pair of clubs at each skill level.")
        pairs.add(encounter_key)
        expected = KINDS[doc["format"]]
        actual = [p["kind"] for p in encounter["pairings"]]
        if len(actual) != len(expected) or set(actual) != set(expected):
            raise ValueError(f"Each {doc['format']} matchup requires these pairings: {', '.join(expected)}.")
        encounter["pairings"].sort(key=lambda p: expected.index(p["kind"]))
        for side in ("a", "b"):
            club_round = (encounter["rotation"], encounter["division"], encounter[f"club_{side}"])
            if club_round in club_rounds:
                raise ValueError("A club cannot play two opponents in the same rotation and skill level.")
            club_rounds.add(club_round)
            base = [p[f"players_{side}"] for p in encounter["pairings"]]
            if all(len(players) == 2 for players in base):
                # Historical completed and replay pairings can use different
                # eligible teams on different dates. Their snapshots remain
                # separate rather than rewriting the already played lineup.
                same_cutoff = _cutoff(encounter["pairings"][0]) == _cutoff(encounter["pairings"][1])
                if same_cutoff and len(set(base[0] + base[1])) != 4:
                    raise ValueError("The two doubles pairings must use four different players.")
                if doc["format"] == "mlp" and (len(set(base[2] + base[3])) != 4 or set(base[0] + base[1]) != set(base[2] + base[3])):
                    raise ValueError("The two mixed doubles games must use the same four team players, once each.")
        for obj in [encounter, *encounter["pairings"], *(g for p in encounter["pairings"] for g in p["games"])]:
            if obj["id"] in ids:
                raise ValueError("Encounter, pairing, and game IDs must be unique within the meet.")
            ids.add(obj["id"])
        for pairing in encounter["pairings"]:
            if len(pairing["games"]) != (3 if doc["phase"] == "regular" else 1):
                raise ValueError("Regular pairings have all three games; MLP pairings have one game.")
            if doc["phase"] == "regular" and pairing.get("court") is not None:
                court_key = (encounter["rotation"], pairing["court"])
                if court_key in occupied:
                    raise ValueError("Two pairings cannot use the same court in the same rotation.")
                occupied.add(court_key)
            for side in ("a", "b"):
                players = pairing[f"players_{side}"]
                if players:
                    _distinct_players(players, 2, "The scheduled doubles lineup")
            for game in pairing["games"]:
                _validate_game(game, pairing, doc, official=official)
                if not official or game["status"] not in PLAYED:
                    continue
                for side in ("a", "b"):
                    club = encounter[f"club_{side}"]
                    actual_players = set(_lineup(pairing, game, side))
                    base_players = set(pairing[f"players_{side}"])
                    key = (club, encounter["division"], pairing["kind"], _cutoff(pairing))
                    previous = current_lineups.get(key, base_players)
                    if actual_players != previous:
                        if not (game.get("injury_reason") or "").strip():
                            raise ValueError("Record an injury reason for every change to the playing lineup.")
                        injury_key = (club, encounter["division"], _cutoff(pairing))
                        if actual_players & removed_players[injury_key]:
                            raise ValueError("An injured player who was replaced cannot return later in this meet.")
                        removed_players[injury_key].update(previous - actual_players)
                    current_lineups[key] = actual_players
                    round_key = (encounter["rotation"], club, encounter["division"], _cutoff(pairing), pairing["kind"])
                    for other_key, other_players in played_in_round.items():
                        if other_key[:4] == round_key[:4] and other_key[4] != pairing["kind"] and doc["phase"] == "regular" and actual_players & other_players:
                            raise ValueError("A player cannot play both regular pairings in the same club matchup.")
                    played_in_round[round_key].update(actual_players)
        tie = encounter.get("tiebreak")
        if doc["phase"] == "regular" and tie is not None:
            raise ValueError("Regular-season matchups do not use a singles tiebreak.")
        result = _encounter_result(encounter, doc["phase"]) if official else None
        if tie and tie["status"] == "completed":
            _score_winner(tie.get("a"), tie.get("b"), 21, "the rotating-singles tiebreak")
            for side in ("a", "b"):
                _distinct_players(tie[f"order_{side}"], 4, "The singles rotation order")
                active = {x for p in encounter["pairings"][2:] for x in _lineup(p, p["games"][-1], side)}
                if set(tie[f"order_{side}"]) != active:
                    raise ValueError("The singles rotation must use all four active team players in a fixed order.")
            if official and (result["pairings_a"] != 2 or result["pairings_b"] != 2):
                raise ValueError("Play rotating singles only when the four doubles games finish 2–2.")
        if official and not result["complete"]:
            raise ValueError("A 2–2 MLP matchup needs its completed rotating-singles tiebreak.")
    return doc


def _entry_lineups(entry: dict, *, format: str = "gender", allow_partial: bool = False) -> dict[str, list[str]]:
    roster = entry.get("roster", entry.get("players", []))
    women, men = [], []
    for player in roster:
        gender = str(player.get("gender") or "").strip().lower()
        entry_id = str(player.get("entry_id") or "")
        if gender in {"f", "female", "woman", "women"}:
            women.append(entry_id)
        elif gender in {"m", "male", "man", "men"}:
            men.append(entry_id)
        else:
            raise ValueError("Confirm each roster player's gender before generating two-women/two-men teams.")
    if allow_partial and len(women + men) == 2:
        _distinct_players(women + men, 2, "The declared available doubles pairing")
        if format == "gender" and len(women) in {0, 2}:
            return {"women": women, "men": men, "mixed_a": [], "mixed_b": []}
        if format == "mixed" and len(women) == len(men) == 1:
            return {"women": [], "men": [], "mixed_a": [women[0], men[0]], "mixed_b": []}
        raise ValueError("A partial team needs two women or two men for gender doubles, or one woman and one man for mixed doubles.")
    _distinct_players(women + men, 4, "Each entered team")
    if len(women) != 2 or len(men) != 2:
        raise ValueError("Each entered team must contain two women and two men.")
    return {"women": women, "men": men, "mixed_a": [women[0], men[0]], "mixed_b": [women[1], men[1]]}


def _new_encounter(meet_id: str, division: str, a: dict, b: dict, format: str, phase: str, rotation: int, played_at: str | None, court_start: int) -> dict:
    club_a, club_b = a["club_id"], b["club_id"]
    encounter_id = _id(meet_id, phase, division, *sorted([club_a, club_b]))
    line_a = _entry_lineups(a, format=format, allow_partial=phase == "regular")
    line_b = _entry_lineups(b, format=format, allow_partial=phase == "regular")
    pairings = []
    next_court = court_start
    for index, kind in enumerate(KINDS[format]):
        pairing_id = _id(encounter_id, kind)
        if not line_a[kind] and not line_b[kind]:
            status, winner = "double_forfeit", None
        elif not line_a[kind] or not line_b[kind]:
            status, winner = "forfeit", "a" if line_a[kind] else "b"
        else:
            status, winner = "pending", None
        court = (next_court if phase == "regular" else court_start) if status == "pending" else None
        if court is not None:
            next_court += 1
        pairings.append({"id": pairing_id, "kind": kind,
                         "court": court,
                         "players_a": line_a[kind], "players_b": line_b[kind],
                         "games": [{"id": _id(pairing_id, game_index), "status": status, "winner": winner, "played_at": played_at}
                                   for game_index in range(3 if phase == "regular" else 1)]})
    return {"id": encounter_id, "division": division, "club_a": club_a, "club_b": club_b,
            "rotation": rotation, "pairings": pairings, "tiebreak": None}


def assign_regular_courts(encounters: list[dict], courts: int | None, schedule_mode: str = "simultaneous") -> None:
    """Keep each opponent round in order, splitting it into court-sized waves.

    The two playable doubles pairings in a matchup start together. Forfeited
    pairings take no court. IDs and lineups are independent of court allocation.
    """
    if schedule_mode not in {"simultaneous", "staggered"}:
        raise ValueError("Choose simultaneous or staggered scheduling.")
    if courts is not None and (isinstance(courts, bool) or not isinstance(courts, int) or not 1 <= courts <= 100):
        raise ValueError("Choose between 1 and 100 available courts.")
    if schedule_mode == "staggered" and courts is None:
        raise ValueError("Set the available court count for a staggered schedule.")
    ordered = sorted(encounters, key=lambda e: (e["rotation"], e["division"], e["club_a"], e["club_b"]))
    next_rotation = 1
    for _, group in groupby(ordered, key=lambda e: e["rotation"]):
        # Materialize before changing rotation, which is also groupby's key.
        round_encounters = list(group)
        waves: list[dict] = []
        needed = sum(bool(p["players_a"] and p["players_b"]) for e in round_encounters for p in e["pairings"])
        if schedule_mode == "simultaneous" and courts is not None and needed > courts:
            raise ValueError(f"This draw needs {needed} courts per rotation; {courts} are available. Choose Staggered under Court schedule, or increase the meet's court count.")
        for encounter in round_encounters:
            playable = [p for p in encounter["pairings"] if p["players_a"] and p["players_b"]]
            players = {player for p in playable for side in ("a", "b") for player in p[f"players_{side}"]}
            if courts is not None and len(playable) > courts:
                raise ValueError(f"This matchup needs {len(playable)} courts so its doubles pairings can play together. Increase the meet's court count to at least {len(playable)}.")
            wave = next((w for w in waves if (courts is None or w["used"] + len(playable) <= courts) and not w["players"] & players), None)
            if wave is None:
                if schedule_mode == "simultaneous" and waves:
                    raise ValueError("A player cannot be assigned to two simultaneous matchups.")
                wave = {"rotation": next_rotation + len(waves), "used": 0, "players": set()}
                waves.append(wave)
            encounter["rotation"] = wave["rotation"]
            for pairing in encounter["pairings"]:
                pairing["court"] = None
                if pairing["players_a"] and pairing["players_b"]:
                    wave["used"] += 1
                    pairing["court"] = wave["used"]
            wave["players"].update(players)
        next_rotation += len(waves)


def generate_round_robin(meet_id: str, entries: list[dict], format: str = "gender", played_at: str | None = None, courts: int | None = None, schedule_mode: str = "simultaneous") -> dict:
    if format not in {"gender", "mixed"}:
        raise ValueError("Choose gender or mixed doubles for a regular meet.")
    divisions: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        if not entry.get("division") or not entry.get("club_id"):
            raise ValueError("Every team entry needs its skill level and club.")
        _entry_lineups(entry, format=format, allow_partial=True)
        divisions[str(entry["division"])].append(entry)
    encounters = []
    court_counts: dict[int, int] = defaultdict(int)
    for division, teams in sorted(divisions.items()):
        if len(teams) < 2 or len(teams) > 4:
            raise ValueError(f"{division} needs two to four entered clubs to schedule a round robin.")
        if len({t["club_id"] for t in teams}) != len(teams):
            raise ValueError("A club can enter only one four-player team per skill level in a meet.")
        # Circle rotation includes one bye for a three-club field.
        circle = sorted(teams, key=lambda t: t["club_id"])
        if len(circle) % 2:
            circle.append(None)
        for rotation in range(1, len(circle)):
            for index in range(len(circle) // 2):
                a, b = circle[index], circle[-index - 1]
                if a is None or b is None:
                    continue
                court_start = court_counts[rotation] + 1
                encounter = _new_encounter(meet_id, division, a, b, format, "regular", rotation, played_at, court_start)
                court_counts[rotation] += sum(pairing["court"] is not None for pairing in encounter["pairings"])
                encounters.append(encounter)
            circle = [circle[0], circle[-1], *circle[1:-1]]
    assign_regular_courts(encounters, courts, schedule_mode)
    return validate_document({"meet_id": str(meet_id), "phase": "regular", "format": format, "schedule_mode": schedule_mode, "encounters": encounters})


def generate_championship(meet_id: str, division: str, club_a: dict, club_b: dict, phase: str = "final", played_at: str | None = None) -> dict:
    if phase not in {"final", "qualifier"}:
        raise ValueError("Choose a championship final or qualification playoff.")
    return validate_document({"meet_id": str(meet_id), "phase": phase, "format": "mlp",
                              "encounters": [_new_encounter(meet_id, division, club_a, club_b, "mlp", phase, 1, played_at, 1)]})


def prepare_reschedule(document: dict, *, played_at: str | None, eligibility_deadline: str) -> dict:
    doc = validate_document(document)
    if doc["phase"] != "regular":
        raise ValueError("Use the organizer's championship plan to reschedule an MLP matchup.")
    doc["weather"] = "rescheduled"
    for encounter in doc["encounters"]:
        for pairing in encounter["pairings"]:
            if all(g["status"] in DECIDED for g in pairing["games"]):
                continue
            pairing["eligibility_deadline"] = eligibility_deadline
            for game in pairing["games"]:
                # Same logical game ID; the document revision keeps superseded scores.
                game.update(status="pending", a=None, b=None, winner=None,
                            players_a=[], players_b=[], injury_reason=None, played_at=played_at)
    return validate_document(doc)


def summarize_document(document: dict) -> dict:
    doc = validate_document(document)
    results = [_encounter_result(e, doc["phase"]) for e in doc["encounters"]]
    statuses: dict[str, int] = {status: 0 for status in ("pending", "completed", "retired", "forfeit", "double_forfeit", "unplayed")}
    for encounter in doc["encounters"]:
        for pairing in encounter["pairings"]:
            for game in pairing["games"]:
                statuses[game["status"]] += 1
    return {"meet_id": doc["meet_id"], "phase": doc["phase"], "format": doc["format"],
            "complete": bool(results) and all(r["complete"] for r in results),
            "encounters": results, "games": statuses,
            "rating_game_count": statuses["completed"]}


def rating_games(document: dict) -> list[dict]:
    doc = validate_document(document, official=True)
    games = []
    scheduled = sorted((encounter["rotation"], game_index, pairing_index, encounter["id"], game["id"])
                       for encounter in doc["encounters"]
                       for pairing_index, pairing in enumerate(encounter["pairings"])
                       for game_index, game in enumerate(pairing["games"]))
    sequence = {scheduled_game[-1]: index for index, scheduled_game in enumerate(scheduled)}
    for encounter in doc["encounters"]:
        for pairing in encounter["pairings"]:
            for game in pairing["games"]:
                if game["status"] != "completed":
                    continue
                games.append({"id": game["id"], "encounter_id": encounter["id"], "pairing_id": pairing["id"],
                              "meet_id": doc["meet_id"], "phase": doc["phase"], "division": encounter["division"],
                              "club_a": encounter["club_a"], "club_b": encounter["club_b"],
                              "players_a": _lineup(pairing, game, "a"), "players_b": _lineup(pairing, game, "b"),
                              "a": game["a"], "b": game["b"], "played_at": game["played_at"], "sequence": sequence[game["id"]]})
    return sorted(games, key=lambda g: (datetime.fromisoformat(g["played_at"].replace("Z", "+00:00")), g["sequence"], g["id"]))


def _official_documents(documents: Iterable[dict]) -> list[dict]:
    docs = [validate_document(document, official=True) for document in documents]
    seen = set()
    for doc in docs:
        key = (doc["meet_id"], doc["phase"])
        if key in seen:
            raise ValueError("Supply only the latest approved revision of each meet and phase.")
        seen.add(key)
    return docs


def _empty_row(club_id: str) -> dict:
    return {"club_id": club_id, "points": 0, "pairings_won": 0, "games_won": 0, "games_lost": 0,
            "point_differential": 0, "head_to_head_points": 0, "meets_played": 0,
            "played_meet_ids": [], "position": 0, "tied": False}


def _club_catalog(clubs: Iterable[str | dict] | None) -> dict[str, str]:
    """Public/admin callers pass either IDs or safe club summary objects."""
    result = {}
    for club in clubs or []:
        club_id = str(club.get("id") or "").strip() if isinstance(club, dict) else str(club).strip()
        if not club_id:
            raise ValueError("Each club summary must include its club ID.")
        result[club_id] = str(club.get("name") or club_id) if isinstance(club, dict) else club_id
    return result


def _base_key(row: dict) -> tuple:
    return tuple(row[key] for key in ("points", "pairings_won", "games_won", "point_differential"))


def _rank_rows(rows: list[dict], encounters: list[dict]) -> list[dict]:
    rows.sort(key=lambda r: tuple(-n for n in _base_key(r)) + (r["club_id"],))
    ranked = []
    for _, grouped in groupby(rows, key=_base_key):
        group = list(grouped)
        clubs = {r["club_id"] for r in group}
        h2h: dict[str, int] = defaultdict(int)
        if len(group) > 1:
            for e in encounters:
                if e["club_a"] in clubs and e["club_b"] in clubs:
                    h2h[e["club_a"]] += e["points_a"]
                    h2h[e["club_b"]] += e["points_b"]
        for row in group:
            row["head_to_head_points"] = h2h[row["club_id"]]
        group.sort(key=lambda r: (-r["head_to_head_points"], r["club_id"]))
        for _, tied_rows in groupby(group, key=lambda r: r["head_to_head_points"]):
            tied_group = list(tied_rows)
            position = len(ranked) + 1
            for row in tied_group:
                row["position"] = position
                row["tied"] = len(tied_group) > 1
            ranked.extend(tied_group)
    return ranked


def _add_result(row: dict, result: dict, side: str, meet_id: str, *, points: int | None = None) -> None:
    row["points"] += result[f"points_{side}"] if points is None else points
    row["pairings_won"] += result[f"pairings_{side}"]
    row["games_won"] += result[f"games_{side}"]
    row["games_lost"] += result[f"losses_{side}"]
    row["point_differential"] += result["point_differential"] * (1 if side == "a" else -1)
    if result[f"played_{side}"] and meet_id not in row["played_meet_ids"]:
        row["played_meet_ids"].append(meet_id)
        row["played_meet_ids"].sort()
        row["meets_played"] = len(row["played_meet_ids"])


def _qualification(rows: list[dict], division: str, qualifier_results: list[dict]) -> dict:
    eligible = [row for row in rows if row["meets_played"] >= 1]
    result = {"qualifiers": [], "playoff_required": [], "eligible": [r["club_id"] for r in eligible], "status": "ready"}
    if len(eligible) < 2:
        result.update(status="insufficient_entries", qualifiers=[r["club_id"] for r in eligible])
        return result
    slots = 2
    # Re-evaluate the cutoff among eligible clubs, preserving all sporting ties.
    for _, grouped in groupby(eligible, key=lambda r: (_base_key(r), r["head_to_head_points"])):
        group = list(grouped)
        if len(group) <= slots:
            result["qualifiers"].extend(r["club_id"] for r in group)
            slots -= len(group)
            if not slots:
                break
            continue
        tied = {r["club_id"] for r in group}
        playoffs = [e for e in qualifier_results if e["division"] == division and {e["club_a"], e["club_b"]} <= tied and e["winner"] in {"a", "b"}]
        resolved, unresolved = _resolve_qualifying_playoffs(tied, slots, playoffs)
        result["qualifiers"].extend(resolved)
        if unresolved:
            result.update(status="playoff_required", playoff_required=unresolved)
        break
    return result


def _resolve_qualifying_playoffs(clubs: set[str], slots: int, results: list[dict]) -> tuple[list[str], list[str]]:
    """Complete MLP round robins settle multiway cutoff ties, without a bracket.

    One match for every club pair is necessary before that round can rank the
    field. If only a smaller cutoff group remains tied, subsequent matches among
    those clubs form the next playoff round. Prior rounds do not get silently
    reused to break the same tie again.
    """
    pending = defaultdict(list)
    for result in results:
        pending[frozenset((result["club_a"], result["club_b"]))].append(result)
    for pair_results in pending.values():
        pair_results.sort(key=lambda e: (e.get("_played_at") or datetime.min.replace(tzinfo=timezone.utc), e["id"]))
        if any(first.get("_played_at") == second.get("_played_at") for first, second in zip(pair_results, pair_results[1:])):
            # Correct the actual play times rather than choosing which playoff
            # round happened first based on arbitrary generated UUIDs.
            return [], sorted(clubs)
    pair_keys = [frozenset(pair) for pair in combinations(sorted(clubs), 2)]
    if any(not pending[key] for key in pair_keys):
        return [], sorted(clubs)
    round_results = [pending[key].pop(0) for key in pair_keys]
    stats = {club: {"club_id": club, "wins": 0, "games_won": 0, "point_differential": 0} for club in clubs}
    for result in round_results:
        stats[result[f"club_{result['winner']}"]]["wins"] += 1
        for side in ("a", "b"):
            row = stats[result[f"club_{side}"]]
            row["games_won"] += result[f"games_{side}"]
            row["point_differential"] += result["point_differential"] * (1 if side == "a" else -1)
    def key(row: dict) -> tuple:
        return row["wins"], row["games_won"], row["point_differential"]

    ranked = sorted(stats.values(), key=lambda row: tuple(-n for n in key(row)) + (row["club_id"],))
    qualifiers = []
    for _, grouped in groupby(ranked, key=key):
        group = list(grouped)
        if len(group) <= slots:
            qualifiers.extend(row["club_id"] for row in group)
            slots -= len(group)
            if not slots:
                return qualifiers, []
            continue
        tied = {row["club_id"] for row in group}
        remaining = [result for pair, pair_results in pending.items() if pair <= tied for result in pair_results]
        if remaining:
            next_qualifiers, unresolved = _resolve_qualifying_playoffs(tied, slots, remaining)
            return qualifiers + next_qualifiers, unresolved
        return qualifiers, sorted(tied)
    return qualifiers, []


def league_standings(documents: Iterable[dict], clubs: Iterable[str | dict] | None = None) -> dict:
    docs = _official_documents(documents)
    catalog = _club_catalog(clubs)
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    encounters: dict[str, list[dict]] = defaultdict(list)
    qualifier_results = []
    for doc in docs:
        if doc["phase"] == "qualifier":
            for encounter in doc["encounters"]:
                result = _encounter_result(encounter, "qualifier")
                times = [datetime.fromisoformat(game["played_at"].replace("Z", "+00:00"))
                         for pairing in encounter["pairings"] for game in pairing["games"]
                         if game.get("played_at")]
                result["_played_at"] = max(times) if times else None
                qualifier_results.append(result)
        if doc["phase"] != "regular":
            continue
        for encounter in doc["encounters"]:
            result = _encounter_result(encounter, "regular")
            division = result["division"]
            encounters[division].append(result)
            for side in ("a", "b"):
                club = result[f"club_{side}"]
                row = rows[division].setdefault(club, _empty_row(club))
                _add_result(row, result, side, doc["meet_id"])
    for division_rows in rows.values():
        for club_id in catalog:
            division_rows.setdefault(club_id, _empty_row(club_id))
        for row in division_rows.values():
            row["name"] = catalog.get(row["club_id"], row["club_id"])
    ranked = {division: _rank_rows(list(division_rows.values()), encounters[division]) for division, division_rows in sorted(rows.items())}
    qualification = {division: _qualification(division_rows, division, qualifier_results) for division, division_rows in ranked.items()}
    return {"divisions": ranked, "qualification": qualification}


def qualifying_clubs(documents: Iterable[dict], clubs: Iterable[str | dict] | None = None) -> dict:
    return league_standings(documents, clubs)["qualification"]


def club_cup(documents: Iterable[dict], clubs: Iterable[str | dict] | None = None) -> dict:
    docs = _official_documents(documents)
    catalog = _club_catalog(clubs)
    regular = league_standings(docs, catalog)
    rows: dict[str, dict] = {}
    h2h = []
    final_divisions = set()
    for doc in docs:
        if doc["phase"] == "qualifier":
            continue
        for encounter in doc["encounters"]:
            result = _encounter_result(encounter, doc["phase"])
            if doc["phase"] == "final":
                if result["division"] in final_divisions:
                    raise ValueError("A skill level can have only one official championship final.")
                final_divisions.add(result["division"])
                result["points_a"] = 6 if result["winner"] == "a" else 3
                result["points_b"] = 6 if result["winner"] == "b" else 3
            h2h.append(result)
            for side in ("a", "b"):
                club = result[f"club_{side}"]
                if club not in rows:
                    rows[club] = {**_empty_row(club), "regular_points": 0, "championship_points": 0}
                row = rows[club]
                _add_result(row, result, side, doc["meet_id"])
                row["regular_points" if doc["phase"] == "regular" else "championship_points"] += result[f"points_{side}"]
    for club_id in catalog:
        rows.setdefault(club_id, {**_empty_row(club_id), "regular_points": 0, "championship_points": 0})
    for row in rows.values():
        row["name"] = catalog.get(row["club_id"], row["club_id"])
    ranked = _rank_rows(list(rows.values()), h2h)
    required = {division for division, qualification in regular["qualification"].items() if len(qualification["eligible"]) >= 2}
    complete = bool(required) and required <= final_divisions and all(q["status"] != "playoff_required" for q in regular["qualification"].values())
    return {"standings": ranked, "champions": [r["club_id"] for r in ranked if r["position"] == 1] if complete else [],
            "status": "complete" if complete else "provisional"}
