from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

TEAM_SLOTS = ("MAN_1", "MAN_2", "WOMAN_1", "WOMAN_2")
MIXED_PAIRINGS = {"STRAIGHT", "CROSS"}
TIEBREAK_MODES = {"SINGLES", "SKINNY_RELAY"}
PLAYOFF_FORMATS = {
    "NONE",
    "TOP_2_FINAL",
    "TOP_4_SEMIFINALS",
    "TOP_4_SEMIFINALS_WITH_BRONZE",
}
REGULATION_GAME_CODES = ("WOMENS", "MENS", "MIXED_1", "MIXED_2")
ACTIVE_TEAM_STATES = {"CONFIRMED"}
ACTIVE_MEMBER_STATES = {"ACCEPTED"}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _player_id(row: dict[str, Any]) -> int:
    try:
        return int(row.get("player_id"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Each accepted team member must have a linked player.") from exc


def validate_four_player_roster(members: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate the exact two-men/two-women accepted-player contract."""

    roster = [dict(row) for row in members]
    if len(roster) != 4:
        raise ValueError("A four-player team requires exactly four players.")
    by_slot: dict[str, dict[str, Any]] = {}
    player_ids: set[int] = set()
    emails: set[str] = set()
    for row in roster:
        slot = _text(row.get("slot")).upper()
        if slot not in TEAM_SLOTS or slot in by_slot:
            raise ValueError("A team must fill each of the two men and two women slots once.")
        status = _text(row.get("status") or "ACCEPTED").upper()
        if status not in ACTIVE_MEMBER_STATES:
            raise ValueError("All four team invitations must be accepted.")
        player_id = _player_id(row)
        if player_id in player_ids:
            raise ValueError("A player may occupy only one team roster slot.")
        email = _text(row.get("invited_email") or row.get("email")).lower()
        if email and email in emails:
            raise ValueError("A roster email may be used only once.")
        player_ids.add(player_id)
        if email:
            emails.add(email)
        normalized = {**row, "slot": slot, "status": status, "player_id": player_id}
        by_slot[slot] = normalized
    missing = [slot for slot in TEAM_SLOTS if slot not in by_slot]
    if missing:
        raise ValueError("A team must include two men and two women.")
    return [by_slot[slot] for slot in TEAM_SLOTS]


def build_locked_team_games(
    members: Iterable[dict[str, Any]],
    *,
    mixed_pairing: str,
    singles_tiebreak_player_id: int | None = None,
    tiebreak_mode: str = "SINGLES",
) -> list[dict[str, Any]]:
    """Create the four regulation child games for a locked lineup."""

    roster = validate_four_player_roster(members)
    pairing = _text(mixed_pairing).upper()
    if pairing not in MIXED_PAIRINGS:
        raise ValueError("Mixed pairing must be Straight or Cross.")
    mode = _text(tiebreak_mode).upper()
    if mode not in TIEBREAK_MODES:
        raise ValueError("Team tiebreak mode must be Singles or Skinny Relay.")
    players = {row["slot"]: row["player_id"] for row in roster}
    if mode == "SINGLES" and singles_tiebreak_player_id not in set(players.values()):
        raise ValueError("The singles tiebreak player must be on the locked roster.")
    if pairing == "STRAIGHT":
        mixed_1 = [players["WOMAN_1"], players["MAN_1"]]
        mixed_2 = [players["WOMAN_2"], players["MAN_2"]]
    else:
        mixed_1 = [players["WOMAN_1"], players["MAN_2"]]
        mixed_2 = [players["WOMAN_2"], players["MAN_1"]]
    return [
        {
            "game_code": "WOMENS",
            "game_order": 1,
            "match_format": "DOUBLES",
            "player_ids": [players["WOMAN_1"], players["WOMAN_2"]],
            "counts_for_rating": True,
        },
        {
            "game_code": "MENS",
            "game_order": 2,
            "match_format": "DOUBLES",
            "player_ids": [players["MAN_1"], players["MAN_2"]],
            "counts_for_rating": True,
        },
        {
            "game_code": "MIXED_1",
            "game_order": 3,
            "match_format": "DOUBLES",
            "player_ids": mixed_1,
            "counts_for_rating": True,
        },
        {
            "game_code": "MIXED_2",
            "game_order": 4,
            "match_format": "DOUBLES",
            "player_ids": mixed_2,
            "counts_for_rating": True,
        },
    ]


def build_team_tiebreak_game(
    members: Iterable[dict[str, Any]],
    *,
    tiebreak_mode: str,
    singles_tiebreak_player_id: int | None = None,
) -> dict[str, Any]:
    roster = validate_four_player_roster(members)
    mode = _text(tiebreak_mode).upper()
    if mode not in TIEBREAK_MODES:
        raise ValueError("Team tiebreak mode must be Singles or Skinny Relay.")
    player_ids = [row["player_id"] for row in roster]
    if mode == "SINGLES":
        if singles_tiebreak_player_id not in set(player_ids):
            raise ValueError("The singles tiebreak player must be on the locked roster.")
        players = [int(singles_tiebreak_player_id)]
    else:
        players = player_ids
    return {
        "game_code": "TIEBREAK",
        "game_order": 5,
        "match_format": mode,
        "player_ids": players,
        "counts_for_rating": mode == "SINGLES",
    }


def evaluate_team_match(
    games: list[dict[str, Any]],
    *,
    tiebreak_mode: str,
) -> dict[str, Any]:
    mode = _text(tiebreak_mode).upper()
    if mode not in TIEBREAK_MODES:
        raise ValueError("Team tiebreak mode must be Singles or Skinny Relay.")
    by_code = {_text(row.get("game_code")).upper(): row for row in games}
    wins_a = wins_b = completed = 0
    for code in REGULATION_GAME_CODES:
        winner = _text((by_code.get(code) or {}).get("winner_side")).upper()
        if winner not in {"A", "B"}:
            continue
        completed += 1
        wins_a += int(winner == "A")
        wins_b += int(winner == "B")
    if completed < 4:
        return {
            "status": "IN_PROGRESS",
            "team_a_wins": wins_a,
            "team_b_wins": wins_b,
            "winner_side": None,
            "tiebreak_required": False,
        }
    if wins_a != wins_b:
        return {
            "status": "FINAL",
            "team_a_wins": wins_a,
            "team_b_wins": wins_b,
            "winner_side": "A" if wins_a > wins_b else "B",
            "tiebreak_required": False,
        }
    winner = _text((by_code.get("TIEBREAK") or {}).get("winner_side")).upper()
    if winner not in {"A", "B"}:
        return {
            "status": "TIEBREAK_REQUIRED",
            "team_a_wins": wins_a,
            "team_b_wins": wins_b,
            "winner_side": None,
            "tiebreak_required": True,
            "tiebreak_mode": mode,
            "tiebreak_counts_for_rating": mode == "SINGLES",
        }
    return {
        "status": "FINAL",
        "team_a_wins": wins_a + int(winner == "A"),
        "team_b_wins": wins_b + int(winner == "B"),
        "winner_side": winner,
        "tiebreak_required": True,
        "tiebreak_mode": mode,
        "tiebreak_counts_for_rating": mode == "SINGLES",
    }


def build_team_standings(
    teams: Iterable[dict[str, Any]],
    matchups: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for team in teams:
        team_id = _text(team.get("id") or team.get("team_id"))
        if not team_id or _text(team.get("status") or "CONFIRMED").upper() not in ACTIVE_TEAM_STATES:
            continue
        rows[team_id] = {
            "team_id": team_id,
            "team_name": _text(team.get("name") or team.get("team_name")) or "Team",
            "match_wins": 0,
            "match_losses": 0,
            "game_wins": 0,
            "game_losses": 0,
            "game_differential": 0,
            "head_to_head": {},
        }
    for matchup in matchups:
        if (
            _text(matchup.get("stage") or "ROUND_ROBIN").upper() != "ROUND_ROBIN"
            or _text(matchup.get("status")).upper() != "FINAL"
        ):
            continue
        team_a = _text(matchup.get("team_a_id"))
        team_b = _text(matchup.get("team_b_id"))
        winner = _text(matchup.get("winner_team_id"))
        loser = _text(matchup.get("loser_team_id"))
        if team_a not in rows or team_b not in rows or winner not in rows or loser not in rows:
            continue
        a_wins = int(matchup.get("team_a_game_wins") or 0)
        b_wins = int(matchup.get("team_b_game_wins") or 0)
        rows[team_a]["game_wins"] += a_wins
        rows[team_a]["game_losses"] += b_wins
        rows[team_b]["game_wins"] += b_wins
        rows[team_b]["game_losses"] += a_wins
        rows[winner]["match_wins"] += 1
        rows[loser]["match_losses"] += 1
        rows[winner]["head_to_head"][loser] = rows[winner]["head_to_head"].get(loser, 0) + 1
    for row in rows.values():
        row["game_differential"] = row["game_wins"] - row["game_losses"]
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows.values():
        grouped[(row["match_wins"], row["match_losses"])].append(row)
    ordered: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (-item[0], item[1])):
        tied = grouped[key]
        tied_ids = {row["team_id"] for row in tied}
        tied.sort(
            key=lambda row: (
                -sum(
                    int(row["head_to_head"].get(opponent_id) or 0)
                    for opponent_id in tied_ids
                    if opponent_id != row["team_id"]
                ),
                -row["game_differential"],
                -row["game_wins"],
                row["team_name"].lower(),
                row["team_id"],
            )
        )
        ordered.extend(tied)
    return [{**row, "rank": index} for index, row in enumerate(ordered, start=1)]


def build_team_round_robin_matchups(team_ids: list[str]) -> list[dict[str, Any]]:
    """Build one deterministic single round robin, including odd-field byes."""

    teams = [_text(value) for value in team_ids if _text(value)]
    if len(teams) < 2:
        raise ValueError("At least two confirmed teams are required.")
    if len(set(teams)) != len(teams):
        raise ValueError("A team may appear only once in the round-robin field.")
    rotation: list[str | None] = list(teams)
    if len(rotation) % 2:
        rotation.append(None)
    rounds = len(rotation) - 1
    matchups: list[dict[str, Any]] = []
    for round_index in range(rounds):
        slot = 0
        for pair_index in range(len(rotation) // 2):
            team_a = rotation[pair_index]
            team_b = rotation[-1 - pair_index]
            if team_a is None or team_b is None:
                continue
            slot += 1
            if pair_index == 0 and round_index % 2:
                team_a, team_b = team_b, team_a
            matchups.append(
                {
                    "stage": "ROUND_ROBIN",
                    "round_number": round_index + 1,
                    "slot_number": slot,
                    "playoff_game_code": None,
                    "team_a_id": team_a,
                    "team_b_id": team_b,
                    "team_a_source": {"type": "TEAM", "team_id": team_a},
                    "team_b_source": {"type": "TEAM", "team_id": team_b},
                }
            )
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return matchups


def build_team_playoff_matchups(
    standings: list[dict[str, Any]],
    *,
    playoff_format: str,
) -> list[dict[str, Any]]:
    playoff = _text(playoff_format).upper()
    if playoff not in PLAYOFF_FORMATS or playoff == "NONE":
        raise ValueError("A supported playoff format is required.")
    ordered = sorted(standings, key=lambda row: int(row.get("rank") or 10_000))
    required = 2 if playoff == "TOP_2_FINAL" else 4
    if len(ordered) < required:
        raise ValueError(f"{required} completed round-robin seeds are required.")
    seed = {index: _text(ordered[index - 1].get("team_id")) for index in range(1, required + 1)}
    if playoff == "TOP_2_FINAL":
        return [
            {
                "stage": "PLAYOFF",
                "round_number": 1,
                "slot_number": 1,
                "playoff_game_code": "FINAL",
                "team_a_id": seed[1],
                "team_b_id": seed[2],
                "team_a_source": {"type": "SEED", "seed": 1},
                "team_b_source": {"type": "SEED", "seed": 2},
            }
        ]
    matchups = [
        {
            "stage": "PLAYOFF",
            "round_number": 1,
            "slot_number": 1,
            "playoff_game_code": "SF1",
            "team_a_id": seed[1],
            "team_b_id": seed[4],
            "team_a_source": {"type": "SEED", "seed": 1},
            "team_b_source": {"type": "SEED", "seed": 4},
        },
        {
            "stage": "PLAYOFF",
            "round_number": 1,
            "slot_number": 2,
            "playoff_game_code": "SF2",
            "team_a_id": seed[2],
            "team_b_id": seed[3],
            "team_a_source": {"type": "SEED", "seed": 2},
            "team_b_source": {"type": "SEED", "seed": 3},
        },
        {
            "stage": "PLAYOFF",
            "round_number": 2,
            "slot_number": 1,
            "playoff_game_code": "FINAL",
            "team_a_id": None,
            "team_b_id": None,
            "team_a_source": {"type": "WINNER", "game_code": "SF1"},
            "team_b_source": {"type": "WINNER", "game_code": "SF2"},
        },
    ]
    if playoff == "TOP_4_SEMIFINALS_WITH_BRONZE":
        matchups.append(
            {
                "stage": "PLAYOFF",
                "round_number": 2,
                "slot_number": 2,
                "playoff_game_code": "BRONZE",
                "team_a_id": None,
                "team_b_id": None,
                "team_a_source": {"type": "LOSER", "game_code": "SF1"},
                "team_b_source": {"type": "LOSER", "game_code": "SF2"},
            }
        )
    return matchups


def calculate_team_podium(
    *,
    playoff_format: str,
    standings: list[dict[str, Any]],
    playoff_matchups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Derive placements from completed results; never accept caller-chosen winners."""

    playoff = _text(playoff_format).upper()
    if playoff not in PLAYOFF_FORMATS:
        raise ValueError("Unsupported playoff format.")
    active_standings = sorted(standings, key=lambda row: int(row.get("rank") or 10_000))
    if playoff == "NONE":
        if len(active_standings) < 3:
            raise ValueError("Three completed standings are required for a podium.")
        return [
            {"placement": index, "team_id": _text(row.get("team_id"))}
            for index, row in enumerate(active_standings[:3], start=1)
        ]
    by_code = {_text(row.get("playoff_game_code")).upper(): row for row in playoff_matchups}
    final = by_code.get("FINAL")
    if not final or _text(final.get("status")).upper() != "FINAL":
        raise ValueError("The configured final must be completed before publishing a podium.")
    winner = _text(final.get("winner_team_id"))
    runner_up = _text(final.get("loser_team_id"))
    if not winner or not runner_up:
        raise ValueError("The configured final is missing a winner or runner-up.")
    podium = [{"placement": 1, "team_id": winner}, {"placement": 2, "team_id": runner_up}]
    if playoff == "TOP_2_FINAL":
        third = next(
            (_text(row.get("team_id")) for row in active_standings if _text(row.get("team_id")) not in {winner, runner_up}),
            "",
        )
    elif playoff == "TOP_4_SEMIFINALS_WITH_BRONZE":
        bronze = by_code.get("BRONZE")
        if not bronze or _text(bronze.get("status")).upper() != "FINAL":
            raise ValueError("The configured bronze match must be completed before publishing a podium.")
        third = _text(bronze.get("winner_team_id"))
    else:
        semifinal_losers = {
            _text((by_code.get("SF1") or {}).get("loser_team_id")),
            _text((by_code.get("SF2") or {}).get("loser_team_id")),
        }
        third = next(
            (_text(row.get("team_id")) for row in active_standings if _text(row.get("team_id")) in semifinal_losers),
            "",
        )
    if not third:
        raise ValueError("A third-place team cannot be derived from completed results.")
    return [*podium, {"placement": 3, "team_id": third}]


# Compatibility names retained for API/service callers.
build_team_lineup_games = build_locked_team_games
evaluate_team_matchup = evaluate_team_match
build_team_playoffs = build_team_playoff_matchups
