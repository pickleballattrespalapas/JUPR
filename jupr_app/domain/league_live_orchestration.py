from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

import pandas as pd

from jupr_app.domain.league_night_roster import suggest_court_sizes
from jupr_app.domain.live_ladder import build_movement_preview, compute_round_stats


class LeagueLiveDomainError(ValueError):
    """Raised when a League Live plan cannot be built safely."""


def _clean_text(value: Any, *, limit: int = 240) -> str:
    return " ".join(str(value or "").replace("<", "").replace(">", "").split())[:limit]


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 1200.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(parsed):
        return float(default)
    return float(parsed)


def _normalized_name(value: Any) -> str:
    return _clean_text(value, limit=160).casefold()


def canonical_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_league_live_roster(roster: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for index, raw in enumerate(roster or [], start=1):
        if not isinstance(raw, dict):
            raise LeagueLiveDomainError(f"Roster row {index} is invalid.")
        player_id = _safe_int(raw.get("player_id", raw.get("id")))
        player_name = _clean_text(raw.get("player_name", raw.get("name")), limit=160)
        if player_id is None or player_id <= 0:
            raise LeagueLiveDomainError(f"Roster row {index} requires a positive player_id.")
        if not player_name:
            raise LeagueLiveDomainError(f"Roster row {index} requires a player_name.")
        normalized_name = _normalized_name(player_name)
        if player_id in seen_ids:
            raise LeagueLiveDomainError(f"Player #{player_id} appears more than once in the roster.")
        if normalized_name in seen_names:
            raise LeagueLiveDomainError(f"Player name {player_name!r} appears more than once in the roster.")
        seen_ids.add(player_id)
        seen_names.add(normalized_name)
        status = _clean_text(raw.get("status"), limit=20).lower() or "active"
        if status not in {"active", "bench"}:
            raise LeagueLiveDomainError(f"Unsupported roster status for player #{player_id}.")
        normalized.append(
            {
                "player_id": int(player_id),
                "player_name": player_name,
                "rating": _safe_float(raw.get("rating", raw.get("rating_jupr", 1200.0))),
                "status": status,
                "court_number": _safe_int(raw.get("court_number", raw.get("court"))),
                "slot": _safe_int(raw.get("slot")),
                "bench_reason": _clean_text(raw.get("bench_reason"), limit=120) or None,
                "source_order": index,
            }
        )
    return normalized


def _default_bench_ids(roster: list[dict[str, Any]], bench_count: int, prefer_keep_player_ids: set[int]) -> list[int]:
    ordered = sorted(
        roster,
        key=lambda row: (
            int(row["player_id"]) in prefer_keep_player_ids,
            float(row.get("rating") or 1200.0),
            int(row.get("source_order") or 0),
            int(row["player_id"]),
        ),
    )
    return [int(row["player_id"]) for row in ordered[: max(0, int(bench_count))]]


def _validate_court_sizes(court_sizes: Iterable[Any] | None, *, roster_count: int) -> list[int]:
    sizes = [_safe_int(value) for value in (court_sizes or [])]
    if any(value not in {4, 5} for value in sizes):
        raise LeagueLiveDomainError("League Live courts must contain exactly four or five players.")
    parsed = [int(value) for value in sizes if value is not None]
    if sum(parsed) > int(roster_count):
        raise LeagueLiveDomainError("Court capacity cannot exceed the roster size.")
    return parsed


def _assign_performance_slots(
    rows: list[dict[str, Any]],
    *,
    current_rows_by_id: dict[int, dict[str, Any]],
    round_stats: dict[int, dict[str, int]],
) -> None:
    """Order next-round courts by movement boundary, then last-round results.

    A player arriving from a higher court anchors slot 1, a player promoted
    from a lower court anchors the final slot, and players staying on the court
    remain between them in wins/differential order. Ratings are deliberately
    excluded from the card-order contract.
    """

    def sort_key(row: dict[str, Any]) -> tuple[int, int, int, int, int, int, int]:
        player_id = int(row["player_id"])
        destination_court = int(row["court_number"])
        current_row = current_rows_by_id.get(player_id)
        source_court = int(current_row["court_number"]) if current_row else None
        if source_court is not None and destination_court > source_court:
            boundary = 0  # Down-movers enter at the top of their new court.
        elif source_court is not None and destination_court == source_court:
            boundary = 1  # Stayers are ranked by the completed round.
        elif source_court is None:
            boundary = 2  # New/returning players sit above the up-mover.
        else:
            boundary = 3  # Up-movers enter at the bottom of their new court.
        performance = round_stats.get(player_id, {})
        return (
            boundary,
            -int(performance.get("w", 0)),
            -int(performance.get("diff", 0)),
            -int(performance.get("pts", 0)),
            int(current_row.get("slot") or 10_000) if current_row else 10_000,
            int(row.get("source_order") or 10_000),
            player_id,
        )

    for court_number in sorted({int(row["court_number"]) for row in rows}):
        scoped = [row for row in rows if int(row["court_number"]) == court_number]
        scoped.sort(key=sort_key)
        for slot, row in enumerate(scoped, start=1):
            row["slot"] = slot


def build_league_live_roster_suggestion(
    roster: Iterable[dict[str, Any]] | None,
    *,
    court_sizes: Iterable[Any] | None = None,
    prefer_keep_player_ids: Iterable[Any] | None = None,
    bench_player_ids: Iterable[Any] | None = None,
    bench_override_reason: str | None = None,
    round_number: int = 1,
    preserve_assignment_order: bool = False,
    require_bench_override_reason: bool = True,
) -> dict[str, Any]:
    normalized = normalize_league_live_roster(roster)
    if len(normalized) < 4:
        raise LeagueLiveDomainError("At least four rostered players are required for League Live.")

    sizes = _validate_court_sizes(court_sizes, roster_count=len(normalized))
    suggestion_note: str | None = None
    if not sizes:
        suggestion = suggest_court_sizes(len(normalized))
        if not suggestion.get("ok") or not suggestion.get("sizes"):
            raise LeagueLiveDomainError(str(suggestion.get("note") or "Unable to suggest League Live courts."))
        sizes = [int(value) for value in suggestion["sizes"]]
        suggestion_note = str(suggestion.get("note") or "") or None

    capacity = sum(sizes)
    bench_count = len(normalized) - capacity
    prefer_keep = {
        int(value)
        for value in (_safe_int(item) for item in (prefer_keep_player_ids or []))
        if value is not None and value > 0
    }
    default_bench_ids = _default_bench_ids(normalized, bench_count, prefer_keep)
    requested_bench_ids = [
        int(value)
        for value in (_safe_int(item) for item in (bench_player_ids or []))
        if value is not None and value > 0
    ]
    if requested_bench_ids:
        if len(set(requested_bench_ids)) != len(requested_bench_ids):
            raise LeagueLiveDomainError("Bench override player IDs must be unique.")
        valid_ids = {int(row["player_id"]) for row in normalized}
        if not set(requested_bench_ids).issubset(valid_ids):
            raise LeagueLiveDomainError("Bench override contains a player outside this roster.")
        if len(requested_bench_ids) != bench_count:
            raise LeagueLiveDomainError(f"Select exactly {bench_count} bench player(s) for this court setup.")
        if (
            require_bench_override_reason
            and set(requested_bench_ids) != set(default_bench_ids)
            and len(_clean_text(bench_override_reason, limit=500)) < 10
        ):
            raise LeagueLiveDomainError("Explain the bench override in at least 10 characters.")
        selected_bench_ids = requested_bench_ids
    else:
        selected_bench_ids = default_bench_ids

    bench_set = set(selected_bench_ids)
    active = [row for row in normalized if int(row["player_id"]) not in bench_set]
    if preserve_assignment_order:
        active.sort(
            key=lambda row: (
                int(row.get("court_number") or 10_000),
                int(row.get("slot") or 10_000),
                int(row.get("source_order") or 0),
                int(row["player_id"]),
            )
        )
    else:
        active.sort(key=lambda row: (-float(row.get("rating") or 1200.0), int(row["player_id"])))
    bench = [row for row in normalized if int(row["player_id"]) in bench_set]
    bench.sort(key=lambda row: (selected_bench_ids.index(int(row["player_id"])), int(row["player_id"])))

    courts: list[dict[str, Any]] = []
    assigned_roster: list[dict[str, Any]] = []
    offset = 0
    safe_round = max(1, _safe_int(round_number, 1) or 1)
    for court_number, size in enumerate(sizes, start=1):
        players = active[offset : offset + size]
        if len(players) != size:
            raise LeagueLiveDomainError("Court suggestion did not consume the expected active roster.")
        court_players: list[dict[str, Any]] = []
        for slot, row in enumerate(players, start=1):
            player = {
                **{key: value for key, value in row.items() if key != "source_order"},
                "status": "active",
                "court_number": int(court_number),
                "slot": int(slot),
                "bench_reason": None,
            }
            court_players.append(player)
            assigned_roster.append(player)
        courts.append(
            {
                "round_number": int(safe_round),
                "court_number": int(court_number),
                "format_type": f"{size}-Player",
                "player_names": [str(row["player_name"]) for row in court_players],
                "players_json": court_players,
            }
        )
        offset += size

    bench_rows = [
        {
            **{key: value for key, value in row.items() if key != "source_order"},
            "status": "bench",
            "court_number": None,
            "slot": None,
            "bench_reason": "operator_override" if set(selected_bench_ids) != set(default_bench_ids) else "court_capacity",
        }
        for row in bench
    ]
    complete_roster = assigned_roster + bench_rows
    fingerprint_payload = {
        "round_number": safe_round,
        "court_sizes": sizes,
        "roster": complete_roster,
        "bench_override_reason": _clean_text(bench_override_reason, limit=500) or None,
    }
    return {
        "ok": True,
        "mode": "league_live_roster_suggestion",
        "round_number": int(safe_round),
        "roster": complete_roster,
        "active_roster": assigned_roster,
        "bench": bench_rows,
        "bench_count": len(bench_rows),
        "bench_player_ids": [int(row["player_id"]) for row in bench_rows],
        "default_bench_player_ids": default_bench_ids,
        "bench_override_applied": set(selected_bench_ids) != set(default_bench_ids),
        "bench_override_reason": _clean_text(bench_override_reason, limit=500) or None,
        "court_sizes": sizes,
        "courts": courts,
        "suggestion_note": suggestion_note,
        "fingerprint": canonical_fingerprint(fingerprint_payload),
    }


def _roster_from_courts(
    courts: Iterable[dict[str, Any]] | None,
    *,
    roster_pool: Iterable[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool = normalize_league_live_roster(roster_pool)
    by_id = {int(row["player_id"]): row for row in pool}
    by_name = {_normalized_name(row["player_name"]): row for row in pool}
    active_rows: list[dict[str, Any]] = []
    normalized_courts: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    court_numbers: list[int] = []
    for index, raw in enumerate(courts or [], start=1):
        if not isinstance(raw, dict):
            raise LeagueLiveDomainError(f"Court row {index} is invalid.")
        court_number = _safe_int(raw.get("court_number", raw.get("court")), index) or index
        if court_number <= 0 or court_number in court_numbers:
            raise LeagueLiveDomainError("Court numbers must be positive and unique.")
        court_numbers.append(court_number)
        raw_players = raw.get("players_json", raw.get("players"))
        players: list[dict[str, Any]] = []
        if isinstance(raw_players, list) and raw_players:
            for player in raw_players:
                if not isinstance(player, dict):
                    raise LeagueLiveDomainError(f"Court {court_number} contains an invalid player row.")
                player_id = _safe_int(player.get("player_id", player.get("id")))
                resolved = by_id.get(int(player_id)) if player_id is not None else None
                if resolved is None:
                    resolved = by_name.get(_normalized_name(player.get("player_name", player.get("name"))))
                if resolved is None:
                    raise LeagueLiveDomainError(f"Court {court_number} contains a player outside the session roster.")
                players.append(resolved)
        else:
            names = raw.get("player_names") or []
            if isinstance(names, str):
                names = [value.strip() for value in names.replace(",", "\n").splitlines() if value.strip()]
            for name in names if isinstance(names, list) else []:
                resolved = by_name.get(_normalized_name(name))
                if resolved is None:
                    raise LeagueLiveDomainError(f"Court {court_number} contains unknown player {str(name)!r}.")
                players.append(resolved)
        if len(players) not in {4, 5}:
            raise LeagueLiveDomainError(f"Court {court_number} must contain exactly four or five players.")
        court_players: list[dict[str, Any]] = []
        for slot, row in enumerate(players, start=1):
            player_id = int(row["player_id"])
            if player_id in seen_ids:
                raise LeagueLiveDomainError(f"Player #{player_id} appears on more than one court.")
            seen_ids.add(player_id)
            normalized = {
                **{key: value for key, value in row.items() if key != "source_order"},
                "status": "active",
                "court_number": int(court_number),
                "slot": int(slot),
                "bench_reason": None,
            }
            court_players.append(normalized)
            active_rows.append(normalized)
        normalized_courts.append(
            {
                "round_number": _safe_int(raw.get("round_number"), 1) or 1,
                "court_number": int(court_number),
                "format_type": f"{len(court_players)}-Player",
                "player_names": [str(row["player_name"]) for row in court_players],
                "players_json": court_players,
            }
        )
    if sorted(court_numbers) != list(range(1, len(court_numbers) + 1)):
        raise LeagueLiveDomainError("Court numbers must be contiguous starting at 1.")
    normalized_courts.sort(key=lambda row: int(row["court_number"]))
    return active_rows, normalized_courts


def _valid_scored_matches(
    matches: Iterable[dict[str, Any]] | None,
    *,
    active_by_id: dict[int, dict[str, Any]],
    court_by_player_id: dict[int, int],
) -> tuple[list[dict[str, Any]], list[str]]:
    valid: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, raw in enumerate(matches or [], start=1):
        if not isinstance(raw, dict):
            warnings.append(f"Match {index} is not an object and was ignored.")
            continue
        score_t1 = _safe_int(raw.get("score_t1", raw.get("s1")))
        score_t2 = _safe_int(raw.get("score_t2", raw.get("s2")))
        if score_t1 is None or score_t2 is None:
            warnings.append(f"Match {index} has no complete score and was ignored.")
            continue
        if score_t1 < 0 or score_t2 < 0 or score_t1 == score_t2 or score_t1 + score_t2 <= 0:
            raise LeagueLiveDomainError(f"Match {index} requires a non-negative, non-tied score.")
        player_ids = [
            _safe_int(raw.get("t1_p1")),
            _safe_int(raw.get("t1_p2")),
            _safe_int(raw.get("t2_p1")),
            _safe_int(raw.get("t2_p2")),
        ]
        if any(player_id is None or player_id <= 0 for player_id in player_ids):
            raise LeagueLiveDomainError(f"Match {index} requires four valid player IDs.")
        parsed_ids = [int(player_id) for player_id in player_ids if player_id is not None]
        if len(set(parsed_ids)) != 4:
            raise LeagueLiveDomainError(f"Match {index} must contain four distinct players.")
        if not set(parsed_ids).issubset(active_by_id):
            raise LeagueLiveDomainError(f"Match {index} contains a player outside the active court roster.")
        player_courts = {court_by_player_id[player_id] for player_id in parsed_ids}
        match_court = _safe_int(raw.get("court", raw.get("court_number")))
        if len(player_courts) != 1 or (match_court is not None and match_court not in player_courts):
            raise LeagueLiveDomainError(f"Match {index} players must all belong to the same court.")
        valid.append(
            {
                **raw,
                "court": int(next(iter(player_courts))),
                "t1_p1": parsed_ids[0],
                "t1_p2": parsed_ids[1],
                "t2_p1": parsed_ids[2],
                "t2_p2": parsed_ids[3],
                "s1": int(score_t1),
                "s2": int(score_t2),
                "score_t1": int(score_t1),
                "score_t2": int(score_t2),
            }
        )
    return valid, warnings


def build_league_live_round_plan(
    *,
    session_id: str,
    round_number: int,
    total_rounds: int,
    session_updated_at: str,
    roster: Iterable[dict[str, Any]] | None,
    courts: Iterable[dict[str, Any]] | None,
    matches: Iterable[dict[str, Any]] | None,
    movement_overrides: Iterable[dict[str, Any]] | None = None,
    override_reason: str | None = None,
    roster_change: dict[str, Any] | None = None,
    bench_player_ids: Iterable[Any] | None = None,
    bench_override_reason: str | None = None,
    comparison_session_updated_at: str | None = None,
) -> dict[str, Any]:
    safe_round = max(1, _safe_int(round_number, 1) or 1)
    safe_total_rounds = max(safe_round, _safe_int(total_rounds, safe_round) or safe_round)
    next_round = min(safe_round + 1, safe_total_rounds)
    roster_pool = normalize_league_live_roster(roster)
    active_rows, normalized_courts = _roster_from_courts(courts, roster_pool=roster_pool)
    if not active_rows:
        raise LeagueLiveDomainError("League Live movement requires at least one active court.")
    active_by_id = {int(row["player_id"]): row for row in active_rows}
    court_by_player_id = {int(row["player_id"]): int(row["court_number"]) for row in active_rows}
    valid_matches, warnings = _valid_scored_matches(
        matches,
        active_by_id=active_by_id,
        court_by_player_id=court_by_player_id,
    )
    if not valid_matches:
        raise LeagueLiveDomainError("Enter at least one valid scored match before building the court board.")

    roster_df = pd.DataFrame(
        [
            {
                "player_id": int(row["player_id"]),
                "name": str(row["player_name"]),
                "rating": float(row.get("rating") or 1200.0),
                "court": int(row["court_number"]),
                "slot": int(row.get("slot") or 0),
            }
            for row in active_rows
        ]
    )
    stats = compute_round_stats(valid_matches, [int(row["player_id"]) for row in active_rows])
    preview = build_movement_preview(roster_df, stats, max_court=len(normalized_courts))
    if preview.empty:
        raise LeagueLiveDomainError("Unable to build the next-round court board.")

    next_active: list[dict[str, Any]] = []
    for _, row in preview.iterrows():
        player_id = int(row["player_id"])
        source_row = active_by_id[player_id]
        next_active.append(
            {
                **source_row,
                "status": "active",
                "court_number": int(row["Proposed Court"]),
                "slot": None,
                "bench_reason": None,
            }
        )

    _assign_performance_slots(
        next_active,
        current_rows_by_id=active_by_id,
        round_stats=stats,
    )
    for court_number in range(1, len(normalized_courts) + 1):
        scoped = [row for row in next_active if int(row["court_number"]) == court_number]
        if len(scoped) not in {4, 5}:
            raise LeagueLiveDomainError(
                f"Manual movement leaves Court {court_number} with {len(scoped)} players; each court requires four or five."
            )

    existing_bench = [row for row in roster_pool if int(row["player_id"]) not in active_by_id]
    roster_change_payload: dict[str, Any] | None = None
    if roster_change:
        if not isinstance(roster_change, dict):
            raise LeagueLiveDomainError("Roster change must be an object.")
        action = _clean_text(roster_change.get("action"), limit=20).lower()
        if action not in {"add", "substitute"}:
            raise LeagueLiveDomainError("Roster change action must be add or substitute.")
        incoming = normalize_league_live_roster([roster_change.get("player") or {}])[0]
        pool_ids = {int(row["player_id"]) for row in next_active + existing_bench}
        if int(incoming["player_id"]) in pool_ids:
            raise LeagueLiveDomainError("Incoming roster-change player is already in this League Live roster.")
        replaced_player_id = _safe_int(roster_change.get("replaced_player_id"))
        if action == "substitute":
            if replaced_player_id is None or replaced_player_id not in {int(row["player_id"]) for row in next_active}:
                raise LeagueLiveDomainError("Substitution requires an active replaced_player_id.")
            next_active = [row for row in next_active if int(row["player_id"]) != replaced_player_id]
        elif replaced_player_id is not None:
            raise LeagueLiveDomainError("Add-player changes must not include replaced_player_id.")
        incoming.update({"status": "active", "court_number": None, "slot": None, "bench_reason": None})
        combined = next_active + existing_bench + [incoming]
        preserved_ids = {int(row["player_id"]) for row in next_active} | {int(incoming["player_id"])}
        current_court_sizes = [
            len([row for row in active_rows if int(row["court_number"]) == court_number])
            for court_number in range(1, len(normalized_courts) + 1)
        ]
        suggested = build_league_live_roster_suggestion(
            combined,
            court_sizes=current_court_sizes if action == "substitute" else None,
            prefer_keep_player_ids=preserved_ids,
            bench_player_ids=bench_player_ids,
            bench_override_reason=bench_override_reason,
            round_number=next_round,
            preserve_assignment_order=True,
            require_bench_override_reason=False,
        )
        next_active = list(suggested["active_roster"])
        existing_bench = list(suggested["bench"])
        roster_change_payload = {
            "action": action,
            "replaced_player_id": replaced_player_id,
            "incoming_player_id": int(incoming["player_id"]),
            "effective_round": next_round,
        }
    else:
        if list(bench_player_ids or []):
            suggested = build_league_live_roster_suggestion(
                next_active + existing_bench,
                court_sizes=[len([row for row in next_active if int(row["court_number"]) == number]) for number in range(1, len(normalized_courts) + 1)],
                prefer_keep_player_ids={int(row["player_id"]) for row in next_active},
                bench_player_ids=bench_player_ids,
                bench_override_reason=bench_override_reason,
                round_number=next_round,
                preserve_assignment_order=True,
                require_bench_override_reason=False,
            )
            next_active = list(suggested["active_roster"])
            existing_bench = list(suggested["bench"])

    _assign_performance_slots(
        next_active,
        current_rows_by_id=active_by_id,
        round_stats=stats,
    )

    default_next_active = [dict(row) for row in next_active]
    default_bench = [dict(row) for row in existing_bench]
    next_court_numbers = sorted({int(row["court_number"]) for row in default_next_active})
    if next_court_numbers != list(range(1, len(next_court_numbers) + 1)):
        raise LeagueLiveDomainError("Next-round court numbers must be contiguous starting at 1.")
    max_next_court = len(next_court_numbers)
    default_location_by_id = {
        int(row["player_id"]): (int(row["court_number"]), int(row.get("slot") or 0))
        for row in default_next_active
    }
    default_location_by_id.update(
        {int(row["player_id"]): (0, 0) for row in default_bench}
    )

    raw_override_rows = list(movement_overrides or [])
    normalized_override_rows: list[dict[str, int]] = []
    seen_override_ids: set[int] = set()
    final_roster_ids = set(default_location_by_id)
    for index, raw in enumerate(raw_override_rows, start=1):
        if not isinstance(raw, dict):
            raise LeagueLiveDomainError(f"Movement override {index} is invalid.")
        player_id = _safe_int(raw.get("player_id"))
        if "to_court" not in raw and "court_number" not in raw:
            raise LeagueLiveDomainError(f"Movement override {index} requires a target court or Bench.")
        raw_to_court = raw.get("to_court", raw.get("court_number"))
        to_court = 0 if raw_to_court in (None, "", "bench", "Bench") else _safe_int(raw_to_court)
        to_slot = _safe_int(raw.get("to_slot", raw.get("slot")))
        if player_id is None or player_id not in final_roster_ids:
            raise LeagueLiveDomainError(f"Movement override {index} contains a player outside the next-round roster.")
        if player_id in seen_override_ids:
            raise LeagueLiveDomainError(f"Movement override for player #{player_id} is duplicated.")
        if to_court is None or to_court < 0 or to_court > max_next_court:
            raise LeagueLiveDomainError(f"Movement override {index} targets an unavailable court.")
        if to_court == 0 and to_slot not in (None, 0):
            raise LeagueLiveDomainError(f"Movement override {index} gives a bench player a court slot.")
        if to_court > 0 and to_slot is not None and to_slot < 1:
            raise LeagueLiveDomainError(f"Movement override {index} requires a positive court slot.")
        seen_override_ids.add(player_id)
        normalized = {"player_id": int(player_id), "to_court": int(to_court)}
        if to_slot is not None:
            normalized["to_slot"] = int(to_slot)
        normalized_override_rows.append(normalized)

    ordered_board_override = any("to_slot" in row for row in normalized_override_rows) or any(
        row["to_court"] == 0 for row in normalized_override_rows
    )
    override_by_id = {int(row["player_id"]): row for row in normalized_override_rows}
    default_bench_ids = {int(row["player_id"]) for row in default_bench}
    if ordered_board_override:
        if seen_override_ids != final_roster_ids:
            raise LeagueLiveDomainError("A reordered court board must assign every next-round player exactly once.")
        overridden_bench_ids = {
            int(row["player_id"]) for row in normalized_override_rows if int(row["to_court"]) == 0
        }
        if overridden_bench_ids != default_bench_ids:
            raise LeagueLiveDomainError("Court-board bench assignments must match the reviewed next-round bench selection.")

    if normalized_override_rows:
        reassigned_active: list[dict[str, Any]] = []
        for row in default_next_active:
            player_id = int(row["player_id"])
            assignment = override_by_id.get(player_id)
            if assignment and int(assignment["to_court"]) == 0:
                raise LeagueLiveDomainError("An active court-board player cannot also be assigned to Bench.")
            reassigned_active.append(
                {
                    **row,
                    "court_number": int(assignment["to_court"]) if assignment else int(row["court_number"]),
                    "slot": int(assignment["to_slot"]) if assignment and "to_slot" in assignment else row.get("slot"),
                }
            )
        next_active = reassigned_active

    for court_number in next_court_numbers:
        scoped = [row for row in next_active if int(row["court_number"]) == court_number]
        if len(scoped) not in {4, 5}:
            raise LeagueLiveDomainError(
                f"Manual movement leaves Court {court_number} with {len(scoped)} players; each court requires four or five."
            )
        if ordered_board_override:
            slots = sorted(int(row.get("slot") or 0) for row in scoped)
            if slots != list(range(1, len(scoped) + 1)):
                raise LeagueLiveDomainError(f"Court {court_number} card order must use every slot from 1 to {len(scoped)} once.")
    if not ordered_board_override:
        _assign_performance_slots(
            next_active,
            current_rows_by_id=active_by_id,
            round_stats=stats,
        )

    next_active.sort(key=lambda row: (int(row["court_number"]), int(row.get("slot") or 0), int(row["player_id"])))
    final_location_by_id = {
        int(row["player_id"]): (int(row["court_number"]), int(row.get("slot") or 0))
        for row in next_active
    }
    final_location_by_id.update({int(row["player_id"]): (0, 0) for row in existing_bench})
    override_applied = any(
        final_location_by_id[player_id] != default_location_by_id[player_id]
        for player_id in final_roster_ids
    )
    clean_override_reason = _clean_text(override_reason, limit=500)

    movement_rows: list[dict[str, Any]] = []
    for _, row in preview.iterrows():
        player_id = int(row["player_id"])
        if player_id not in final_location_by_id:
            continue
        from_court = int(row["court"])
        suggested_court, suggested_slot = default_location_by_id[player_id]
        to_court, to_slot = final_location_by_id[player_id]
        movement_rows.append(
            {
                "player_id": player_id,
                "player_name": str(row["name"]),
                "from_court": from_court,
                "suggested_court": suggested_court or None,
                "suggested_slot": suggested_slot or None,
                "to_court": to_court or None,
                "to_slot": to_slot or None,
                "wins": int(row["Round Wins"]),
                "differential": int(row["Round Diff"]),
                "points": int(row["Round Pts"]),
                "direction": "bench" if to_court == 0 else "up" if to_court < from_court else "down" if to_court > from_court else "stay",
                "overridden": (to_court, to_slot) != (suggested_court, suggested_slot),
            }
        )

    next_courts: list[dict[str, Any]] = []
    for court_number in sorted({int(row["court_number"]) for row in next_active}):
        players = sorted(
            [row for row in next_active if int(row["court_number"]) == court_number],
            key=lambda row: int(row.get("slot") or 0),
        )
        next_courts.append(
            {
                "round_number": next_round,
                "court_number": court_number,
                "format_type": f"{len(players)}-Player",
                "player_names": [str(row["player_name"]) for row in players],
                "players_json": players,
            }
        )

    next_roster = next_active + existing_bench
    # Publish metadata (for example deterministic context IDs) is intentionally
    # excluded. The movement plan key covers only the scored result and court
    # inputs that can change Python's next-round decision.
    operation_matches = [
        {
            "court": int(row["court"]),
            "t1_p1": int(row["t1_p1"]),
            "t1_p2": int(row["t1_p2"]),
            "t2_p1": int(row["t2_p1"]),
            "t2_p2": int(row["t2_p2"]),
            "score_t1": int(row["score_t1"]),
            "score_t2": int(row["score_t2"]),
        }
        for row in valid_matches
    ]
    operation_payload = {
        "session_id": str(session_id),
        "session_updated_at": str(session_updated_at),
        "round_number": safe_round,
        "courts": normalized_courts,
        "matches": operation_matches,
        "movement_overrides": sorted(normalized_override_rows, key=lambda row: int(row["player_id"])),
        "override_reason": clean_override_reason or None,
        "roster_change": roster_change_payload,
        "bench_player_ids": [int(row["player_id"]) for row in existing_bench],
        "bench_override_reason": _clean_text(bench_override_reason, limit=500) or None,
    }
    operation_key = canonical_fingerprint(operation_payload)
    result = {
        "ok": True,
        "mode": "league_live_round_plan",
        "operation_key": operation_key,
        "session_id": str(session_id),
        "session_updated_at": str(session_updated_at),
        "round_number": safe_round,
        "next_round": next_round,
        "ready_to_save": True,
        "scored_match_count": len(valid_matches),
        "warnings": warnings,
        "current_courts": normalized_courts,
        "movement": {
            "strategy": "top_up_bottom_down",
            "authority": "python_fastapi",
            "applied": any(row["direction"] != "stay" for row in movement_rows),
            "override_applied": override_applied,
            "override_reason": clean_override_reason or None,
            "next_round": next_round,
            "rows": movement_rows,
            "next_courts": next_courts,
            "operation_key": operation_key,
        },
        "roster_change": roster_change_payload,
        "next_roster": next_roster,
        "next_courts": next_courts,
        "bench": existing_bench,
        "bench_player_ids": [int(row["player_id"]) for row in existing_bench],
        "recovery": {
            "session_detail": f"/admin/clubs/{{club_id}}/league-manager/live-sessions/{session_id}",
            "match_log": "/admin/match-log",
            "replay_history": "/admin/replay-history",
        },
    }
    if comparison_session_updated_at:
        result["comparison_operation_key"] = canonical_fingerprint(
            {
                **operation_payload,
                "session_updated_at": str(comparison_session_updated_at),
            }
        )
    return result
