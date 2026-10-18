from __future__ import annotations

import csv
import hashlib
import io
import json
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

from jupr_app.domain.league_match_structure import (
    normalize_league_match_structure,
    validate_league_series_matches,
)


class LeagueLivePublishError(ValueError):
    """A League Live publish request violates the durable submit contract."""


UNUSUAL_SCORE_POINT_THRESHOLD = 30
UNUSUAL_SCORE_MARGIN_THRESHOLD = 20
PLAYER_KEYS = ("t1_p1", "t1_p2", "t2_p1", "t2_p2")


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _clean_text(value: Any, *, limit: int = 240) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def stable_payload_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def league_live_match_context_id(*, session_id: str, round_number: int, match_index: int) -> str:
    seed = f"jupr:league-live:{str(session_id)}:round:{int(round_number)}:match:{int(match_index)}"
    return str(uuid5(NAMESPACE_URL, seed))


def league_live_unusual_score_reason(score_t1: Any, score_t2: Any) -> str | None:
    """Return an operator-review reason without rejecting legitimate long games."""
    left = _safe_int(score_t1)
    right = _safe_int(score_t2)
    if left is None or right is None or left < 0 or right < 0:
        return None
    if max(left, right) >= UNUSUAL_SCORE_POINT_THRESHOLD:
        return f"one side has {max(left, right)} points (usual review threshold: {UNUSUAL_SCORE_POINT_THRESHOLD})"
    margin = abs(left - right)
    if margin >= UNUSUAL_SCORE_MARGIN_THRESHOLD:
        return f"the winning margin is {margin} points (usual review threshold: {UNUSUAL_SCORE_MARGIN_THRESHOLD})"
    return None


def _positive_integer(value: Any, label: str) -> int:
    try:
        number = int(value)
        if isinstance(value, bool) or float(value) != number or number < 1:
            raise ValueError
        return number
    except (TypeError, ValueError, OverflowError):
        raise LeagueLivePublishError(f"{label} must be a positive whole number.") from None


def _player_ids(values: Any, label: str) -> list[int]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise LeagueLivePublishError(f"{label} must contain four ordered player IDs.")
    ids = [_positive_integer(value, label) for value in values]
    if len(set(ids)) != 4:
        raise LeagueLivePublishError(f"{label} must contain four distinct players.")
    return ids


def _normalize_unplayed_games(games: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized = []
    for index, raw in enumerate(games or [], start=1):
        if not isinstance(raw, dict):
            raise LeagueLivePublishError(f"Unplayed game {index} is invalid.")
        series_key = str(raw.get("series_key") or "").strip()
        if not series_key or len(series_key) > 160:
            raise LeagueLivePublishError(f"Unplayed game {index} needs its generated series reference.")
        reason = _clean_text(raw.get("reason"), limit=500)
        if not reason:
            raise LeagueLivePublishError(f"Unplayed game {index} needs a reason.")
        if any(raw.get(key) not in (None, "") for key in ("score_t1", "score_t2", "s1", "s2")):
            raise LeagueLivePublishError("Unplayed games cannot have scores or be published as forfeits.")
        normalized.append({
            "series_key": series_key,
            "game_number": _positive_integer(raw.get("game_number"), "Unplayed game number"),
            "court": _positive_integer(raw.get("court"), "Unplayed game court"),
            "reason": reason,
        })
    return normalized


def validate_league_live_game_slots(
    matches: Iterable[dict[str, Any]] | None,
    *,
    preview: dict[str, Any] | None,
    match_structure: Any = None,
    unplayed_games: Iterable[dict[str, Any]] | None = None,
    require_complete: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check actual participants and exclusions against a server-generated schedule.

    The caller must supply an authoritative preview, regenerated from the saved
    court assignments. Scheduled IDs identify a slot for movement only; actual
    IDs remain the scored participants used by ratings and player history.
    """
    structure = normalize_league_match_structure(match_structure)
    skipped = _normalize_unplayed_games(unplayed_games)
    if skipped and structure["kind"] == "best_of":
        raise LeagueLivePublishError("Best-of series must finish normally; unplayed game exclusions are supported for fixed-game rounds.")
    generated: dict[str, dict[str, Any]] = {}
    for court in (preview or {}).get("courts") or []:
        if not isinstance(court, dict):
            raise LeagueLivePublishError("The generated court preview is invalid. Regenerate the round.")
        for raw in court.get("matches") or []:
            if not isinstance(raw, dict):
                raise LeagueLivePublishError("The generated match preview is invalid. Regenerate the round.")
            series_key = str(raw.get("row_id") or raw.get("series_key") or "").strip()
            if not series_key or len(series_key) > 160 or series_key in generated:
                raise LeagueLivePublishError("The generated preview has missing or duplicate series references.")
            generated[series_key] = {
                "court": _positive_integer(raw.get("court", court.get("court")), "Generated court"),
                "player_ids": _player_ids([raw.get(key) for key in PLAYER_KEYS], "Generated players"),
            }
    if not generated:
        raise LeagueLivePublishError("A verified generated preview is required for substitutions or unplayed games.")

    seen: set[tuple[str, int]] = set()

    def validate_slot(raw: dict[str, Any]) -> tuple[str, int, dict[str, Any]]:
        series_key = str(raw.get("series_key") or raw.get("row_id") or "").strip()
        game_number = _positive_integer(raw.get("game_number", 1), "Game number")
        slot = (series_key, game_number)
        if series_key not in generated or game_number > int(structure["games"]):
            raise LeagueLivePublishError("A game does not match a generated slot. Regenerate the round.")
        if slot in seen:
            raise LeagueLivePublishError("A generated game slot is duplicated or marked both scored and unplayed.")
        expected = generated[series_key]
        if _positive_integer(raw.get("court"), "Game court") != expected["court"]:
            raise LeagueLivePublishError(f"Series {series_key} does not match its generated court.")
        seen.add(slot)
        return series_key, game_number, expected

    scored: list[dict[str, Any]] = []
    for raw in matches or []:
        if not isinstance(raw, dict):
            raise LeagueLivePublishError("Each scored game must be a match row.")
        series_key, game_number, expected = validate_slot(raw)
        supplied_kind = str(raw.get("series_kind") or structure["kind"]).strip().lower()
        supplied_games = _positive_integer(raw.get("series_games", structure["games"]), "Series game count")
        if supplied_kind != structure["kind"] or supplied_games != int(structure["games"]):
            raise LeagueLivePublishError("The scored series does not match the league settings. Reload the league.")
        actual = _player_ids([raw.get(key) for key in PLAYER_KEYS], "Actual players")
        has_scheduled = "scheduled_player_ids" in raw
        scheduled = _player_ids(raw.get("scheduled_player_ids"), "Scheduled players") if has_scheduled else actual
        if scheduled != expected["player_ids"]:
            raise LeagueLivePublishError(f"Series {series_key} does not match its generated scheduled players.")
        reason = _clean_text(raw.get("substitution_reason"), limit=500)
        if actual != scheduled and not reason:
            raise LeagueLivePublishError(f"Series {series_key}, Game {game_number} needs a substitution reason.")
        row = {
            **raw,
            "series_key": series_key,
            "series_kind": structure["kind"],
            "series_games": int(structure["games"]),
            "game_number": game_number,
        }
        if has_scheduled:
            row["scheduled_player_ids"] = scheduled
            row["substitution_reason"] = reason or None
        scored.append(row)
    for raw in skipped:
        validate_slot(raw)
    if require_complete:
        if structure["kind"] == "fixed_games":
            expected_slots = {
                (series_key, number)
                for series_key in generated
                for number in range(1, int(structure["games"]) + 1)
            }
            if seen != expected_slots:
                raise LeagueLivePublishError("Every generated game must have a final score or an explicit unplayed reason.")
        elif {key for key, _ in seen} != set(generated):
            raise LeagueLivePublishError("Every generated best-of series must be completed before publishing.")
    order = {key: index for index, key in enumerate(generated)}
    scored.sort(key=lambda row: (order[row["series_key"]], row["game_number"]))
    skipped.sort(key=lambda row: (order[row["series_key"]], row["game_number"]))
    return scored, skipped


def normalize_league_live_publish_matches(
    matches: Iterable[dict[str, Any]] | None,
    *,
    session_id: str,
    round_number: int,
    league_name: str,
    week_tag: str,
    match_date: str,
    expected_match_count: int,
    match_structure: Any = None,
    preview: dict[str, Any] | None = None,
    unplayed_games: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    structure = normalize_league_match_structure(match_structure)
    source_rows = list(matches or [])
    skipped = _normalize_unplayed_games(unplayed_games)
    flexible_games = unplayed_games is not None or any(isinstance(row, dict) and "scheduled_player_ids" in row for row in source_rows)
    if flexible_games:
        raw_rows, skipped = validate_league_live_game_slots(
            source_rows, preview=preview, match_structure=structure, unplayed_games=skipped,
        )
        if structure["kind"] == "best_of":
            # Validate the scheduled teams' series outcome, while keeping the
            # actual game participants in the official rows below.
            validate_league_series_matches(
                [{**row, **dict(zip(PLAYER_KEYS, row.get("scheduled_player_ids") or [row.get(key) for key in PLAYER_KEYS]))} for row in raw_rows],
                match_structure=structure,
            )
    else:
        raw_rows = validate_league_series_matches(source_rows, match_structure=structure)
    expected = int(expected_match_count or 0)
    if expected < 1:
        raise LeagueLivePublishError("expected_match_count must include every generated match slot.")
    if len(raw_rows) + len(skipped) != expected:
        raise LeagueLivePublishError(
            f"All generated matches must be scored or explicitly unplayed before publish ({len(raw_rows) + len(skipped)} of {expected} supplied)."
        )
    if not raw_rows:
        raise LeagueLivePublishError("At least one completed game is required; cancel an entirely unplayed round instead.")
    if expected > 200:
        raise LeagueLivePublishError("A League Live round cannot publish more than 200 matches.")

    normalized: list[dict[str, Any]] = []
    signatures: set[tuple[Any, ...]] = set()
    for index, raw in enumerate(raw_rows, start=1):
        player_ids = [_safe_int(raw.get(key)) for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")]
        if any(player_id is None or int(player_id) <= 0 for player_id in player_ids):
            raise LeagueLivePublishError(f"Match {index} has an unresolved player.")
        resolved_ids = [int(player_id or 0) for player_id in player_ids]
        if len(set(resolved_ids)) != 4:
            raise LeagueLivePublishError(f"Match {index} must contain four distinct players.")
        score_t1 = _safe_int(raw.get("score_t1", raw.get("s1")))
        score_t2 = _safe_int(raw.get("score_t2", raw.get("s2")))
        if score_t1 is None or score_t2 is None or score_t1 < 0 or score_t2 < 0:
            raise LeagueLivePublishError(f"Match {index} has an invalid score.")
        if score_t1 == score_t2 or (score_t1 + score_t2) <= 0:
            raise LeagueLivePublishError(f"Match {index} must have a non-tied final score.")
        series_key = _clean_text(raw.get("series_key"), limit=160)
        game_number = _safe_int(raw.get("game_number"))
        signature = (
            *resolved_ids,
            int(score_t1),
            int(score_t2),
            series_key,
            int(game_number or 0),
        )
        if signature in signatures:
            raise LeagueLivePublishError(f"Match {index} duplicates another scored row in this round.")
        signatures.add(signature)
        context_id = league_live_match_context_id(
            session_id=str(session_id),
            round_number=int(round_number),
            match_index=index,
        )
        normalized_row = {
            "date": _clean_text(match_date, limit=40),
            "league": _clean_text(league_name, limit=120),
            "week_tag": _clean_text(week_tag, limit=80),
            "match_type": "League Manager Live",
            "context_type": "league_live_session",
            "context_id": context_id,
            "court": _safe_int(raw.get("court"), index) or index,
            "t1_p1": resolved_ids[0],
            "t1_p2": resolved_ids[1],
            "t2_p1": resolved_ids[2],
            "t2_p2": resolved_ids[3],
            "score_t1": int(score_t1),
            "score_t2": int(score_t2),
        }
        if series_key:
            normalized_row.update(
                {
                    "series_key": series_key,
                    "series_kind": structure["kind"],
                    "series_games": int(structure["games"]),
                    "game_number": int(game_number or 0),
                }
            )
        if "scheduled_player_ids" in raw:
            normalized_row["scheduled_player_ids"] = list(raw["scheduled_player_ids"])
            normalized_row["substitution_reason"] = raw.get("substitution_reason")
        normalized.append(normalized_row)
    return normalized


def build_league_live_publish_request(
    *,
    session_id: str,
    round_number: int,
    league_name: str,
    week_tag: str,
    match_date: str,
    matches: list[dict[str, Any]],
    expected_match_count: int,
    expected_updated_at: str,
    expected_operation_key: str,
    unusual_score_acknowledgement: bool = False,
    round_label: str | None = None,
    preview: dict[str, Any] | None = None,
    courts: list[dict[str, Any]] | None = None,
    movement_overrides: list[dict[str, Any]] | None = None,
    override_reason: str | None = None,
    roster_change: dict[str, Any] | None = None,
    bench_player_ids: list[Any] | None = None,
    bench_override_reason: str | None = None,
    match_structure: Any = None,
    unplayed_games: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    structure = normalize_league_match_structure(match_structure)
    normalized_matches = normalize_league_live_publish_matches(
        matches,
        session_id=str(session_id),
        round_number=int(round_number),
        league_name=str(league_name),
        week_tag=str(week_tag),
        match_date=str(match_date),
        expected_match_count=int(expected_match_count),
        match_structure=structure,
        preview=preview,
        unplayed_games=unplayed_games,
    )
    request = {
        "session_id": str(session_id),
        "round_number": int(round_number),
        "round_label": _clean_text(round_label, limit=80) or f"Round {int(round_number)}",
        "match_date": _clean_text(match_date, limit=40),
        "expected_match_count": int(expected_match_count),
        "expected_updated_at": _clean_text(expected_updated_at, limit=120),
        "expected_operation_key": _clean_text(expected_operation_key, limit=128),
        "preview": dict(preview or {}),
        "matches": normalized_matches,
        "courts": [dict(row) for row in (courts or []) if isinstance(row, dict)],
        "movement_overrides": [dict(row) for row in (movement_overrides or []) if isinstance(row, dict)],
        "override_reason": _clean_text(override_reason, limit=500) or None,
        "roster_change": dict(roster_change or {}) or None,
        "bench_player_ids": sorted({int(value) for value in (bench_player_ids or []) if _safe_int(value) is not None}),
        "bench_override_reason": _clean_text(bench_override_reason, limit=500) or None,
        "match_structure": structure,
    }
    if unplayed_games is not None:
        request["unplayed_games"] = _normalize_unplayed_games(unplayed_games)
        request["preview"]["unplayed_games"] = request["unplayed_games"]
    if not request["expected_updated_at"]:
        raise LeagueLivePublishError("expected_updated_at is required; reload the session before publish.")
    if len(str(request["expected_operation_key"])) != 64:
        raise LeagueLivePublishError("A verified 64-character Python plan operation key is required.")
    unusual_scores = [
        {
            "match_index": index,
            "court": row["court"],
            "score_t1": row["score_t1"],
            "score_t2": row["score_t2"],
            "reason": reason,
        }
        for index, row in enumerate(normalized_matches, start=1)
        if (reason := league_live_unusual_score_reason(row["score_t1"], row["score_t2"])) is not None
    ]
    if unusual_scores and not unusual_score_acknowledgement:
        first = unusual_scores[0]
        raise LeagueLivePublishError(
            f"Court {first['court']} has an unusual score {first['score_t1']}-{first['score_t2']}; "
            "review it and explicitly acknowledge unusual scores before publishing."
        )
    if unusual_scores or unusual_score_acknowledgement:
        request["unusual_score_acknowledgement"] = bool(unusual_score_acknowledgement)
        request["unusual_score_findings"] = unusual_scores
    request["match_context_ids"] = [row["context_id"] for row in normalized_matches]
    request["request_fingerprint"] = stable_payload_fingerprint(request)
    return request


def build_rating_review(
    *,
    before_rows: Iterable[dict[str, Any]] | None,
    after_rows: Iterable[dict[str, Any]] | None,
    expected_player_ids: Iterable[int],
    published_match_count: int,
) -> dict[str, Any]:
    before = {int(row["id"]): dict(row) for row in (before_rows or []) if _safe_int(row.get("id")) is not None}
    after = {int(row["id"]): dict(row) for row in (after_rows or []) if _safe_int(row.get("id")) is not None}
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for player_id in sorted({int(value) for value in expected_player_ids}):
        old = before.get(player_id, {})
        new = after.get(player_id, {})
        rating_before = old.get("rating")
        rating_after = new.get("rating")
        delta: float | None = None
        try:
            if rating_before is not None and rating_after is not None:
                delta = float(rating_after) - float(rating_before)
        except Exception:
            delta = None
        if not new:
            warnings.append(f"Player {player_id} could not be read back after publish.")
        rows.append(
            {
                "player_id": player_id,
                "player_name": new.get("name") or old.get("name") or f"Player {player_id}",
                "rating_before": rating_before,
                "rating_after": rating_after,
                "rating_delta": delta,
                "matches_played_before": old.get("matches_played"),
                "matches_played_after": new.get("matches_played"),
            }
        )
    return {
        "status": "review_required" if warnings else "verified_readback",
        "published_match_count": int(published_match_count),
        "affected_player_count": len(rows),
        "requires_replay_review": bool(warnings),
        "rows": rows,
        "warnings": warnings,
        "recovery": {"match_log": "/admin/match-log", "replay_history": "/admin/replay-history"},
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    text = "" if value is None else str(value)
    if text.startswith(("=", "+", "-", "@")):
        return f"'{text}"
    return text


def rows_to_safe_csv(rows: Iterable[dict[str, Any]] | None) -> str:
    materialized = [dict(row) for row in (rows or []) if isinstance(row, dict)]
    if not materialized:
        return ""
    columns: list[str] = []
    for row in materialized:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in materialized:
        writer.writerow({key: _csv_value(row.get(key)) for key in columns})
    return output.getvalue()
