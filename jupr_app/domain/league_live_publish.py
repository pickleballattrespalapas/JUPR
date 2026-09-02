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
) -> list[dict[str, Any]]:
    structure = normalize_league_match_structure(match_structure)
    raw_rows = validate_league_series_matches(matches, match_structure=structure)
    expected = int(expected_match_count or 0)
    if expected < 1:
        raise LeagueLivePublishError("expected_match_count must include every generated match slot.")
    if len(raw_rows) != expected:
        raise LeagueLivePublishError(
            f"All generated matches must be scored before publish ({len(raw_rows)} of {expected} supplied)."
        )
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
