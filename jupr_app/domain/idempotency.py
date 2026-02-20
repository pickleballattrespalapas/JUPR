from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, TypedDict


class CanonicalMatch(TypedDict, total=False):
    club_id: str
    date: str
    context_type: str
    context_id: str
    competition_id: str
    division_id: str
    tournament_id: str
    tournament_game_id: str
    match_type: str
    match_format: str
    best_of: int
    t1_p1: int
    t1_p2: int
    t2_p1: int
    t2_p2: int
    score_t1: int
    score_t2: int
    score_json: Any
    games: Any


def _normalize_utc_iso(raw_value: Any) -> str:
    if raw_value is None:
        raise ValueError("match_datetime (date) is required for idempotency")

    text = str(raw_value).strip()
    if not text:
        raise ValueError("match_datetime (date) is required for idempotency")

    normalized = text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _canonical_team_side(player_ids: list[Any]) -> list[int]:
    parsed = sorted(int(pid) for pid in player_ids if pid is not None)
    if not parsed:
        raise ValueError("match team players are required for idempotency")
    return parsed


def _canonical_scoreline(match: CanonicalMatch) -> dict[str, Any]:
    if match.get("games") is not None:
        return {"games": match.get("games")}
    if match.get("score_json") is not None:
        return {"score_json": match.get("score_json")}
    return {
        "score_t1": int(match.get("score_t1") or 0),
        "score_t2": int(match.get("score_t2") or 0),
    }


def build_match_idempotency_key_v1(match: CanonicalMatch) -> str:
    canonical_payload = {
        "club_id": str(match.get("club_id") or "").strip(),
        "match_datetime_utc": _normalize_utc_iso(match.get("date")),
        "context": {
            "context_type": str(match.get("context_type") or "").strip().lower(),
            "context_id": str(match.get("context_id") or "").strip(),
            "competition_id": str(match.get("competition_id") or "").strip(),
            "division_id": str(match.get("division_id") or "").strip(),
            "tournament_id": str(match.get("tournament_id") or "").strip(),
            "tournament_game_id": str(match.get("tournament_game_id") or "").strip(),
        },
        "teams": {
            "team_1": _canonical_team_side([match.get("t1_p1"), match.get("t1_p2")]),
            "team_2": _canonical_team_side([match.get("t2_p1"), match.get("t2_p2")]),
        },
        "scoreline": _canonical_scoreline(match),
        "format": {
            "match_type": str(match.get("match_type") or "").strip(),
            "match_format": str(match.get("match_format") or "").strip(),
            "best_of": int(match.get("best_of") or 0),
        },
    }

    if not canonical_payload["club_id"]:
        raise ValueError("club_id is required for idempotency")

    normalized = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"), default=str)
    return f"match:v1:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"
