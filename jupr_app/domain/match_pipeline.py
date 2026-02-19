from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from jupr_app.data.sb_write import sb_update, sb_upsert
from jupr_app.domain.ratings import calculate_hybrid_elo


def record_match(
    supabase,
    *,
    club_id: str,
    team_a_player_ids: list[int],
    team_b_player_ids: list[int],
    score_a: int,
    score_b: int,
    played_at: str | None = None,
    context_id: str | None = None,
    source: str = "manual",
) -> dict:
    if not str(club_id or "").strip():
        raise ValueError("club_id is required")

    if not team_a_player_ids or not team_b_player_ids:
        raise ValueError("team_a_player_ids and team_b_player_ids are required")

    club_id = str(club_id).strip()
    normalized_team_a = [int(pid) for pid in team_a_player_ids]
    normalized_team_b = [int(pid) for pid in team_b_player_ids]
    score_a = int(score_a)
    score_b = int(score_b)
    played_at_iso = _normalize_played_at(played_at)
    normalized_context_id = _normalize_optional_str(context_id)
    source = _normalize_optional_str(source) or "manual"

    all_player_ids = sorted(set(normalized_team_a + normalized_team_b))
    player_rows = _fetch_and_validate_players(supabase, club_id=club_id, player_ids=all_player_ids)

    idempotency_key = _build_idempotency_key(
        club_id=club_id,
        team_a_player_ids=normalized_team_a,
        team_b_player_ids=normalized_team_b,
        score_a=score_a,
        score_b=score_b,
        played_at=played_at_iso,
        context_id=normalized_context_id,
        source=source,
    )

    snapshot = _build_rating_snapshot(player_rows, normalized_team_a, normalized_team_b)
    rating_delta = _compute_rating_delta(
        snapshot=snapshot,
        team_a_player_ids=normalized_team_a,
        team_b_player_ids=normalized_team_b,
        score_a=score_a,
        score_b=score_b,
    )

    match_payload = {
        "club_id": club_id,
        "date": played_at_iso,
        "t1_p1": _nth_or_none(normalized_team_a, 0),
        "t1_p2": _nth_or_none(normalized_team_a, 1),
        "t2_p1": _nth_or_none(normalized_team_b, 0),
        "t2_p2": _nth_or_none(normalized_team_b, 1),
        "score_t1": score_a,
        "score_t2": score_b,
        "elo_delta_t1": float(rating_delta["team_a_delta"]),
        "elo_delta_t2": float(rating_delta["team_b_delta"]),
        "elo_delta": float(max(abs(rating_delta["team_a_delta"]), abs(rating_delta["team_b_delta"]))),
        "t1_p1_r": _snapshot_rating(snapshot, normalized_team_a, 0),
        "t1_p2_r": _snapshot_rating(snapshot, normalized_team_a, 1),
        "t2_p1_r": _snapshot_rating(snapshot, normalized_team_b, 0),
        "t2_p2_r": _snapshot_rating(snapshot, normalized_team_b, 1),
        "t1_p1_r_end": _end_rating(snapshot, normalized_team_a, rating_delta["team_a_delta"], 0),
        "t1_p2_r_end": _end_rating(snapshot, normalized_team_a, rating_delta["team_a_delta"], 1),
        "t2_p1_r_end": _end_rating(snapshot, normalized_team_b, rating_delta["team_b_delta"], 0),
        "t2_p2_r_end": _end_rating(snapshot, normalized_team_b, rating_delta["team_b_delta"], 1),
        "context_type": "league" if normalized_context_id else "admin",
        "context_id": normalized_context_id,
        "idempotency_key": idempotency_key,
        "match_type": source,
    }

    inserted_match_resp = (
        supabase.table("matches")
        .upsert(
            match_payload,
            on_conflict="idempotency_key",
            ignore_duplicates=True,
        )
        .execute()
    )
    inserted_rows = getattr(inserted_match_resp, "data", None) or []
    if not inserted_rows:
        existing_match = _fetch_existing_match(supabase, idempotency_key=idempotency_key)
        print("[PIPELINE] idempotent hit — skipping delta")
        return {
            "status": "exists",
            "idempotency_key": idempotency_key,
            "match": existing_match,
        }

    inserted_match = dict(inserted_rows[0])
    inserted_match_id = str(inserted_match.get("id") or idempotency_key)

    updates_by_player = _build_player_updates(
        snapshot=snapshot,
        team_a_player_ids=normalized_team_a,
        team_b_player_ids=normalized_team_b,
        team_a_delta=rating_delta["team_a_delta"],
        team_b_delta=rating_delta["team_b_delta"],
        played_at_iso=played_at_iso,
    )

    for player_id, update_payload in updates_by_player.items():
        result = sb_update(
            supabase,
            "players",
            update_payload,
            filters={"club_id": club_id, "id": int(player_id)},
        )
        if not (getattr(result, "data", None) or []):
            raise RuntimeError(f"Failed to update players row for player_id={player_id}")

    if normalized_context_id:
        for player_id, update_payload in updates_by_player.items():
            league_payload = {
                "club_id": club_id,
                "player_id": int(player_id),
                "league_name": normalized_context_id,
                "rating": float(update_payload["rating"]),
                "is_active": True,
                "inactive_at": None,
            }
            sb_upsert(
                supabase,
                "league_ratings",
                league_payload,
                conflict="club_id,player_id,league_name",
            )

    badge_payload = {
        "club_id": club_id,
        "context_id": normalized_context_id or "overall",
        "event_type": "match_recorded",
        "player_ids": all_player_ids,
        "match_id": inserted_match_id,
        "payload_json": {
            "source": source,
            "score_a": score_a,
            "score_b": score_b,
            "played_at": played_at_iso,
            "idempotency_key": idempotency_key,
        },
        "status": "pending",
    }
    badge_resp = sb_upsert(
        supabase,
        "badge_eval_queue",
        badge_payload,
        conflict="event_type,match_id",
    )
    if getattr(badge_resp, "data", None) is None:
        raise RuntimeError("Failed to enqueue badge_eval_queue row")

    return {
        "status": "inserted",
        "idempotency_key": idempotency_key,
        "match": inserted_match,
        "rating_delta": rating_delta,
    }


def update_match(*args: Any, **kwargs: Any) -> dict:
    raise NotImplementedError("update_match will be implemented in a future phase")


def void_match(*args: Any, **kwargs: Any) -> dict:
    raise NotImplementedError("void_match will be implemented in a future phase")


def _normalize_played_at(played_at: str | None) -> str:
    if played_at is None or str(played_at).strip() == "":
        return datetime.now(timezone.utc).isoformat()

    raw = str(played_at).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat()


def _normalize_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _fetch_existing_match(supabase, *, idempotency_key: str) -> dict[str, Any]:
    existing = (
        supabase.table("matches")
        .select("*")
        .eq("idempotency_key", idempotency_key)
        .limit(1)
        .execute()
    )
    existing_rows = getattr(existing, "data", None) or []
    if not existing_rows:
        raise RuntimeError("Failed to fetch existing idempotent match row")
    return dict(existing_rows[0])


def _fetch_and_validate_players(supabase, *, club_id: str, player_ids: list[int]) -> dict[int, dict[str, Any]]:
    resp = (
        supabase.table("players")
        .select("id,club_id,rating,last_game_at")
        .eq("club_id", club_id)
        .in_("id", player_ids)
        .execute()
    )
    rows = getattr(resp, "data", None) or []
    if len(rows) != len(player_ids):
        found = {int(row.get("id")) for row in rows if row.get("id") is not None}
        missing = [pid for pid in player_ids if int(pid) not in found]
        raise ValueError(f"Unknown or out-of-club player IDs: {missing}")

    validated: dict[int, dict[str, Any]] = {}
    for row in rows:
        pid = int(row["id"])
        if str(row.get("club_id") or "") != club_id:
            raise ValueError(f"player_id={pid} does not belong to club_id={club_id}")
        validated[pid] = dict(row)
    return validated


def _build_idempotency_key(
    *,
    club_id: str,
    team_a_player_ids: list[int],
    team_b_player_ids: list[int],
    score_a: int,
    score_b: int,
    played_at: str,
    context_id: str | None,
    source: str,
) -> str:
    payload = {
        "club_id": club_id,
        "team_a_player_ids": [int(pid) for pid in team_a_player_ids],
        "team_b_player_ids": [int(pid) for pid in team_b_player_ids],
        "score_a": int(score_a),
        "score_b": int(score_b),
        "played_at": played_at,
        "context_id": context_id or "",
        "source": source,
    }
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"canonical:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


def _build_rating_snapshot(
    players_by_id: dict[int, dict[str, Any]], team_a_player_ids: list[int], team_b_player_ids: list[int]
) -> dict[int, float]:
    snapshot: dict[int, float] = {}
    for pid in team_a_player_ids + team_b_player_ids:
        snapshot[int(pid)] = float(players_by_id[int(pid)].get("rating") or 1200.0)
    return snapshot


def _compute_rating_delta(
    *,
    snapshot: dict[int, float],
    team_a_player_ids: list[int],
    team_b_player_ids: list[int],
    score_a: int,
    score_b: int,
) -> dict[str, float]:
    team_a_avg = sum(snapshot[int(pid)] for pid in team_a_player_ids) / float(len(team_a_player_ids))
    team_b_avg = sum(snapshot[int(pid)] for pid in team_b_player_ids) / float(len(team_b_player_ids))
    delta_a, delta_b = calculate_hybrid_elo(team_a_avg, team_b_avg, int(score_a), int(score_b))
    return {"team_a_delta": float(delta_a), "team_b_delta": float(delta_b)}


def _build_player_updates(
    *,
    snapshot: dict[int, float],
    team_a_player_ids: list[int],
    team_b_player_ids: list[int],
    team_a_delta: float,
    team_b_delta: float,
    played_at_iso: str,
) -> dict[int, dict[str, Any]]:
    updates: dict[int, dict[str, Any]] = {}
    for pid in team_a_player_ids:
        updates[int(pid)] = {"rating": float(snapshot[int(pid)] + float(team_a_delta)), "last_game_at": played_at_iso}
    for pid in team_b_player_ids:
        updates[int(pid)] = {"rating": float(snapshot[int(pid)] + float(team_b_delta)), "last_game_at": played_at_iso}
    return updates


def _nth_or_none(values: list[int], index: int) -> int | None:
    if index < len(values):
        return int(values[index])
    return None


def _snapshot_rating(snapshot: dict[int, float], team: list[int], index: int) -> float | None:
    pid = _nth_or_none(team, index)
    if pid is None:
        return None
    return float(snapshot[int(pid)])


def _end_rating(snapshot: dict[int, float], team: list[int], delta: float, index: int) -> float | None:
    start = _snapshot_rating(snapshot, team, index)
    if start is None:
        return None
    return float(start + float(delta))
