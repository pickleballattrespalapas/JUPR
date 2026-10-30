"""Review unplayed starting estimates without replacing calculated ratings.

The database owns the evidence snapshot and compare-and-set confirmation. This
module only presents that snapshot and handles the optional CourtReserve text.
"""
from __future__ import annotations

import math
import re
from typing import Any

from jupr_app.domain.admin.roles import PERMISSION_MANAGE_PLAYERS, has_permission


class StartingRatingReviewRequired(ValueError):
    def __init__(self, review: dict[str, Any]):
        super().__init__("Review starting estimates for players with no rated history before starting rated play.")
        self.detail = {
            "code": "STARTING_RATING_REVIEW_REQUIRED",
            "message": str(self),
            "match_format": review["match_format"],
            "players": [row for row in review["players"] if row["review_required"]],
        }


class StartingRatingReviewConflict(ValueError):
    pass


def _format(value: str) -> str:
    if value not in {"doubles", "singles"}:
        raise ValueError("Choose singles or doubles for starting-rating review.")
    return value


def _ids(values: list[int]) -> list[int]:
    if len(values) > 500:
        raise ValueError("Review up to 500 players at once.")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError("Every selected player must have a positive whole-number ID.")
    return sorted(set(values))


def _requested(value: Any) -> tuple[str | None, float | None]:
    raw = " ".join(str(value or "").split())
    if len(raw) > 120:
        raise ValueError("Requested skill level must be 120 characters or fewer.")
    if not raw:
        return None, None
    # Keep ranges and descriptive answers as text for a human decision. Never
    # silently select an endpoint or average a requested skill range.
    number = float(raw) if re.fullmatch(r"[1-7](?:\.\d{1,4})?", raw) else None
    return raw, number if number is not None and 1 <= number <= 7 else None


def _payload(response: Any) -> dict[str, Any]:
    value = getattr(response, "data", None)
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict) or not isinstance(value.get("players"), list):
        raise RuntimeError("Starting-rating review returned no authoritative player snapshot.")
    return value


def _number(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _present(raw: dict[str, Any], match_format: str, request: Any = None) -> dict[str, Any]:
    player = raw["player"]
    confirmation = raw.get("confirmation") or {}
    requested, requested_jupr = _requested(request)
    rating = _number(player.get("rating"))
    starting = _number(player.get("starting_rating"))
    rated_matches = int(raw.get("rated_matches") or 0)
    has_history = rated_matches > 0
    approved = _number(confirmation.get("rating_elo"))
    confirmed = (
        approved is not None and rating is not None and starting is not None
        and abs(approved - rating) < 0.00005 and abs(approved - starting) < 0.00005
    )
    requested_differs = bool(
        not has_history and requested and (
            requested_jupr is None or starting is None
            or abs(requested_jupr * 400 - starting) >= 0.00005
        )
    )
    # An administrator can deliberately choose a different seed from the
    # player's request. Remember the request that was considered, too.
    reviewed_request, reviewed_jupr = _requested(confirmation.get("requested_rating"))
    same_request = requested == reviewed_request or (
        requested_jupr is not None and reviewed_jupr is not None
        and abs(requested_jupr - reviewed_jupr) < 0.0000001
    )
    request_needs_review = requested_differs and not same_request
    can_edit = match_format == "doubles" and not has_history
    return {
        "player_id": int(player["id"]),
        "name": str(player.get("name") or ""),
        "rating_jupr": rating / 400 if rating is not None else None,
        "starting_jupr": starting / 400 if starting is not None else None,
        "rated_matches": rated_matches,
        "recorded_rated_matches": int(raw.get("recorded_rated_matches") or 0),
        "has_rated_history": has_history,
        "review_required": not has_history and (not confirmed or request_needs_review),
        "requested_rating": requested,
        "requested_jupr": requested_jupr,
        "requested_rating_differs": requested_differs,
        "state_fingerprint": str(raw["state_fingerprint"]),
        "confirmed_at": confirmation.get("confirmed_at") if confirmed else None,
        "can_edit_starting_rating": can_edit,
        "review_note": (
            "Uses calculated JUPR from rated history."
            if has_history else
            "Confirm the existing singles starting estimate. Singles replay baselines cannot be changed here."
            if match_format == "singles" else
            "Starting estimate only; no rated match history."
        ),
    }


def get_starting_rating_review(
    supabase: Any, *, club_id: str, player_ids: list[int],
    match_format: str = "doubles", requested_ratings: dict[Any, Any] | None = None,
) -> dict[str, Any]:
    format_name = _format(match_format)
    ids = _ids(player_ids)
    if not ids:
        return {"match_format": format_name, "players": [], "review_required": False}
    requests = {str(key): value for key, value in (requested_ratings or {}).items()}
    if any(key not in {str(pid) for pid in ids} for key in requests):
        raise ValueError("Requested ratings must belong to selected players.")
    for value in requests.values():
        _requested(value)
    try:
        result = _payload(supabase.rpc("preview_starting_rating_review_v1", {
            "p_club_id": str(club_id), "p_player_ids": ids, "p_match_format": format_name,
        }).execute())
    except Exception as exc:
        if "STARTING_REVIEW_PLAYER_NOT_FOUND" in str(exc):
            raise ValueError("Choose current player profiles from this club.") from exc
        raise
    rows = [_present(raw, format_name, requests.get(str(raw["player"]["id"]))) for raw in result["players"]]
    if sorted(row["player_id"] for row in rows) != ids:
        raise RuntimeError("Starting-rating review did not return every selected player.")
    return {"match_format": format_name, "players": rows, "review_required": any(row["review_required"] for row in rows)}


def require_starting_rating_review(supabase: Any, **kwargs: Any) -> dict[str, Any]:
    result = get_starting_rating_review(supabase, **kwargs)
    if result["review_required"]:
        raise StartingRatingReviewRequired(result)
    return result


def confirm_starting_ratings(
    supabase: Any, *, club_id: str, match_format: str, reviews: list[dict[str, Any]],
    actor_email: str, actor_role: str, source: str = "starting_rating_review",
) -> dict[str, Any]:
    format_name = _format(match_format)
    if not has_permission(actor_role, PERMISSION_MANAGE_PLAYERS):
        raise PermissionError("Only staff with player-management permission can confirm starting estimates.")
    if not str(actor_email or "").strip():
        raise PermissionError("An authenticated reviewer is required.")
    if not 1 <= len(reviews) <= 500:
        raise ValueError("Confirm 1–500 starting estimates at once.")
    ids = _ids([row.get("player_id") for row in reviews])
    if len(ids) != len(reviews):
        raise ValueError("Confirm each player only once.")
    clean = []
    requests = {}
    for row in reviews:
        value = row.get("rating_jupr")
        if isinstance(value, bool):
            raise ValueError("Starting JUPR must be between 1.0 and 7.0.")
        try:
            rating = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Starting JUPR must be between 1.0 and 7.0.") from exc
        if not math.isfinite(rating) or not 1 <= rating <= 7:
            raise ValueError("Starting JUPR must be between 1.0 and 7.0.")
        fingerprint = str(row.get("expected_state_fingerprint") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("Refresh the starting-rating review before saving.")
        requested, _ = _requested(row.get("requested_rating"))
        requests[str(row["player_id"])] = requested
        clean.append({"player_id": row["player_id"], "rating_elo": round(rating * 400, 4),
                      "expected_state_fingerprint": fingerprint, "requested_rating": requested})
    try:
        result = _payload(supabase.rpc("confirm_starting_rating_reviews_v1", {
            "p_club_id": str(club_id), "p_match_format": format_name, "p_reviews": clean,
            "p_actor_email": str(actor_email).strip(), "p_actor_role": str(actor_role),
            "p_source": str(source or "starting_rating_review")[:120],
        }).execute())
    except Exception as exc:
        detail = str(exc)
        error_code = str(getattr(exc, "code", ""))
        if error_code in {"55P03", "40P01", "57014"} or any(text in detail.lower() for text in ("lock timeout", "deadlock detected", "statement timeout")):
            raise StartingRatingReviewConflict("Scores are being saved right now. Refresh the starting-rating review and try again.") from exc
        if any(code in detail for code in ("STARTING_REVIEW_CONFLICT", "STARTING_REVIEW_HISTORY_EXISTS", "STARTING_REVIEW_REPLAY_ACTIVE")):
            raise StartingRatingReviewConflict("This player's rating or match history changed. Refresh the review before saving.") from exc
        if "STARTING_REVIEW_SINGLES_IMMUTABLE" in detail:
            raise ValueError("Singles replay baselines cannot be changed here. Confirm the existing singles estimate.") from exc
        if "STARTING_REVIEW_PLAYER_NOT_FOUND" in detail:
            raise ValueError("Choose current player profiles from this club.") from exc
        raise
    if result.get("ok") is not True or int(result.get("confirmed_count") or 0) != len(clean):
        raise RuntimeError("Starting-rating confirmation returned no complete commit receipt.")
    rows = [_present(raw, format_name, requests.get(str(raw["player"]["id"]))) for raw in result["players"]]
    if sorted(row["player_id"] for row in rows) != ids or any(row["review_required"] for row in rows):
        raise RuntimeError("Starting-rating confirmation did not verify every selected player.")
    return {"ok": True, "match_format": format_name, "players": rows, "review_required": False}
