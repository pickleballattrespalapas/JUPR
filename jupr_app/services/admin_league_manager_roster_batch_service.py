from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any
from uuid import uuid4

CONFIRM_ROSTER_BATCH = "SAVE LEAGUE ROSTER BATCH"
IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$")


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload(response: Any) -> dict[str, Any]:
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return dict(data[0])
    return {}


def update_admin_league_manager_roster_batch(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    action: str,
    player_ids: list[int],
    starting_rating: float | None,
    idempotency_key: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_league_manager_bulk_roster_editor",
) -> dict[str, Any]:
    if os.getenv("JUPR_ENV", "").strip().lower() == "production":
        raise PermissionError(
            "League roster batches are staging-only and disabled in production."
        )
    if str(confirmation_text or "").strip() != CONFIRM_ROSTER_BATCH:
        raise ValueError(f"Type {CONFIRM_ROSTER_BATCH} to update the roster.")
    clean_action = str(action or "").strip().lower()
    if clean_action not in {"activate", "deactivate"}:
        raise ValueError("Roster action must be activate or deactivate.")
    clean_ids = sorted({int(player_id) for player_id in player_ids})
    if len(clean_ids) != len(player_ids) or not 1 <= len(clean_ids) <= 500:
        raise ValueError("Select 1–500 distinct club players.")
    if any(player_id <= 0 for player_id in clean_ids):
        raise ValueError("Every player ID must be positive.")
    key = str(idempotency_key or "").strip()
    if not IDEMPOTENCY_KEY_RE.fullmatch(key):
        raise ValueError("A valid 8–160 character idempotency key is required.")
    rating = None if starting_rating is None else float(starting_rating)
    if rating is not None and not (
        1.0 <= rating <= 7.0 or 400.0 <= rating <= 2800.0
    ):
        raise ValueError("Starting rating must be JUPR 1.0–7.0 or Elo 400–2800.")
    request = {
        "league_name": str(league_name).strip(),
        "action": clean_action,
        "player_ids": clean_ids,
        "starting_rating": rating,
    }
    try:
        response = supabase.rpc(
            "admin_apply_league_roster_batch_atomic_v3",
            {
                "p_operation_id": str(uuid4()),
                "p_club_id": str(club_id),
                "p_league_name": request["league_name"],
                "p_idempotency_key": key,
                "p_request_fingerprint": _fingerprint(request),
                "p_action": clean_action,
                "p_player_ids": clean_ids,
                "p_starting_rating": rating,
                "p_actor_email": str(actor_email or "").strip(),
                "p_actor_role": str(actor_role or "").strip(),
                "p_source": str(source or "").strip(),
            },
        ).execute()
    except Exception as exc:
        detail = str(exc)
        if (
            "IDEMPOTENCY_CONFLICT" in detail
            or "CONCURRENT_CONFLICT" in detail
            or "WRITE_CONFLICT" in detail
        ):
            raise RuntimeError(
                "Roster data changed. Reload before another bulk update."
            ) from exc
        if "REPLAY_IN_PROGRESS" in detail:
            raise RuntimeError(
                "Replay History is rebuilding this club. Wait for it to finish."
            ) from exc
        if "LEAGUE_NOT_FOUND" in detail:
            raise ValueError("league not found") from exc
        if "LIFECYCLE_INVALID" in detail:
            raise ValueError(
                "League lifecycle state is inconsistent; reload before editing the roster."
            ) from exc
        if "READ_ONLY" in detail:
            raise ValueError(
                "League roster is read-only unless the league is draft or active."
            ) from exc
        if "PLAYER_NOT_FOUND" in detail:
            raise ValueError("Every selected player must belong to this club.") from exc
        if "PLAYER_INACTIVE" in detail:
            raise ValueError(
                "Inactive club players cannot be added to a league."
            ) from exc
        if "ALREADY_ACTIVE" in detail:
            raise ValueError(
                "At least one selected player is already active in this league."
            ) from exc
        if "NOT_ACTIVE" in detail:
            raise ValueError(
                "Every selected player must currently be active in this league."
            ) from exc
        if "RATING_INVALID" in detail:
            raise ValueError(
                "Every new league member needs a valid starting rating."
            ) from exc
        if "OVERALL_RATING_REQUIRED" in detail:
            raise ValueError(
                "Set a reviewed Overall JUPR in Player Editor before adding this player to a league."
            ) from exc
        if "FORMAT_INVALID" in detail:
            raise ValueError("League match format must be singles or doubles.") from exc
        raise
    result = _payload(response)
    if (
        not result
        or not bool(result.get("ok"))
        or not bool(result.get("committed"))
        or int(result.get("updated_count") or 0) != len(clean_ids)
    ):
        raise RuntimeError(
            "The roster batch returned no authoritative commit receipt."
        )
    return result
