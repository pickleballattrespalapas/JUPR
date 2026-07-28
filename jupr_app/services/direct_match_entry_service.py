from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any
from uuid import uuid4

from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.matches.side_effects import (
    queue_player_updates,
    run_badge_side_effects,
)
from jupr_app.domain.singles_match_processing import process_singles_matches

logger = logging.getLogger(__name__)

DIRECT_MATCH_RPC = "admin_apply_direct_match_entry_atomic_v1"
IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$")


class DirectMatchConflictError(RuntimeError):
    """The request was rejected before commit because durable state changed."""


class DirectMatchRecoveryRequiredError(RuntimeError):
    """The transport did not provide authoritative commit evidence."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def stable_direct_match_request(
    *,
    club_id: str,
    match_format: str,
    matches: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    request_json = _json_safe(
        {
            "contract_version": 1,
            "club_id": str(club_id),
            "match_format": str(match_format),
            "matches": list(matches),
        }
    )
    encoded = json.dumps(
        request_json,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return request_json, hashlib.sha256(encoded).hexdigest()


def _rpc_payload(response: Any) -> dict[str, Any]:
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return dict(data[0])
    return {}


def _rpc_error_text(exc: Exception) -> str:
    details = [str(exc)]
    args = getattr(exc, "args", ())
    if args and isinstance(args[0], dict):
        details.extend(str(value) for value in args[0].values())
    return " ".join(details)


def _existing_receipt(
    supabase: Any,
    *,
    club_id: str,
    idempotency_key: str,
    request_fingerprint: str,
    match_format: str,
) -> dict[str, Any] | None:
    """Return a completed receipt before rebuilding any rating projection."""

    try:
        response = (
            supabase.table("admin_direct_match_entry_operations")
            .select("request_fingerprint,match_format,result_json")
            .eq("club_id", str(club_id))
            .eq("idempotency_key", str(idempotency_key))
            .limit(1)
            .execute()
        )
        rows = list(getattr(response, "data", None) or [])
    except Exception as exc:  # noqa: BLE001 - the transactional RPC remains safe
        logger.info("Direct match receipt preflight was unavailable: %s", exc)
        return None
    if not rows:
        return None
    row = dict(rows[0])
    if (
        str(row.get("request_fingerprint") or "") != request_fingerprint
        or str(row.get("match_format") or "") != match_format
    ):
        raise DirectMatchConflictError(
            "This idempotency key already belongs to a different match "
            "request. Nothing was written."
        )
    result = dict(row.get("result_json") or {})
    if (
        not bool(result.get("ok"))
        or not bool(result.get("committed"))
        or str(result.get("request_fingerprint") or "")
        != request_fingerprint
    ):
        raise DirectMatchRecoveryRequiredError(
            "The stored operation receipt is incomplete. Stop and reconcile "
            "it before any new submission."
        )
    return {
        **result,
        "idempotent": True,
        "duplicate_request": False,
    }


def _apply_atomic_plan(
    supabase: Any,
    *,
    club_id: str,
    idempotency_key: str,
    request_fingerprint: str,
    match_format: str,
    source: str,
    actor_email: str,
    actor_role: str,
    request_json: dict[str, Any],
    result_summary: dict[str, Any],
    write_plan: dict[str, Any],
) -> dict[str, Any]:
    try:
        response = supabase.rpc(
            DIRECT_MATCH_RPC,
            {
                "p_operation_id": str(uuid4()),
                "p_club_id": str(club_id),
                "p_idempotency_key": str(idempotency_key),
                "p_request_fingerprint": str(request_fingerprint),
                "p_match_format": str(match_format),
                "p_source": str(source),
                "p_actor_email": str(actor_email or ""),
                "p_actor_role": str(actor_role or ""),
                "p_request_json": dict(request_json),
                "p_result_summary": dict(result_summary),
                "p_match_rows": list(write_plan.get("match_rows") or []),
                "p_player_updates": list(write_plan.get("player_updates") or []),
                "p_league_rating_updates": list(
                    write_plan.get("league_rating_updates") or []
                ),
                "p_league_metadata_expectations": list(
                    write_plan.get("league_metadata_expectations") or []
                ),
            },
        ).execute()
    except Exception as exc:
        detail = _rpc_error_text(exc)
        invalid_markers = (
            "JUPR_DIRECT_MATCH_PLAN_INVALID",
            "JUPR_DIRECT_MATCH_ROWS_INVALID",
            "JUPR_DIRECT_MATCH_PLAYER_PLAN_INVALID",
            "JUPR_DIRECT_MATCH_LEAGUE_PLAN_INVALID",
        )
        conflict_markers = (
            "JUPR_DIRECT_MATCH_IDEMPOTENCY_CONFLICT",
            "JUPR_DIRECT_MATCH_CONCURRENT_CONFLICT",
            "JUPR_DIRECT_MATCH_PLAYER_STALE",
            "JUPR_DIRECT_MATCH_LEAGUE_METADATA_STALE",
            "JUPR_DIRECT_MATCH_LEAGUE_RATING_STALE",
            "JUPR_DIRECT_MATCH_INSERT_INCOMPLETE",
            "JUPR_DIRECT_MATCH_PLAYER_WRITE_INCOMPLETE",
            "JUPR_DIRECT_MATCH_LEAGUE_RATING_WRITE_INCOMPLETE",
            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_RATING_LOCK",
        )
        if any(marker in detail for marker in invalid_markers):
            raise ValueError(
                "The match plan was rejected before any data was written."
            ) from exc
        if any(marker in detail for marker in conflict_markers):
            raise DirectMatchConflictError(
                "Match or rating data changed before commit. Nothing from this "
                "request was written; reload and submit again."
            ) from exc
        raise DirectMatchRecoveryRequiredError(
            "The server response was interrupted. Retry the exact unchanged "
            "request; its idempotency key prevents duplicate matches."
        ) from exc

    result = _rpc_payload(response)
    if (
        not result
        or not bool(result.get("ok"))
        or not bool(result.get("committed"))
        or str(result.get("request_fingerprint") or "") != request_fingerprint
    ):
        raise DirectMatchRecoveryRequiredError(
            "The server returned no authoritative commit receipt. Retry the "
            "exact unchanged request; its idempotency key is safe."
        )
    return result


def _frame_name_map(frame: Any) -> dict[int, str]:
    if frame is None or getattr(frame, "empty", True):
        return {}
    result: dict[int, str] = {}
    try:
        rows = frame.to_dict("records")
    except Exception:
        return result
    for row in rows:
        try:
            player_id = int(row.get("id"))
        except Exception:
            continue
        name = str(row.get("name") or "").strip()
        if name:
            result[player_id] = name
    return result


def _feedback_from_receipt(
    *,
    match_format: str,
    player_updates: list[dict[str, Any]],
    match_ids: list[Any],
    player_names: dict[int, str],
) -> dict[str, Any]:
    rating_key = "singles_rating" if match_format == "singles" else "rating"
    matches_key = (
        "singles_matches_played"
        if match_format == "singles"
        else "matches_played"
    )
    affected_players: list[dict[str, Any]] = []
    ratings_updated = False
    for update in player_updates:
        player_id = int(update.get("player_id"))
        expected = dict(update.get("expected") or {})
        after = dict(update.get("after") or {})
        before_rating = expected.get(rating_key)
        after_rating = after.get(rating_key)
        try:
            rating_delta = float(after_rating) - float(before_rating)
        except Exception:
            rating_delta = None
        if rating_delta not in (None, 0.0):
            ratings_updated = True
        affected_players.append(
            {
                "id": player_id,
                "name": player_names.get(player_id) or f"Player {player_id}",
                "rating_before": before_rating,
                "rating_after": after_rating,
                "rating_delta": rating_delta,
                "matches_played_before": expected.get(matches_key),
                "matches_played_after": after.get(matches_key),
            }
        )
    return {
        "ratings_updated": ratings_updated,
        "rating_type": "singles" if match_format == "singles" else "doubles",
        "affected_players": affected_players,
        "latest_match_id": match_ids[-1] if match_ids else None,
    }


def _post_commit_side_effects(
    supabase: Any,
    *,
    club_id: str,
    write_plan: dict[str, Any],
    side_effect_context: dict[str, Any],
    match_ids: list[Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    warnings: list[str] = []
    db_matches = list(write_plan.get("match_rows") or [])
    affected_players = {
        int(player_id)
        for player_id in (
            side_effect_context.get("affected_player_ids") or []
        )
    }
    try:
        badge_summary = run_badge_side_effects(
            supabase=supabase,
            club_id=str(club_id),
            has_badge_eligible_match=bool(
                side_effect_context.get("has_badge_eligible_match")
            ),
            affected_players=affected_players,
            db_matches=db_matches,
            match_payloads=list(
                side_effect_context.get("match_payloads") or []
            ),
            dedupe_match_id=str(match_ids[0]) if match_ids else None,
        )
    except Exception as exc:  # noqa: BLE001 - core is already committed
        logger.warning("Direct match badge handoff failed after commit: %s", exc)
        badge_summary = {"mode": "error", "error": type(exc).__name__}
        warnings.append(
            "Match data is committed, but badge processing needs a later retry."
        )

    try:
        player_update_queue = queue_player_updates(
            supabase=supabase,
            club_id=str(club_id),
            db_matches=db_matches,
            affected_players=affected_players,
            successful_match_dates=list(
                side_effect_context.get("successful_match_dates") or []
            ),
        )
    except Exception as exc:  # noqa: BLE001 - core is already committed
        logger.warning(
            "Direct match player-update queue handoff failed after commit: %s",
            exc,
        )
        player_update_queue = {"mode": "error", "error": type(exc).__name__}
        warnings.append(
            "Match data is committed, but player-update queueing needs a "
            "later retry."
        )
    return badge_summary, player_update_queue, warnings


def submit_atomic_direct_matches(
    supabase: Any,
    *,
    club_id: str,
    matches: list[dict[str, Any]],
    match_format: str,
    idempotency_key: str,
    actor_email: str,
    actor_role: str,
    source: str,
    name_to_id: dict[str, int],
    df_players_all: Any,
    df_leagues: Any = None,
    df_meta: Any = None,
    default_k_factor: int = 32,
    min_win_delta_elo: float = 1.0,
    cap_loser_gain_elo: float | None = 16.0,
) -> dict[str, Any]:
    clean_format = str(match_format or "").strip().lower()
    clean_idempotency_key = str(idempotency_key or "").strip()
    if clean_format not in {"doubles", "singles"}:
        raise ValueError("Match format must be doubles or singles.")
    if not IDEMPOTENCY_KEY_RE.fullmatch(clean_idempotency_key):
        raise ValueError(
            "A valid 8–160 character match idempotency key is required."
        )
    if not isinstance(matches, list) or not (1 <= len(matches) <= 200):
        raise ValueError("Submit between 1 and 200 match rows.")

    request_json, request_fingerprint = stable_direct_match_request(
        club_id=str(club_id),
        match_format=clean_format,
        matches=matches,
    )
    receipt = _existing_receipt(
        supabase,
        club_id=str(club_id),
        idempotency_key=clean_idempotency_key,
        request_fingerprint=request_fingerprint,
        match_format=clean_format,
    )
    calculated: dict[str, Any] = {}
    write_plan: dict[str, Any] = {}
    if receipt is not None:
        result_summary = dict(receipt.get("result_summary") or {})
    elif clean_format == "singles":
        calculated = process_singles_matches(
            matches,
            supabase=supabase,
            club_id=str(club_id),
            name_to_id=name_to_id,
            df_players_all=df_players_all,
            df_meta=df_meta,
            default_k_factor=default_k_factor,
            min_win_delta_elo=min_win_delta_elo,
            cap_loser_gain_elo=cap_loser_gain_elo,
            build_write_plan_only=True,
        )
    else:
        calculated = process_matches(
            matches,
            supabase=supabase,
            club_id=str(club_id),
            name_to_id=name_to_id,
            df_players_all=df_players_all,
            df_leagues=df_leagues,
            df_meta=df_meta,
            default_k_factor=default_k_factor,
            min_win_delta_elo=min_win_delta_elo,
            cap_loser_gain_elo=cap_loser_gain_elo,
            build_write_plan_only=True,
        )

    if receipt is None:
        write_plan = dict(calculated.get("write_plan") or {})
        match_rows = list(write_plan.get("match_rows") or [])
        if (
            int(calculated.get("inserted") or 0) != len(matches)
            or len(match_rows) != len(matches)
        ):
            raise ValueError(
                "Every submitted row must be a complete, non-tied doubles or "
                "singles match. Nothing was written."
            )
        result_summary = {
            str(key): _json_safe(value)
            for key, value in calculated.items()
            if key not in {"write_plan", "side_effect_context"}
        }
        receipt = _apply_atomic_plan(
            supabase,
            club_id=str(club_id),
            idempotency_key=clean_idempotency_key,
            request_fingerprint=request_fingerprint,
            match_format=clean_format,
            source=str(source),
            actor_email=str(actor_email or ""),
            actor_role=str(actor_role or ""),
            request_json=request_json,
            result_summary=result_summary,
            write_plan=write_plan,
        )

    stored_summary = dict(receipt.get("result_summary") or result_summary)
    stored_player_updates = list(receipt.get("player_updates") or [])
    match_ids = list(receipt.get("match_ids") or [])
    warnings: list[str] = []
    if bool(receipt.get("idempotent")):
        badge_summary = {"mode": "idempotent_retry_skipped"}
        player_update_queue = {"mode": "idempotent_retry_skipped"}
    else:
        (
            badge_summary,
            player_update_queue,
            post_commit_warnings,
        ) = _post_commit_side_effects(
            supabase,
            club_id=str(club_id),
            write_plan=write_plan,
            side_effect_context=dict(
                calculated.get("side_effect_context") or {}
            ),
            match_ids=match_ids,
        )
        warnings.extend(post_commit_warnings)

    result = {
        **stored_summary,
        "badge_summary": badge_summary,
        "player_update_queue": player_update_queue,
    }
    operation = {
        str(key): value
        for key, value in receipt.items()
        if key not in {"result_summary", "player_updates"}
    }
    return {
        "ok": True,
        "match_write_committed": True,
        "submitted_count": len(matches),
        "result": result,
        "feedback": _feedback_from_receipt(
            match_format=clean_format,
            player_updates=stored_player_updates,
            match_ids=match_ids,
            player_names=_frame_name_map(df_players_all),
        ),
        "operation": operation,
        "warnings": warnings,
    }
