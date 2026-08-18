from __future__ import annotations

import os
from typing import Any

from jupr_app.data.load import load_data
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.events import upsert_or_get_active_event
from jupr_app.domain.schedule import (
    EXPECTED_DOUBLES_GAMES_BY_FORMAT,
    SCHEDULE_MODE_FULL,
    SUPPORTED_DOUBLES_FORMAT_TYPES,
    get_match_schedule,
)
from jupr_app.services.direct_match_entry_service import (
    submit_atomic_direct_matches,
)
from jupr_app.services.admin_guarded_write_service import (
    GuardedWriteRecoveryRequired,
    begin_guarded_operation,
    canonical_fingerprint,
    get_guarded_operation,
    operation_result,
    update_guarded_operation,
)

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
MAX_MATCH_UPLOADER_BATCH_ROWS = 200
MAX_MATCH_UPLOADER_RR_COURTS = 10
DEFAULT_NEW_PLAYER_JUPR = 3.5


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def is_admin_match_uploader_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER")


def is_admin_match_uploader_singles_enabled() -> bool:
    """Expose direct singles only when its reviewed atomic write gate is open."""
    return is_admin_match_uploader_enabled() and _truthy_env(
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES"
    )


def is_admin_match_uploader_preview_enabled() -> bool:
    """Allow the read-only round-robin planner without opening uploader writes."""
    return is_admin_match_uploader_enabled() or _truthy_env(
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_PREVIEW"
    )


def is_api_audit_log_required() -> bool:
    return _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG")


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _normalize_name(value: Any) -> str:
    return " ".join(str(value or "").replace("\u00A0", " ").split()).strip()


def _reviewed_name_key(value: Any) -> str:
    """Canonical browser-compatible membership key for reviewed batches."""
    return _normalize_name(value).lower()


def _normalized_new_player_batch(players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requested: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for row in players or []:
        if not isinstance(row, dict):
            continue
        name = _normalize_name(row.get("name"))
        if not name:
            raise ValueError("New player name is required.")
        # Keep reviewed-batch identity aligned with browser
        # ``toLocaleLowerCase("en-US")``. Python ``casefold`` is broader (for
        # example Straße == STRASSE) and would change list membership/hash.
        name_key = _reviewed_name_key(name)
        if name_key in seen_names:
            continue
        requested.append(
            {
                "name": name,
                "starting_jupr": _coerce_starting_jupr(row.get("starting_jupr")),
            }
        )
        seen_names.add(name_key)
    if not requested:
        raise ValueError("Provide at least one new player to create.")
    return requested


def match_uploader_player_batch_fingerprint(players: list[dict[str, Any]]) -> str:
    """Return the cross-runtime reviewed fingerprint for a normalized player batch.

    Ratings use four fixed decimal places so Python and browser JSON encoders do
    not disagree about integer-valued floats (for example, 3.0 versus 3).
    """
    requested = _normalized_new_player_batch(players)
    review_payload = {
        "players": [
            {
                "name": item["name"],
                "starting_jupr": f"{float(item['starting_jupr']):.4f}",
            }
            for item in requested
        ]
    }
    return canonical_fingerprint(review_payload)


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _score_entry_player_ids(matches: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for match in matches or []:
        for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            pid = _safe_int(match.get(key))
            if pid is not None and int(pid) not in ids:
                ids.append(int(pid))
    return ids


def _fetch_players(supabase: Any, *, club_id: str, player_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not player_ids:
        return {}
    try:
        rows = _safe_rows(supabase.table("players").select("id,name,rating,wins,losses,matches_played").eq("club_id", str(club_id)).execute())
    except Exception:
        return {}
    allowed = {int(pid) for pid in player_ids}
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        pid = _safe_int(row.get("id"))
        if pid is not None and int(pid) in allowed:
            result[int(pid)] = dict(row)
    return result


def _fetch_all_players(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("players")
            .select("id,name,rating,wins,losses,matches_played,active")
            .eq("club_id", str(club_id))
            .execute()
        )
    except Exception:
        return []
    players: list[dict[str, Any]] = []
    for row in rows:
        pid = _safe_int(row.get("id"))
        name = _normalize_name(row.get("name"))
        if pid is None or not name:
            continue
        players.append(
            {
                "id": int(pid),
                "club_id": str(row.get("club_id") or club_id),
                "name": name,
                "rating": row.get("rating"),
                "wins": row.get("wins"),
                "losses": row.get("losses"),
                "matches_played": row.get("matches_played"),
                "is_active": row.get("active", row.get("is_active", True)),
            }
        )
    return sorted(players, key=lambda row: str(row.get("name") or "").lower())


def _fetch_all_players_for_guarded_write(
    supabase: Any,
    *,
    club_id: str,
) -> list[dict[str, Any]]:
    """Read players without converting transport/readback errors into an empty club."""
    rows = _safe_rows(
        supabase.table("players")
        .select("id,name,rating,starting_rating,wins,losses,matches_played,active,club_id")
        .eq("club_id", str(club_id))
        .execute()
    )
    players: list[dict[str, Any]] = []
    for row in rows:
        pid = _safe_int(row.get("id"))
        name = _normalize_name(row.get("name"))
        if pid is None or not name:
            continue
        players.append(
            {
                "id": int(pid),
                "club_id": str(row.get("club_id") or club_id),
                "name": name,
                "rating": row.get("rating"),
                "starting_rating": row.get("starting_rating"),
                "wins": row.get("wins"),
                "losses": row.get("losses"),
                "matches_played": row.get("matches_played"),
                "is_active": row.get("active", row.get("is_active", True)),
            }
        )
    return sorted(players, key=lambda row: str(row.get("name") or "").casefold())


def get_admin_match_uploader_player_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_key: str,
) -> dict[str, Any] | None:
    return get_guarded_operation(
        supabase,
        club_id=str(club_id),
        workflow="match_uploader_player_batch",
        operation_key=str(operation_key),
    )


def _player_batch_reconcile_evidence(
    operation: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before_json = operation.get("before_json") or {}
    result_json = operation.get("result_json") or {}
    reviewed = before_json.get("reviewed_players") if isinstance(before_json, dict) else None
    preflight = result_json.get("preflight") if isinstance(result_json, dict) else None
    if not isinstance(preflight, dict) and isinstance(before_json, dict):
        preflight = before_json.get("preflight")
    if not isinstance(reviewed, list) or not reviewed or not isinstance(preflight, dict):
        raise ValueError(
            "This player-batch operation predates proof-based reconciliation. Inspect Player Editor manually."
        )
    return _normalized_new_player_batch(reviewed), dict(preflight)


def reconcile_admin_match_uploader_player_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_key: str,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    source: str = "next_match_uploader_player_reconcile",
) -> dict[str, Any]:
    if not is_admin_match_uploader_enabled():
        raise PermissionError("Next Match Uploader is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != "RECONCILE PLAYER BATCH":
        raise ValueError("Type RECONCILE PLAYER BATCH to reconcile this exact operation.")
    operation = get_admin_match_uploader_player_operation(
        supabase,
        club_id=str(club_id),
        operation_key=str(operation_key),
    )
    if operation is None:
        raise ValueError("Player batch operation was not found.")
    status = str(operation.get("status") or "")
    if status == "completed":
        return operation_result(operation)
    if status not in {"intent_recorded", "recovery_required"}:
        raise ValueError(f"Player batch operation is {status or 'unknown'} and cannot be reconciled.")

    requested, preflight = _player_batch_reconcile_evidence(operation)
    to_create = preflight.get("to_create")
    preexisting = preflight.get("preexisting")
    if not isinstance(to_create, list) or not isinstance(preexisting, list):
        raise ValueError("Player batch preflight evidence is incomplete.")
    to_create_by_key = {
        _reviewed_name_key(item.get("name")): item
        for item in to_create
        if isinstance(item, dict)
    }
    preexisting_by_key = {
        _reviewed_name_key(item.get("name")): item
        for item in preexisting
        if isinstance(item, dict)
    }
    current_players = _fetch_all_players_for_guarded_write(
        supabase,
        club_id=str(club_id),
    )
    current_by_key: dict[str, list[dict[str, Any]]] = {}
    for player in current_players:
        current_by_key.setdefault(_reviewed_name_key(player.get("name")), []).append(player)

    proven: list[dict[str, Any]] = []
    absent_created: list[str] = []
    ambiguous: list[str] = []
    for item in requested:
        key = _reviewed_name_key(item.get("name"))
        candidates = current_by_key.get(key, [])
        if len(candidates) != 1:
            if not candidates and key in to_create_by_key:
                absent_created.append(str(item.get("name") or ""))
            else:
                ambiguous.append(str(item.get("name") or ""))
            continue
        player = candidates[0]
        if key in preexisting_by_key:
            expected_id = _safe_int(preexisting_by_key[key].get("id"))
            if expected_id is None or _safe_int(player.get("id")) != expected_id:
                ambiguous.append(str(item.get("name") or ""))
                continue
        elif key in to_create_by_key:
            expected_elo = float(item["starting_jupr"]) * 400.0
            if (
                _safe_int(player.get("wins")) != 0
                or _safe_int(player.get("losses")) != 0
                or _safe_int(player.get("matches_played")) != 0
                or abs(float(player.get("rating") or 0) - expected_elo) > 1e-9
                or abs(float(player.get("starting_rating") or 0) - expected_elo) > 1e-9
                or player.get("is_active") is False
            ):
                ambiguous.append(str(item.get("name") or ""))
                continue
        else:
            ambiguous.append(str(item.get("name") or ""))
            continue
        proven.append(player)

    if (
        absent_created
        and not ambiguous
        and len(absent_created) == len(to_create_by_key)
        and len(proven) == len(preexisting_by_key)
    ):
        update_guarded_operation(
            supabase,
            operation_id=operation.get("id"),
            operation_key=str(operation_key),
            status="failed",
            result_json={
                "reconciled": True,
                "proof": "expected_created_players_absent",
                "absent_players": absent_created,
            },
            error_text="Authoritative readback proves the expected created player rows are absent.",
        )
        return {
            "ok": False,
            "mode": "match_uploader_player_batch_reconciled_failed",
            "operation_key": str(operation_key),
            "status": "failed",
            "absent_players": absent_created,
            "recovery_required": False,
        }
    if ambiguous or len(proven) != len(requested):
        raise GuardedWriteRecoveryRequired(
            str(operation_key),
            "Current player state does not exactly prove this batch outcome. Keep the operation blocked and inspect Player Editor.",
        )

    audit_write = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=str(actor_email or ""),
            actor_role=str(actor_role or ""),
            action_type="reconcile_match_uploader_player_batch",
            entity_type="players",
            entity_id="batch",
            before_json=operation.get("before_json"),
            after_json={
                "source_client": "fastapi/nextjs",
                "source_page": source,
                "operation_key": str(operation_key),
                "proof": "authoritative_player_readback",
                "players": [{"id": row.get("id"), "name": row.get("name")} for row in proven],
            },
            source_page=source,
            flagged_for_review=True,
        ),
    )
    if not audit_write.ok:
        _mark_player_batch_recovery(
            supabase,
            operation=operation,
            operation_key=str(operation_key),
            result_json={**(operation.get("result_json") or {}), "reconcile_audit_failed": True},
            error_text="Authoritative player proof succeeded, but reconciliation audit did not persist.",
        )
        raise GuardedWriteRecoveryRequired(
            str(operation_key),
            "Player proof succeeded, but the reconciliation audit is unavailable. Do not retry with a new key.",
        )
    calculated_fingerprint = match_uploader_player_batch_fingerprint(requested)
    result = {
        "ok": True,
        "mode": "match_uploader_new_players",
        "requested_count": len(requested),
        "accepted_count": len(proven),
        "created_count": len(to_create_by_key),
        "unchanged_count": len(preexisting_by_key),
        "reviewed_fingerprint": calculated_fingerprint,
        "operation_key": str(operation_key),
        "player_insert_atomic": True,
        "idempotent_replay": True,
        "reconciled": True,
        "players": proven,
        "recovery": {
            "player_editor": "/admin/players",
            "operator_rule": "This operation was finalized from authoritative player readback and reconciliation audit.",
        },
        "warnings": [],
    }
    update_guarded_operation(
        supabase,
        operation_id=operation.get("id"),
        operation_key=str(operation_key),
        status="completed",
        after_json={"players": proven, "reconciled": True},
        result_json=result,
    )
    return result


def _mark_player_batch_recovery(
    supabase: Any,
    *,
    operation: dict[str, Any],
    operation_key: str,
    result_json: Any = None,
    error_text: str,
) -> None:
    try:
        update_guarded_operation(
            supabase,
            operation_id=operation.get("id"),
            operation_key=str(operation_key),
            status="recovery_required",
            result_json=result_json,
            error_text=error_text,
        )
    except Exception:
        pass


def _latest_match_id(supabase: Any, *, club_id: str) -> Any:
    try:
        rows = _safe_rows(supabase.table("matches").select("id").eq("club_id", str(club_id)).order("date", desc=True).limit(1).execute())
        return rows[0].get("id") if rows else None
    except Exception:
        return None


def _score_feedback(*, before: dict[int, dict[str, Any]], after: dict[int, dict[str, Any]], player_ids: list[int], latest_match_id: Any = None) -> dict[str, Any]:
    affected = []
    ratings_updated = False
    for pid in player_ids:
        b = before.get(int(pid), {})
        a = after.get(int(pid), {})
        rb = b.get("rating")
        ra = a.get("rating")
        try:
            delta = None if rb is None or ra is None else float(ra) - float(rb)
        except Exception:
            delta = None
        if delta not in (None, 0):
            ratings_updated = True
        affected.append(
            {
                "id": int(pid),
                "name": a.get("name") or b.get("name") or f"Player {int(pid)}",
                "rating_before": rb,
                "rating_after": ra,
                "rating_delta": delta,
                "matches_played_before": b.get("matches_played"),
                "matches_played_after": a.get("matches_played"),
            }
        )
    return {"ratings_updated": ratings_updated, "affected_players": affected, "latest_match_id": latest_match_id}


def _normalize_match(row: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    pids = {key: _safe_int(row.get(key)) for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")}
    if any(value is None for value in pids.values()):
        return None
    score_t1 = _safe_int(row.get("score_t1", row.get("s1"))) or 0
    score_t2 = _safe_int(row.get("score_t2", row.get("s2"))) or 0
    if (score_t1 + score_t2) <= 0:
        return None
    match_type = _clean_text(row.get("match_type") or "Live Match", limit=80)
    league = _clean_text(row.get("league") or ("POPUP" if match_type == "PopUp" else "Open"), limit=120)
    payload = {
        "date": _clean_text(row.get("date"), limit=80) or None,
        "league": league,
        "match_type": match_type,
        "week_tag": _clean_text(row.get("week_tag"), limit=80),
        "t1_p1": int(pids["t1_p1"] or 0),
        "t1_p2": int(pids["t1_p2"] or 0),
        "t2_p1": int(pids["t2_p1"] or 0),
        "t2_p2": int(pids["t2_p2"] or 0),
        "score_t1": int(score_t1),
        "score_t2": int(score_t2),
        "is_popup": bool(row.get("is_popup") or match_type == "PopUp"),
        "context_type": _clean_text(row.get("context_type"), limit=80) or None,
        "context_id": row.get("context_id"),
    }
    rating_scope = _clean_text(row.get("rating_scope"), limit=40)
    if rating_scope:
        payload["rating_scope"] = rating_scope
    context_name = _clean_text(row.get("context_name") or row.get("event_name"), limit=160)
    if context_name:
        payload["context_name"] = context_name
    return payload


def _normalize_batch(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = [_normalize_match(row) for row in (matches or [])]
    clean_rows = [row for row in clean if row]
    if not clean_rows:
        raise ValueError("No valid match rows were provided.")
    if len(clean_rows) > MAX_MATCH_UPLOADER_BATCH_ROWS:
        raise ValueError(f"No more than {MAX_MATCH_UPLOADER_BATCH_ROWS} matches can be submitted at once.")
    return clean_rows


def _apply_event_contexts(supabase: Any, *, club_id: str, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    event_ids_by_name: dict[str, str] = {}
    hydrated: list[dict[str, Any]] = []
    for row in matches:
        clean = dict(row)
        context_name = _clean_text(clean.pop("context_name", None), limit=160)
        is_popup_event = bool(clean.get("is_popup") or clean.get("match_type") == "PopUp" or clean.get("context_type") == "event")
        if is_popup_event and context_name and not clean.get("context_id"):
            if context_name not in event_ids_by_name:
                event_ids_by_name[context_name] = upsert_or_get_active_event(
                    supabase,
                    club_id=str(club_id),
                    name=context_name,
                )
            clean["context_type"] = "event"
            clean["context_id"] = event_ids_by_name[context_name]
        hydrated.append(clean)
    return hydrated


def _round_robin_format_options() -> list[str]:
    return list(SUPPORTED_DOUBLES_FORMAT_TYPES)


def _round_robin_expected_games() -> dict[str, int]:
    return {str(key): int(value) for key, value in EXPECTED_DOUBLES_GAMES_BY_FORMAT.items()}


def _normalize_league_match_format(value: Any) -> str:
    return "singles" if str(value or "").strip().casefold() == "singles" else "doubles"


def build_admin_match_uploader_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    round_robin_formats = _round_robin_format_options()
    round_robin_expected_games = _round_robin_expected_games()
    if not is_admin_match_uploader_enabled():
        return {
            "enabled": False,
            "singles_write_enabled": False,
            "status": "guarded_off",
            "submit_endpoint": None,
            "singles_submit_endpoint": None,
            "max_batch_rows": MAX_MATCH_UPLOADER_BATCH_ROWS,
            "league_options": ["Open", "POPUP"],
            "doubles_league_options": ["Open"],
            "singles_league_options": [],
            "week_tag_options": [f"Week {idx}" for idx in range(1, 13)] + ["Playoffs", "Finals", "Event"],
            "round_robin_format_options": round_robin_formats,
            "round_robin_expected_games": round_robin_expected_games,
            "warnings": ["Next Match Uploader is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER on FastAPI for the closed-club pilot."],
        }

    active_names: dict[str, list[str]] = {"doubles": [], "singles": []}
    try:
        try:
            rows = _safe_rows(
                supabase.table("leagues_metadata")
                .select("league_name,is_active,status,ended_at,match_format")
                .eq("club_id", str(club_id))
                .execute()
            )
        except Exception:
            rows = _safe_rows(
                supabase.table("leagues_metadata")
                .select("league_name,is_active,status,ended_at")
                .eq("club_id", str(club_id))
                .execute()
            )
        seen_names: dict[str, set[str]] = {"doubles": set(), "singles": set()}
        for row in rows:
            name = _clean_text(row.get("league_name"), limit=120)
            normalized_name = name.casefold()
            if not name or normalized_name in {"overall", "popup"}:
                continue
            ended_at = row.get("ended_at")
            if ended_at not in (None, "") and str(ended_at) not in {"<NA>", "NaT", "nan"}:
                continue
            status = str(row.get("status") or "").strip().casefold()
            if status in {"inactive", "disabled", "ended", "completed", "archived", "paused"}:
                continue
            is_active = row.get("is_active", True)
            if isinstance(is_active, str):
                is_active = is_active.strip().casefold() not in {"0", "false", "no", "off"}
            if not bool(is_active):
                continue
            match_format = _normalize_league_match_format(row.get("match_format"))
            if normalized_name in seen_names[match_format]:
                continue
            seen_names[match_format].add(normalized_name)
            active_names[match_format].append(name)
    except Exception:
        pass

    doubles_league_options = sorted(active_names["doubles"], key=str.casefold)
    singles_league_options = sorted(active_names["singles"], key=str.casefold)
    legacy_league_options = doubles_league_options + ["POPUP"]
    return {
        "enabled": True,
        "singles_write_enabled": is_admin_match_uploader_singles_enabled(),
        "status": "ready_for_manual_batch_and_round_robin",
        "submit_endpoint": "/admin/clubs/{club_id}/match-uploader/batch",
        "singles_submit_endpoint": (
            "/admin/clubs/{club_id}/match-uploader/singles"
            if is_admin_match_uploader_singles_enabled()
            else None
        ),
        "round_robin_preview_endpoint": "/admin/clubs/{club_id}/match-uploader/round-robin/preview",
        "player_create_endpoint": "/admin/clubs/{club_id}/match-uploader/players",
        "max_batch_rows": MAX_MATCH_UPLOADER_BATCH_ROWS,
        "league_options": legacy_league_options,
        "doubles_league_options": doubles_league_options,
        "singles_league_options": singles_league_options,
        "week_tag_options": [f"Week {idx}" for idx in range(1, 21)] + ["Playoffs", "Finals", "Event"],
        "round_robin_format_options": round_robin_formats,
        "round_robin_expected_games": round_robin_expected_games,
        "warnings": (
            []
            if is_admin_match_uploader_singles_enabled()
            else ["Direct singles submission is disabled for the current write wave."]
        ),
    }

def _court_player_names(court: dict[str, Any]) -> list[str]:
    raw_names = court.get("player_names")
    if isinstance(raw_names, list):
        values = raw_names
    else:
        names_text = court.get("names") or court.get("players_text") or ""
        values = str(names_text).replace("\n", ",").split(",")
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        name = _normalize_name(raw)
        if name and name not in seen:
            result.append(name)
            seen.add(name)
    return result


def _format_type(value: Any) -> str:
    clean = _clean_text(value, limit=40)
    if clean not in SUPPORTED_DOUBLES_FORMAT_TYPES:
        raise ValueError(f"Unsupported round-robin format: {clean or 'blank'}")
    return clean


def _build_player_lookups(players: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    exact_by_name: dict[str, dict[str, Any]] = {}
    normalized_by_name: dict[str, dict[str, Any]] = {}
    by_id: dict[int, dict[str, Any]] = {}
    for player in players:
        name = _normalize_name(player.get("name"))
        pid = _safe_int(player.get("id"))
        if not name or pid is None:
            continue
        clean = {**player, "id": int(pid), "name": name}
        exact_by_name.setdefault(name, clean)
        normalized_by_name.setdefault(_normalize_name(name), clean)
        by_id[int(pid)] = clean
    return exact_by_name, normalized_by_name, by_id


def _resolve_player(name: str, exact_by_name: dict[str, dict[str, Any]], normalized_by_name: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if name in exact_by_name:
        return exact_by_name[name]
    return normalized_by_name.get(_normalize_name(name))


def _round_robin_player_payload(player: dict[str, Any] | None, fallback_id: Any) -> dict[str, Any]:
    pid = _safe_int(player.get("id") if player else fallback_id)
    return {
        "id": int(pid or 0),
        "name": str((player or {}).get("name") or f"Player {pid or fallback_id}"),
        "rating": (player or {}).get("rating"),
    }


def build_admin_match_uploader_round_robin_preview(
    supabase: Any,
    *,
    club_id: str,
    courts: list[dict[str, Any]],
    custom_schedule: str = "",
    schedule_mode: str = SCHEDULE_MODE_FULL,
    source: str = "next_match_uploader_round_robin_preview",
) -> dict[str, Any]:
    if not is_admin_match_uploader_preview_enabled():
        raise PermissionError("Next Match Uploader preview is disabled.")
    if not courts:
        raise ValueError("Add at least one round-robin court.")
    if len(courts) > MAX_MATCH_UPLOADER_RR_COURTS:
        raise ValueError(f"No more than {MAX_MATCH_UPLOADER_RR_COURTS} round-robin courts can be generated at once.")

    players = _fetch_all_players(supabase, club_id=str(club_id))
    exact_by_name, normalized_by_name, by_id = _build_player_lookups(players)
    prepared_courts: list[dict[str, Any]] = []
    missing_names: list[str] = []
    seen_missing: set[str] = set()

    for index, court in enumerate(courts, start=1):
        format_type = _format_type(court.get("format_type"))
        player_names = _court_player_names(court)
        if not player_names:
            raise ValueError(f"Court {index}: enter player names before generating a schedule.")
        try:
            needed = int(format_type.split("-", 1)[0])
        except Exception:
            needed = len(player_names)
        if len(player_names) < needed:
            raise ValueError(f"Court {index}: {format_type} requires {needed} players.")
        for name in player_names:
            if _resolve_player(name, exact_by_name, normalized_by_name) is None and name not in seen_missing:
                missing_names.append(name)
                seen_missing.add(name)
        prepared_courts.append(
            {
                "court": _safe_int(court.get("court")) or index,
                "format_type": format_type,
                "player_names": player_names,
                "expected_games": EXPECTED_DOUBLES_GAMES_BY_FORMAT.get(format_type),
            }
        )

    if missing_names:
        return {
            "ok": True,
            "mode": "round_robin_preview",
            "source": source,
            "missing_players": sorted(missing_names),
            "courts": [],
            "match_count": 0,
        }

    response_courts: list[dict[str, Any]] = []
    match_count = 0
    schedule_mode = _clean_text(schedule_mode, limit=80) or SCHEDULE_MODE_FULL
    for prepared in prepared_courts:
        resolved_players = [
            _resolve_player(name, exact_by_name, normalized_by_name)
            for name in prepared["player_names"]
        ]
        player_ids = [int(player["id"]) for player in resolved_players if player is not None]
        schedule = get_match_schedule(
            prepared["format_type"],
            player_ids,
            custom_text=custom_schedule,
            schedule_mode=schedule_mode,
        )
        if not schedule:
            raise ValueError(f"Court {prepared['court']}: unable to generate a schedule for {prepared['format_type']}.")
        matches: list[dict[str, Any]] = []
        for match_index, match in enumerate(schedule, start=1):
            t1_ids = [_safe_int(value) for value in (match.get("t1") or [])]
            t2_ids = [_safe_int(value) for value in (match.get("t2") or [])]
            if len(t1_ids) != 2 or len(t2_ids) != 2 or any(value is None for value in [*t1_ids, *t2_ids]):
                continue
            t1_p1, t1_p2 = int(t1_ids[0] or 0), int(t1_ids[1] or 0)
            t2_p1, t2_p2 = int(t2_ids[0] or 0), int(t2_ids[1] or 0)
            matches.append(
                {
                    "row_id": f"rr-{prepared['court']}-{match_index}",
                    "court": prepared["court"],
                    "match_index": match_index,
                    "label": _clean_text(match.get("desc"), limit=120) or f"Game {match_index}",
                    "t1": [_round_robin_player_payload(by_id.get(t1_p1), t1_p1), _round_robin_player_payload(by_id.get(t1_p2), t1_p2)],
                    "t2": [_round_robin_player_payload(by_id.get(t2_p1), t2_p1), _round_robin_player_payload(by_id.get(t2_p2), t2_p2)],
                    "t1_p1": t1_p1,
                    "t1_p2": t1_p2,
                    "t2_p1": t2_p1,
                    "t2_p2": t2_p2,
                }
            )
        match_count += len(matches)
        response_courts.append(
            {
                "court": prepared["court"],
                "format_type": prepared["format_type"],
                "expected_games": prepared.get("expected_games"),
                "player_names": prepared["player_names"],
                "matches": matches,
            }
        )

    return {
        "ok": True,
        "mode": "round_robin_preview",
        "source": source,
        "missing_players": [],
        "courts": response_courts,
        "match_count": match_count,
    }


def _coerce_starting_jupr(value: Any) -> float:
    if value in (None, ""):
        return DEFAULT_NEW_PLAYER_JUPR
    try:
        rating = float(value)
    except Exception as exc:
        raise ValueError("Starting JUPR must be a number.") from exc
    if rating < 1.0 or rating > 7.0:
        raise ValueError("Starting JUPR must be between 1.0 and 7.0.")
    if abs(rating - round(rating, 4)) > 1e-9:
        raise ValueError("Starting JUPR may use at most four decimal places.")
    return rating


def create_admin_match_uploader_players(
    supabase: Any,
    *,
    club_id: str,
    players: list[dict[str, Any]],
    actor_email: str,
    actor_role: str,
    reviewed_fingerprint: str,
    idempotency_key: str,
    confirmation_text: str,
    source: str = "next_match_uploader_new_players",
) -> dict[str, Any]:
    if not is_admin_match_uploader_enabled():
        raise PermissionError("Next Match Uploader is disabled.")
    requested = _normalized_new_player_batch(players)

    if _clean_text(confirmation_text, limit=80).upper() != "CREATE PLAYERS":
        raise ValueError("Type CREATE PLAYERS to create the reviewed player batch.")
    request_payload = {
        "players": [
            {
                "name": item["name"],
                "starting_jupr": f"{float(item['starting_jupr']):.4f}",
            }
            for item in requested
        ]
    }
    calculated_fingerprint = canonical_fingerprint(request_payload)
    if str(reviewed_fingerprint or "").strip().lower() != calculated_fingerprint:
        raise ValueError("The player batch changed after review. Review the list again before creating players.")

    try:
        existing_players = _fetch_all_players_for_guarded_write(
            supabase,
            club_id=str(club_id),
        )
    except Exception as exc:
        raise RuntimeError("The player batch was not started because current players could not be loaded.") from exc
    existing_by_name = {
        _reviewed_name_key(player.get("name")): player
        for player in existing_players
    }
    to_create = [
        item for item in requested
        if _reviewed_name_key(item["name"]) not in existing_by_name
    ]
    preflight = {
        "reviewed_fingerprint": calculated_fingerprint,
        "preexisting": [
            {"id": player.get("id"), "name": player.get("name")}
            for key, player in existing_by_name.items()
            if key in {_reviewed_name_key(item["name"]) for item in requested}
        ],
        "to_create": to_create,
    }

    operation, idempotent = begin_guarded_operation(
        supabase,
        club_id=str(club_id),
        workflow="match_uploader_player_batch",
        action="create_match_uploader_players",
        operation_key=str(idempotency_key),
        request_payload={
            **request_payload,
            "reviewed_fingerprint": calculated_fingerprint,
            "confirmation_text": "CREATE PLAYERS",
        },
        actor_email=actor_email,
        actor_role=actor_role,
        source=source,
        before_json={
            "reviewed_player_names": [item["name"] for item in requested],
            "reviewed_players": requested,
            "preflight": preflight,
        },
    )
    if idempotent:
        return operation_result(operation)

    update_guarded_operation(
        supabase,
        operation_id=operation.get("id"),
        operation_key=str(idempotency_key),
        status="intent_recorded",
        result_json={"phase": "preflight", "preflight": preflight},
    )

    if to_create:
        insert_payloads = [
            {
                "club_id": str(club_id),
                "name": item["name"],
                "rating": float(item["starting_jupr"]) * 400.0,
                "starting_rating": float(item["starting_jupr"]) * 400.0,
                "wins": 0,
                "losses": 0,
                "matches_played": 0,
                "active": True,
                "last_game_at": None,
                "inactive_at": None,
            }
            for item in to_create
        ]
        try:
            # PostgREST sends this as one multi-row INSERT statement. PostgreSQL
            # rolls the complete statement back when any row fails.
            inserted = _safe_rows(supabase.table("players").insert(insert_payloads).execute())
        except Exception as exc:
            readback: list[dict[str, Any]] | None = None
            try:
                current_players = _fetch_all_players_for_guarded_write(
                    supabase,
                    club_id=str(club_id),
                )
                requested_names = {
                    _reviewed_name_key(item["name"])
                    for item in requested
                }
                readback = [
                    player
                    for player in current_players
                    if _reviewed_name_key(player.get("name")) in requested_names
                ]
            except Exception:
                readback = None
            _mark_player_batch_recovery(
                supabase,
                operation=operation,
                operation_key=str(idempotency_key),
                result_json={
                    "preflight": preflight,
                    "readback_verified": readback is not None,
                    "matched_count": None if readback is None else len(readback),
                    "players": readback or [],
                },
                error_text=(
                    "Bulk player insert returned an ambiguous transport result; "
                    "the operation must be inspected before retrying."
                ),
            )
            raise GuardedWriteRecoveryRequired(
                str(idempotency_key),
                "The player batch may have committed. Inspect this exact operation before retrying; do not use a new key.",
            ) from exc
        if len(inserted) != len(insert_payloads):
            _mark_player_batch_recovery(
                supabase,
                operation=operation,
                operation_key=str(idempotency_key),
                result_json={"preflight": preflight, "inserted_count": len(inserted)},
                error_text="Bulk player insert readback did not match the reviewed batch.",
            )
            raise GuardedWriteRecoveryRequired(
                str(idempotency_key),
                "The player batch outcome could not be verified. Inspect the Player Editor before retrying.",
            )

    try:
        refreshed_players = _fetch_all_players_for_guarded_write(
            supabase,
            club_id=str(club_id),
        )
    except Exception as exc:
        _mark_player_batch_recovery(
            supabase,
            operation=operation,
            operation_key=str(idempotency_key),
            result_json={"preflight": preflight},
            error_text="Post-insert player readback could not be completed.",
        )
        raise GuardedWriteRecoveryRequired(
            str(idempotency_key),
            "The player batch outcome could not be read back. Inspect this exact operation before retrying.",
        ) from exc
    requested_names = {_reviewed_name_key(item["name"]) for item in requested}
    matching_players = [
        player for player in refreshed_players
        if _reviewed_name_key(player.get("name")) in requested_names
    ]
    if len(matching_players) != len(requested):
        _mark_player_batch_recovery(
            supabase,
            operation=operation,
            operation_key=str(idempotency_key),
            result_json={"preflight": preflight, "matched_count": len(matching_players)},
            error_text="Post-insert player readback did not match the reviewed batch.",
        )
        raise GuardedWriteRecoveryRequired(
            str(idempotency_key),
            "The player batch committed but its readback was incomplete. Inspect the Player Editor before retrying.",
        )
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="create_match_uploader_players",
        entity_type="players",
        entity_id="batch",
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "requested_count": len(requested),
            "accepted_count": len(matching_players),
            "players": [{"id": player.get("id"), "name": player.get("name")} for player in matching_players],
        },
        source_page=source,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and is_api_audit_log_required():
        _mark_player_batch_recovery(
            supabase,
            operation=operation,
            operation_key=str(idempotency_key),
            result_json={"preflight": preflight, "players": matching_players},
            error_text="Required completion audit did not persist.",
        )
        raise GuardedWriteRecoveryRequired(
            str(idempotency_key),
            "The players were created but their required completion audit is unavailable. Inspect the Player Editor before retrying.",
        )
    result = {
        "ok": True,
        "mode": "match_uploader_new_players",
        "requested_count": len(requested),
        "accepted_count": len(matching_players),
        "created_count": len(to_create),
        "unchanged_count": len(requested) - len(to_create),
        "reviewed_fingerprint": calculated_fingerprint,
        "operation_key": str(idempotency_key),
        "player_insert_atomic": True,
        "idempotent_replay": False,
        "players": matching_players,
        "recovery": {
            "player_editor": "/admin/players",
            "operator_rule": "Retry the exact unchanged request with the same idempotency key after an interrupted response.",
        },
        "warnings": warnings,
    }
    update_guarded_operation(
        supabase,
        operation_id=operation.get("id"),
        operation_key=str(idempotency_key),
        status="completed",
        after_json={"players": matching_players},
        result_json=result,
    )
    return result


def submit_admin_match_uploader_batch(
    supabase: Any,
    *,
    club_id: str,
    matches: list[dict[str, Any]],
    actor_email: str,
    actor_role: str,
    idempotency_key: str,
    match_format: str = "doubles",
    source: str = "next_match_uploader",
) -> dict[str, Any]:
    if not is_admin_match_uploader_enabled():
        raise PermissionError("Next Match Uploader is disabled.")
    clean_match_format = str(match_format or "").strip().casefold()
    if clean_match_format not in {"singles", "doubles"}:
        raise ValueError("match_format must be singles or doubles.")
    clean_matches = _apply_event_contexts(supabase, club_id=str(club_id), matches=_normalize_batch(matches))
    (
        df_players_all,
        _df_players_active,
        df_leagues,
        _df_matches,
        df_meta,
        _df_badges,
        _df_player_badges,
        name_to_id,
        _id_to_name,
        _schema_degraded,
        _schema_degraded_reason,
    ) = load_data(supabase, str(club_id))
    result = submit_atomic_direct_matches(
        supabase,
        club_id=str(club_id),
        matches=clean_matches,
        match_format=clean_match_format,
        idempotency_key=str(idempotency_key),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        source=source,
        name_to_id=name_to_id,
        df_players_all=df_players_all,
        df_leagues=df_leagues,
        df_meta=df_meta,
    )
    return {
        **result,
        "mode": "match_uploader_batch",
    }
