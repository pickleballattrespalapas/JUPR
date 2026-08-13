from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.league_live_orchestration import (
    LeagueLiveDomainError,
    build_league_live_roster_suggestion,
    build_league_live_round_plan,
    normalize_league_live_roster,
)
from jupr_app.services.admin_guarded_write_service import (
    GuardedWriteRecoveryRequired,
    canonical_fingerprint,
    operation_result,
    update_guarded_operation,
)
from jupr_app.services.admin_league_manager_service import is_admin_league_manager_enabled

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
CONFIRM_CREATE_SESSION = "CREATE LIVE SESSION"
CONFIRM_SAVE_SESSION = "SAVE SESSION"
CONFIRM_SAVE_ROUND = "SAVE ROUND"
SESSION_STATUSES = {"setup", "active", "paused", "complete", "archived"}
ROUND_STATUSES = {"draft", "generated", "submitted", "voided"}


class LeagueLiveConflictError(RuntimeError):
    """Optimistic-state or idempotency conflict."""


class LeagueLivePersistenceError(RuntimeError):
    """A required League Live write did not persist."""


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def is_admin_league_live_domain_enabled() -> bool:
    return is_admin_league_manager_enabled() and _truthy_env("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN")


def is_admin_league_live_submit_enabled() -> bool:
    runtime = str(os.getenv("JUPR_ENV") or os.getenv("ENVIRONMENT") or "").strip().lower()
    return (
        runtime == "staging"
        and is_admin_league_live_domain_enabled()
        and _truthy_env("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT")
    )


def ensure_admin_league_live_submit_enabled() -> None:
    if not is_admin_league_live_submit_enabled():
        raise PermissionError(
            "Next League Live all-match publish is disabled. Keep the Streamlit League Manager fallback available."
        )


def _ensure_live_domain_enabled() -> None:
    if not is_admin_league_live_domain_enabled():
        raise PermissionError(
            "Next League Live domain writes are disabled. Use the Streamlit League Manager fallback."
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _first_row(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _clean_text(value: Any, *, limit: int = 240) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _session_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "club_id": str(row.get("club_id") or ""),
        "league_name": str(row.get("league_name") or ""),
        "week_tag": str(row.get("week_tag") or ""),
        "status": str(row.get("status") or "setup"),
        "total_rounds": _safe_int(row.get("total_rounds"), 1) or 1,
        "current_round": _safe_int(row.get("current_round"), 1) or 1,
        "roster_json": _as_list(row.get("roster_json")),
        "current_court_state_json": _as_list(row.get("current_court_state_json")),
        "notes": row.get("notes"),
        "created_by": row.get("created_by"),
        "updated_by": row.get("updated_by"),
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _round_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "club_id": str(row.get("club_id") or ""),
        "session_id": str(row.get("session_id") or ""),
        "round_number": _safe_int(row.get("round_number"), 1) or 1,
        "round_label": row.get("round_label"),
        "status": str(row.get("status") or "draft"),
        "match_date": row.get("match_date"),
        "preview_json": _as_dict(row.get("preview_json")),
        "matches_json": _as_list(row.get("matches_json")),
        "movement_json": _as_dict(row.get("movement_json")),
        "operation_key": str(row.get("operation_key") or _as_dict(row.get("movement_json")).get("operation_key") or ""),
        "submitted_match_count": _safe_int(row.get("submitted_match_count"), 0) or 0,
        "submitted_match_ids": _as_list(row.get("submitted_match_ids")),
        "submitted_at": row.get("submitted_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _court_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "club_id": str(row.get("club_id") or ""),
        "session_id": str(row.get("session_id") or ""),
        "round_number": _safe_int(row.get("round_number"), 1) or 1,
        "court_number": _safe_int(row.get("court_number"), 1) or 1,
        "format_type": str(row.get("format_type") or "4-Player"),
        "player_names": _as_list(row.get("player_names")),
        "players_json": _as_list(row.get("players_json")),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _normalize_courts(courts: list[dict[str, Any]], *, default_round: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, raw in enumerate(courts or [], start=1):
        if not isinstance(raw, dict):
            continue
        court_number = _safe_int(raw.get("court_number", raw.get("court")), index) or index
        if court_number in seen:
            continue
        seen.add(court_number)
        player_names = raw.get("player_names")
        if isinstance(player_names, str):
            names = [name.strip() for name in player_names.replace(",", "\n").split("\n") if name.strip()]
        else:
            names = [str(name or "").strip() for name in _as_list(player_names) if str(name or "").strip()]
        format_type = _clean_text(raw.get("format_type", raw.get("formatType") or ""), limit=80)
        if not format_type:
            format_type = f"{len(names)}-Player" if len(names) in {4, 5} else "4-Player"
        elif format_type.lower() in {"4-player", "5-player"}:
            format_type = f"{format_type[0]}-Player"
        normalized.append(
            {
                "round_number": _safe_int(raw.get("round_number"), default_round) or default_round,
                "court_number": int(court_number),
                "format_type": format_type,
                "player_names": names,
                "players_json": _as_list(raw.get("players_json", raw.get("players"))),
            }
        )
    normalized.sort(key=lambda row: int(row.get("court_number") or 0))
    return normalized


def _expected_session_version(before: dict[str, Any], expected_updated_at: Any) -> str:
    expected = _clean_text(expected_updated_at, limit=120)
    current = _clean_text(before.get("updated_at"), limit=120)
    if not expected:
        raise LeagueLiveConflictError("expected_updated_at is required; reload this League Live session and retry.")
    if expected != current:
        raise LeagueLiveConflictError("League Live session changed in another browser. Reload before continuing.")
    return expected


def _session_restore_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "league_name",
            "week_tag",
            "status",
            "total_rounds",
            "current_round",
            "roster_json",
            "current_court_state_json",
            "notes",
            "created_by",
            "updated_by",
            "started_at",
            "completed_at",
            "created_at",
            "updated_at",
        )
        if key in row
    }


def _round_restore_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "club_id",
            "session_id",
            "round_number",
            "round_label",
            "status",
            "match_date",
            "preview_json",
            "matches_json",
            "movement_json",
            "operation_key",
            "submitted_match_count",
            "submitted_match_ids",
            "submitted_at",
            "created_at",
            "updated_at",
        )
        if key in row
    }


def _normalize_roster_and_courts(
    *,
    roster: list[dict[str, Any]],
    courts: list[dict[str, Any]],
    round_number: int,
    bench_player_ids: list[Any] | None = None,
    bench_override_reason: str | None = None,
) -> dict[str, Any]:
    normalized_roster = normalize_league_live_roster(roster)
    if not normalized_roster:
        raise LeagueLiveDomainError("League Live requires a roster before a session can be saved.")
    if courts:
        sizes: list[int] = []
        requested_bench: list[int] = []
        court_player_ids: list[int] = []
        by_name = {str(row["player_name"]).strip().casefold(): int(row["player_id"]) for row in normalized_roster}
        for raw in courts:
            players = raw.get("players_json", raw.get("players")) if isinstance(raw, dict) else None
            ids: list[int] = []
            if isinstance(players, list) and players:
                ids = [
                    int(_safe_int(player.get("player_id", player.get("id"))) or 0)
                    for player in players
                    if isinstance(player, dict)
                ]
            else:
                names = raw.get("player_names") if isinstance(raw, dict) else []
                if isinstance(names, str):
                    names = [value.strip() for value in names.replace(",", "\n").splitlines() if value.strip()]
                ids = [by_name.get(str(name).strip().casefold(), 0) for name in (names or [])]
            if any(player_id <= 0 for player_id in ids):
                raise LeagueLiveDomainError("Every court player must resolve to the session roster.")
            sizes.append(len(ids))
            court_player_ids.extend(ids)
        if len(set(court_player_ids)) != len(court_player_ids):
            raise LeagueLiveDomainError("A rostered player cannot appear on multiple courts.")
        roster_ids = {int(row["player_id"]) for row in normalized_roster}
        if not set(court_player_ids).issubset(roster_ids):
            raise LeagueLiveDomainError("Court assignments contain a player outside the session roster.")
        assignment_by_id: dict[int, tuple[int, int]] = {}
        cursor = 0
        for court_number, size in enumerate(sizes, start=1):
            for slot, player_id in enumerate(court_player_ids[cursor : cursor + size], start=1):
                assignment_by_id[int(player_id)] = (court_number, slot)
            cursor += size
        normalized_roster = [
            {
                **row,
                "court_number": assignment_by_id.get(int(row["player_id"]), (None, None))[0],
                "slot": assignment_by_id.get(int(row["player_id"]), (None, None))[1],
                "status": "active" if int(row["player_id"]) in assignment_by_id else "bench",
            }
            for row in normalized_roster
        ]
        requested_bench = sorted(roster_ids - set(court_player_ids))
        return build_league_live_roster_suggestion(
            normalized_roster,
            court_sizes=sizes,
            prefer_keep_player_ids=court_player_ids,
            bench_player_ids=list(bench_player_ids or requested_bench),
            bench_override_reason=bench_override_reason or ("Preserve operator-provided court assignments." if requested_bench else None),
            round_number=round_number,
            preserve_assignment_order=True,
        )
    return build_league_live_roster_suggestion(
        normalized_roster,
        bench_player_ids=bench_player_ids,
        bench_override_reason=bench_override_reason,
        round_number=round_number,
    )


def _verified_club_roster(supabase: Any, *, club_id: str, roster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = normalize_league_live_roster(roster)
    requested_ids = [int(row["player_id"]) for row in normalized]
    if not requested_ids:
        raise LeagueLiveDomainError("League Live requires at least four club players.")
    try:
        rows = _safe_rows(
            supabase.table("players")
            .select("*")
            .eq("club_id", str(club_id))
            .in_("id", requested_ids)
            .execute()
        )
    except Exception as exc:
        raise LeagueLivePersistenceError("Unable to verify the League Live roster against club players.") from exc
    verified = {int(row.get("id")): row for row in rows if _safe_int(row.get("id")) is not None}
    missing = sorted(set(requested_ids) - set(verified))
    if missing:
        raise LeagueLiveDomainError(f"Roster contains player IDs outside this club: {missing}.")
    return [
        {
            **row,
            "player_name": _clean_text(verified[int(row["player_id"])].get("name"), limit=160) or row["player_name"],
            "rating": verified[int(row["player_id"])].get("rating")
            if verified[int(row["player_id"])].get("rating") is not None
            else verified[int(row["player_id"])].get("rating_jupr", row.get("rating")),
        }
        for row in normalized
    ]


def _fetch_session_row(supabase: Any, *, club_id: str, session_id: str) -> dict[str, Any] | None:
    try:
        return _first_row(
            supabase.table("league_live_sessions")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("id", str(session_id))
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise LeagueLivePersistenceError("Unable to load the League Live session from private storage.") from exc


def _fetch_rounds(supabase: Any, *, club_id: str, session_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("league_live_rounds")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("session_id", str(session_id))
            .execute()
        )
    except Exception as exc:
        raise LeagueLivePersistenceError("Unable to load League Live round history from private storage.") from exc
    return sorted((_round_payload(row) for row in rows), key=lambda row: int(row.get("round_number") or 0))


def _fetch_courts(supabase: Any, *, club_id: str, session_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("league_live_courts")
            .select("*")
            .eq("club_id", str(club_id))
            .eq("session_id", str(session_id))
            .execute()
        )
    except Exception as exc:
        raise LeagueLivePersistenceError("Unable to load League Live court snapshots from private storage.") from exc
    return sorted(
        (_court_payload(row) for row in rows),
        key=lambda row: (int(row.get("round_number") or 0), int(row.get("court_number") or 0)),
    )


def _replace_court_snapshots(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    round_number: int,
    courts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized = _normalize_courts(courts, default_round=int(round_number))
    (
        supabase.table("league_live_courts")
        .delete()
        .eq("club_id", str(club_id))
        .eq("session_id", str(session_id))
        .eq("round_number", int(round_number))
        .execute()
    )
    if not normalized:
        return []
    payloads = [
        {
            "id": str(uuid4()),
            "club_id": str(club_id),
            "session_id": str(session_id),
            "round_number": int(round_number),
            "court_number": int(row["court_number"]),
            "format_type": str(row.get("format_type") or "4-Player"),
            "player_names": row.get("player_names") or [],
            "players_json": row.get("players_json") or [],
            "updated_at": _now_iso(),
        }
        for row in normalized
    ]
    inserted = _safe_rows(supabase.table("league_live_courts").insert(payloads).execute())
    if len(inserted) != len(payloads):
        raise LeagueLivePersistenceError("League Live court snapshots did not persist completely.")
    return [_court_payload(row) for row in inserted]


def _court_rows_for_round(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    round_number: int,
) -> list[dict[str, Any]]:
    return [
        row
        for row in _fetch_courts(supabase, club_id=club_id, session_id=session_id)
        if int(row.get("round_number") or 0) == int(round_number)
    ]


def _restore_court_snapshots(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    round_number: int,
    courts: list[dict[str, Any]],
) -> None:
    _replace_court_snapshots(
        supabase,
        club_id=club_id,
        session_id=session_id,
        round_number=round_number,
        courts=courts,
    )


def _audit(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action_type: str,
    entity_id: str,
    before_json: dict[str, Any] | None = None,
    after_json: dict[str, Any] | None = None,
    source: str,
) -> list[str]:
    payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type=action_type,
        entity_type="league_live_session",
        entity_id=str(entity_id or ""),
        before_json=before_json or {},
        after_json={"source_client": "fastapi/nextjs", "source_page": source, **(after_json or {})},
        source_page=source,
        flagged_for_review=True,
    )
    write = write_admin_activity_log(supabase, payload)
    if not write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return [write.warning] if write.warning else []


def build_admin_league_live_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    # This endpoint is intentionally configuration-only. It is rendered before
    # browser authentication and must never project private session metadata.
    del supabase, club_id
    if not is_admin_league_manager_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "sessions_endpoint": None,
            "roster_suggestion_endpoint": None,
            "round_plan_endpoint": None,
            "submit_enabled": False,
            "round_submit_endpoint": None,
            "round_reconcile_endpoint": None,
            "round_compensate_endpoint": None,
            "guest_endpoint": None,
            "export_endpoint": None,
            "streamlit_fallback": "league_manager",
            "warnings": ["Next League Manager is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER on FastAPI."],
        }
    if not is_admin_league_live_domain_enabled():
        return {
            "enabled": False,
            "status": "streamlit_fallback",
            "sessions_endpoint": None,
            "roster_suggestion_endpoint": None,
            "round_plan_endpoint": None,
            "submit_enabled": False,
            "round_submit_endpoint": None,
            "round_reconcile_endpoint": None,
            "round_compensate_endpoint": None,
            "guest_endpoint": None,
            "export_endpoint": None,
            "streamlit_fallback": "league_manager",
            "warnings": [
                "Python-authoritative League Live is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN on staging FastAPI only."
            ],
        }
    submit_enabled = is_admin_league_live_submit_enabled()
    return {
        "enabled": True,
        "status": "ready_for_durable_all_match_publish" if submit_enabled else "ready_for_python_authoritative_league_live",
        "sessions_endpoint": "/admin/clubs/{club_id}/league-manager/live-sessions",
        "roster_suggestion_endpoint": "/admin/clubs/{club_id}/league-manager/live/roster-suggestion",
        "round_plan_endpoint": "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/plan",
        "submit_enabled": submit_enabled,
        "round_submit_endpoint": (
            "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/submit"
            if submit_enabled
            else None
        ),
        "round_reconcile_endpoint": (
            "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/reconcile"
            if submit_enabled
            else None
        ),
        "round_compensate_endpoint": (
            "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/compensate"
            if submit_enabled
            else None
        ),
        "guest_endpoint": (
            "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/guests" if submit_enabled else None
        ),
        "export_endpoint": (
            "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/export" if submit_enabled else None
        ),
        "movement_authority": "python_fastapi",
        "publish_authority": "python_fastapi",
        "streamlit_fallback": "league_manager",
        "warnings": (
            []
            if submit_enabled
            else [
                "Durable all-match publish is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT on staging FastAPI only."
            ]
        ),
    }


def list_admin_league_live_sessions(
    supabase: Any,
    *,
    club_id: str,
    status: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    _ensure_live_domain_enabled()
    query: Any | None = None
    try:
        query = supabase.table("league_live_sessions").select("*").eq("club_id", str(club_id))
        clean_status = _clean_text(status, limit=40).lower()
        if clean_status:
            query = query.eq("status", clean_status)
        rows = _safe_rows(query.order("updated_at", desc=True).limit(max(1, min(int(limit or 50), 200))).execute())
    except Exception as first_exc:
        try:
            if query is None:
                raise first_exc
            rows = _safe_rows(query.execute())
        except Exception as exc:
            raise LeagueLivePersistenceError("Unable to list League Live sessions from private storage.") from exc
    sessions = [_session_payload(row) for row in rows]
    return {"ok": True, "mode": "league_live_sessions_list", "sessions": sessions, "count": len(sessions)}


def get_admin_league_live_session(supabase: Any, *, club_id: str, session_id: str) -> dict[str, Any]:
    _ensure_live_domain_enabled()
    row = _fetch_session_row(supabase, club_id=str(club_id), session_id=str(session_id))
    if row is None:
        raise ValueError("league live session not found")
    session = _session_payload(row)
    rounds = _fetch_rounds(supabase, club_id=str(club_id), session_id=str(session_id))
    courts = _fetch_courts(supabase, club_id=str(club_id), session_id=str(session_id))
    return {"ok": True, "mode": "league_live_session_detail", "session": session, "rounds": rounds, "courts": courts}


def suggest_admin_league_live_roster(
    supabase: Any,
    *,
    club_id: str,
    roster: list[dict[str, Any]],
    court_sizes: list[int] | None = None,
    prefer_keep_player_ids: list[Any] | None = None,
    bench_player_ids: list[Any] | None = None,
    bench_override_reason: str | None = None,
    round_number: int = 1,
) -> dict[str, Any]:
    _ensure_live_domain_enabled()
    canonical_roster = _verified_club_roster(supabase, club_id=club_id, roster=roster)
    return build_league_live_roster_suggestion(
        canonical_roster,
        court_sizes=court_sizes,
        prefer_keep_player_ids=prefer_keep_player_ids,
        bench_player_ids=bench_player_ids,
        bench_override_reason=bench_override_reason,
        round_number=round_number,
    )


def build_admin_league_live_round_plan(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    round_number: int,
    expected_updated_at: str,
    matches: list[dict[str, Any]],
    courts: list[dict[str, Any]] | None = None,
    movement_overrides: list[dict[str, Any]] | None = None,
    override_reason: str | None = None,
    roster_change: dict[str, Any] | None = None,
    bench_player_ids: list[Any] | None = None,
    bench_override_reason: str | None = None,
) -> dict[str, Any]:
    _ensure_live_domain_enabled()
    session_row = _fetch_session_row(supabase, club_id=str(club_id), session_id=str(session_id))
    if session_row is None:
        raise LeagueLiveDomainError("league live session not found")
    _expected_session_version(session_row, expected_updated_at)
    safe_round = max(1, _safe_int(round_number, 1) or 1)
    if safe_round != (_safe_int(session_row.get("current_round"), 1) or 1):
        raise LeagueLiveConflictError("This round is no longer current. Reload the League Live session.")
    if str(session_row.get("status") or "").lower() not in {"active", "setup"}:
        raise LeagueLiveConflictError("League Live movement is available only for an active session.")

    resolved_roster_change = dict(roster_change or {}) or None
    if resolved_roster_change:
        incoming = resolved_roster_change.get("player")
        if not isinstance(incoming, dict):
            raise LeagueLiveDomainError("Roster change requires a player.")
        incoming_id = _safe_int(incoming.get("player_id", incoming.get("id")))
        if incoming_id is None or incoming_id <= 0:
            raise LeagueLiveDomainError("Roster change requires a positive incoming player ID.")
        try:
            verified = _first_row(
                supabase.table("players")
                .select("*")
                .eq("club_id", str(club_id))
                .eq("id", int(incoming_id))
                .limit(1)
                .execute()
            )
        except Exception as exc:
            raise LeagueLivePersistenceError("Unable to verify the incoming roster player.") from exc
        if verified is None:
            raise LeagueLiveDomainError("Incoming roster player does not belong to this club.")
        resolved_roster_change["player"] = {
            "player_id": int(incoming_id),
            "player_name": _clean_text(verified.get("name"), limit=160),
            "rating": verified.get("rating")
            if verified.get("rating") is not None
            else verified.get("rating_jupr", 1200.0),
        }

    effective_courts = list(courts or _as_list(session_row.get("current_court_state_json")))
    return build_league_live_round_plan(
        session_id=str(session_id),
        round_number=safe_round,
        total_rounds=_safe_int(session_row.get("total_rounds"), safe_round) or safe_round,
        session_updated_at=str(session_row.get("updated_at") or ""),
        roster=_as_list(session_row.get("roster_json")),
        courts=effective_courts,
        matches=matches,
        movement_overrides=movement_overrides,
        override_reason=override_reason,
        roster_change=resolved_roster_change,
        bench_player_ids=bench_player_ids,
        bench_override_reason=bench_override_reason,
    )


def prepare_admin_league_live_session_create(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    league_name: str,
    week_tag: str,
    total_rounds: int,
    current_round: int = 1,
    roster: list[dict[str, Any]] | None = None,
    courts: list[dict[str, Any]] | None = None,
    bench_player_ids: list[Any] | None = None,
    bench_override_reason: str | None = None,
    notes: str | None = None,
    actor_email: str,
) -> dict[str, Any]:
    """Build the exact create payload before the first domain mutation."""
    _ensure_live_domain_enabled()
    clean_league = _clean_text(league_name, limit=120)
    if not clean_league:
        raise ValueError("league_name is required")
    safe_total_rounds = max(1, min(_safe_int(total_rounds, 1) or 1, 50))
    safe_current_round = max(1, min(_safe_int(current_round, 1) or 1, safe_total_rounds))
    canonical_roster = _verified_club_roster(
        supabase,
        club_id=str(club_id),
        roster=list(roster or []),
    )
    roster_plan = _normalize_roster_and_courts(
        roster=canonical_roster,
        courts=list(courts or []),
        round_number=safe_current_round,
        bench_player_ids=bench_player_ids,
        bench_override_reason=bench_override_reason,
    )
    normalized_courts = list(roster_plan["courts"])
    now = _now_iso()
    return {
        "session_payload": {
            "id": str(session_id),
            "club_id": str(club_id),
            "league_name": clean_league,
            "week_tag": _clean_text(week_tag, limit=80),
            "status": "active",
            "total_rounds": safe_total_rounds,
            "current_round": safe_current_round,
            "roster_json": list(roster_plan["roster"]),
            "current_court_state_json": normalized_courts,
            "notes": _clean_text(notes, limit=1000) or None,
            "created_by": str(actor_email or ""),
            "updated_by": str(actor_email or ""),
            "started_at": now,
            "created_at": now,
            "updated_at": now,
        },
        "courts": normalized_courts,
        "roster_suggestion": roster_plan,
    }


def _semantic_court_rows(courts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "round_number": _safe_int(row.get("round_number"), 1) or 1,
            "court_number": _safe_int(row.get("court_number"), 1) or 1,
            "format_type": str(row.get("format_type") or "4-Player"),
            "player_names": _as_list(row.get("player_names")),
            "players_json": _as_list(row.get("players_json")),
        }
        for row in sorted(
            (dict(item) for item in courts if isinstance(item, dict)),
            key=lambda item: (_safe_int(item.get("round_number"), 1) or 1, _safe_int(item.get("court_number"), 1) or 1),
        )
    ]


def reconcile_admin_league_live_session_create(
    supabase: Any,
    *,
    operation: dict[str, Any],
    club_id: str,
    operation_key: str,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    source: str = "next_league_live_operation_reconcile",
) -> dict[str, Any]:
    """Resolve an uncertain create only from exact preflight and authoritative storage."""
    _ensure_live_domain_enabled()
    if _clean_text(confirmation_text, limit=80).upper() != "RECONCILE LIVE SESSION":
        raise ValueError("Type RECONCILE LIVE SESSION to reconcile this exact create.")
    status = str(operation.get("status") or "")
    if status == "completed":
        return operation_result(operation)
    if status not in {"intent_recorded", "recovery_required"}:
        raise ValueError(f"League Live create operation is {status or 'unknown'} and cannot be reconciled.")
    evidence = operation.get("result_json") or {}
    planned = evidence.get("planned") if isinstance(evidence, dict) else None
    expected_session = planned.get("session_payload") if isinstance(planned, dict) else None
    expected_courts = planned.get("courts") if isinstance(planned, dict) else None
    roster_suggestion = planned.get("roster_suggestion") if isinstance(planned, dict) else None
    if not isinstance(expected_session, dict) or not isinstance(expected_courts, list) or not isinstance(roster_suggestion, dict):
        raise GuardedWriteRecoveryRequired(
            str(operation_key),
            "This League Live create predates exact preflight evidence and requires manual inspection.",
        )
    session_id = str(expected_session.get("id") or "")
    authoritative_row = _fetch_session_row(supabase, club_id=str(club_id), session_id=session_id)
    authoritative_courts = _fetch_courts(supabase, club_id=str(club_id), session_id=session_id)
    if authoritative_row is None and not authoritative_courts:
        update_guarded_operation(
            supabase,
            operation_id=operation.get("id"),
            operation_key=str(operation_key),
            status="failed",
            result_json={"reconciled": True, "proof": "session_and_courts_absent", "session_id": session_id},
            error_text="Authoritative readback proves the League Live create did not commit.",
        )
        return {
            "ok": False,
            "mode": "league_live_session_create_reconciled_failed",
            "operation_key": str(operation_key),
            "status": "failed",
            "recovery_required": False,
            "session_id": session_id,
        }
    actual_session = _session_payload(authoritative_row or {}) if authoritative_row else None
    comparable_fields = (
        "id", "club_id", "league_name", "week_tag", "status", "total_rounds", "current_round",
        "roster_json", "current_court_state_json", "notes", "created_by", "updated_by", "started_at",
        "created_at", "updated_at",
    )
    expected_comparable = {field: expected_session.get(field) for field in comparable_fields}
    actual_comparable = {field: (actual_session or {}).get(field) for field in comparable_fields}
    session_proven = actual_session is not None and canonical_fingerprint(actual_comparable) == canonical_fingerprint(expected_comparable)
    courts_proven = canonical_fingerprint(_semantic_court_rows(authoritative_courts)) == canonical_fingerprint(_semantic_court_rows(expected_courts))
    if not session_proven or not courts_proven:
        raise GuardedWriteRecoveryRequired(
            str(operation_key),
            "Authoritative League Live state does not exactly prove the reviewed session and court create.",
        )
    try:
        audit_write = write_admin_activity_log(
            supabase,
            build_activity_payload(
                club_id=str(club_id),
                actor_email=str(actor_email or ""),
                actor_role=str(actor_role or ""),
                action_type="reconcile_league_live_session_create",
                entity_type="league_live_session",
                entity_id=session_id,
                before_json=operation.get("before_json") or {},
                after_json={
                    "source_client": "fastapi/nextjs",
                    "source_page": source,
                    "operation_key": str(operation_key),
                    "proof": "authoritative_session_and_court_readback",
                    "session": actual_session,
                    "court_count": len(authoritative_courts),
                },
                source_page=source,
                flagged_for_review=True,
            ),
        )
        if not audit_write.ok:
            raise RuntimeError("League Live reconciliation audit did not persist.")
        warnings = [audit_write.warning] if audit_write.warning else []
    except Exception as exc:
        try:
            update_guarded_operation(
                supabase,
                operation_id=operation.get("id"),
                operation_key=str(operation_key),
                status="recovery_required",
                result_json={**evidence, "reconcile_audit_failed": True},
                error_text="Authoritative League Live proof succeeded, but reconciliation audit did not persist.",
            )
        except GuardedWriteRecoveryRequired:
            pass
        raise GuardedWriteRecoveryRequired(
            str(operation_key),
            "League Live proof succeeded, but its reconciliation audit is unavailable.",
        ) from exc
    result = {
        "ok": True,
        "mode": "league_live_session_create",
        "session": actual_session,
        "courts": authoritative_courts,
        "roster_suggestion": roster_suggestion,
        "operation_key": str(operation_key),
        "idempotent_replay": True,
        "reconciled": True,
        "recovery": {
            "streamlit_fallback": "league_manager",
            "session_detail": f"/admin/clubs/{{club_id}}/league-manager/live-sessions/{session_id}",
            "operator_rule": "This create was finalized from authoritative session and court readback plus a reconciliation audit.",
        },
        "warnings": warnings,
    }
    update_guarded_operation(
        supabase,
        operation_id=operation.get("id"),
        operation_key=str(operation_key),
        status="completed",
        after_json={"session": actual_session, "courts": authoritative_courts, "reconciled": True},
        result_json=result,
    )
    return result


def create_admin_league_live_session(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    week_tag: str,
    total_rounds: int,
    current_round: int = 1,
    roster: list[dict[str, Any]] | None = None,
    courts: list[dict[str, Any]] | None = None,
    bench_player_ids: list[Any] | None = None,
    bench_override_reason: str | None = None,
    notes: str | None = None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_league_live_session_create",
    session_id: str | None = None,
    prepared_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ensure_live_domain_enabled()
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_CREATE_SESSION:
        raise ValueError(f"Type {CONFIRM_CREATE_SESSION} to create a persisted league live session.")
    clean_league = _clean_text(league_name, limit=120)
    if not clean_league:
        raise ValueError("league_name is required")
    safe_total_rounds = max(1, min(_safe_int(total_rounds, 1) or 1, 50))
    safe_current_round = max(1, min(_safe_int(current_round, 1) or 1, safe_total_rounds))
    if prepared_plan is None:
        prepared_plan = prepare_admin_league_live_session_create(
            supabase,
            club_id=str(club_id),
            session_id=str(session_id or uuid4()),
            league_name=clean_league,
            week_tag=week_tag,
            total_rounds=safe_total_rounds,
            current_round=safe_current_round,
            roster=list(roster or []),
            courts=list(courts or []),
            bench_player_ids=bench_player_ids,
            bench_override_reason=bench_override_reason,
            notes=notes,
            actor_email=actor_email,
        )
    payload = dict(prepared_plan.get("session_payload") or {})
    roster_plan = dict(prepared_plan.get("roster_suggestion") or {})
    normalized_courts = list(prepared_plan.get("courts") or [])
    session_id = str(payload.get("id") or "")
    if not session_id or str(payload.get("club_id") or "") != str(club_id):
        raise ValueError("Prepared League Live create evidence is invalid.")
    inserted = _first_row(supabase.table("league_live_sessions").insert(payload).execute())
    if inserted is None:
        raise LeagueLivePersistenceError("League Live session did not persist.")
    try:
        court_rows = _replace_court_snapshots(
            supabase,
            club_id=str(club_id),
            session_id=session_id,
            round_number=safe_current_round,
            courts=normalized_courts,
        )
        warnings = _audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="create_league_live_session_admin",
            entity_id=session_id,
            after_json={
                "session": _session_payload(inserted),
                "court_count": len(court_rows),
                "bench_player_ids": roster_plan["bench_player_ids"],
                "roster_fingerprint": roster_plan["fingerprint"],
            },
            source=source,
        )
    except Exception:
        try:
            supabase.table("league_live_courts").delete().eq("club_id", str(club_id)).eq("session_id", session_id).execute()
            supabase.table("league_live_sessions").delete().eq("club_id", str(club_id)).eq("id", session_id).execute()
        except Exception:
            pass
        raise
    return {
        "ok": True,
        "mode": "league_live_session_create",
        "session": _session_payload(inserted),
        "courts": court_rows,
        "roster_suggestion": roster_plan,
        "recovery": {"streamlit_fallback": "league_manager", "session_detail": f"/admin/clubs/{{club_id}}/league-manager/live-sessions/{session_id}"},
        "warnings": warnings,
    }


def update_admin_league_live_session_snapshot(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    patch: dict[str, Any],
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_league_live_session_snapshot",
) -> dict[str, Any]:
    _ensure_live_domain_enabled()
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_SAVE_SESSION:
        raise ValueError(f"Type {CONFIRM_SAVE_SESSION} to save the league live session snapshot.")
    before = _fetch_session_row(supabase, club_id=str(club_id), session_id=str(session_id))
    if before is None:
        raise ValueError("league live session not found")
    expected_updated_at = _expected_session_version(before, patch.get("expected_updated_at"))
    current_round = _safe_int(patch.get("current_round"), _safe_int(before.get("current_round"), 1) or 1) or 1
    total_rounds = max(1, min(_safe_int(patch.get("total_rounds"), _safe_int(before.get("total_rounds"), 1) or 1) or 1, 50))
    current_round = max(1, min(current_round, total_rounds))
    status = _clean_text(patch.get("status") or before.get("status") or "active", limit=40).lower()
    if status not in SESSION_STATUSES:
        raise ValueError("unsupported session status")
    current_status = _clean_text(before.get("status") or "setup", limit=40).lower()
    allowed_transitions = {
        "setup": {"setup", "active", "archived"},
        "active": {"active", "paused", "complete"},
        "paused": {"paused", "active", "complete"},
        "complete": {"complete", "archived"},
        "archived": {"archived"},
    }
    if status not in allowed_transitions.get(current_status, {current_status}):
        raise LeagueLiveConflictError(f"Cannot change League Live status from {current_status} to {status}.")

    requested_roster = patch.get("roster", patch.get("roster_json", before.get("roster_json")))
    requested_courts = patch.get("courts", patch.get("current_court_state_json", before.get("current_court_state_json")))
    canonical_roster = _verified_club_roster(
        supabase,
        club_id=str(club_id),
        roster=_as_list(requested_roster),
    )
    roster_plan = _normalize_roster_and_courts(
        roster=canonical_roster,
        courts=_as_list(requested_courts),
        round_number=current_round,
        bench_player_ids=_as_list(patch.get("bench_player_ids")),
        bench_override_reason=patch.get("bench_override_reason"),
    )
    normalized_courts = list(roster_plan["courts"])
    payload: dict[str, Any] = {
        "status": status,
        "total_rounds": total_rounds,
        "current_round": current_round,
        "updated_by": str(actor_email or ""),
        "updated_at": _now_iso(),
        "roster_json": list(roster_plan["roster"]),
        "current_court_state_json": normalized_courts,
    }
    if "week_tag" in patch:
        payload["week_tag"] = _clean_text(patch.get("week_tag"), limit=80)
    if "notes" in patch:
        payload["notes"] = _clean_text(patch.get("notes"), limit=1000) or None
    if status == "complete" and not before.get("completed_at"):
        payload["completed_at"] = _now_iso()
    before_courts = _court_rows_for_round(
        supabase,
        club_id=str(club_id),
        session_id=str(session_id),
        round_number=current_round,
    )
    updated = _first_row(
        supabase.table("league_live_sessions")
        .update(payload)
        .eq("club_id", str(club_id))
        .eq("id", str(session_id))
        .eq("updated_at", expected_updated_at)
        .execute()
    )
    if updated is None:
        raise LeagueLiveConflictError("League Live session changed while this snapshot was saving. Reload and retry.")
    try:
        court_rows = _replace_court_snapshots(
            supabase,
            club_id=str(club_id),
            session_id=str(session_id),
            round_number=current_round,
            courts=normalized_courts,
        )
        warnings = _audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="update_league_live_session_snapshot_admin",
            entity_id=str(session_id),
            before_json={"session": _session_payload(before), "courts": before_courts},
            after_json={
                "session": _session_payload(updated),
                "court_count": len(court_rows),
                "bench_player_ids": roster_plan["bench_player_ids"],
                "roster_fingerprint": roster_plan["fingerprint"],
            },
            source=source,
        )
    except Exception:
        try:
            supabase.table("league_live_sessions").update(_session_restore_payload(before)).eq("club_id", str(club_id)).eq("id", str(session_id)).execute()
            _restore_court_snapshots(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                round_number=current_round,
                courts=before_courts,
            )
        except Exception:
            pass
        raise
    return {
        "ok": True,
        "mode": "league_live_session_snapshot_update",
        "session": _session_payload(updated),
        "courts": court_rows,
        "roster_suggestion": roster_plan,
        "recovery": {"streamlit_fallback": "league_manager", "previous_session": _session_payload(before)},
        "warnings": warnings,
    }


def save_admin_league_live_round(
    supabase: Any,
    *,
    club_id: str,
    session_id: str,
    round_number: int,
    round_label: str | None = None,
    match_date: str | None = None,
    preview: dict[str, Any] | None = None,
    matches: list[dict[str, Any]] | None = None,
    movement: dict[str, Any] | None = None,
    movement_overrides: list[dict[str, Any]] | None = None,
    override_reason: str | None = None,
    roster_change: dict[str, Any] | None = None,
    bench_player_ids: list[Any] | None = None,
    bench_override_reason: str | None = None,
    expected_updated_at: str = "",
    expected_operation_key: str = "",
    submitted_match_count: int | None = None,
    submitted_match_ids: list[Any] | None = None,
    courts: list[dict[str, Any]] | None = None,
    advance_after_save: bool = True,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_league_live_round_save",
) -> dict[str, Any]:
    _ensure_live_domain_enabled()
    if (
        is_admin_league_live_submit_enabled()
        and source not in {"next_league_live_round_publish", "next_league_live_round_reconcile"}
        and (_as_list(matches) or submitted_match_count is not None)
    ):
        raise PermissionError(
            "Direct League Live round submission is disabled. Use the durable all-match publish endpoint."
        )
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_SAVE_ROUND:
        raise ValueError(f"Type {CONFIRM_SAVE_ROUND} to save the league live round state.")
    if _as_dict(movement):
        raise LeagueLiveDomainError("Client-supplied movement is not accepted; use the Python round-plan contract.")
    session_row = _fetch_session_row(supabase, club_id=str(club_id), session_id=str(session_id))
    if session_row is None:
        raise ValueError("league live session not found")
    safe_round = max(1, _safe_int(round_number, 1) or 1)
    existing = _first_row(
        supabase.table("league_live_rounds")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("session_id", str(session_id))
        .eq("round_number", safe_round)
        .limit(1)
        .execute()
    )
    existing_operation_key = str(
        (existing or {}).get("operation_key")
        or _as_dict((existing or {}).get("movement_json")).get("operation_key")
        or ""
    )
    if existing and existing_operation_key and existing_operation_key == str(expected_operation_key):
        return {
            "ok": True,
            "mode": "league_live_round_save",
            "idempotent_replay": True,
            "operation_key": existing_operation_key,
            "session": _session_payload(session_row),
            "round": _round_payload(existing),
            "courts": _court_rows_for_round(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                round_number=safe_round,
            ),
            "movement": _as_dict(existing.get("movement_json")),
            "bench": [
                row for row in _as_list(session_row.get("roster_json")) if str(row.get("status") or "") == "bench"
            ],
            "recovery": {
                "session_detail": f"/admin/clubs/{{club_id}}/league-manager/live-sessions/{session_id}",
                "match_log": "/admin/match-log",
                "replay_history": "/admin/replay-history",
            },
            "warnings": [],
        }
    expected_version = _expected_session_version(session_row, expected_updated_at)
    matches_list = _as_list(matches)
    plan = build_admin_league_live_round_plan(
        supabase,
        club_id=str(club_id),
        session_id=str(session_id),
        round_number=safe_round,
        expected_updated_at=expected_version,
        matches=matches_list,
        courts=list(courts or []),
        movement_overrides=movement_overrides,
        override_reason=override_reason,
        roster_change=roster_change,
        bench_player_ids=bench_player_ids,
        bench_override_reason=bench_override_reason,
    )
    operation_key = str(plan["operation_key"])
    if not _clean_text(expected_operation_key, limit=128):
        raise LeagueLiveConflictError("expected_operation_key is required; preview Python movement again.")
    if str(expected_operation_key) != operation_key:
        raise LeagueLiveConflictError("League Live plan changed. Preview Python movement again before saving.")
    count = plan["scored_match_count"]
    if submitted_match_count is not None and int(submitted_match_count) != int(count):
        raise LeagueLiveConflictError("Submitted match count does not match the Python round plan.")
    status = "submitted"
    now = _now_iso()
    if existing and str(existing.get("status") or "") == "submitted":
        raise LeagueLiveConflictError("This round was already saved with a different operation key. Reload before retrying.")
    payload = {
        "club_id": str(club_id),
        "session_id": str(session_id),
        "round_number": safe_round,
        "round_label": _clean_text(round_label, limit=80) or f"Round {safe_round}",
        "status": status,
        "match_date": _clean_text(match_date, limit=20) or None,
        "preview_json": _as_dict(preview),
        "matches_json": matches_list,
        "movement_json": plan["movement"],
        "operation_key": operation_key,
        "submitted_match_count": count,
        "submitted_match_ids": _as_list(submitted_match_ids),
        "submitted_at": now if count > 0 else None,
        "updated_at": now,
    }
    before_courts = _court_rows_for_round(
        supabase,
        club_id=str(club_id),
        session_id=str(session_id),
        round_number=safe_round,
    )
    next_round = int(plan["next_round"])
    session_patch: dict[str, Any] = {
        "current_court_state_json": list(plan["next_courts"]),
        "roster_json": list(plan["next_roster"]),
        "updated_by": str(actor_email or ""),
        "updated_at": now,
    }
    if advance_after_save and next_round > (_safe_int(session_row.get("current_round"), 1) or 1):
        session_patch["current_round"] = next_round
    if count > 0:
        session_patch["status"] = "active"
    updated_session = _first_row(
        supabase.table("league_live_sessions")
        .update(session_patch)
        .eq("club_id", str(club_id))
        .eq("id", str(session_id))
        .eq("updated_at", expected_version)
        .execute()
    )
    if updated_session is None:
        raise LeagueLiveConflictError("League Live session changed while the round was saving. Reload and recover through Match Log if scores were already submitted.")
    saved: dict[str, Any] | None = None
    court_rows: list[dict[str, Any]] = []
    try:
        if existing:
            saved = _first_row(
                supabase.table("league_live_rounds")
                .update(payload)
                .eq("id", str(existing.get("id")))
                .execute()
            )
        else:
            insert_payload = {"id": str(uuid4()), "created_at": now, **payload}
            saved = _first_row(supabase.table("league_live_rounds").insert(insert_payload).execute())
        if saved is None:
            raise LeagueLivePersistenceError("League Live round state did not persist.")
        court_rows = _replace_court_snapshots(
            supabase,
            club_id=str(club_id),
            session_id=str(session_id),
            round_number=safe_round,
            courts=list(plan["current_courts"]),
        )
        warnings = _audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="save_league_live_round_admin",
            entity_id=str(session_id),
            before_json={
                "session": _session_payload(session_row),
                "round": _round_payload(existing or {}) if existing else None,
                "courts": before_courts,
            },
            after_json={
                "session": _session_payload(updated_session),
                "round": _round_payload(saved),
                "court_count": len(court_rows),
                "operation_key": operation_key,
                "movement_authority": "python_fastapi",
                "override_reason": plan["movement"].get("override_reason"),
                "bench_player_ids": plan["bench_player_ids"],
            },
            source=source,
        )
    except Exception:
        try:
            supabase.table("league_live_sessions").update(_session_restore_payload(session_row)).eq("club_id", str(club_id)).eq("id", str(session_id)).execute()
            if existing:
                supabase.table("league_live_rounds").update(_round_restore_payload(existing)).eq("id", str(existing.get("id"))).execute()
            elif saved:
                supabase.table("league_live_rounds").delete().eq("id", str(saved.get("id"))).execute()
            _restore_court_snapshots(
                supabase,
                club_id=str(club_id),
                session_id=str(session_id),
                round_number=safe_round,
                courts=before_courts,
            )
        except Exception:
            pass
        raise
    return {
        "ok": True,
        "mode": "league_live_round_save",
        "idempotent_replay": False,
        "operation_key": operation_key,
        "session": _session_payload(updated_session),
        "round": _round_payload(saved),
        "courts": court_rows,
        "movement": plan["movement"],
        "bench": plan["bench"],
        "recovery": {
            **plan["recovery"],
            "previous_session": _session_payload(session_row),
            "submitted_match_ids": _as_list(submitted_match_ids),
            "streamlit_fallback": "league_manager",
        },
        "warnings": warnings,
    }
