from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os
import uuid

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_tournament_draw_service import _draw_payload
from jupr_app.services.admin_tournament_service import (
    TOURNAMENT_SELECT,
    _clean_text,
    _first_row,
    is_admin_tournament_admin_enabled,
)

CONFIRM_SAVE_TEAMS = "SAVE TEAMS"
MAX_TEAM_ROWS = 64


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _fetch_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> dict[str, Any] | None:
    try:
        rows = _safe_rows(
            supabase.table("tournament_event_draws")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("id", str(draw_id))
            .limit(1)
            .execute()
        )
    except Exception:
        rows = []
    return rows[0] if rows else None


def _games_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table("tournament_games")
            .select("id")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception:
        return []


def _existing_teams_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("tournament_teams")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception:
        rows = []
    return sorted(rows, key=lambda row: int(_safe_int(row.get("team_number")) or 0))


def _team_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _clean_text(row.get("id"), limit=120),
        "tournament_id": _clean_text(row.get("tournament_id"), limit=120),
        "draw_id": _clean_text(row.get("draw_id"), limit=120) or None,
        "registration_day_id": _clean_text(row.get("registration_day_id"), limit=120) or None,
        "event_option_id": _clean_text(row.get("event_option_id"), limit=120) or None,
        "team_number": _safe_int(row.get("team_number")),
        "player1_id": _safe_int(row.get("player1_id")),
        "player2_id": _safe_int(row.get("player2_id")),
        "seed": _safe_int(row.get("seed")),
        "source": _clean_text(row.get("source") or "MANUAL", limit=60),
        "notes": _clean_text(row.get("notes"), limit=500),
    }


def _normalize_team_rows(team_rows: list[dict[str, Any]], *, draw: dict[str, Any], tournament_id: str) -> list[dict[str, Any]]:
    if not team_rows:
        raise ValueError("At least one team row is required.")
    if len(team_rows) > MAX_TEAM_ROWS:
        raise ValueError(f"A draw can have at most {MAX_TEAM_ROWS} team rows in this editor.")

    seen_numbers: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(team_rows, start=1):
        team_number = _safe_int(row.get("team_number")) or index
        if team_number <= 0:
            raise ValueError("Team numbers must be positive integers.")
        if team_number in seen_numbers:
            raise ValueError(f"Duplicate team number: {team_number}")
        seen_numbers.add(team_number)

        player1_id = _safe_int(row.get("player1_id"))
        player2_id = _safe_int(row.get("player2_id"))
        if player1_id is None:
            raise ValueError(f"Team {team_number} requires Player 1.")
        if player2_id is not None and player2_id == player1_id:
            raise ValueError(f"Team {team_number} cannot use the same player twice.")

        normalized.append(
            {
                "id": str(row.get("id") or uuid.uuid4()),
                "tournament_id": str(tournament_id),
                "draw_id": str(draw.get("id")),
                "registration_day_id": _clean_text(draw.get("registration_day_id"), limit=120) or None,
                "event_option_id": _clean_text(draw.get("event_option_id"), limit=120) or None,
                "team_number": int(team_number),
                "player1_id": int(player1_id),
                "player2_id": int(player2_id) if player2_id is not None else None,
                "seed": _safe_int(row.get("seed")),
                "source": _clean_text(row.get("source") or "MANUAL", limit=60) or "MANUAL",
                "notes": _clean_text(row.get("notes"), limit=500) or None,
            }
        )
    return sorted(normalized, key=lambda row: int(row.get("team_number") or 0))


def replace_admin_tournament_draw_teams(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    teams: list[dict[str, Any]],
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_replace_teams",
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_SAVE_TEAMS:
        raise ValueError(f"Type {CONFIRM_SAVE_TEAMS} to save teams for this draw.")

    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_draw_id = _clean_text(draw_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    draw = _fetch_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not draw:
        raise ValueError("draw not found for this tournament")
    if _games_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id):
        raise ValueError("This draw already has games. Clear or recreate the games before replacing teams.")

    before_rows = _existing_teams_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    next_rows = _normalize_team_rows(list(teams or []), draw=draw, tournament_id=clean_tournament_id)

    (
        supabase.table("tournament_teams")
        .delete()
        .eq("tournament_id", clean_tournament_id)
        .eq("draw_id", clean_draw_id)
        .execute()
    )
    inserted = _safe_rows(supabase.table("tournament_teams").insert(next_rows).execute()) if next_rows else []
    output_rows = [_team_payload(row) for row in (inserted or next_rows)]

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="replace_tournament_draw_teams_admin",
        entity_type="tournament_event_draw",
        entity_id=clean_draw_id,
        before_json={"draw": _draw_payload(draw), "teams": [_team_payload(row) for row in before_rows]},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "draw": _draw_payload(draw),
            "teams": output_rows,
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "tournament_draw_team_replace", "draw_id": clean_draw_id, "teams": output_rows, "updated_count": len(output_rows), "warnings": warnings}
