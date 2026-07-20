from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os
import uuid

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournaments import build_playoff_games, compute_round_robin_standings
from jupr_app.services.admin_tournament_draw_service import _draw_payload
from jupr_app.services.admin_tournament_game_service import (
    _game_payload,
    _insert_tournament_draw_games_atomic,
    _require_reviewed_draw_version,
    _require_reviewed_row_versions,
)
from jupr_app.services.admin_tournament_service import TOURNAMENT_SELECT, _clean_text, _first_row, is_admin_tournament_admin_enabled

CONFIRM_GENERATE_PLAYOFFS = "GENERATE PLAYOFFS"
SUPPORTED_ADVANCE_COUNTS = {4, 5, 6}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


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
    except Exception as exc:
        raise RuntimeError("Could not verify the tournament draw; playoff generation was refused.") from exc
    return rows[0] if rows else None


def _teams_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("tournament_teams")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not load draw teams; playoff generation was refused.") from exc
    return sorted(rows, key=lambda row: int(_safe_int(row.get("team_number")) or 0))


def _games_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("tournament_games")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not load draw games; playoff generation was refused.") from exc
    return rows


def _podium_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table("tournament_podium")
            .select("id")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not verify podium state; playoff generation was refused.") from exc


def _round_robin_complete(games: list[dict[str, Any]]) -> bool:
    rr_games = [row for row in games if _clean_text(row.get("stage"), limit=80).upper() == "ROUND_ROBIN"]
    return bool(rr_games) and all(row.get("winner_team_id") and row.get("score_a") is not None and row.get("score_b") is not None for row in rr_games)


def generate_admin_tournament_playoff_games(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    advance_count: Any = None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    expected_draw_updated_at: str | None = None,
    expected_team_versions: list[dict[str, Any]] | None = None,
    expected_source_game_versions: list[dict[str, Any]] | None = None,
    source: str = "next_tournament_admin_generate_playoffs",
    dry_run: bool = False,
    atomic: bool = False,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_GENERATE_PLAYOFFS:
        raise ValueError(f"Type {CONFIRM_GENERATE_PLAYOFFS} to generate playoff games.")

    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_draw_id = _clean_text(draw_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    draw = _fetch_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not draw:
        raise ValueError("draw not found for this tournament")
    reviewed_draw_version = _require_reviewed_draw_version(
        draw,
        expected_draw_updated_at=expected_draw_updated_at,
        atomic=atomic,
    )

    teams = _teams_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    games = _games_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    reviewed_team_versions = _require_reviewed_row_versions(
        teams,
        expected_team_versions,
        label="team set",
        atomic=atomic,
    )
    reviewed_source_game_versions = _require_reviewed_row_versions(
        games,
        expected_source_game_versions,
        label="source game set",
        atomic=atomic,
    )
    if _podium_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id):
        raise ValueError("This draw already has a podium. Playoff generation is locked after podium review or awards.")
    if any(_clean_text(row.get("stage"), limit=80).upper() == "PLAYOFF" for row in games):
        raise ValueError("This draw already has playoff games.")
    if not _round_robin_complete(games):
        raise ValueError("All round-robin games must be scored before generating playoffs.")

    count = _safe_int(advance_count) or min(4, len(teams))
    if count not in SUPPORTED_ADVANCE_COUNTS:
        raise ValueError("Playoff generation currently supports advance counts of 4, 5, or 6.")
    if count > len(teams):
        raise ValueError("Advance count cannot exceed the number of teams in the draw.")

    standings = compute_round_robin_standings(teams, [row for row in games if _clean_text(row.get("stage"), limit=80).upper() == "ROUND_ROBIN"])
    now = _now_iso()
    playoff_rows: list[dict[str, Any]] = []
    for row in build_playoff_games(tournament_id=clean_tournament_id, advance_count=count, standings=standings):
        playoff_rows.append(
            {
                **row,
                "id": str(uuid.uuid4()),
                "draw_id": clean_draw_id,
                "registration_day_id": _clean_text(draw.get("registration_day_id"), limit=120) or None,
                "event_option_id": _clean_text(draw.get("event_option_id"), limit=120) or None,
                "created_at": now,
                "updated_at": now,
            }
        )
    if dry_run:
        playoff_games = [_game_payload(row) for row in playoff_rows]
        return {
            "ok": True,
            "mode": "tournament_playoff_generate_preview",
            "dry_run": True,
            "write_count": 0,
            "draw_id": clean_draw_id,
            "advance_count": count,
            "standings": standings,
            "game_count": len(playoff_games),
            "games": playoff_games,
            "warnings": [],
        }
    inserted = (
        _insert_tournament_draw_games_atomic(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            draw_id=clean_draw_id,
            expected_draw_updated_at=reviewed_draw_version,
            expected_team_versions=reviewed_team_versions,
            expected_source_game_versions=reviewed_source_game_versions,
            mode="PLAYOFF",
            rows=playoff_rows,
        )
        if atomic
        else (_safe_rows(supabase.table("tournament_games").insert(playoff_rows).execute()) if playoff_rows else [])
    )
    playoff_games = [_game_payload(row) for row in (inserted or playoff_rows)]

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="generate_tournament_playoff_games_admin",
        entity_type="tournament_event_draw",
        entity_id=clean_draw_id,
        before_json={"draw": _draw_payload(draw), "advance_count": count, "existing_game_count": len(games)},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "draw": _draw_payload(draw),
            "advance_count": count,
            "standings": standings,
            "playoff_game_count": len(playoff_games),
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
    return {"ok": True, "mode": "tournament_playoff_generate", "draw_id": clean_draw_id, "advance_count": count, "standings": standings, "game_count": len(playoff_games), "games": playoff_games, "warnings": warnings}
