from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os
import uuid

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournaments import SUPPORTED_TEAM_COUNTS, build_round_robin_games
from jupr_app.services.admin_tournament_draw_service import _draw_payload
from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError
from jupr_app.services.admin_tournament_service import TOURNAMENT_SELECT, _clean_text, _first_row, is_admin_tournament_admin_enabled

CONFIRM_GENERATE_GAMES = "GENERATE GAMES"


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
        raise RuntimeError("Could not verify the tournament draw; game generation was refused.") from exc
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
        raise RuntimeError("Could not load draw teams; game generation was refused.") from exc
    return sorted(rows, key=lambda row: int(_safe_int(row.get("team_number")) or 0))


def _games_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table("tournament_games")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not verify whether this draw already has games; game generation was refused.") from exc


def _game_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _clean_text(row.get("id"), limit=120),
        "tournament_id": _clean_text(row.get("tournament_id"), limit=120),
        "draw_id": _clean_text(row.get("draw_id"), limit=120) or None,
        "registration_day_id": _clean_text(row.get("registration_day_id"), limit=120) or None,
        "event_option_id": _clean_text(row.get("event_option_id"), limit=120) or None,
        "stage": _clean_text(row.get("stage"), limit=80),
        "rr_round_number": _safe_int(row.get("rr_round_number")),
        "rr_slot_number": _safe_int(row.get("rr_slot_number")),
        "team_a_id": _clean_text(row.get("team_a_id"), limit=120) or None,
        "team_b_id": _clean_text(row.get("team_b_id"), limit=120) or None,
        "score_a": _safe_int(row.get("score_a")),
        "score_b": _safe_int(row.get("score_b")),
        "winner_team_id": _clean_text(row.get("winner_team_id"), limit=120) or None,
        "loser_team_id": _clean_text(row.get("loser_team_id"), limit=120) or None,
        "finalized_at": row.get("finalized_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _require_reviewed_draw_version(
    draw: dict[str, Any],
    *,
    expected_draw_updated_at: str | None,
    atomic: bool,
) -> str:
    reviewed = str(expected_draw_updated_at or "").strip()
    if atomic and not reviewed:
        raise StaleTournamentAdminStateError(
            "A reviewed draw version is required for staging game generation. Reload the Ops snapshot."
        )
    if reviewed and str(draw.get("updated_at") or "") != reviewed:
        raise StaleTournamentAdminStateError(
            "This tournament draw changed after it was reviewed. Reload the Ops snapshot before generating games."
        )
    return reviewed


def _canonical_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")
    except Exception:
        return text


def _require_reviewed_row_versions(
    current_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]] | None,
    *,
    label: str,
    atomic: bool,
) -> list[dict[str, str]]:
    reviewed = [
        {"id": str(row.get("id") or "").strip(), "updated_at": str(row.get("updated_at") or "").strip()}
        for row in (expected_rows or [])
    ]
    if atomic and not reviewed:
        raise StaleTournamentAdminStateError(
            f"A reviewed {label} snapshot is required for this staging mutation. Reload the Ops snapshot."
        )
    expected_map = {
        row["id"]: _canonical_timestamp(row["updated_at"])
        for row in reviewed
        if row["id"] and row["updated_at"]
    }
    if len(expected_map) != len(reviewed):
        raise StaleTournamentAdminStateError(
            f"The reviewed {label} snapshot is incomplete or duplicated. Reload the Ops snapshot."
        )
    current_map = {
        str(row.get("id") or "").strip(): _canonical_timestamp(row.get("updated_at"))
        for row in current_rows
        if str(row.get("id") or "").strip() and str(row.get("updated_at") or "").strip()
    }
    if reviewed and (len(current_map) != len(current_rows) or current_map != expected_map):
        raise StaleTournamentAdminStateError(
            f"The tournament {label} changed after review. Reload the Ops snapshot before continuing."
        )
    return sorted(reviewed, key=lambda row: row["id"])


def _insert_tournament_draw_games_atomic(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    expected_draw_updated_at: str,
    expected_team_versions: list[dict[str, str]],
    expected_source_game_versions: list[dict[str, str]],
    mode: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        response = supabase.rpc(
            "admin_insert_tournament_draw_games_cas",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament_id),
                "p_draw_id": str(draw_id),
                "p_expected_draw_updated_at": str(expected_draw_updated_at),
                "p_mode": str(mode),
                "p_expected_teams": list(expected_team_versions),
                "p_expected_source_games": list(expected_source_game_versions),
                "p_games": list(rows),
            },
        ).execute()
    except Exception as exc:
        detail = str(exc)
        if any(
            marker in detail
            for marker in (
                "JUPR_TOURNAMENT_DRAW_STALE",
                "JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE",
                "JUPR_TOURNAMENT_SOURCE_GAME_SNAPSHOT_STALE",
            )
        ):
            raise StaleTournamentAdminStateError(
                "The draw, team set, or source game set changed while games were being generated. Reload the Ops snapshot."
            ) from exc
        raise RuntimeError("Atomic tournament game generation failed; no game set was committed.") from exc
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        saved = data.get("games")
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        saved = data[0].get("games")
    else:
        saved = None
    if not isinstance(saved, list):
        raise RuntimeError("Atomic tournament game generation returned no saved game set.")
    return [dict(row) for row in saved if isinstance(row, dict)]


def generate_admin_tournament_round_robin_games(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    expected_draw_updated_at: str | None = None,
    expected_team_versions: list[dict[str, Any]] | None = None,
    source: str = "next_tournament_admin_generate_round_robin",
    dry_run: bool = False,
    atomic: bool = False,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_GENERATE_GAMES:
        raise ValueError(f"Type {CONFIRM_GENERATE_GAMES} to generate round-robin games.")

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
    if _games_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id):
        raise ValueError("This draw already has games. Delete/recreate the draw or clear games before regenerating.")

    teams = _teams_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    reviewed_team_versions = _require_reviewed_row_versions(
        teams,
        expected_team_versions,
        label="team set",
        atomic=atomic,
    )
    team_count = len(teams)
    if team_count not in SUPPORTED_TEAM_COUNTS:
        raise ValueError(f"Round-robin generation supports {SUPPORTED_TEAM_COUNTS}; this draw has {team_count} teams.")
    team_ids_by_number: dict[int, str] = {}
    for team in teams:
        team_number = _safe_int(team.get("team_number"))
        team_id = _clean_text(team.get("id"), limit=120)
        if team_number is None or not team_id:
            raise ValueError("Every team must have a team number and id before generating games.")
        team_ids_by_number[int(team_number)] = team_id
    if sorted(team_ids_by_number) != list(range(1, team_count + 1)):
        raise ValueError("Team numbers must be contiguous from 1 through the draw size before generating games.")

    now = _now_iso()
    game_rows: list[dict[str, Any]] = []
    for row in build_round_robin_games(tournament_id=clean_tournament_id, team_ids_by_number=team_ids_by_number):
        game_rows.append(
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
        games = [_game_payload(row) for row in game_rows]
        return {
            "ok": True,
            "mode": "tournament_round_robin_generate_preview",
            "dry_run": True,
            "write_count": 0,
            "draw_id": clean_draw_id,
            "game_count": len(games),
            "games": games,
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
            expected_source_game_versions=[],
            mode="ROUND_ROBIN",
            rows=game_rows,
        )
        if atomic
        else (_safe_rows(supabase.table("tournament_games").insert(game_rows).execute()) if game_rows else [])
    )
    games = [_game_payload(row) for row in (inserted or game_rows)]

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="generate_tournament_round_robin_games_admin",
        entity_type="tournament_event_draw",
        entity_id=clean_draw_id,
        before_json={"draw": _draw_payload(draw), "teams": len(teams), "games": 0},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "draw": _draw_payload(draw),
            "game_count": len(games),
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
    return {"ok": True, "mode": "tournament_round_robin_generate", "draw_id": clean_draw_id, "game_count": len(games), "games": games, "warnings": warnings}
