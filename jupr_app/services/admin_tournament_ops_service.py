from __future__ import annotations

from typing import Any

from jupr_app.services.admin_tournament_service import (
    TOURNAMENT_SELECT,
    _clean_text,
    _first_row,
    _tournament_payload,
    is_admin_tournament_admin_enabled,
)

OPS_TABLES = {
    "draws": "tournament_event_draws",
    "teams": "tournament_teams",
    "games": "tournament_games",
    "podium": "tournament_podium",
}


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _table_rows(supabase: Any, table_name: str, *, tournament_id: str, limit: int = 1000) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        rows = _safe_rows(
            supabase.table(table_name)
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .limit(max(1, min(int(limit), 2000)))
            .execute()
        )
        return rows, []
    except Exception as exc:  # noqa: BLE001 - surface operational table availability without failing the whole ops view
        return [], [f"{table_name} unavailable: {exc.__class__.__name__}"]


def _maybe_filter_draw_id(rows: list[dict[str, Any]], draw_id: str | None) -> list[dict[str, Any]]:
    if not draw_id:
        return rows
    clean_draw_id = str(draw_id)
    return [row for row in rows if str(row.get("draw_id") or row.get("id") or "") == clean_draw_id]


def _sort_rows(rows: list[dict[str, Any]], *keys: str) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(row.get(key) or "") for key in keys)

    return sorted(rows, key=sort_key)


def get_admin_tournament_ops_snapshot(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str | None = None,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    if not clean_tournament_id:
        raise ValueError("tournament_id is required")
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")

    warnings: list[str] = []
    draws, draw_warnings = _table_rows(supabase, OPS_TABLES["draws"], tournament_id=clean_tournament_id)
    teams, team_warnings = _table_rows(supabase, OPS_TABLES["teams"], tournament_id=clean_tournament_id)
    games, game_warnings = _table_rows(supabase, OPS_TABLES["games"], tournament_id=clean_tournament_id)
    podium, podium_warnings = _table_rows(supabase, OPS_TABLES["podium"], tournament_id=clean_tournament_id)
    warnings.extend([*draw_warnings, *team_warnings, *game_warnings, *podium_warnings])

    clean_draw_id = _clean_text(draw_id, limit=120) or None
    if clean_draw_id:
        teams = [row for row in teams if str(row.get("draw_id") or "") == clean_draw_id]
        games = [row for row in games if str(row.get("draw_id") or "") == clean_draw_id]
        podium = [row for row in podium if str(row.get("draw_id") or "") == clean_draw_id]
        draws = [row for row in draws if str(row.get("id") or "") == clean_draw_id]

    draws = _sort_rows(draws, "registration_day_id", "event_option_id", "name", "id")
    teams = _sort_rows(teams, "draw_id", "team_number", "id")
    games = _sort_rows(games, "draw_id", "stage", "rr_round_number", "rr_slot_number", "game_number", "id")
    podium = _sort_rows(podium, "draw_id", "placement", "id")

    return {
        "ok": True,
        "mode": "tournament_ops_snapshot",
        "tournament": _tournament_payload(tournament),
        "draw_id": clean_draw_id,
        "summary": {
            "draws": len(draws),
            "teams": len(teams),
            "games": len(games),
            "podium": len(podium),
            "completed_games": len([row for row in games if str(row.get("status") or "").lower() in {"complete", "completed", "final"} or row.get("winner_team_id")]),
        },
        "draws": draws,
        "teams": teams,
        "games": games,
        "podium": podium,
        "warnings": warnings,
    }
