from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from jupr_app.domain.admin.role_assignments import (
    delete_role_assignment,
    has_other_super_admin_support,
    list_role_assignments,
    normalize_email,
    upsert_role_assignment,
)
from jupr_app.domain.admin.roles import ALL_ROLES, ROLE_SUPER_ADMIN, normalize_role
from jupr_app.domain.admin_activity_log import (
    RETENTION_DAYS,
    build_activity_payload,
    list_recent_admin_activity_logs,
    retention_cutoff_iso,
    write_admin_activity_log,
)
from jupr_app.domain.gamification.badge_worker import (
    process_badge_eval_queue,
    process_badge_eval_queue_until_empty,
)
from jupr_app.domain.gamification.recompute import run_badge_recompute
from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.tournament_match_payload import build_tournament_match_payload

TRUTHY = {"1", "true", "yes", "y", "on"}
CONFIRM_ROLE = "SAVE ROLE"
CONFIRM_REVOKE = "REVOKE ROLE"
CONFIRM_QUEUE_BATCH = "PROCESS BADGE QUEUE"
CONFIRM_QUEUE_DRAIN = "DRAIN BADGE QUEUE"
CONFIRM_BADGE_RECOMPUTE = "RUN BADGE RECOMPUTE"
CONFIRM_TOURNAMENT_MATCH_BACKFILL = "BACKFILL TOURNAMENT MATCHES"
MAX_TOURNAMENT_MATCH_BACKFILL_APPLY = 100
MAX_ADMIN_RATING_REPORT_ROWS = 10000
SNAPSHOT_COLUMNS = {
    "t1_p1_r",
    "t1_p2_r",
    "t2_p1_r",
    "t2_p2_r",
    "t1_p1_r_end",
    "t1_p2_r_end",
    "t2_p1_r_end",
    "t2_p2_r_end",
}
QUEUE_STATUS_VALUES = ("pending", "processing", "done", "error")
BADGE_RECOMPUTE_MODES = ("dry-run", "append-only", "strict")


def is_admin_tools_enabled() -> bool:
    return os.getenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "").strip().lower() in TRUTHY


def build_admin_tools_status(*, club_id: str) -> dict[str, Any]:
    enabled = is_admin_tools_enabled()
    return {
        "ok": True,
        "enabled": enabled,
        "status": "ready" if enabled else "disabled",
        "club_id": str(club_id),
        "roles": list(ALL_ROLES),
        "retention_days": RETENTION_DAYS,
        "retention_cutoff": retention_cutoff_iso(),
        "required_permissions": {
            "overview": "view_audit_log",
            "roles": "manage_roles",
            "workers": "run_replay",
            "rating_reports": "view_audit_log",
            "tournament_backfill_preview": "view_audit_log",
            "tournament_backfill_apply": "run_replay",
        },
        "confirmation_text": {
            "save": CONFIRM_ROLE,
            "revoke": CONFIRM_REVOKE,
            "process_queue": CONFIRM_QUEUE_BATCH,
            "drain_queue": CONFIRM_QUEUE_DRAIN,
            "badge_recompute": CONFIRM_BADGE_RECOMPUTE,
            "tournament_match_backfill": CONFIRM_TOURNAMENT_MATCH_BACKFILL,
        },
    }


def _sample_match_schema(supabase: Any, *, club_id: str) -> dict[str, Any]:
    try:
        rows = supabase.table("matches").select("*").eq("club_id", str(club_id)).limit(1).execute().data or []
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return {"ok": False, "message": f"Unable to inspect matches table: {exc}"}
    if not rows:
        return {"ok": True, "sample_found": False, "snapshot_columns_present": False, "missing_snapshot_columns": sorted(SNAPSHOT_COLUMNS)}
    keys = set((rows[0] or {}).keys())
    missing = sorted(SNAPSHOT_COLUMNS - keys)
    return {"ok": True, "sample_found": True, "snapshot_columns_present": not missing, "missing_snapshot_columns": missing}


def _null_snapshot_count(supabase: Any, *, club_id: str) -> dict[str, Any]:
    try:
        rows = (
            supabase.table("matches")
            .select("id,t1_p1_r,t1_p1_r_end")
            .eq("club_id", str(club_id))
            .is_("t1_p1_r", None)
            .limit(5000)
            .execute()
            .data
            or []
        )
        return {"ok": True, "sample_limit": 5000, "null_snapshot_count": len(rows), "sample_match_ids": [row.get("id") for row in rows[:25]]}
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return {"ok": False, "message": f"Unable to inspect null snapshots: {exc}"}


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except Exception:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _chunks(values: list[str], size: int = 100) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _preview_fingerprint(*, club_id: str, candidate_limit: int, rows: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        {
            "club_id": str(club_id),
            "candidate_limit": int(candidate_limit),
            "candidates": rows,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _strict_table_frame(supabase: Any, table_name: str, *, club_id: str) -> pd.DataFrame:
    try:
        rows = _safe_rows(
            supabase.table(table_name)
            .select("*")
            .eq("club_id", str(club_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to load club-scoped {table_name} for tournament match backfill.") from exc
    return pd.DataFrame(rows)


def _table_count(supabase: Any, table: str, *, club_id: str, status: str | None = None, limit: int = 10000) -> dict[str, Any]:
    try:
        query = supabase.table(table).select("id,status").eq("club_id", str(club_id))
        if status is not None:
            query = query.eq("status", str(status))
        rows = _safe_rows(query.limit(int(limit)).execute())
        return {"ok": True, "count": len(rows), "sample_limit": int(limit)}
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return {"ok": False, "count": None, "message": str(exc)}


def _badge_eval_run_count(supabase: Any, *, club_id: str, limit: int = 1000) -> dict[str, Any]:
    """Count recompute runs scoped to a club in badge_eval_runs.scope_json."""
    try:
        rows = _safe_rows(
            supabase.table("badge_eval_runs")
            .select("id,status,scope_json")
            .limit(int(limit))
            .execute()
        )
        scoped_rows = [
            row
            for row in rows
            if str((row.get("scope_json") or {}).get("club_id") or "") == str(club_id)
        ]
        return {"ok": True, "count": len(scoped_rows), "sample_limit": int(limit)}
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return {"ok": False, "count": None, "message": str(exc)}


def build_admin_worker_status(supabase: Any, *, club_id: str) -> dict[str, Any]:
    queue_counts = {
        status: _table_count(supabase, "badge_eval_queue", club_id=str(club_id), status=status)
        for status in QUEUE_STATUS_VALUES
    }
    eval_runs = _badge_eval_run_count(supabase, club_id=str(club_id), limit=1000)
    return {
        "ok": True,
        "queue_counts": queue_counts,
        "badge_recompute_run_count": eval_runs,
        "queue_modes": ["batch", "drain"],
        "badge_recompute_modes": list(BADGE_RECOMPUTE_MODES),
        "confirmation_text": {
            "batch": CONFIRM_QUEUE_BATCH,
            "drain": CONFIRM_QUEUE_DRAIN,
            "badge_recompute": CONFIRM_BADGE_RECOMPUTE,
        },
    }


def build_admin_rating_report(
    supabase: Any,
    *,
    club_id: str,
    league_name: str = "OVERALL",
) -> dict[str, Any]:
    """Build the club-scoped read-only ratings report exposed by Streamlit Admin Tools."""
    if not is_admin_tools_enabled():
        raise PermissionError("Next Admin Tools are disabled.")
    selected_scope = str(league_name or "OVERALL").strip() or "OVERALL"
    is_overall = selected_scope.upper() == "OVERALL"
    try:
        player_rows = _safe_rows(
            supabase.table("players")
            .select("id,name,rating,starting_rating,wins,losses,matches_played,active,inactive_at")
            .eq("club_id", str(club_id))
            .limit(MAX_ADMIN_RATING_REPORT_ROWS + 1)
            .execute()
        )
        metadata_rows = _safe_rows(
            supabase.table("leagues_metadata")
            .select("league_name")
            .eq("club_id", str(club_id))
            .limit(MAX_ADMIN_RATING_REPORT_ROWS + 1)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Unable to load club-scoped rating report inputs.") from exc

    available_leagues = sorted(
        {
            str(row.get("league_name") or "").strip()
            for row in metadata_rows[:MAX_ADMIN_RATING_REPORT_ROWS]
            if str(row.get("league_name") or "").strip()
            and str(row.get("league_name") or "").strip().upper() != "OVERALL"
        },
        key=str.casefold,
    )
    if not is_overall and selected_scope not in available_leagues:
        raise ValueError("Select a league available to this club's rating report.")

    players_truncated = len(player_rows) > MAX_ADMIN_RATING_REPORT_ROWS
    metadata_truncated = len(metadata_rows) > MAX_ADMIN_RATING_REPORT_ROWS
    player_rows = player_rows[:MAX_ADMIN_RATING_REPORT_ROWS]
    visible_players = [
        row
        for row in player_rows
        if "(merged into " not in str(row.get("name") or "").casefold()
    ]
    player_names = {
        int(player_id): str(row.get("name") or "")
        for row in visible_players
        if (player_id := _safe_int(row.get("id"))) is not None
    }

    source_rows: list[dict[str, Any]]
    source_truncated = players_truncated
    if is_overall:
        has_inactive_at = any("inactive_at" in row for row in visible_players)
        source_rows = [
            {
                **row,
                "player_id": row.get("id"),
                "report_name": str(row.get("name") or ""),
            }
            for row in visible_players
            if (
                not row.get("inactive_at")
                if has_inactive_at
                else row.get("active", row.get("is_active", True)) is not False
            )
        ]
        normalized_scope = "OVERALL"
    else:
        try:
            league_rows = _safe_rows(
                supabase.table("league_ratings")
                .select("player_id,league_name,rating,starting_rating,wins,losses,matches_played,is_active")
                .eq("club_id", str(club_id))
                .eq("league_name", selected_scope)
                .limit(MAX_ADMIN_RATING_REPORT_ROWS + 1)
                .execute()
            )
        except Exception as exc:
            raise RuntimeError("Unable to load the selected club league rating report.") from exc
        source_truncated = players_truncated or len(league_rows) > MAX_ADMIN_RATING_REPORT_ROWS
        source_rows = [
            {
                **row,
                "report_name": player_names.get(_safe_int(row.get("player_id")) or -1, ""),
            }
            for row in league_rows[:MAX_ADMIN_RATING_REPORT_ROWS]
        ]
        normalized_scope = selected_scope

    report_rows: list[dict[str, Any]] = []
    for row in source_rows:
        rating = _safe_float(row.get("rating"), 1200.0)
        starting_rating = _safe_float(row.get("starting_rating"), rating)
        wins = _safe_int(row.get("wins")) or 0
        losses = _safe_int(row.get("losses")) or 0
        matches_played = _safe_int(row.get("matches_played"))
        if matches_played is None:
            matches_played = wins + losses
        report_rows.append(
            {
                "_rating_sort": rating,
                "player_id": _safe_int(row.get("player_id")),
                "name": str(row.get("report_name") or ""),
                "jupr": round(rating / 400.0, 4),
                "wins": wins,
                "losses": losses,
                "matches_played": matches_played,
                "win_percent": round((float(wins) / float(max(matches_played, 1))) * 100.0, 2),
                "gain": round((rating - starting_rating) / 400.0, 4),
            }
        )
    report_rows.sort(
        key=lambda row: (
            -float(row.get("_rating_sort") or 0.0),
            str(row.get("name") or "").casefold(),
            int(row.get("player_id") or 0),
        )
    )
    for row in report_rows:
        row.pop("_rating_sort", None)
    truncated = bool(source_truncated or metadata_truncated)
    warnings = (
        [f"The report reached its {MAX_ADMIN_RATING_REPORT_ROWS:,}-row safety limit and may be incomplete."]
        if truncated
        else []
    )
    return {
        "ok": True,
        "mode": "admin_rating_report",
        "read_only": True,
        "scope": normalized_scope,
        "available_scopes": ["OVERALL", *available_leagues],
        "generated_on": datetime.now(timezone.utc).date().isoformat(),
        "summary": {
            "row_count": len(report_rows),
            "row_limit": MAX_ADMIN_RATING_REPORT_ROWS,
            "truncated": truncated,
        },
        "columns": ["name", "jupr", "wins", "losses", "matches_played", "win_percent", "gain"],
        "rows": report_rows,
        "warnings": warnings,
    }


def build_admin_tournament_match_backfill_preview(
    supabase: Any,
    *,
    club_id: str,
    candidate_limit: int = 500,
) -> dict[str, Any]:
    """Read-only inventory of finalized tournament games missing official matches."""
    if not is_admin_tools_enabled():
        raise PermissionError("Next Admin Tools are disabled.")
    safe_candidate_limit = max(1, min(int(candidate_limit or 500), 1000))
    try:
        tournaments = _safe_rows(
            supabase.table("tournaments")
            .select("id,name,club_id")
            .eq("club_id", str(club_id))
            .limit(1001)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Unable to inspect club tournaments for backfill preview.") from exc
    tournaments_truncated = len(tournaments) > 1000
    tournaments = tournaments[:1000]
    tournaments_by_id = {
        str(row.get("id")): row
        for row in tournaments
        if str(row.get("id") or "").strip()
    }
    tournament_ids = sorted(tournaments_by_id)
    if not tournament_ids:
        fingerprint = _preview_fingerprint(
            club_id=str(club_id),
            candidate_limit=safe_candidate_limit,
            rows=[],
        )
        return {
            "ok": True,
            "mode": "tournament_match_backfill_preview",
            "read_only": True,
            "preview_fingerprint": fingerprint,
            "confirmation_text": CONFIRM_TOURNAMENT_MATCH_BACKFILL,
            "summary": {
                "tournament_count": 0,
                "finalized_game_count": 0,
                "already_published_count": 0,
                "missing_match_count": 0,
                "ready_count": 0,
                "blocked_count": 0,
                "candidate_count": 0,
                "candidate_limit": safe_candidate_limit,
                "apply_limit": MAX_TOURNAMENT_MATCH_BACKFILL_APPLY,
                "truncated": tournaments_truncated,
            },
            "candidates": [],
            "warnings": ["Tournament scan reached its 1,000-row safety limit."] if tournaments_truncated else [],
        }

    games: list[dict[str, Any]] = []
    teams: list[dict[str, Any]] = []
    players: list[dict[str, Any]] = []
    scan_truncated = tournaments_truncated
    try:
        for tournament_chunk in _chunks(tournament_ids):
            game_chunk = _safe_rows(
                supabase.table("tournament_games")
                .select("id,tournament_id,team_a_id,team_b_id,score_a,score_b,finalized_at")
                .in_("tournament_id", tournament_chunk)
                .limit(5001)
                .execute()
            )
            team_chunk = _safe_rows(
                supabase.table("tournament_teams")
                .select("id,tournament_id,player1_id,player2_id")
                .in_("tournament_id", tournament_chunk)
                .limit(10001)
                .execute()
            )
            scan_truncated = scan_truncated or len(game_chunk) > 5000 or len(team_chunk) > 10000
            games.extend(game_chunk[:5000])
            teams.extend(team_chunk[:10000])
        players = _safe_rows(
            supabase.table("players")
            .select("id,club_id")
            .eq("club_id", str(club_id))
            .limit(10001)
            .execute()
        )
        scan_truncated = scan_truncated or len(players) > 10000
        players = players[:10000]
    except Exception as exc:
        raise RuntimeError("Unable to inspect tournament games, teams, and players for backfill preview.") from exc

    finalized_games = [row for row in games if row.get("finalized_at") is not None]
    game_ids = sorted(
        {
            str(row.get("id"))
            for row in finalized_games
            if str(row.get("id") or "").strip()
        }
    )
    published_game_ids: set[str] = set()
    try:
        for game_chunk in _chunks(game_ids, size=200):
            rows = _safe_rows(
                supabase.table("matches")
                .select("id,tournament_game_id")
                .eq("club_id", str(club_id))
                .in_("tournament_game_id", game_chunk)
                .execute()
            )
            published_game_ids.update(
                str(row.get("tournament_game_id"))
                for row in rows
                if str(row.get("tournament_game_id") or "").strip()
            )
    except Exception as exc:
        raise RuntimeError("Unable to verify existing official tournament matches.") from exc

    teams_by_key = {
        (str(row.get("tournament_id") or ""), str(row.get("id") or "")): row
        for row in teams
        if str(row.get("id") or "").strip()
    }
    club_player_ids = {
        str(row.get("id"))
        for row in players
        if str(row.get("id") or "").strip()
    }
    candidates: list[dict[str, Any]] = []
    fingerprint_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    missing_games = [row for row in finalized_games if str(row.get("id") or "") not in published_game_ids]
    for game in sorted(missing_games, key=lambda row: (str(row.get("tournament_id") or ""), str(row.get("id") or ""))):
        tournament_id = str(game.get("tournament_id") or "")
        game_id = str(game.get("id") or "")
        tournament = tournaments_by_id.get(tournament_id)
        score_a = _safe_int(game.get("score_a"))
        score_b = _safe_int(game.get("score_b"))
        team_a = teams_by_key.get((tournament_id, str(game.get("team_a_id") or "")))
        team_b = teams_by_key.get((tournament_id, str(game.get("team_b_id") or "")))
        match_payload: dict[str, Any] | None = None
        if not tournament or not game_id:
            status, reason = "incomplete_tournament", "Tournament or game identity is incomplete."
        elif score_a is None or score_b is None or score_a + score_b <= 0:
            status, reason = "empty_score", "Finalized game has no non-zero score."
        elif score_a == score_b:
            status, reason = "tied_score", "Official tournament matches cannot use a tied score."
        elif not team_a or not team_b:
            status, reason = "incomplete_team", "One or both linked teams are missing."
        else:
            match_payload = build_tournament_match_payload(
                tournament,
                game,
                {
                    team_a.get("id"): team_a,
                    str(team_a.get("id")): team_a,
                    team_b.get("id"): team_b,
                    str(team_b.get("id")): team_b,
                },
                score_a=score_a,
                score_b=score_b,
            )
            required_players = ("t1_p1", "t1_p2", "t2_p1", "t2_p2")
            if any(match_payload.get(key) is None for key in required_players):
                status, reason = "incomplete_team", "Backfill parity requires two linked players on each team."
                match_payload = None
            elif any(str(match_payload.get(key)) not in club_player_ids for key in required_players):
                status, reason = "missing_player", "One or more linked players do not belong to this club."
                match_payload = None
            elif len({str(match_payload.get(key)) for key in required_players}) != len(required_players):
                status, reason = "invalid_players", "A player cannot occupy multiple positions in one official match."
                match_payload = None
            else:
                status, reason = "ready", "Finalized doubles game can be submitted through the existing Python match service."
        status_counts[status] = status_counts.get(status, 0) + 1
        fingerprint_rows.append(
            {
                "game_id": game_id or None,
                "tournament_id": tournament_id or None,
                "tournament_name": str((tournament or {}).get("name") or ""),
                "team_a_id": str(game.get("team_a_id") or ""),
                "team_b_id": str(game.get("team_b_id") or ""),
                "team_a_players": [
                    (team_a or {}).get("player1_id"),
                    (team_a or {}).get("player2_id"),
                ],
                "team_b_players": [
                    (team_b or {}).get("player1_id"),
                    (team_b or {}).get("player2_id"),
                ],
                "score_a": score_a,
                "score_b": score_b,
                "finalized_at": str(game.get("finalized_at") or ""),
                "status": status,
            }
        )
        if len(candidates) < safe_candidate_limit:
            candidates.append(
                {
                    "game_id": game_id or None,
                    "tournament_id": tournament_id or None,
                    "tournament_name": str((tournament or {}).get("name") or ""),
                    "team_a_id": game.get("team_a_id"),
                    "team_b_id": game.get("team_b_id"),
                    "score_a": score_a,
                    "score_b": score_b,
                    "status": status,
                    "reason": reason,
                    "match_payload": match_payload,
                }
            )
    candidates_truncated = len(missing_games) > safe_candidate_limit
    scan_truncated = scan_truncated or candidates_truncated
    warnings: list[str] = []
    if scan_truncated:
        warnings.append("The preview reached a scan or candidate safety limit; do not use it as a complete backfill count.")
    warnings.append("Preview only: no official match, rating, snapshot, badge, or tournament row was written.")
    ready_count = int(status_counts.get("ready", 0))
    fingerprint = _preview_fingerprint(
        club_id=str(club_id),
        candidate_limit=safe_candidate_limit,
        rows=fingerprint_rows,
    )
    return {
        "ok": True,
        "mode": "tournament_match_backfill_preview",
        "read_only": True,
        "preview_fingerprint": fingerprint,
        "confirmation_text": CONFIRM_TOURNAMENT_MATCH_BACKFILL,
        "summary": {
            "tournament_count": len(tournaments),
            "finalized_game_count": len(finalized_games),
            "already_published_count": len(published_game_ids.intersection(set(game_ids))),
            "missing_match_count": len(missing_games),
            "ready_count": ready_count,
            "blocked_count": len(missing_games) - ready_count,
            "candidate_count": len(candidates),
            "candidate_limit": safe_candidate_limit,
            "apply_limit": MAX_TOURNAMENT_MATCH_BACKFILL_APPLY,
            "truncated": scan_truncated,
            "status_counts": status_counts,
        },
        "candidates": candidates,
        "warnings": warnings,
    }


def apply_admin_tournament_match_backfill(
    supabase: Any,
    *,
    club_id: str,
    game_ids: list[str],
    preview_fingerprint: str,
    preview_limit: int = 500,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    source: str = "next_admin_tools_tournament_match_backfill",
) -> dict[str, Any]:
    """Apply an explicitly reviewed subset of ready legacy tournament games."""
    if not is_admin_tools_enabled():
        raise PermissionError("Next Admin Tools are disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_TOURNAMENT_MATCH_BACKFILL:
        raise ValueError(
            f"Type {CONFIRM_TOURNAMENT_MATCH_BACKFILL} to apply the tournament match backfill."
        )
    selected_game_ids = sorted(
        {
            str(value).strip()
            for value in (game_ids or [])
            if str(value or "").strip()
        }
    )
    if not selected_game_ids:
        raise ValueError("Select at least one ready tournament game to backfill.")
    if len(selected_game_ids) > MAX_TOURNAMENT_MATCH_BACKFILL_APPLY:
        raise ValueError(
            f"Tournament match backfill is capped at {MAX_TOURNAMENT_MATCH_BACKFILL_APPLY} games per reviewed apply."
        )

    preview = build_admin_tournament_match_backfill_preview(
        supabase,
        club_id=str(club_id),
        candidate_limit=int(preview_limit),
    )
    if bool((preview.get("summary") or {}).get("truncated")):
        raise ValueError("The tournament backfill preview is truncated; narrow the data set before applying it.")
    if str(preview_fingerprint or "").strip() != str(preview.get("preview_fingerprint") or ""):
        raise ValueError("Tournament backfill preview is stale. Reload and review the candidates again.")
    candidates_by_id = {
        str(row.get("game_id")): row
        for row in (preview.get("candidates") or [])
        if str(row.get("game_id") or "").strip()
    }
    missing_ids = [game_id for game_id in selected_game_ids if game_id not in candidates_by_id]
    if missing_ids:
        raise ValueError("Selected tournament games are no longer missing official matches: " + ", ".join(missing_ids))
    blocked = [
        f"{game_id} ({candidates_by_id[game_id].get('status')})"
        for game_id in selected_game_ids
        if str(candidates_by_id[game_id].get("status") or "") != "ready"
    ]
    if blocked:
        raise ValueError("Only ready tournament games can be backfilled: " + ", ".join(blocked))

    try:
        duplicate_rows = _safe_rows(
            supabase.table("matches")
            .select("id,club_id,tournament_game_id")
            .in_("tournament_game_id", selected_game_ids)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Unable to recheck tournament match duplicates before backfill.") from exc
    duplicate_ids = sorted(
        {
            str(row.get("tournament_game_id"))
            for row in duplicate_rows
            if str(row.get("tournament_game_id") or "").strip()
        }
    )
    if duplicate_ids:
        raise ValueError("Selected tournament games already have official matches: " + ", ".join(duplicate_ids))

    selected_candidates = [candidates_by_id[game_id] for game_id in selected_game_ids]
    match_payloads = [dict(row.get("match_payload") or {}) for row in selected_candidates]
    required_player_ids = {
        int(payload[key])
        for payload in match_payloads
        for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")
    }
    df_players_all = _strict_table_frame(supabase, "players", club_id=str(club_id))
    df_leagues = _strict_table_frame(supabase, "league_ratings", club_id=str(club_id))
    df_meta = _strict_table_frame(supabase, "leagues_metadata", club_id=str(club_id))
    loaded_player_ids = {
        int(coerced)
        for value in (df_players_all.get("id", pd.Series(dtype="int64")).tolist())
        if (coerced := _safe_int(value)) is not None
    }
    missing_player_ids = sorted(required_player_ids - loaded_player_ids)
    if missing_player_ids:
        raise ValueError(
            "Selected tournament games reference players outside this club: "
            + ", ".join(str(value) for value in missing_player_ids)
        )

    operation_id = hashlib.sha256(
        f"{club_id}:{preview_fingerprint}:{','.join(selected_game_ids)}".encode("utf-8")
    ).hexdigest()[:24]
    intent_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="tournament_match_backfill_apply_started",
        entity_type="tournament_match_backfill",
        entity_id=operation_id,
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "preview_fingerprint": str(preview_fingerprint),
            "selected_game_ids": selected_game_ids,
            "selected_count": len(selected_game_ids),
        },
        source_page=source,
        flagged_for_review=True,
    )
    intent_write = write_admin_activity_log(supabase, intent_payload)
    warnings: list[str] = []
    if intent_write.warning:
        warnings.append(intent_write.warning)
    if not intent_write.ok and os.getenv("JUPR_REQUIRE_API_AUDIT_LOG", "").strip().lower() in TRUTHY:
        raise RuntimeError("audit log write required but unavailable")

    process_result = process_matches(
        match_payloads,
        supabase=supabase,
        club_id=str(club_id),
        name_to_id={},
        df_players_all=df_players_all,
        df_leagues=df_leagues,
        df_meta=df_meta,
    )
    inserted_count = int(process_result.get("inserted") or 0)
    if inserted_count != len(match_payloads):
        raise RuntimeError(
            f"Tournament match backfill inserted {inserted_count} of {len(match_payloads)} reviewed games. "
            "Stop further writes and use Match Log plus Replay History to recover."
        )
    try:
        persisted_rows = _safe_rows(
            supabase.table("matches")
            .select("id,club_id,tournament_game_id")
            .eq("club_id", str(club_id))
            .in_("tournament_game_id", selected_game_ids)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Unable to verify persisted tournament backfill matches.") from exc
    persisted_ids = {
        str(row.get("tournament_game_id"))
        for row in persisted_rows
        if str(row.get("tournament_game_id") or "").strip()
    }
    if persisted_ids != set(selected_game_ids):
        raise RuntimeError(
            "Tournament match backfill persisted IDs differ from the reviewed selection. "
            "Stop further writes and use Match Log plus Replay History to recover."
        )

    completion_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="tournament_match_backfill_applied",
        entity_type="tournament_match_backfill",
        entity_id=operation_id,
        before_json={
            "preview_fingerprint": str(preview_fingerprint),
            "selected_game_ids": selected_game_ids,
        },
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "inserted_count": inserted_count,
            "persisted_game_ids": sorted(persisted_ids),
            "process_result": process_result,
        },
        source_page=source,
        flagged_for_review=True,
    )
    completion_write = write_admin_activity_log(supabase, completion_payload)
    if completion_write.warning:
        warnings.append(completion_write.warning)
    if not completion_write.ok and os.getenv("JUPR_REQUIRE_API_AUDIT_LOG", "").strip().lower() in TRUTHY:
        raise RuntimeError("audit log write required but unavailable")
    warnings.append(
        "If any post-apply verification disagrees, stop additional writes and recover through Match Log and Replay History."
    )
    return {
        "ok": True,
        "mode": "tournament_match_backfill_apply",
        "operation_id": operation_id,
        "preview_fingerprint": str(preview_fingerprint),
        "selected_game_ids": selected_game_ids,
        "inserted_count": inserted_count,
        "process_result": process_result,
        "warnings": warnings,
    }


def build_admin_tools_overview(
    supabase: Any,
    *,
    club_id: str,
    include_flagged_only: bool = False,
    activity_limit: int = 200,
) -> dict[str, Any]:
    if not is_admin_tools_enabled():
        raise PermissionError("Next Admin Tools are disabled.")
    role_rows = list_role_assignments(supabase, str(club_id))
    activity_rows, activity_warning = list_recent_admin_activity_logs(
        supabase,
        club_id=str(club_id),
        include_flagged_only=bool(include_flagged_only),
        limit=max(1, min(int(activity_limit or 200), 500)),
    )
    health = {
        "match_schema": _sample_match_schema(supabase, club_id=str(club_id)),
        "null_snapshots": _null_snapshot_count(supabase, club_id=str(club_id)),
        "workers": build_admin_worker_status(supabase, club_id=str(club_id)),
    }
    return {
        "ok": True,
        "roles": role_rows,
        "activity": activity_rows,
        "activity_warning": activity_warning,
        "health": health,
        "role_options": list(ALL_ROLES),
        "retention_days": RETENTION_DAYS,
        "retention_cutoff": retention_cutoff_iso(),
    }


def update_admin_role_assignment(
    supabase: Any,
    *,
    club_id: str,
    target_email: str,
    target_role: str,
    user_id: str | None,
    action: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_admin_tools_roles",
) -> dict[str, Any]:
    if not is_admin_tools_enabled():
        raise PermissionError("Next Admin Tools are disabled.")
    normalized_email = normalize_email(target_email)
    if "@" not in normalized_email or "." not in normalized_email.split("@")[-1]:
        raise ValueError("Enter a valid email address.")
    normalized_action = str(action or "").strip().lower()
    rows = list_role_assignments(supabase, str(club_id))
    existing = next((row for row in rows if normalize_email(row.get("email")) == normalized_email), None)

    if normalized_action == "upsert":
        if str(confirmation_text or "").strip().upper() != CONFIRM_ROLE:
            raise ValueError(f"Type {CONFIRM_ROLE} to save role assignments.")
        selected_role = normalize_role(target_role)
        if existing and normalize_role(existing.get("role")) == ROLE_SUPER_ADMIN and selected_role != ROLE_SUPER_ADMIN:
            if not has_other_super_admin_support(rows=rows, target_email=normalized_email, admin_allowlist=set()):
                raise ValueError("Unsafe change blocked: this would remove the final super_admin access.")
        upsert_role_assignment(supabase, str(club_id), normalized_email, selected_role, user_id=str(user_id or "").strip() or None)
        after = {"email": normalized_email, "role": selected_role, "user_id": str(user_id or "").strip() or None}
        action_type = "role_assignment_upsert"
        note = "Admin role assignment create/update"
    elif normalized_action == "revoke":
        if str(confirmation_text or "").strip().upper() != CONFIRM_REVOKE:
            raise ValueError(f"Type {CONFIRM_REVOKE} to revoke role assignments.")
        if existing and normalize_role(existing.get("role")) == ROLE_SUPER_ADMIN:
            if not has_other_super_admin_support(rows=rows, target_email=normalized_email, admin_allowlist=set()):
                raise ValueError("Unsafe revoke blocked: this would remove the final super_admin access.")
        delete_role_assignment(supabase, str(club_id), normalized_email)
        after = None
        action_type = "role_assignment_revoke"
        note = "Admin role assignment revoked"
    else:
        raise ValueError("action must be upsert or revoke")

    log_result = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type=action_type,
            entity_type="admin_role_assignment",
            entity_id=normalized_email,
            before_json={"role": str((existing or {}).get("role") or "") or None, "user_id": (existing or {}).get("user_id")},
            after_json=after,
            note=note,
            source_page=source,
            flagged_for_review=True,
        ),
    )
    return {
        "ok": True,
        "mode": "admin_role_assignment_update",
        "action": normalized_action,
        "target_email": normalized_email,
        "audit_warning": log_result.warning,
        "roles": list_role_assignments(supabase, str(club_id)),
    }


def run_admin_badge_queue_worker(
    supabase: Any,
    *,
    club_id: str,
    mode: str,
    max_jobs: int,
    time_budget_seconds: float,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_admin_tools_workers",
) -> dict[str, Any]:
    if not is_admin_tools_enabled():
        raise PermissionError("Next Admin Tools are disabled.")
    normalized_mode = str(mode or "batch").strip().lower()
    if normalized_mode not in {"batch", "drain"}:
        raise ValueError("mode must be batch or drain")
    expected = CONFIRM_QUEUE_DRAIN if normalized_mode == "drain" else CONFIRM_QUEUE_BATCH
    if str(confirmation_text or "").strip().upper() != expected:
        raise ValueError(f"Type {expected} to run the badge queue worker.")

    safe_max_jobs = max(1, min(int(max_jobs or 10), 5000))
    safe_budget = max(1.0, min(float(time_budget_seconds or 5.0), 180.0))
    before = build_admin_worker_status(supabase, club_id=str(club_id))
    if normalized_mode == "drain":
        result = process_badge_eval_queue_until_empty(
            supabase,
            str(club_id),
            max_total_jobs=safe_max_jobs,
            batch_max_jobs=min(50, safe_max_jobs),
            per_batch_time_budget_seconds=min(5.0, safe_budget),
            max_wall_clock_seconds=safe_budget,
            max_errors=max(1, min(50, safe_max_jobs)),
        )
    else:
        result = process_badge_eval_queue(
            supabase,
            max_jobs=safe_max_jobs,
            time_budget_seconds=int(safe_budget),
        )
    after = build_admin_worker_status(supabase, club_id=str(club_id))
    log_result = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="admin_badge_queue_worker_run",
            entity_type="badge_eval_queue",
            entity_id=normalized_mode,
            before_json={"worker_status": before},
            after_json={"mode": normalized_mode, "result": result, "worker_status": after},
            note="Admin Tools badge eval queue worker run",
            source_page=source,
            flagged_for_review=True,
        ),
    )
    return {"ok": True, "mode": normalized_mode, "result": result, "worker_status": after, "audit_warning": log_result.warning}


def run_admin_badge_recompute_job(
    supabase: Any,
    *,
    club_id: str,
    mode: str,
    player_id: int | None,
    badge_id: str | None,
    league_id: str | None,
    context_id: str | None,
    since: str | None,
    until: str | None,
    include_non_live: bool,
    allow_strict_global: bool,
    match_limit: int,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_admin_tools_badge_recompute",
) -> dict[str, Any]:
    if not is_admin_tools_enabled():
        raise PermissionError("Next Admin Tools are disabled.")
    normalized_mode = str(mode or "dry-run").strip().lower()
    if normalized_mode not in BADGE_RECOMPUTE_MODES:
        raise ValueError("mode must be dry-run, append-only, or strict")
    if normalized_mode != "dry-run" and str(confirmation_text or "").strip().upper() != CONFIRM_BADGE_RECOMPUTE:
        raise ValueError(f"Type {CONFIRM_BADGE_RECOMPUTE} to run an applying badge recompute.")
    summary = run_badge_recompute(
        supabase,
        club_id=str(club_id),
        mode=normalized_mode,
        league_id=str(league_id).strip() or None if league_id is not None else None,
        context_id=str(context_id).strip() or None if context_id is not None else None,
        player_id=int(player_id) if player_id is not None else None,
        badge_id=str(badge_id).strip() or None if badge_id is not None else None,
        since=str(since).strip() or None if since is not None else None,
        until=str(until).strip() or None if until is not None else None,
        created_by=actor_email,
        allow_strict_global=bool(allow_strict_global),
        match_limit=max(100, min(int(match_limit or 5000), 50000)),
        include_non_live=bool(include_non_live),
    )
    log_result = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="admin_badge_recompute_run",
            entity_type="player_badges",
            entity_id=f"badge_recompute:{normalized_mode}",
            after_json={"mode": normalized_mode, "summary": summary},
            note="Admin Tools badge recompute job",
            source_page=source,
            flagged_for_review=normalized_mode != "dry-run",
        ),
    )
    return {"ok": True, "mode": normalized_mode, "summary": summary, "audit_warning": log_result.warning}
