from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os
import uuid

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournaments import compute_podium_from_playoffs, compute_podium_from_rr
from jupr_app.services.admin_tournament_draw_service import _draw_payload
from jupr_app.services.admin_tournament_game_service import _require_reviewed_row_versions
from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError
from jupr_app.services.admin_tournament_service import TOURNAMENT_SELECT, _clean_text, _first_row, is_admin_tournament_admin_enabled

CONFIRM_GENERATE_PODIUM = "GENERATE PODIUM"


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
        raise RuntimeError("Could not verify the tournament draw; podium generation was refused.") from exc
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
        raise RuntimeError("Could not load draw teams; podium generation was refused.") from exc
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
        raise RuntimeError("Could not load draw games; podium generation was refused.") from exc
    return rows


def _podium_for_draw(supabase: Any, *, tournament_id: str, draw_id: str) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("tournament_podium")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .eq("draw_id", str(draw_id))
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not load the current draw podium; podium generation was refused.") from exc
    return rows


def _awarded_badges_for_draw(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
) -> list[dict[str, Any]]:
    try:
        rows = _safe_rows(
            supabase.table("player_badges")
            .select("id,context_type,context_id")
            .eq("club_id", str(club_id))
            .eq("context_type", "tournament")
            .execute()
        )
    except Exception as exc:
        raise RuntimeError("Could not verify tournament badge state; podium generation was refused.") from exc
    prefix = f"{tournament_id}:draw:{draw_id}:podium:"
    return [row for row in rows if str(row.get("context_id") or "").startswith(prefix)]


def _atomic_replace_podium(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    expected_draw_updated_at: str,
    expected_team_versions: list[dict[str, str]],
    expected_source_game_versions: list[dict[str, str]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        response = supabase.rpc(
            "admin_replace_tournament_draw_podium_cas",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament_id),
                "p_draw_id": str(draw_id),
                "p_expected_draw_updated_at": str(expected_draw_updated_at),
                "p_expected_teams": list(expected_team_versions),
                "p_expected_source_games": list(expected_source_game_versions),
                "p_podium": list(rows),
            },
        ).execute()
    except Exception as exc:
        detail = str(exc)
        if "JUPR_TOURNAMENT_DRAW_STALE" in detail or "JUPR_TOURNAMENT_PODIUM_SNAPSHOT_STALE" in detail:
            raise StaleTournamentAdminStateError(
                "The draw, team set, or source game set changed while the podium was being saved. Reload the Ops snapshot."
            ) from exc
        raise
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        saved = data.get("podium")
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        saved = data[0].get("podium")
    else:
        saved = None
    if not isinstance(saved, list):
        raise RuntimeError("Atomic tournament podium RPC returned no saved podium.")
    return [dict(row) for row in saved if isinstance(row, dict)]


def _podium_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _clean_text(row.get("id"), limit=120),
        "tournament_id": _clean_text(row.get("tournament_id"), limit=120),
        "draw_id": _clean_text(row.get("draw_id"), limit=120) or None,
        "placement": _safe_int(row.get("placement")),
        "team_id": _clean_text(row.get("team_id"), limit=120),
        "source": _clean_text(row.get("source"), limit=80),
        "created_at": row.get("created_at"),
    }


def _all_games_complete(games: list[dict[str, Any]], *, stage: str) -> bool:
    scoped = [row for row in games if _clean_text(row.get("stage"), limit=80).upper() == stage]
    return bool(scoped) and all(row.get("winner_team_id") and row.get("score_a") is not None and row.get("score_b") is not None for row in scoped)


def generate_admin_tournament_draw_podium(
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
    expected_source_game_versions: list[dict[str, Any]] | None = None,
    source: str = "next_tournament_admin_generate_podium",
    dry_run: bool = False,
    atomic: bool = False,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_GENERATE_PODIUM:
        raise ValueError(f"Type {CONFIRM_GENERATE_PODIUM} to generate the draw podium.")

    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_draw_id = _clean_text(draw_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    draw = _fetch_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not draw:
        raise ValueError("draw not found for this tournament")
    reviewed_draw_version = str(expected_draw_updated_at or "").strip()
    if atomic and not reviewed_draw_version:
        raise StaleTournamentAdminStateError(
            "A reviewed draw version is required for staging podium generation. Reload the Ops snapshot."
        )
    if reviewed_draw_version and str(draw.get("updated_at") or "") != reviewed_draw_version:
        raise StaleTournamentAdminStateError(
            "This tournament draw changed after it was reviewed. Reload the Ops snapshot before generating the podium."
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
    existing_podium = _podium_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if _awarded_badges_for_draw(
        supabase,
        club_id=str(club_id),
        tournament_id=clean_tournament_id,
        draw_id=clean_draw_id,
    ):
        raise ValueError(
            "Podium badges have already been awarded for this draw. Use badge recovery before replacing the podium."
        )
    playoff_games = [row for row in games if _clean_text(row.get("stage"), limit=80).upper() == "PLAYOFF"]
    round_robin_games = [row for row in games if _clean_text(row.get("stage"), limit=80).upper() == "ROUND_ROBIN"]

    podium_source = "PLAYOFF" if playoff_games else "ROUND_ROBIN"
    if playoff_games:
        podium = compute_podium_from_playoffs(playoff_games)
        if not podium:
            raise ValueError("Playoff final and bronze games must be scored before generating the podium.")
    else:
        if not _all_games_complete(round_robin_games, stage="ROUND_ROBIN"):
            raise ValueError("All round-robin games must be scored before generating a round-robin podium.")
        podium = compute_podium_from_rr(teams, round_robin_games)

    if not podium:
        raise ValueError("No podium placements could be computed for this draw.")

    now = _now_iso()
    rows: list[dict[str, Any]] = []
    for item in podium[:3]:
        placement = _safe_int(item.get("placement"))
        team_id = _clean_text(item.get("team_id"), limit=120)
        if placement is None or placement < 1 or placement > 3 or not team_id:
            continue
        rows.append(
            {
                "id": str(uuid.uuid4()),
                "tournament_id": clean_tournament_id,
                "draw_id": clean_draw_id,
                "placement": placement,
                "team_id": team_id,
                "source": podium_source,
                "created_at": now,
            }
        )
    if not rows:
        raise ValueError("No valid podium placements could be saved for this draw.")

    if dry_run:
        saved = [_podium_payload(row) for row in rows]
        return {
            "ok": True,
            "mode": "tournament_draw_podium_generate_preview",
            "dry_run": True,
            "write_count": 0,
            "draw_id": clean_draw_id,
            "podium_source": podium_source,
            "podium": saved,
            "warnings": [],
        }

    try:
        if atomic:
            saved_rows = _atomic_replace_podium(
                supabase,
                club_id=str(club_id),
                tournament_id=clean_tournament_id,
                draw_id=clean_draw_id,
                expected_draw_updated_at=reviewed_draw_version,
                expected_team_versions=reviewed_team_versions,
                expected_source_game_versions=reviewed_source_game_versions,
                rows=rows,
            )
        else:
            supabase.table("tournament_podium").delete().eq("tournament_id", clean_tournament_id).eq("draw_id", clean_draw_id).execute()
            saved_rows = _safe_rows(supabase.table("tournament_podium").insert(rows).execute())
    except StaleTournamentAdminStateError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface schema/cache problems as operator-visible API errors
        raise RuntimeError(f"Could not save draw-scoped podium: {exc.__class__.__name__}") from exc
    saved = [_podium_payload(row) for row in (saved_rows or rows)]

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="generate_tournament_draw_podium_admin",
        entity_type="tournament_event_draw",
        entity_id=clean_draw_id,
        before_json={"draw": _draw_payload(draw), "podium": [_podium_payload(row) for row in existing_podium]},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "draw": _draw_payload(draw),
            "podium_source": podium_source,
            "podium": saved,
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
    return {"ok": True, "mode": "tournament_draw_podium_generate", "draw_id": clean_draw_id, "podium_source": podium_source, "podium": saved, "warnings": warnings}
