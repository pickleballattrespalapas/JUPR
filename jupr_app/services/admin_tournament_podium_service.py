from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os
import uuid

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournaments import compute_podium_from_playoffs, compute_podium_from_rr
from jupr_app.services.admin_tournament_draw_service import _draw_payload
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
    except Exception:
        rows = []
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
    except Exception:
        rows = []
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
    except Exception:
        rows = []
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
    except Exception:
        rows = []
    return rows


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
    source: str = "next_tournament_admin_generate_podium",
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

    teams = _teams_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    games = _games_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    existing_podium = _podium_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
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

    try:
        supabase.table("tournament_podium").delete().eq("tournament_id", clean_tournament_id).eq("draw_id", clean_draw_id).execute()
        saved_rows = _safe_rows(supabase.table("tournament_podium").insert(rows).execute())
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
