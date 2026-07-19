from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import os

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_podium import award_tournament_trophies_from_podium, build_tournament_podium_candidates
from jupr_app.services.admin_tournament_draw_service import _draw_payload
from jupr_app.services.admin_tournament_podium_service import _podium_payload
from jupr_app.services.admin_tournament_service import TOURNAMENT_SELECT, _clean_text, _first_row, is_admin_tournament_admin_enabled

CONFIRM_AWARD_PODIUM = "AWARD PODIUM"


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


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
    except Exception as exc:
        raise RuntimeError("Could not verify the tournament draw; podium awards were refused.") from exc
    return rows[0] if rows else None


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
        raise RuntimeError("Could not load the draw podium; podium awards were refused.") from exc
    return rows


def award_admin_tournament_draw_podium(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_award_podium",
    dry_run: bool = False,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_AWARD_PODIUM:
        raise ValueError(f"Type {CONFIRM_AWARD_PODIUM} to award draw podium trophies.")

    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_draw_id = _clean_text(draw_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    draw = _fetch_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not draw:
        raise ValueError("draw not found for this tournament")
    podium_rows = _podium_for_draw(supabase, tournament_id=clean_tournament_id, draw_id=clean_draw_id)
    if not podium_rows:
        raise ValueError("Generate a draw-scoped podium before awarding trophies.")

    ctx = SimpleNamespace(supabase=supabase, club_id=str(club_id))
    candidates = build_tournament_podium_candidates(ctx, clean_tournament_id, str(tournament.get("name") or ""), draw_id=clean_draw_id)
    if not candidates:
        raise ValueError("No podium badge candidates could be built for this draw.")
    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_draw_podium_award_preview",
            "dry_run": True,
            "write_count": 0,
            "draw_id": clean_draw_id,
            "candidate_count": len(candidates),
            "awarded_count": 0,
            "badge_ids": sorted({str(candidate.badge_id) for candidate in candidates}),
            "warnings": [],
        }
    awarded = award_tournament_trophies_from_podium(ctx, clean_tournament_id, str(tournament.get("name") or ""), draw_id=clean_draw_id)

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="award_tournament_draw_podium_badges_admin",
        entity_type="tournament_event_draw",
        entity_id=clean_draw_id,
        before_json={"draw": _draw_payload(draw), "candidate_count": len(candidates)},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "draw": _draw_payload(draw),
            "podium": [_podium_payload(row) for row in podium_rows],
            "candidate_count": len(candidates),
            "awarded_count": len(awarded),
            "badge_ids": sorted({str(candidate.badge_id) for candidate in candidates}),
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
    return {
        "ok": True,
        "mode": "tournament_draw_podium_award",
        "draw_id": clean_draw_id,
        "candidate_count": len(candidates),
        "awarded_count": len(awarded),
        "warnings": warnings,
    }
